#!/usr/bin/env python3
"""Measure the 2024 Zbb event-level AK8 trigger scale factor.

The selection is evaluated with ROOT RDataFrame.  Data and MC efficiencies are
then accumulated in probe ``(pT, msoftdrop)`` bins, with the highest-pT AK8 jet
after muon cross-cleaning used as the probe coordinate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DERIVE_DIR = Path(__file__).resolve().parent
REPO_DIR = DERIVE_DIR.parents[2]
DEFAULT_CONFIG = DERIVE_DIR / "config.json"
DEFAULT_PROXY = "/afs/cern.ch/user/j/jiehan/x509up_u152815"
DEFAULT_DATA_MANIFEST = REPO_DIR / "jobs/zmm_2024_v15_DATA/job_manifest.json"
DEFAULT_MC_DATASET = "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
DEFAULT_MC_SAMPLE = "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8"
DEFAULT_GOLDEN = "/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions24/latest/Cert_Collisions2024_378981_386951_Golden.json"
PATHS = ("500", "380", "or", "both")
PATH_BRANCHES = {
    "reference": "HLT_IsoMu24",
    "500": "HLT_AK8PFJet500",
    "380": "HLT_AK8PFJet380_SoftDropMass30",
}
COMMON_BRANCHES = {
    "run",
    "luminosityBlock",
    "event",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_pfRelIso04_all",
    "Muon_tightId",
    "Muon_looseId",
    "Electron_pt",
    "Electron_eta",
    "Electron_cutBased",
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_btagUParTAK4B",
    "Jet_neHEF",
    "Jet_neEmEF",
    "Jet_chHEF",
    "Jet_chMultiplicity",
    "Jet_neMultiplicity",
    "FatJet_pt",
    "FatJet_eta",
    "FatJet_phi",
    "FatJet_neHEF",
    "FatJet_neEmEF",
    "FatJet_chHEF",
    "FatJet_chMultiplicity",
    "FatJet_neMultiplicity",
    *PATH_BRANCHES.values(),
}
BASE_BRANCHES = COMMON_BRANCHES - set(PATH_BRANCHES.values())
MASS_DEFINITIONS = {
    "msoftdrop": {
        "branches": ("FatJet_msoftdrop",),
        "expression": "FatJet_msoftdrop[probe_index]",
        "label": r"m_{SD}",
        "axis_label": r"AK8 jet $m_{SD}$ [GeV]",
        "root_label": "m_{SD}",
        "correction_input": "msd",
        "correction_names": ("zbb_2024_or_trigger_sf", "zbb_2024_or_trigger_sf_valid",
                              "zbb_2024_or_trigger_sf_status"),
    },
    "gptmass_x2p": {
        "branches": ("FatJet_mass", "FatJet_rawFactor", "FatJet_globalParT3_massCorrX2p"),
        "expression": "FatJet_mass[probe_index] * (1.f - FatJet_rawFactor[probe_index]) * "
                      "FatJet_globalParT3_massCorrX2p[probe_index]",
        "label": r"m_{GloParT}^{X2p}",
        "axis_label": r"AK8 jet $m_{GloParT}^{X2p}$ [GeV]",
        "root_label": "m_{GloParT}^{X2p}",
        "correction_input": "gptmass_x2p",
        "correction_names": ("zbb_2024_or_trigger_sf_gptmass_x2p",
                              "zbb_2024_or_trigger_sf_gptmass_x2p_valid",
                              "zbb_2024_or_trigger_sf_gptmass_x2p_status"),
    },
}


@dataclass
class SampleResult:
    stats: "Stats"
    report: dict[str, Any]
    records: dict[str, np.ndarray]


class MissingBranches(ValueError):
    def __init__(self, path: str, missing: Sequence[str]):
        self.path = path
        self.missing = list(missing)
        super().__init__(f"missing branches: {', '.join(self.missing)}")


class Stats:
    """Per-region weighted counters and their sum-of-squares."""

    def __init__(self, pt_bins: Sequence[float], mass_bins: Sequence[float]):
        self.pt_bins = np.asarray(pt_bins, dtype=float)
        self.mass_bins = np.asarray(mass_bins, dtype=float)
        shape = (2, len(pt_bins) - 1, len(mass_bins) - 1)
        self.total_n = np.zeros(shape, dtype=np.int64)
        self.total_w = np.zeros(shape, dtype=float)
        self.total_w2 = np.zeros(shape, dtype=float)
        self.pass_n = {name: np.zeros(shape, dtype=np.int64) for name in PATHS}
        self.pass_w = {name: np.zeros(shape, dtype=float) for name in PATHS}
        self.pass_w2 = {name: np.zeros(shape, dtype=float) for name in PATHS}

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.total_n.shape

    def fill(self, pt: np.ndarray, mass: np.ndarray, eta: np.ndarray, weight: np.ndarray,
             pass_500: np.ndarray, pass_380: np.ndarray) -> None:
        region = np.where(np.abs(eta) < 1.5, 0, 1)
        p_index, p_in_range = _bin_indices(self.pt_bins, pt)
        m_index, m_in_range = _bin_indices(self.mass_bins, mass)
        valid = (
            p_in_range & m_in_range &
            np.isfinite(weight) & np.isfinite(pt) & np.isfinite(mass) & np.isfinite(eta) &
            (np.abs(eta) < 2.4)
        )
        if not np.any(valid):
            return
        region, p_index, m_index, weight = region[valid], p_index[valid], m_index[valid], weight[valid]
        passed = {
            "500": pass_500[valid].astype(bool),
            "380": pass_380[valid].astype(bool),
        }
        passed["both"] = passed["500"] & passed["380"]
        passed["or"] = passed["500"] | passed["380"]
        np.add.at(self.total_n, (region, p_index, m_index), 1)
        np.add.at(self.total_w, (region, p_index, m_index), weight)
        np.add.at(self.total_w2, (region, p_index, m_index), weight * weight)
        for name, mask in passed.items():
            np.add.at(self.pass_n[name], (region[mask], p_index[mask], m_index[mask]), 1)
            np.add.at(self.pass_w[name], (region[mask], p_index[mask], m_index[mask]), weight[mask])
            np.add.at(self.pass_w2[name], (region[mask], p_index[mask], m_index[mask]), weight[mask] ** 2)

    def efficiency(self, name: str) -> dict[str, np.ndarray]:
        denominator = self.total_w
        valid = denominator > 1e-12
        value = np.full(self.shape, np.nan)
        error = np.full(self.shape, np.nan)
        value[valid] = self.pass_w[name][valid] / denominator[valid]
        valid &= np.isfinite(value) & (value >= 0.0) & (value <= 1.0)
        residual_w2 = self.pass_w2[name] * (1.0 - 2.0 * value) + self.total_w2 * value * value
        error[valid] = np.sqrt(np.maximum(0.0, residual_w2[valid])) / np.abs(denominator[valid])
        return {"value": value, "error": error, "valid": valid}

    def payload(self) -> dict[str, Any]:
        return {
            "total_n": self.total_n.tolist(),
            "total_weight": self.total_w.tolist(),
            "total_weight2": self.total_w2.tolist(),
            "pass_n": {name: values.tolist() for name, values in self.pass_n.items()},
            "pass_weight": {name: values.tolist() for name, values in self.pass_w.items()},
            "pass_weight2": {name: values.tolist() for name, values in self.pass_w2.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], pt_bins: Sequence[float], mass_bins: Sequence[float]) -> "Stats":
        result = cls(pt_bins, mass_bins)
        result.total_n = np.asarray(payload["total_n"], dtype=np.int64)
        result.total_w = np.asarray(payload["total_weight"], dtype=float)
        result.total_w2 = np.asarray(payload["total_weight2"], dtype=float)
        for name in PATHS:
            result.pass_n[name] = np.asarray(payload["pass_n"][name], dtype=np.int64)
            result.pass_w[name] = np.asarray(payload["pass_weight"][name], dtype=float)
            result.pass_w2[name] = np.asarray(payload["pass_weight2"][name], dtype=float)
        return result


RECORD_FIELDS = (
    "run", "luminosityBlock", "event", "muon_pt", "probe_pt", "probe_eta", "probe_mass",
    "ak8_count", "pass_500", "pass_380", "weight", "period",
)
RECORD_DTYPES = {
    "run": np.int64, "luminosityBlock": np.int64, "event": np.int64,
    "muon_pt": float, "probe_pt": float, "probe_eta": float, "probe_mass": float,
    "ak8_count": np.int64, "pass_500": bool, "pass_380": bool, "weight": float, "period": "U12",
}


def concat_records(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate selected-event records while keeping an explicit schema."""
    if not parts:
        return {name: np.asarray([], dtype=RECORD_DTYPES[name]) for name in RECORD_FIELDS}
    result = {}
    for name in RECORD_FIELDS:
        values = [np.asarray(part[name]) for part in parts if name in part]
        if not values:
            raise ValueError(f"record part is missing {name}")
        result[name] = np.concatenate(values)
    return result


def save_records(path: Path, records: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.asarray(records[name]) for name in RECORD_FIELDS})


def load_records(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in RECORD_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"record cache {path} is missing {', '.join(missing)}")
        return {name: np.asarray(payload[name]) for name in RECORD_FIELDS}


def deduplicate_records(records: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int, int]:
    """Deduplicate Data records and reject inconsistent duplicate event content."""
    count = len(records["run"])
    seen: dict[tuple[int, int, int], int] = {}
    keep: list[int] = []
    conflicts = 0
    compare = ("muon_pt", "probe_pt", "probe_eta", "probe_mass", "ak8_count", "pass_500", "pass_380")
    for index in range(count):
        key = (int(records["run"][index]), int(records["luminosityBlock"][index]), int(records["event"][index]))
        previous = seen.get(key)
        if previous is None:
            seen[key] = index
            keep.append(index)
            continue
        consistent = True
        for name in compare:
            first, second = records[name][previous], records[name][index]
            if np.issubdtype(np.asarray(first).dtype, np.floating):
                if not np.isclose(float(first), float(second), rtol=1e-6, atol=1e-5, equal_nan=True):
                    consistent = False
                    break
            elif first != second:
                consistent = False
                break
        if not consistent:
            conflicts += 1
    if conflicts:
        raise ValueError(f"found {conflicts} inconsistent duplicate Data event records")
    unique = {name: np.asarray(records[name])[keep] for name in RECORD_FIELDS}
    return unique, count, count - len(keep)


def _bin_indices(edges: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(edges, values, side="right") - 1
    # Include an event exactly on the final edge in the final bin.
    indices = np.where(np.isclose(values, edges[-1], rtol=0.0, atol=1e-9), len(edges) - 2, indices)
    in_range = (indices >= 0) & (indices < len(edges) - 1)
    return indices, in_range


def stats_from_records(records: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> Stats:
    stats = Stats(config["pt_bins"], config["mass_bins"])
    finite = np.isfinite(records["weight"])
    if np.any(finite):
        stats.fill(records["probe_pt"][finite], records["probe_mass"][finite], records["probe_eta"][finite],
                   records["weight"][finite], records["pass_500"][finite], records["pass_380"][finite])
    return stats


def rebuild_from_records(report: dict[str, Any], records: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> Stats:
    """Rebuild post-selection counters and diagnostics from globally unique records."""
    stats = stats_from_records(records, config)
    count = len(records["run"])
    weight = np.asarray(records["weight"], dtype=float)
    finite_weight = np.isfinite(weight)
    pt = np.asarray(records["probe_pt"], dtype=float)
    mass = np.asarray(records["probe_mass"], dtype=float)
    eta = np.asarray(records["probe_eta"], dtype=float)
    coordinate_valid = np.isfinite(pt) & np.isfinite(mass) & np.isfinite(eta) & (np.abs(eta) < 2.4)
    p_index, p_in_range = _bin_indices(stats.pt_bins, pt)
    m_index, m_in_range = _bin_indices(stats.mass_bins, mass)
    in_range = coordinate_valid & p_in_range & m_in_range
    report["events_after_unique"] = int(count)
    report["events_binned"] = int(np.count_nonzero(in_range & finite_weight))
    report["invalid_weights"] = int(np.count_nonzero(~finite_weight))
    report["invalid_coordinates"] = int(np.count_nonzero(~coordinate_valid & finite_weight))
    report["out_of_range_coordinates"] = int(np.count_nonzero(coordinate_valid & ~(p_in_range & m_in_range) & finite_weight))
    report["record_count"] = int(count)
    report["run_summary"] = {}
    report["ak8_multiplicity"] = {"barrel": {}, "endcap": {}}
    report["muon_pt"] = new_muon_diagnostic(config["muon_pt_bins"])
    periods = np.asarray(records["period"]).astype(str)
    passed_500 = np.asarray(records["pass_500"], dtype=bool)
    passed_380 = np.asarray(records["pass_380"], dtype=bool)
    passed_or = passed_500 | passed_380
    for index in range(count):
        run = str(int(records["run"][index]))
        summary = report["run_summary"].setdefault(run, {"events": 0, "pass_500": 0, "pass_380": 0, "pass_or": 0})
        summary["events"] += 1
        summary["pass_500"] += int(passed_500[index])
        summary["pass_380"] += int(passed_380[index])
        summary["pass_or"] += int(passed_or[index])
        if not finite_weight[index]:
            continue
        region = "barrel" if abs(float(eta[index])) < 1.5 else "endcap"
        item = report["ak8_multiplicity"][region].setdefault(str(int(records["ak8_count"][index])), {"n": 0, "sumw": 0.0, "sumw2": 0.0})
        item["n"] += 1
        item["sumw"] += float(weight[index])
        item["sumw2"] += float(weight[index]) ** 2
    # Vectorized muon diagnostic accumulation keeps the full-run merge cheap.
    fill_muon_diagnostic(report["muon_pt"], np.asarray(records["muon_pt"], dtype=float)[finite_weight],
                         weight[finite_weight], passed_500[finite_weight], passed_380[finite_weight])
    finalize_muon_diagnostic(report["muon_pt"])
    for period in np.unique(periods):
        mask = periods == period
        item = report.setdefault("period_summary", {}).setdefault(
            str(period), {"files": 0, "selected": 0, "after_golden": 0, "unique": 0,
                          "pass_500": 0, "pass_380": 0, "pass_or": 0})
        item["unique"] = int(np.count_nonzero(mask))
        item["pass_500"] = int(np.count_nonzero(passed_500[mask]))
        item["pass_380"] = int(np.count_nonzero(passed_380[mask]))
        item["pass_or"] = int(np.count_nonzero(passed_or[mask]))
    return stats


def new_muon_diagnostic(edges: Sequence[float]) -> dict[str, Any]:
    size = len(edges) - 1
    return {
        "edges": list(edges),
        "total_n": np.zeros(size, dtype=np.int64),
        "total_weight": np.zeros(size, dtype=float),
        "total_weight2": np.zeros(size, dtype=float),
        "pass_n": {name: np.zeros(size, dtype=np.int64) for name in PATHS},
        "pass_weight": {name: np.zeros(size, dtype=float) for name in PATHS},
        "pass_weight2": {name: np.zeros(size, dtype=float) for name in PATHS},
        "out_of_range": 0,
    }


def fill_muon_diagnostic(diagnostic: dict[str, Any], values: np.ndarray, weights: np.ndarray,
                         pass_500: np.ndarray, pass_380: np.ndarray) -> None:
    edges = np.asarray(diagnostic["edges"], dtype=float)
    indices = np.searchsorted(edges, values, side="right") - 1
    valid = np.isfinite(values) & np.isfinite(weights) & (indices >= 0) & (indices < len(edges) - 1)
    diagnostic["out_of_range"] += int(np.count_nonzero(~valid))
    if not np.any(valid):
        return
    indices, weights = indices[valid], weights[valid]
    passed = {"500": pass_500[valid].astype(bool), "380": pass_380[valid].astype(bool)}
    passed["both"] = passed["500"] & passed["380"]
    passed["or"] = passed["500"] | passed["380"]
    np.add.at(diagnostic["total_n"], indices, 1)
    np.add.at(diagnostic["total_weight"], indices, weights)
    np.add.at(diagnostic["total_weight2"], indices, weights * weights)
    for name, mask in passed.items():
        np.add.at(diagnostic["pass_n"][name], indices[mask], 1)
        np.add.at(diagnostic["pass_weight"][name], indices[mask], weights[mask])
        np.add.at(diagnostic["pass_weight2"][name], indices[mask], weights[mask] ** 2)


def finalize_muon_diagnostic(diagnostic: dict[str, Any]) -> None:
    for key in ("total_n", "total_weight", "total_weight2"):
        diagnostic[key] = diagnostic[key].tolist()
    for name in PATHS:
        for key in ("pass_n", "pass_weight", "pass_weight2"):
            diagnostic[key][name] = diagnostic[key][name].tolist()


def flatten(values: Sequence[str] | None) -> list[str]:
    return [item for value in values or [] for item in value.split(",") if item]


def manifest_files(path: str | os.PathLike[str], sample: str | None = None) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        jobs = json.load(handle).get("jobs", [])
    files = []
    for job in jobs:
        if sample is None or job.get("nickname") == sample or sample in str(job.get("nickname", "")):
            files.extend(job.get("inputs", []))
    return files


def das_files(dataset: str, limit: int | None = None) -> list[str]:
    os.environ.setdefault("X509_USER_PROXY", DEFAULT_PROXY)
    if not os.path.isfile(os.environ["X509_USER_PROXY"]):
        raise RuntimeError(f"DAS discovery requires proxy {os.environ['X509_USER_PROXY']}")
    command = ["dasgoclient", f"--query=file dataset={dataset}", f"--limit={0 if limit is None else limit}"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("dasgoclient is required to discover MC files; pass --mc-files or --mc-manifest") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"DAS query failed for {dataset}: {message}") from error
    files = []
    for line in result.stdout.splitlines():
        lfn = line.strip()
        if not lfn:
            continue
        files.append(
            lfn if lfn.startswith("root://") else f"root://cms-xrd-global.cern.ch//{lfn.lstrip('/')}"
        )
    if not files:
        raise RuntimeError(f"DAS returned no files for {dataset}")
    return files


def resolve_files(values: Sequence[str] | None, manifest: str | None, sample: str | None,
                  dataset: str | None = None, limit: int | None = None) -> list[str]:
    files = flatten(values)
    from_manifest = False
    if not files and manifest:
        if os.path.isfile(manifest):
            files = manifest_files(manifest, sample)
            from_manifest = True
        elif dataset is None:
            raise ValueError(f"input manifest does not exist: {manifest}")
    if from_manifest and dataset:
        components = [part for part in dataset.strip("/").split("/") if part]
        files = [path for path in files if all(component in path for component in components)]
    if not files and dataset:
        files = das_files(dataset, limit)
    unique_files: dict[str, str] = {}
    for path in files:
        unique_files.setdefault(input_identity(path), path)
    files = list(unique_files.values())
    if not files:
        raise ValueError("no input files found")
    return files


def input_identity(path: str) -> str:
    """Return the logical LFN so alternate XRootD endpoints compare equal."""
    value = str(path)
    if "://" in value:
        value = value.split("://", 1)[1]
        value = value.split("/", 1)[1] if "/" in value else value
    return value.lstrip("/")


def load_config(path: str | None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("mass_variable", "msoftdrop")
    if config["mass_variable"] not in MASS_DEFINITIONS:
        raise ValueError(f"unsupported mass_variable: {config['mass_variable']}")
    for key in ("pt_bins", "mass_bins", "muon_pt_bins", "muon_wp", "electron_wp", "ak4_btag_wp", "pu_file", "pu_correction"):
        if key not in config:
            raise ValueError(f"configuration is missing {key}")
    if any(right <= left for left, right in zip(config["pt_bins"], config["pt_bins"][1:])):
        raise ValueError("pt bins must be strictly increasing")
    if any(right <= left for left, right in zip(config["mass_bins"], config["mass_bins"][1:])):
        raise ValueError("mass bins must be strictly increasing")
    if any(right <= left for left, right in zip(config["muon_pt_bins"], config["muon_pt_bins"][1:])):
        raise ValueError("muon pT bins must be strictly increasing")
    muon_wp = config["muon_wp"]
    if (float(muon_wp.get("pt_min", 0)) != 55.0 or float(muon_wp.get("eta_max", 0)) != 2.4 or
            float(muon_wp.get("iso_max", 0)) != 0.15 or int(config["electron_wp"]) != 2):
        raise ValueError("config muon/electron working points must match the 2024 control-region definition")
    pu_file = config.get("pu_file")
    if pu_file and os.path.isfile(pu_file):
        config.setdefault("pu_file_sha256", hashlib.sha256(Path(pu_file).read_bytes()).hexdigest())
    return config


def mass_definition(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return MASS_DEFINITIONS[str(config.get("mass_variable", "msoftdrop"))]


def correction_names(config: Mapping[str, Any]) -> tuple[str, str, str]:
    return mass_definition(config)["correction_names"]


def trigger_branches(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "reference": str(config.get("reference_trigger", PATH_BRANCHES["reference"])),
        "500": str(config.get("trigger_500", PATH_BRANCHES["500"])),
        "380": str(config.get("trigger_380", PATH_BRANCHES["380"])),
    }


def load_golden(path: str | None) -> dict[int, list[tuple[int, int]]]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(run): [(int(start), int(end)) for start, end in ranges] for run, ranges in payload.items()}


def is_golden(golden: Mapping[int, Sequence[tuple[int, int]]], run: int, lumi: int) -> bool:
    return not golden or any(start <= lumi <= end for start, end in golden.get(run, []))


def root_module():
    try:
        import ROOT
    except ImportError as error:  # pragma: no cover - experiment runtime dependency
        raise RuntimeError("PyROOT is required") from error
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)
    return ROOT


def declare_helpers(ROOT: Any) -> None:
    include = str((DERIVE_DIR / "selection.hxx").resolve())
    ROOT.gInterpreter.Declare(f'#include "{include}"')


def branch_names(path: str, tree_name: str) -> set[str]:
    ROOT = root_module()
    handle = ROOT.TFile.Open(path)
    if not handle or handle.IsZombie():
        raise OSError(f"cannot open {path}")
    tree = handle.Get(tree_name)
    if not tree:
        handle.Close()
        raise ValueError(f"{path} has no {tree_name!r} tree")
    result = {branch.GetName() for branch in tree.GetListOfBranches()}
    handle.Close()
    return result


def entry_count(path: str, tree_name: str) -> int:
    ROOT = root_module()
    handle = ROOT.TFile.Open(path)
    if not handle or handle.IsZombie():
        raise OSError(f"cannot open {path}")
    tree = handle.Get(tree_name)
    if not tree:
        handle.Close()
        raise ValueError(f"{path} has no {tree_name!r} tree")
    count = int(tree.GetEntries())
    handle.Close()
    return count


def make_rdf(path: str, tree_name: str, config: Mapping[str, Any], is_data: bool, max_events: int) -> tuple[Any, set[str], list[str]]:
    ROOT = root_module()
    names = branch_names(path, tree_name)
    triggers = trigger_branches(config)
    required = BASE_BRANCHES | set(triggers.values()) | set(mass_definition(config)["branches"])
    if not is_data:
        required |= {"genWeight"}
        if "puWeight" not in names:
            required.add("Pileup_nTrueInt")
    missing = sorted(required - names)
    if missing:
        raise MissingBranches(path, missing)
    rdf = ROOT.RDataFrame(tree_name, path)
    if max_events >= 0:
        rdf = rdf.Range(max_events)
    rdf = rdf.Filter(triggers["reference"], "HLT_IsoMu24")
    rdf = rdf.Define(
        "muon_index",
        "boosted_2024_sf::selected_muon(Muon_pt, Muon_eta, Muon_pfRelIso04_all, Muon_tightId)",
    )
    rdf = rdf.Filter("muon_index >= 0", "exactly one tight isolated muon")
    rdf = rdf.Filter(
        "boosted_2024_sf::no_extra_muon(muon_index, Muon_pt, Muon_eta, Muon_pfRelIso04_all, Muon_looseId)",
        "extra loose muon veto",
    )
    rdf = rdf.Filter(
        "boosted_2024_sf::no_extra_electron(Electron_pt, Electron_eta, Electron_cutBased)",
        "extra loose electron veto",
    )
    rdf = rdf.Define("muon_eta", "Muon_eta[muon_index]").Define("muon_phi", "Muon_phi[muon_index]")
    rdf = rdf.Define("selected_muon_pt", "Muon_pt[muon_index]")
    rdf = rdf.Define(
        "n_bjets",
        "boosted_2024_sf::loose_b_jet_count(Jet_pt, Jet_eta, Jet_phi, Jet_btagUParTAK4B, "
        f"{float(config['ak4_btag_wp'])}, muon_eta, muon_phi, Jet_neHEF, Jet_neEmEF, Jet_chHEF, "
        "Jet_chMultiplicity, Jet_neMultiplicity)",
    )
    rdf = rdf.Filter("n_bjets >= 2", "two cross-cleaned UParT loose b jets")
    rdf = rdf.Define(
        "probe_index",
        "boosted_2024_sf::leading_ak8(FatJet_pt, FatJet_eta, FatJet_phi, muon_eta, muon_phi, "
        "FatJet_neHEF, FatJet_neEmEF, FatJet_chHEF, FatJet_chMultiplicity, FatJet_neMultiplicity)",
    )
    rdf = rdf.Filter("probe_index >= 0", "cross-cleaned AK8 probe")
    rdf = rdf.Define("probe_pt", "FatJet_pt[probe_index]")
    rdf = rdf.Define("probe_eta", "FatJet_eta[probe_index]")
    rdf = rdf.Define("probe_mass", mass_definition(config)["expression"])
    rdf = rdf.Define(
        "selected_ak8_count",
        "boosted_2024_sf::selected_ak8_count(FatJet_pt, FatJet_eta, FatJet_phi, muon_eta, muon_phi, "
        "FatJet_neHEF, FatJet_neEmEF, FatJet_chHEF, FatJet_chMultiplicity, FatJet_neMultiplicity)",
    )
    columns = [
        "run", "luminosityBlock", "event", "selected_muon_pt", "probe_pt", "probe_eta", "probe_mass", "selected_ak8_count",
        triggers["500"], triggers["380"],
    ]
    if is_data:
        columns.append(triggers["reference"])
    else:
        columns.append("genWeight")
        if "puWeight" in names:
            columns.append("puWeight")
        else:
            columns.append("Pileup_nTrueInt")
    return rdf, names, columns


def load_pu_weights(values: np.ndarray, config: Mapping[str, Any], cache: dict[str, Any]) -> np.ndarray:
    if "correction" not in cache:
        try:
            import correctionlib
        except ImportError as error:  # pragma: no cover - experiment runtime dependency
            raise RuntimeError("correctionlib is required when raw MC has no puWeight branch") from error
        pu_file = str(config["pu_file"])
        if not os.path.isfile(pu_file):
            raise OSError(f"pileup payload does not exist: {pu_file}")
        cache["correction"] = correctionlib.CorrectionSet.from_file(pu_file)[config["pu_correction"]]
    return np.asarray(cache["correction"].evaluate(np.asarray(values, dtype=float), "nominal"), dtype=float)


def parse_report(rdf_report: Any) -> dict[str, int]:
    result = {}
    for cut in rdf_report:
        try:
            result[str(cut.GetName())] = int(cut.GetPass())
        except AttributeError:
            continue
    return result


def period_from_path(path: str) -> str:
    match = re.search(r"Run2024([B-I])", path)
    return f"Run2024{match.group(1)}" if match else "unknown"


def process_sample(files: Sequence[str], *, is_data: bool, config: Mapping[str, Any], tree_name: str,
                   max_events: int, golden_json: str | None, skip_golden: bool) -> SampleResult:
    ROOT = root_module()
    declare_helpers(ROOT)
    triggers = trigger_branches(config)
    stats = Stats(config["pt_bins"], config["mass_bins"])
    report: dict[str, Any] = {
        "mass_variable": config.get("mass_variable", "msoftdrop"),
        "mass_formula": mass_definition(config)["expression"],
        "files_requested": len(files),
        "files_requested_list": list(files),
        "files_opened_list": [],
        "files_opened": 0,
        "entries_seen": 0,
        "events_after_reference": 0,
        "events_after_selection": 0,
        "events_after_golden": 0,
        "events_after_unique": 0,
        "events_binned": 0,
        "invalid_weights": 0,
        "invalid_coordinates": 0,
        "out_of_range_coordinates": 0,
        "missing_branches": {},
        "branch_diagnostics": {},
        "cutflow": {},
        "run_summary": {},
        "period_summary": {},
        "ak8_multiplicity": {"barrel": {}, "endcap": {}},
        "muon_pt": new_muon_diagnostic(config["muon_pt_bins"]),
        "warnings": [],
        "weight": "1.0 (data)" if is_data else "genWeight * puWeight",
    }
    golden = {} if skip_golden else load_golden(golden_json)
    seen: set[tuple[int, int, int]] = set()
    pu_cache: dict[str, Any] = {}
    record_parts: list[dict[str, np.ndarray]] = []
    remaining = max_events
    for path in files:
        if remaining == 0:
            break
        limit = -1 if remaining < 0 else remaining
        try:
            rdf, names, columns = make_rdf(path, tree_name, config, is_data, limit)
            raw_entries = entry_count(path, tree_name)
            report["entries_seen"] += raw_entries if limit < 0 else min(raw_entries, limit)
            arrays = rdf.AsNumpy(columns)
            report["files_opened"] += 1
            report["files_opened_list"].append(path)
            report["branch_diagnostics"][path] = {
                "reference": triggers["reference"] in names,
                "500": triggers["500"] in names,
                "380": triggers["380"] in names,
                "prescale_candidates": sorted(name for name in names if "prescale" in name.lower()),
            }
            report["events_after_selection"] += len(arrays["run"])
            period = period_from_path(path)
            period_summary = report["period_summary"].setdefault(
                period, {"files": 0, "selected": 0, "after_golden": 0, "unique": 0,
                         "pass_500": 0, "pass_380": 0, "pass_or": 0}
            )
            period_summary["files"] += 1
            period_summary["selected"] += len(arrays["run"])
            if remaining > 0:
                remaining -= min(raw_entries, remaining)
            try:
                cutflow = parse_report(rdf.Report())
                report["cutflow"][path] = cutflow
                report["events_after_reference"] += cutflow.get(triggers["reference"], len(arrays["run"]))
            except Exception as error:
                report["warnings"].append(f"cutflow unavailable for {path}: {error}")
                report["events_after_reference"] += len(arrays["run"])
        except MissingBranches as error:
            report["missing_branches"][path] = error.missing
            period = period_from_path(path)
            period_summary = report["period_summary"].setdefault(
                period, {"files": 0, "selected": 0, "after_golden": 0, "unique": 0,
                         "pass_500": 0, "pass_380": 0, "pass_or": 0, "missing_branches": []}
            )
            period_summary["files"] += 1
            period_summary.setdefault("missing_branches", []).extend(error.missing)
            report["warnings"].append(f"{path}: {error}")
            continue
        except (OSError, RuntimeError, ValueError) as error:
            report["warnings"].append(f"{path}: {error}")
            continue
        run = np.asarray(arrays["run"], dtype=np.int64)
        lumi = np.asarray(arrays["luminosityBlock"], dtype=np.int64)
        event = np.asarray(arrays["event"], dtype=np.int64)
        keep = np.ones(len(run), dtype=bool)
        if is_data and not skip_golden:
            keep = np.fromiter((is_golden(golden, int(run_value), int(lumi_value))
                                for run_value, lumi_value in zip(run, lumi)), dtype=bool, count=len(run))
        report["events_after_golden"] += int(np.count_nonzero(keep))
        period_summary["after_golden"] += int(np.count_nonzero(keep))
        unique_keep = np.zeros(len(run), dtype=bool)
        for i, key in enumerate(zip(run, lumi, event)):
            key_tuple = tuple(int(value) for value in key)
            if keep[i] and (not is_data or key_tuple not in seen):
                unique_keep[i] = True
                if is_data:
                    seen.add(key_tuple)
        report["events_after_unique"] += int(np.count_nonzero(unique_keep))
        period_summary["unique"] += int(np.count_nonzero(unique_keep))
        if not np.any(unique_keep):
            continue
        probe_pt = np.asarray(arrays["probe_pt"], dtype=float)[unique_keep]
        probe_eta = np.asarray(arrays["probe_eta"], dtype=float)[unique_keep]
        probe_mass = np.asarray(arrays["probe_mass"], dtype=float)[unique_keep]
        pass_500 = np.asarray(arrays[triggers["500"]], dtype=bool)[unique_keep]
        pass_380 = np.asarray(arrays[triggers["380"]], dtype=bool)[unique_keep]
        if is_data:
            weight = np.ones(len(probe_pt), dtype=float)
        else:
            gen_weight = np.asarray(arrays["genWeight"], dtype=float)[unique_keep]
            if "puWeight" in arrays:
                pu_weight = np.asarray(arrays["puWeight"], dtype=float)[unique_keep]
            else:
                pu_weight = load_pu_weights(np.asarray(arrays["Pileup_nTrueInt"], dtype=float)[unique_keep], config, pu_cache)
            weight = gen_weight * pu_weight
        valid_weight = np.isfinite(weight)
        report["invalid_weights"] += int(np.count_nonzero(~valid_weight))
        record_parts.append({
            "run": run[unique_keep],
            "luminosityBlock": lumi[unique_keep],
            "event": event[unique_keep],
            "muon_pt": np.asarray(arrays["selected_muon_pt"], dtype=float)[unique_keep],
            "probe_pt": probe_pt,
            "probe_eta": probe_eta,
            "probe_mass": probe_mass,
            "ak8_count": np.asarray(arrays["selected_ak8_count"], dtype=np.int64)[unique_keep],
            "pass_500": pass_500,
            "pass_380": pass_380,
            "weight": weight,
            "period": np.full(len(run[unique_keep]), period, dtype="U12"),
        })
        selected_muon_pt = np.asarray(arrays["selected_muon_pt"], dtype=float)[unique_keep][valid_weight]
        fill_muon_diagnostic(report["muon_pt"], selected_muon_pt, weight[valid_weight],
                             pass_500[valid_weight], pass_380[valid_weight])
        selected_counts = np.asarray(arrays["selected_ak8_count"], dtype=np.int64)[unique_keep][valid_weight]
        selected_eta = probe_eta[valid_weight]
        selected_weight = weight[valid_weight]
        for count, eta_value, event_weight in zip(selected_counts, selected_eta, selected_weight):
            region_name = "barrel" if abs(float(eta_value)) < 1.5 else "endcap"
            item = report["ak8_multiplicity"][region_name].setdefault(str(int(count)), {"n": 0, "sumw": 0.0, "sumw2": 0.0})
            item["n"] += 1
            item["sumw"] += float(event_weight)
            item["sumw2"] += float(event_weight) ** 2
        coordinate_valid = np.isfinite(probe_pt[valid_weight]) & np.isfinite(probe_mass[valid_weight]) & np.isfinite(probe_eta[valid_weight])
        coordinate_valid &= np.abs(probe_eta[valid_weight]) < 2.4
        p_index, p_in_range = _bin_indices(stats.pt_bins, probe_pt[valid_weight])
        m_index, m_in_range = _bin_indices(stats.mass_bins, probe_mass[valid_weight])
        in_range = coordinate_valid & p_in_range & m_in_range
        report["invalid_coordinates"] += int(np.count_nonzero(~coordinate_valid))
        report["out_of_range_coordinates"] += int(np.count_nonzero(coordinate_valid & ~in_range))
        report["events_binned"] += int(np.count_nonzero(in_range))
        stats.fill(probe_pt[valid_weight], probe_mass[valid_weight], probe_eta[valid_weight], weight[valid_weight],
                   pass_500[valid_weight], pass_380[valid_weight])
        run_values = run[unique_keep]
        period_summary["pass_500"] += int(np.count_nonzero(pass_500))
        period_summary["pass_380"] += int(np.count_nonzero(pass_380))
        period_summary["pass_or"] += int(np.count_nonzero(pass_500 | pass_380))
        for value in np.unique(run_values):
            key = str(int(value))
            run_mask = run_values == value
            summary = report["run_summary"].setdefault(key, {"events": 0, "pass_500": 0, "pass_380": 0, "pass_or": 0})
            summary["events"] += int(np.count_nonzero(run_mask))
            summary["pass_500"] += int(np.count_nonzero(pass_500[run_mask]))
            summary["pass_380"] += int(np.count_nonzero(pass_380[run_mask]))
            summary["pass_or"] += int(np.count_nonzero((pass_500 | pass_380)[run_mask]))
    if files and report["files_opened"] == 0:
        raise RuntimeError("no usable input files were opened")
    if report["events_after_selection"] > 0 and report["events_after_golden"] == 0 and not skip_golden:
        report["warnings"].append("golden JSON removed every selected event")
    records = concat_records(record_parts)
    report["record_count"] = int(len(records["run"]))
    finalize_muon_diagnostic(report["muon_pt"])
    report["coverage"] = {
        "files_opened_fraction": report["files_opened"] / report["files_requested"] if report["files_requested"] else 0.0,
        "files_failed": report["files_requested"] - report["files_opened"],
        "complete": report["files_opened"] == report["files_requested"] and not report["warnings"],
    }
    return SampleResult(stats, report, records)


def efficiency_interval(stats: Stats, name: str, one_sided_zero: bool = False) -> dict[str, np.ndarray]:
    result = stats.efficiency(name)
    lower = result["value"] - result["error"]
    upper = result["value"] + result["error"]
    zero = one_sided_zero & (stats.pass_n[name] == 0) & result["valid"]
    effective_n = np.divide(stats.total_w ** 2, stats.total_w2, out=np.zeros(stats.shape), where=stats.total_w2 > 0)
    upper_zero = 1.0 - np.exp(np.log(0.05) / np.maximum(1.0, effective_n))
    lower[zero] = 0.0
    upper[zero] = upper_zero[zero]
    # A zero numerator has a one-sided interval, never a zero uncertainty.
    result["error"][zero] = upper_zero[zero]
    return {**result, "lower": lower, "upper": upper}


def sf_value(data: Stats, mc: Stats, name: str) -> dict[str, np.ndarray]:
    d = efficiency_interval(data, name, one_sided_zero=True)
    m = efficiency_interval(mc, name)
    valid = d["valid"] & m["valid"] & (m["value"] > 0)
    value = np.full(data.shape, np.nan)
    error = np.full(data.shape, np.nan)
    upper = np.full(data.shape, np.nan)
    value[valid] = d["value"][valid] / m["value"][valid]
    positive = valid & (d["value"] > 0)
    error[positive] = value[positive] * np.hypot(
        d["error"][positive] / d["value"][positive], m["error"][positive] / m["value"][positive]
    )
    upper[valid & (d["value"] == 0)] = d["upper"][valid & (d["value"] == 0)] / m["value"][valid & (d["value"] == 0)]
    return {"value": value, "error": error, "upper": upper, "valid": valid}


def inclusive_stats(stats: Stats) -> Stats:
    """Return a two-region-shaped Stats object whose regions both contain the inclusive sum."""
    result = Stats(stats.pt_bins, stats.mass_bins)
    for name in ("total_n", "total_w", "total_w2"):
        setattr(result, name, np.repeat(np.sum(getattr(stats, name), axis=0, keepdims=True), 2, axis=0))
    for name in PATHS:
        for prefix in ("pass_n", "pass_w", "pass_w2"):
            values = getattr(stats, prefix)[name]
            getattr(result, prefix)[name] = np.repeat(np.sum(values, axis=0, keepdims=True), 2, axis=0)
    return result


def overlap_diagnostic(stats: Stats) -> dict[str, float | bool]:
    count_residual = stats.pass_n["or"] - stats.pass_n["500"] - stats.pass_n["380"] + stats.pass_n["both"]
    weight_residual = stats.pass_w["or"] - stats.pass_w["500"] - stats.pass_w["380"] + stats.pass_w["both"]
    return {
        "counts_max_abs_residual": float(np.max(np.abs(count_residual))),
        "sumw_max_abs_residual": float(np.max(np.abs(weight_residual))),
        "counts_identity_ok": bool(np.all(count_residual == 0)),
        # Weighted counters are accumulated in independent floating-point
        # arrays; allow the round-off level seen for large MC sums.
        "sumw_identity_ok": bool(np.allclose(weight_residual, 0.0, atol=1e-6, rtol=1e-12)),
    }


def rows(data: Stats, mc: Stats, low_effective_entries: float = 10.0) -> list[dict[str, Any]]:
    regions = (("inclusive", inclusive_stats(data), inclusive_stats(mc)),
               ("barrel", data, mc), ("endcap", data, mc))
    result = []
    for region_name, region_data, region_mc in regions:
        data_eff = {name: efficiency_interval(region_data, name, one_sided_zero=True) for name in PATHS}
        mc_eff = {name: efficiency_interval(region_mc, name) for name in PATHS}
        scales = {name: sf_value(region_data, region_mc, name) for name in PATHS}
        data_neff = np.divide(region_data.total_w ** 2, region_data.total_w2, out=np.zeros(region_data.shape), where=region_data.total_w2 > 0)
        mc_neff = np.divide(region_mc.total_w ** 2, region_mc.total_w2, out=np.zeros(region_mc.shape), where=region_mc.total_w2 > 0)
        for p in range(len(data.pt_bins) - 1):
            for m in range(len(data.mass_bins) - 1):
                index = (0 if region_name == "inclusive" else (0 if region_name == "barrel" else 1), p, m)
                row: dict[str, Any] = {
                    "eta_region": region_name,
                    "pt_low": float(data.pt_bins[p]),
                    "pt_high": float(data.pt_bins[p + 1]),
                    "mass_low": float(data.mass_bins[m]),
                    "mass_high": float(data.mass_bins[m + 1]),
                    "data_total_n": int(region_data.total_n[index]),
                    "data_total_weight": float(region_data.total_w[index]),
                    "data_total_weight2": float(region_data.total_w2[index]),
                    "mc_total_n": int(region_mc.total_n[index]),
                    "mc_total_weight": float(region_mc.total_w[index]),
                    "mc_total_weight2": float(region_mc.total_w2[index]),
                    "data_effective_entries": float(data_neff[index]),
                    "mc_effective_entries": float(mc_neff[index]),
                    "low_statistics": bool(
                        (data_neff[index] < low_effective_entries) or (mc_neff[index] < low_effective_entries)
                    ),
                    "paths": {},
                }
                for name in PATHS:
                    de, me, sf = data_eff[name], mc_eff[name], scales[name]
                    row["paths"][name] = {
                        "data_pass_n": int(region_data.pass_n[name][index]),
                        "data_pass_weight": float(region_data.pass_w[name][index]),
                        "data_pass_weight2": float(region_data.pass_w2[name][index]),
                        "mc_pass_n": int(region_mc.pass_n[name][index]),
                        "mc_pass_weight": float(region_mc.pass_w[name][index]),
                        "mc_pass_weight2": float(region_mc.pass_w2[name][index]),
                        "data_efficiency": _result_scalar(de, "value", index),
                        "data_error": _result_scalar(de, "error", index),
                        "data_interval_low": _result_scalar(de, "lower", index),
                        "data_interval_high": _result_scalar(de, "upper", index),
                        "mc_efficiency": _result_scalar(me, "value", index),
                        "mc_error": _result_scalar(me, "error", index),
                        "sf": _scalar(sf["value"][index]),
                        "sf_error": _scalar(sf["error"][index]),
                        "sf_upper": _scalar(sf["upper"][index]),
                        "data_valid": bool(de["valid"][index]),
                        "mc_valid": bool(me["valid"][index]),
                        "sf_valid": bool(sf["valid"][index]),
                        "invalid_reason": ("valid" if sf["valid"][index] else
                                            "data_efficiency_invalid" if not de["valid"][index] else
                                            "mc_efficiency_invalid" if not me["valid"][index] else
                                            "mc_efficiency_zero"),
                    }
                result.append(row)
    return result


def _scalar(value: Any) -> float | None:
    return float(value) if np.isfinite(value) else None


def _result_scalar(result: Mapping[str, Any], field: str, index: tuple[int, int, int]) -> float | None:
    return _scalar(result[field][index]) if bool(result["valid"][index]) else None


def write_csv(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "eta_region", "pt_low", "pt_high", "mass_low", "mass_high",
        "data_total_n", "data_total_weight", "data_total_weight2",
        "mc_total_n", "mc_total_weight", "mc_total_weight2",
        "data_effective_entries", "mc_effective_entries", "low_statistics",
    ]
    suffixes = ("data_pass_n", "data_pass_weight", "data_pass_weight2", "mc_pass_n", "mc_pass_weight", "mc_pass_weight2",
                "data_efficiency", "data_error", "data_interval_low", "data_interval_high", "mc_efficiency", "mc_error",
                "sf", "sf_error", "sf_upper", "data_valid", "mc_valid", "sf_valid", "invalid_reason")
    fields.extend(f"{name}_{suffix}" for name in PATHS for suffix in suffixes)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            output = {field: row.get(field) for field in fields if field in row}
            for name in PATHS:
                output.update({f"{name}_{suffix}": row["paths"][name].get(suffix) for suffix in suffixes})
            writer.writerow(output)


def write_metadata(output_dir: Path, report: Mapping[str, Any]) -> None:
    with open(output_dir / "config_used.json", "w", encoding="utf-8") as handle:
        json.dump(report.get("config", {}), handle, indent=2, sort_keys=True)
    inputs = report.get("inputs", {})
    for sample in ("data", "mc"):
        with open(output_dir / f"inputs_{sample}.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(str(value) for value in inputs.get(sample, [])))
            handle.write("\n")
        sample_report = report.get(sample, {}).get("report", {})
        for name in ("cutflow", "period_summary"):
            with open(output_dir / f"{name}_{sample}.json", "w", encoding="utf-8") as handle:
                json.dump(sample_report.get(name, {}), handle, indent=2, sort_keys=True)


def write_root(path: Path, data: Stats, mc: Stats, values: Sequence[Mapping[str, Any]],
               data_report: Mapping[str, Any] | None = None, mc_report: Mapping[str, Any] | None = None,
               mass_label: str = "m_{SD}") -> None:
    ROOT = root_module()
    output = ROOT.TFile(str(path), "RECREATE")
    for label, stats in (("data", data), ("mc", mc)):
        counters: dict[str, np.ndarray] = {
            "total_n": stats.total_n,
            "total_weight": stats.total_w,
            "total_weight2": stats.total_w2,
        }
        for name in PATHS:
            counters[f"{name}_pass_n"] = stats.pass_n[name]
            counters[f"{name}_pass_weight"] = stats.pass_w[name]
            counters[f"{name}_pass_weight2"] = stats.pass_w2[name]
        for counter_name, counter in counters.items():
            for region, region_name in enumerate(("barrel", "endcap")):
                hist = ROOT.TH2D(
                    f"{label}_{counter_name}_{region_name}", f"{label} {counter_name};AK8 p_{{T}} [GeV];{mass_label} [GeV]",
                    len(stats.pt_bins) - 1, array("d", stats.pt_bins),
                    len(stats.mass_bins) - 1, array("d", stats.mass_bins),
                )
                for p in range(len(stats.pt_bins) - 1):
                    for m in range(len(stats.mass_bins) - 1):
                        hist.SetBinContent(p + 1, m + 1, float(counter[region, p, m]))
                hist.Write()
        for name in PATHS:
            efficiency = efficiency_interval(stats, name, one_sided_zero=(label == "data"))
            for region, region_name in enumerate(("barrel", "endcap")):
                hist = ROOT.TH2D(
                    f"{label}_eff_{name}_{region_name}",
                    f"{label} {name} efficiency;AK8 p_{{T}} [GeV];{mass_label} [GeV]",
                    len(stats.pt_bins) - 1, array("d", stats.pt_bins),
                    len(stats.mass_bins) - 1, array("d", stats.mass_bins),
                )
                for p in range(len(stats.pt_bins) - 1):
                    for m in range(len(stats.mass_bins) - 1):
                        hist.SetBinContent(p + 1, m + 1, float(efficiency["value"][region, p, m]) if efficiency["valid"][region, p, m] else 0.0)
                        hist.SetBinError(p + 1, m + 1, float(efficiency["error"][region, p, m]) if efficiency["valid"][region, p, m] else 0.0)
                hist.Write()
                valid_hist = ROOT.TH2D(
                    f"{label}_valid_eff_{name}_{region_name}", f"{label} {name} efficiency validity;AK8 p_{{T}} [GeV];{mass_label} [GeV]",
                    len(stats.pt_bins) - 1, array("d", stats.pt_bins),
                    len(stats.mass_bins) - 1, array("d", stats.mass_bins),
                )
                for p in range(len(stats.pt_bins) - 1):
                    for m in range(len(stats.mass_bins) - 1):
                        valid_hist.SetBinContent(p + 1, m + 1, int(efficiency["valid"][region, p, m]))
                valid_hist.Write()
        sample_report = data_report if label == "data" else mc_report
        if sample_report:
            muon = sample_report.get("muon_pt")
            if muon:
                for name, values in (("total", muon["total_weight"]), ("or", muon["pass_weight"]["or"])):
                    hist = ROOT.TH1D(
                        f"{label}_muon_pt_{name}", f"{label} selected muon p_{{T}} ({name});p_{{T}}^{{#mu}} [GeV];sumw",
                        len(muon["edges"]) - 1, array("d", muon["edges"]),
                    )
                    for index, value in enumerate(values, start=1):
                        hist.SetBinContent(index, float(value))
                    hist.Write()
            for region_name in ("barrel", "endcap"):
                values_by_count = sample_report.get("ak8_multiplicity", {}).get(region_name, {})
                max_count = max((int(value) for value in values_by_count), default=0)
                hist = ROOT.TH1D(
                    f"{label}_ak8_multiplicity_{region_name}",
                    f"{label} selected AK8 multiplicity ({region_name});N_{{AK8}};events",
                    max_count + 1, -0.5, max_count + 0.5,
                )
                for count, values_by_weight in values_by_count.items():
                    hist.SetBinContent(int(count) + 1, float(values_by_weight.get("sumw", values_by_weight.get("n", 0))))
                hist.Write()
        inclusive = inclusive_stats(stats)
        for name in PATHS:
            efficiency = efficiency_interval(inclusive, name, one_sided_zero=(label == "data"))
            hist = ROOT.TH2D(
                f"{label}_eff_{name}_inclusive", f"{label} {name} efficiency (inclusive);AK8 p_{{T}} [GeV];{mass_label} [GeV]",
                len(stats.pt_bins) - 1, array("d", stats.pt_bins), len(stats.mass_bins) - 1, array("d", stats.mass_bins),
            )
            for p in range(len(stats.pt_bins) - 1):
                for m in range(len(stats.mass_bins) - 1):
                    if efficiency["valid"][0, p, m]:
                        hist.SetBinContent(p + 1, m + 1, float(efficiency["value"][0, p, m]))
                        hist.SetBinError(p + 1, m + 1, float(efficiency["error"][0, p, m]))
            hist.SetMinimum(0.0)
            hist.SetMaximum(1.0)
            hist.Write()
    for name in PATHS:
        sf = sf_value(data, mc, name)
        for region, region_name in enumerate(("barrel", "endcap")):
            hist = ROOT.TH2D(
                f"sf_{name}_{region_name}", f"{name} trigger SF;AK8 p_{{T}} [GeV];{mass_label} [GeV]",
                len(data.pt_bins) - 1, array("d", data.pt_bins), len(data.mass_bins) - 1, array("d", data.mass_bins),
            )
            for p in range(len(data.pt_bins) - 1):
                for m in range(len(data.mass_bins) - 1):
                    if sf["valid"][region, p, m]:
                        hist.SetBinContent(p + 1, m + 1, float(sf["value"][region, p, m]))
                        if np.isfinite(sf["error"][region, p, m]):
                            hist.SetBinError(p + 1, m + 1, float(sf["error"][region, p, m]))
            hist.Write()
            valid_hist = ROOT.TH2D(
                f"sf_valid_{name}_{region_name}", f"{name} SF validity;AK8 p_{{T}} [GeV];{mass_label} [GeV]",
                len(data.pt_bins) - 1, array("d", data.pt_bins), len(data.mass_bins) - 1, array("d", data.mass_bins),
            )
            for p in range(len(data.pt_bins) - 1):
                for m in range(len(data.mass_bins) - 1):
                    valid_hist.SetBinContent(p + 1, m + 1, int(sf["valid"][region, p, m]))
            valid_hist.Write()
        inclusive_sf = sf_value(inclusive_stats(data), inclusive_stats(mc), name)
        hist = ROOT.TH2D(
            f"sf_{name}_inclusive", f"{name} trigger SF (inclusive);AK8 p_{{T}} [GeV];{mass_label} [GeV]",
            len(data.pt_bins) - 1, array("d", data.pt_bins), len(data.mass_bins) - 1, array("d", data.mass_bins),
        )
        for p in range(len(data.pt_bins) - 1):
            for m in range(len(data.mass_bins) - 1):
                if inclusive_sf["valid"][0, p, m]:
                    hist.SetBinContent(p + 1, m + 1, float(inclusive_sf["value"][0, p, m]))
                    if np.isfinite(inclusive_sf["error"][0, p, m]):
                        hist.SetBinError(p + 1, m + 1, float(inclusive_sf["error"][0, p, m]))
        hist.Write()
        valid_hist = ROOT.TH2D(
            f"sf_valid_{name}_inclusive", f"{name} SF validity (inclusive);AK8 p_{{T}} [GeV];{mass_label} [GeV]",
            len(data.pt_bins) - 1, array("d", data.pt_bins), len(data.mass_bins) - 1, array("d", data.mass_bins),
        )
        for p in range(len(data.pt_bins) - 1):
            for m in range(len(data.mass_bins) - 1):
                valid_hist.SetBinContent(p + 1, m + 1, int(inclusive_sf["valid"][0, p, m]))
        valid_hist.Write()
    output.Write()
    output.Close()


def _plot_array(values: np.ndarray, valid: np.ndarray, edges: Mapping[str, Sequence[float]],
                path: Path, title: str, zmin: float, zmax: float, cmap: str = "viridis",
                errors: np.ndarray | None = None, mass_axis_label: str = r"AK8 jet $m_{SD}$ [GeV]") -> None:
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    masked = np.ma.masked_where(~valid, values)
    figure, axis = plt.subplots(figsize=(10.8, 7.2), constrained_layout=True)
    mesh = axis.pcolormesh(edges["pt"], edges["mass"], masked.T, cmap=cmap, vmin=zmin, vmax=zmax, shading="flat")
    colorbar_label = ("Efficiency" if "efficiency" in title else
                      "Statistical uncertainty" if "uncertainty" in title else "Scale factor")
    figure.colorbar(mesh, ax=axis, label=colorbar_label)
    for p in range(values.shape[0]):
        for m in range(values.shape[1]):
            if valid[p, m] and np.isfinite(values[p, m]):
                text = f"{values[p, m]:.2f}"
                if errors is not None and np.isfinite(errors[p, m]):
                    text += f"\n±{errors[p, m]:.2f}"
                axis.text(0.5 * (edges["pt"][p] + edges["pt"][p + 1]),
                          0.5 * (edges["mass"][m] + edges["mass"][m + 1]), text,
                          ha="center", va="center", fontsize=7, rotation=45, color="black")
    axis.set_xlabel(r"AK8 jet $p_T$ [GeV]", fontsize=14)
    axis.set_ylabel(mass_axis_label, fontsize=14)
    axis.tick_params(labelsize=11)
    plot_title = title.replace("CMS Preliminary — ", "", 1)
    axis.set_title(plot_title, loc="left", fontsize=14, pad=10)
    axis.text(0.0, 1.09, "CMS Preliminary", transform=axis.transAxes,
              ha="left", va="bottom", fontsize=14, fontweight="bold")
    axis.text(1.0, 1.09, "2024", transform=axis.transAxes, ha="right", va="bottom", fontsize=14)
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_pair(left: np.ndarray, left_valid: np.ndarray, right: np.ndarray, right_valid: np.ndarray,
               edges: Mapping[str, Sequence[float]], path: Path, left_title: str, right_title: str,
               zlabel: str, zmin: float, zmax: float,
               mass_axis_label: str = r"AK8 jet $m_{SD}$ [GeV]") -> None:
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 2, figsize=(16.0, 7.0), sharey=True, constrained_layout=True)
    meshes = []
    for axis, values, valid, title in zip(axes, (left, right), (left_valid, right_valid), (left_title, right_title)):
        mesh = axis.pcolormesh(edges["pt"], edges["mass"], np.ma.masked_where(~valid, values).T,
                               cmap="viridis", vmin=zmin, vmax=zmax, shading="flat")
        meshes.append(mesh)
        for p in range(values.shape[0]):
            for m in range(values.shape[1]):
                if valid[p, m] and np.isfinite(values[p, m]):
                    axis.text(0.5 * (edges["pt"][p] + edges["pt"][p + 1]),
                              0.5 * (edges["mass"][m] + edges["mass"][m + 1]), f"{values[p, m]:.2f}",
                              ha="center", va="center", fontsize=7, rotation=45, color="black")
        axis.set_title(title, fontsize=14, pad=8)
        axis.set_xlabel(r"AK8 jet $p_T$ [GeV]", fontsize=14)
        axis.tick_params(labelsize=11)
    axes[0].set_ylabel(mass_axis_label, fontsize=14)
    figure.colorbar(meshes[-1], ax=axes, label=zlabel, fraction=0.025, pad=0.02)
    figure.text(0.02, 0.995, "CMS Preliminary", ha="left", va="top", fontsize=14, fontweight="bold")
    figure.text(0.98, 0.995, "2024", ha="right", va="top", fontsize=14)
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_plots(output_dir: Path, data: Stats, mc: Stats, names: Sequence[str] = PATHS,
                mass_axis_label: str = r"AK8 jet $m_{SD}$ [GeV]") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = {"pt": data.pt_bins.tolist(), "mass": data.mass_bins.tolist()}
    inclusive_data, inclusive_mc = inclusive_stats(data), inclusive_stats(mc)
    for name in names:
        data_eff = efficiency_interval(inclusive_data, name, one_sided_zero=True)
        mc_eff = efficiency_interval(inclusive_mc, name)
        sf = sf_value(inclusive_data, inclusive_mc, name)
        _plot_pair(data_eff["value"][0], data_eff["valid"][0], mc_eff["value"][0], mc_eff["valid"][0],
                   edges, output_dir / f"efficiency_{name}_inclusive_data_mc",
                   f"Data: {name} efficiency", f"MC: {name} efficiency", "Efficiency", 0.0, 1.0,
                   mass_axis_label)
        for quantity, values, valid, errors, label, zmin, zmax, cmap in (
            ("data_efficiency", data_eff["value"][0], data_eff["valid"][0], data_eff["error"][0], "Data", 0.0, 1.0, "viridis"),
            ("mc_efficiency", mc_eff["value"][0], mc_eff["valid"][0], mc_eff["error"][0], "MC", 0.0, 1.0, "viridis"),
            ("sf", sf["value"][0], sf["valid"][0], sf["error"][0], "Data/MC", 0.0, 2.0, "RdYlBu_r"),
        ):
            title = f"CMS Preliminary — {label} {name} trigger {quantity.replace('_', ' ')}"
            _plot_array(values, valid, edges, output_dir / f"{quantity}_{name}_inclusive", title, zmin, zmax, cmap, errors, mass_axis_label)
        sf_error = sf["error"][0]
        error_valid = sf["valid"][0] & np.isfinite(sf_error)
        finite_errors = sf_error[error_valid]
        error_max = max(0.5, float(np.max(finite_errors) * 1.05)) if finite_errors.size else 1.0
        _plot_array(sf_error, error_valid, edges, output_dir / f"sf_stat_uncertainty_{name}_inclusive",
                    f"CMS Preliminary — Data/MC {name} trigger SF statistical uncertainty",
                    0.0, error_max, "magma", mass_axis_label=mass_axis_label)
        for region, region_name in enumerate(("barrel", "endcap")):
            data_region = efficiency_interval(data, name, one_sided_zero=True)
            mc_region = efficiency_interval(mc, name)
            sf_region = sf_value(data, mc, name)
            for quantity, values, valid, errors, label, zmin, zmax in (
                ("data_efficiency", data_region["value"][region], data_region["valid"][region], data_region["error"][region], "Data", 0.0, 1.0),
                ("mc_efficiency", mc_region["value"][region], mc_region["valid"][region], mc_region["error"][region], "MC", 0.0, 1.0),
                ("sf", sf_region["value"][region], sf_region["valid"][region], sf_region["error"][region], "Data/MC", 0.0, 2.0),
            ):
                title = f"CMS Preliminary — {label} {name} {region_name} {quantity.replace('_', ' ')}"
                cmap = "RdYlBu_r" if quantity == "sf" else "viridis"
                _plot_array(values, valid, edges, output_dir / f"{quantity}_{name}_{region_name}", title, zmin, zmax, cmap, errors, mass_axis_label)
            region_error = sf_region["error"][region]
            region_error_valid = sf_region["valid"][region] & np.isfinite(region_error)
            finite_region_errors = region_error[region_error_valid]
            region_error_max = max(0.5, float(np.max(finite_region_errors) * 1.05)) if finite_region_errors.size else 1.0
            _plot_array(region_error, region_error_valid, edges,
                        output_dir / f"sf_stat_uncertainty_{name}_{region_name}",
                        f"CMS Preliminary — Data/MC {name} trigger {region_name} SF statistical uncertainty",
                        0.0, region_error_max, "magma", mass_axis_label=mass_axis_label)


def write_diagnostics(output_dir: Path, data_report: Mapping[str, Any], mc_report: Mapping[str, Any]) -> None:
    """Write compact period, AK8-multiplicity and muon-pT diagnostics."""
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")

    def save(figure: Any, stem: str) -> None:
        figure.savefig(output_dir / f"{stem}.png", dpi=180)
        figure.savefig(output_dir / f"{stem}.pdf")
        plt.close(figure)

    periods = [f"Run2024{letter}" for letter in "CDEFGHI"]
    period_labels = list("CDEFGHI")
    summary = data_report.get("period_summary", {})
    selected = np.asarray([summary.get(period, {}).get("unique", 0) for period in periods], dtype=float)
    passed = np.asarray([summary.get(period, {}).get("pass_or", 0) for period in periods], dtype=float)
    efficiency = np.divide(passed, selected, out=np.full_like(passed, np.nan), where=selected > 0)
    figure, axis = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    axis.plot(period_labels, efficiency, "o-", color="tab:blue")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("OR efficiency")
    axis.set_xlabel("Data period (Run2024)")
    figure.text(0.02, 0.995, "CMS Preliminary", ha="left", va="top", fontsize=14, fontweight="bold")
    figure.text(0.50, 0.995, "Data control-region period diagnostic", ha="center", va="top", fontsize=14)
    figure.text(0.98, 0.995, "2024", ha="right", va="top", fontsize=14)
    save(figure, "period_efficiency_or_data")

    def multiplicity(report: Mapping[str, Any]) -> tuple[list[int], np.ndarray]:
        values = report.get("ak8_multiplicity", {})
        counts = sorted({int(key) for region in values.values() for key in region})
        data_values = np.asarray([
            sum(float(values.get(region, {}).get(str(count), {}).get("sumw", 0.0))
                for region in ("barrel", "endcap")) for count in counts
        ])
        return counts, data_values

    data_counts, data_mult = multiplicity(data_report)
    mc_counts, mc_mult = multiplicity(mc_report)
    counts = sorted(set(data_counts) | set(mc_counts))
    data_mult = np.asarray([data_mult[data_counts.index(count)] if count in data_counts else 0.0 for count in counts])
    mc_mult = np.asarray([mc_mult[mc_counts.index(count)] if count in mc_counts else 0.0 for count in counts])
    if np.sum(data_mult) > 0:
        data_mult /= np.sum(data_mult)
    if np.sum(mc_mult) > 0:
        mc_mult /= np.sum(mc_mult)
    x = np.arange(len(counts), dtype=float)
    width = 0.38
    figure, axis = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    axis.bar(x - width / 2.0, data_mult, width, label="Data")
    axis.bar(x + width / 2.0, mc_mult, width, label="MC")
    axis.set_xticks(x, [str(count) for count in counts])
    axis.set_xlabel("Selected AK8 multiplicity")
    axis.set_ylabel("Fraction of selected events")
    figure.text(0.02, 0.995, "CMS Preliminary", ha="left", va="top", fontsize=14, fontweight="bold")
    figure.text(0.50, 0.995, "Selected AK8 multiplicity diagnostic", ha="center", va="top", fontsize=14)
    figure.text(0.98, 0.995, "2024", ha="right", va="top", fontsize=14)
    axis.legend(frameon=False)
    save(figure, "ak8_multiplicity_data_mc")

    def muon_efficiency(report: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        diagnostic = report.get("muon_pt", {})
        edges = np.asarray(diagnostic.get("edges", []), dtype=float)
        total = np.asarray(diagnostic.get("total_weight", []), dtype=float)
        passed = np.asarray(diagnostic.get("pass_weight", {}).get("or", []), dtype=float)
        return 0.5 * (edges[:-1] + edges[1:]), np.divide(passed, total, out=np.full_like(total, np.nan), where=total > 0)

    data_muon_pt, data_muon_eff = muon_efficiency(data_report)
    mc_muon_pt, mc_muon_eff = muon_efficiency(mc_report)
    figure, axis = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    axis.plot(data_muon_pt, data_muon_eff, "o-", label="Data")
    axis.plot(mc_muon_pt, mc_muon_eff, "s--", label="MC")
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel(r"Selected muon $p_T$ [GeV]")
    axis.set_ylabel("OR efficiency")
    figure.text(0.02, 0.995, "CMS Preliminary", ha="left", va="top", fontsize=14, fontweight="bold")
    figure.text(0.50, 0.995, "Muon $p_T$ control-region diagnostic", ha="center", va="top", fontsize=14)
    figure.text(0.98, 0.995, "2024", ha="right", va="top", fontsize=14)
    axis.legend(frameon=False)
    save(figure, "muon_pt_efficiency_or_data_mc")


def write_correctionlib(output_dir: Path, data: Stats, mc: Stats, config: Mapping[str, Any]) -> dict[str, Any]:
    """Write schema-v2 corrections; invalid bins remain -1 and are never clamped."""
    try:
        from correctionlib import schemav2 as cs
    except ImportError as error:  # pragma: no cover - runtime LCG dependency
        raise RuntimeError("correctionlib is required to write the standard JSON payload") from error
    definition = mass_definition(config)
    sf_name, validity_name, status_name = correction_names(config)
    mass_input = str(definition["correction_input"])
    inclusive_data, inclusive_mc = inclusive_stats(data), inclusive_stats(mc)
    sf = sf_value(inclusive_data, inclusive_mc, "or")
    data_eff = efficiency_interval(inclusive_data, "or", one_sided_zero=True)
    low = float(config.get("low_effective_entries", 10.0))
    data_neff = np.divide(inclusive_data.total_w ** 2, inclusive_data.total_w2,
                          out=np.zeros(inclusive_data.shape), where=inclusive_data.total_w2 > 0)[0]
    mc_neff = np.divide(inclusive_mc.total_w ** 2, inclusive_mc.total_w2,
                        out=np.zeros(inclusive_mc.shape), where=inclusive_mc.total_w2 > 0)[0]
    valid = sf["valid"][0]
    lowstat = valid & ((data_neff < low) | (mc_neff < low))

    def flatten(values: np.ndarray, valid_mask: np.ndarray, variation: str) -> list[float]:
        output: list[float] = []
        for p in range(values.shape[0]):
            for m in range(values.shape[1]):
                output.append(float(values[p, m]) if valid_mask[p, m] and np.isfinite(values[p, m]) else -1.0)
        return output

    def node(values: np.ndarray, valid_mask: np.ndarray) -> Any:
        return cs.MultiBinning(nodetype="multibinning", inputs=["pt", mass_input],
                               edges=[list(map(float, config["pt_bins"])), list(map(float, config["mass_bins"]))],
                               content=flatten(values, valid_mask, "nominal"), flow="error")

    nominal = np.where(valid, sf["value"][0], -1.0)
    stat_up = np.where(valid, sf["value"][0] + np.nan_to_num(sf["error"][0], nan=0.0), -1.0)
    stat_down = np.where(valid, np.maximum(0.0, sf["value"][0] - np.nan_to_num(sf["error"][0], nan=0.0)), -1.0)
    zero_data = valid & (data_eff["value"][0] == 0.0)
    stat_up[zero_data] = sf["upper"][0][zero_data]
    stat_down[zero_data] = 0.0
    variation = cs.Category(nodetype="category", input="variation", content=[
        cs.CategoryItem(key="nominal", value=node(nominal, valid)),
        cs.CategoryItem(key="stat_up", value=node(stat_up, valid)),
        cs.CategoryItem(key="stat_down", value=node(stat_down, valid)),
    ], default=-1.0)
    validity = node(valid.astype(float), np.ones(valid.shape, dtype=bool))
    status = node(np.where(~valid, 0.0, np.where(lowstat, 2.0, 1.0)), np.ones(valid.shape, dtype=bool))
    payload = cs.CorrectionSet(schema_version=2,
        description=("Complete selected 2024 Muon0/Muon1 Run2024C-I and requested TTtoLNu2Q "
                     "NanoAODv15 inputs. Event-level HLT_AK8PFJet500 OR "
                     "HLT_AK8PFJet380_SoftDropMass30 SF from the raw NanoAOD "
                     "semileptonic ttbar control region; statistical variations only. "
                     "NanoAOD does not encode trigger prescales, so per-period prescale "
                     "verification remains required. -1 denotes an invalid bin; check "
                     "the validity correction before applying."),
        corrections=[
            cs.Correction(name=sf_name, version=1,
                          description="Inclusive event-level OR trigger SF; apply once per event.",
                          inputs=[cs.Variable(name="pt", type="real"), cs.Variable(name=mass_input, type="real"), cs.Variable(name="variation", type="string")],
                          output=cs.Variable(name="scale_factor", type="real"), data=variation),
            cs.Correction(name=validity_name, version=1,
                          description="1 for a statistically valid SF bin, 0 otherwise.",
                          inputs=[cs.Variable(name="pt", type="real"), cs.Variable(name=mass_input, type="real")],
                          output=cs.Variable(name="valid", type="real"), data=validity),
            cs.Correction(name=status_name, version=1,
                          description="0 invalid, 1 valid, 2 valid but low effective statistics.",
                          inputs=[cs.Variable(name="pt", type="real"), cs.Variable(name=mass_input, type="real")],
                          output=cs.Variable(name="status", type="real"), data=status),
        ])
    text = payload.model_dump_json(indent=2)
    json_path = output_dir / "trigger_sf_correctionlib.json"
    gzip_path = output_dir / "trigger_sf_correctionlib.json.gz"
    json_path.write_text(text + "\n", encoding="utf-8")
    with gzip.open(gzip_path, "wt", encoding="utf-8") as handle:
        handle.write(text + "\n")
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    return {"json": str(json_path), "json_gz": str(gzip_path), "sha256": digest,
            "correction": sf_name, "mass_input": mass_input, "invalid_sentinel": -1.0,
            "validity_correction": validity_name, "status_correction": status_name}


def report_from_results(config: Mapping[str, Any], data: SampleResult, mc: SampleResult,
                        data_files: Sequence[str], mc_files: Sequence[str], mc_dataset: str | None) -> dict[str, Any]:
    threshold = float(config.get("low_effective_entries", 10.0))
    expected_periods = [f"Run2024{letter}" for letter in "CDEFGHI"]
    observed_periods = sorted(data.report["period_summary"])
    unconfirmed_periods = sorted(set(expected_periods) - set(observed_periods))
    if "unknown" in observed_periods:
        unconfirmed_periods.append("unknown")
    unconfirmed_periods = sorted(set(unconfirmed_periods) | {
        period for period, values in data.report["period_summary"].items() if values.get("missing_branches")
    })
    prescale_candidates = sorted({name for values in data.report.get("branch_diagnostics", {}).values()
                                  for name in values.get("prescale_candidates", [])})
    return {
        "year": 2024,
        "config": dict(config),
        "mass_definition": {
            "variable": config.get("mass_variable", "msoftdrop"),
            "formula": mass_definition(config)["expression"],
            "branches": list(mass_definition(config)["branches"]),
            "correction_input": mass_definition(config)["correction_input"],
        },
        "paths": trigger_branches(config),
        "inputs": {"data": list(data_files), "mc": list(mc_files), "mc_dataset": mc_dataset},
        "normalization": "MC cross section normalization is omitted because it cancels in the efficiency ratio",
        "control_region_composition": {
            "data_purity_estimate": None,
            "status": "not measured by this tool; validate semileptonic ttbar purity with 2024 control-region studies",
        },
        "selection": {
            "reference_trigger": trigger_branches(config)["reference"],
            "target_trigger_or": [trigger_branches(config)["500"], trigger_branches(config)["380"]],
            "muon": "exactly one: pt > 55 GeV, abs(eta) < 2.4, pfRelIso04_all < 0.15, tightId",
            "extra_muon_veto": "pt > 10 GeV, abs(eta) < 2.4, looseId, pfRelIso04_all < 0.25",
            "extra_electron_veto": "pt > 10 GeV, abs(eta) < 2.5, cutBased >= 2",
            "ak4": "pt > 30 GeV, abs(eta) < 2.4, v15 tight central ID, dR(muon,jet) > 0.4, UParT loose",
            "ak8": "pt > 200 GeV, abs(eta) < 2.4, v15 tight central ID, dR(muon,jet) > 0.8",
            "probe": "highest-pT selected AK8 jet; no mass or bb-score cut",
            "eta_regions": {"barrel": "abs(eta) < 1.5", "endcap": "1.5 <= abs(eta) < 2.4"},
        },
        "bin_edges": {"pt": config["pt_bins"], "mass": config["mass_bins"]},
        "trigger_diagnostics": {
            "measurement_uses_event_bits": True,
            "object_matching_applied": False,
            "prescale_status": "not encoded in NanoAOD; verify per Run2024 period before physics use",
            "prescale_candidates": prescale_candidates,
            "prescale_verified": False,
            "expected_data_periods": expected_periods,
            "observed_data_periods": observed_periods,
            "unconfirmed_data_periods": unconfirmed_periods,
        },
        "data": {"report": data.report, "stats": data.stats.payload()},
        "mc": {"report": mc.report, "stats": mc.stats.payload()},
        "bins": rows(data.stats, mc.stats, threshold),
        "overlap": {"data": overlap_diagnostic(data.stats), "mc": overlap_diagnostic(mc.stats)},
        "warnings": [*data.report["warnings"], *mc.report["warnings"]],
    }


def merge_reports(paths: Sequence[str]) -> dict[str, Any]:
    if not paths:
        raise ValueError("merge requires at least one report")
    with open(paths[0], encoding="utf-8") as handle:
        merged = json.load(handle)
    pt_bins, mass_bins = merged["bin_edges"]["pt"], merged["bin_edges"]["mass"]
    data_stats = Stats.from_payload(merged["data"]["stats"], pt_bins, mass_bins)
    mc_stats = Stats.from_payload(merged["mc"]["stats"], pt_bins, mass_bins)

    def merge_details(target: dict[str, Any], source: Mapping[str, Any]) -> None:
        for key in ("files_requested", "files_opened", "entries_seen", "events_after_reference", "events_after_selection",
                    "events_after_golden", "events_after_unique", "events_binned", "invalid_weights",
                    "invalid_coordinates", "out_of_range_coordinates"):
            target[key] = int(target.get(key, 0)) + int(source.get(key, 0))
        for key in ("files_requested_list", "files_opened_list"):
            target.setdefault(key, [])
            target[key].extend(source.get(key, []))
        target["coverage"] = {
            "files_opened_fraction": target["files_opened"] / target["files_requested"] if target["files_requested"] else 0.0,
            "files_failed": target["files_requested"] - target["files_opened"],
            "complete": target["files_opened"] == target["files_requested"] and not target.get("warnings"),
        }
        target.setdefault("warnings", []).extend(source.get("warnings", []))
        target["coverage"]["complete"] = target["files_opened"] == target["files_requested"] and not target.get("warnings")
        target.setdefault("missing_branches", {}).update(source.get("missing_branches", {}))
        target.setdefault("branch_diagnostics", {}).update(source.get("branch_diagnostics", {}))
        target.setdefault("cutflow", {}).update(source.get("cutflow", {}))
        for key in ("run_summary", "period_summary"):
            combined = target.setdefault(key, {})
            for label, values in source.get(key, {}).items():
                if label not in combined:
                    combined[label] = dict(values)
                    continue
                for name, value in values.items():
                    if isinstance(value, list):
                        combined[label].setdefault(name, []).extend(value)
                    else:
                        combined[label][name] = combined[label].get(name, 0) + value
        combined_multiplicity = target.setdefault("ak8_multiplicity", {"barrel": {}, "endcap": {}})
        for region, counts in source.get("ak8_multiplicity", {}).items():
            for count, values in counts.items():
                item = combined_multiplicity[region].setdefault(count, {"n": 0, "sumw": 0.0, "sumw2": 0.0})
                item["n"] += int(values.get("n", 0))
                item["sumw"] += float(values.get("sumw", 0.0))
                item["sumw2"] += float(values.get("sumw2", 0.0))
        source_muon = source.get("muon_pt")
        if source_muon:
            target_muon = target.setdefault("muon_pt", new_muon_diagnostic(source_muon["edges"]))
            if target_muon["edges"] != source_muon["edges"]:
                raise ValueError("incompatible muon pT diagnostic bins")
            for key in ("total_n", "total_weight", "total_weight2"):
                target_muon[key] = (np.asarray(target_muon[key]) + np.asarray(source_muon[key])).tolist()
            for name in PATHS:
                for key in ("pass_n", "pass_weight", "pass_weight2"):
                    target_muon[key][name] = (np.asarray(target_muon[key][name]) +
                                               np.asarray(source_muon[key][name])).tolist()
            target_muon["out_of_range"] += int(source_muon.get("out_of_range", 0))

    for path in paths[1:]:
        with open(path, encoding="utf-8") as handle:
            item = json.load(handle)
        if (item.get("mass_definition") != merged.get("mass_definition") or
                item["bin_edges"] != merged["bin_edges"] or item["paths"] != merged["paths"] or
                item.get("config") != merged.get("config")):
            raise ValueError(f"incompatible mass definition, binning or configuration: {path}")
        for target, source in ((data_stats, item["data"]["stats"]), (mc_stats, item["mc"]["stats"])):
            other = Stats.from_payload(source, pt_bins, mass_bins)
            target.total_n += other.total_n
            target.total_w += other.total_w
            target.total_w2 += other.total_w2
            for name in PATHS:
                target.pass_n[name] += other.pass_n[name]
                target.pass_w[name] += other.pass_w[name]
                target.pass_w2[name] += other.pass_w2[name]
        merge_details(merged["data"]["report"], item["data"]["report"])
        merge_details(merged["mc"]["report"], item["mc"]["report"])
        if "inputs" in merged and "inputs" in item:
            for sample in ("data", "mc"):
                merged["inputs"].setdefault(sample, []).extend(item["inputs"].get(sample, []))
        merged["warnings"].extend(item.get("warnings", []))
    for sample in ("data", "mc"):
        inputs = merged.get("inputs", {}).get(sample, [])
        duplicate_inputs = len(inputs) - len({input_identity(path) for path in inputs})
        if duplicate_inputs:
            raise ValueError(f"merged {sample} input list contains {duplicate_inputs} duplicate files")
    # Rebuild from the selected-event caches when available.  This is required
    # for Muon0/Muon1 overlap: chunk-local de-duplication is not sufficient.
    record_paths = {"data": [], "mc": []}
    for report_path in paths:
        with open(report_path, encoding="utf-8") as handle:
            item = json.load(handle)
        processing = item.get("processing", {})
        for sample in record_paths:
            cache_path = processing.get(f"{sample}_records_file")
            if cache_path:
                record_paths[sample].append(str(cache_path))
    if all(record_paths.values()) and all(os.path.isfile(path) for paths_ in record_paths.values() for path in paths_):
        data_records = concat_records([load_records(path) for path in record_paths["data"]])
        unique_data, before, removed = deduplicate_records(data_records)
        mc_records = concat_records([load_records(path) for path in record_paths["mc"]])
        data_stats = rebuild_from_records(merged["data"]["report"], unique_data, merged["config"])
        mc_stats = rebuild_from_records(merged["mc"]["report"], mc_records, merged["config"])
        merged["data"]["report"]["record_count_before_global_deduplication"] = int(before)
        merged["data"]["report"]["duplicates_removed"] = int(removed)
        merged["mc"]["report"]["duplicates_removed"] = 0
        merged.setdefault("processing", {})["global_data_deduplication"] = True
    else:
        merged.setdefault("processing", {})["global_data_deduplication"] = False
        merged.setdefault("warnings", []).append("selected-event caches unavailable; merged Data statistics are only chunk-local deduplicated")
    merged["data"]["stats"] = data_stats.payload()
    merged["mc"]["stats"] = mc_stats.payload()
    merged["bins"] = rows(data_stats, mc_stats, float(merged.get("config", {}).get("low_effective_entries", 10.0)))
    merged["overlap"] = {"data": overlap_diagnostic(data_stats), "mc": overlap_diagnostic(mc_stats)}
    merged.setdefault("processing", {})["merged_reports"] = len(paths)
    pu_file = merged.get("config", {}).get("pu_file")
    if pu_file and os.path.isfile(pu_file):
        merged.setdefault("config", {})["pu_file_sha256"] = hashlib.sha256(Path(pu_file).read_bytes()).hexdigest()
    observed_periods = sorted(merged["data"]["report"].get("period_summary", {}))
    diagnostics = merged.setdefault("trigger_diagnostics", {})
    diagnostics["observed_data_periods"] = observed_periods
    expected_periods = diagnostics.get("expected_data_periods", [f"Run2024{letter}" for letter in "CDEFGHI"])
    diagnostics["unconfirmed_data_periods"] = sorted(
        (set(expected_periods) - set(observed_periods)) |
        ({"unknown"} if "unknown" in observed_periods else set()) |
        {period for period, values in merged["data"]["report"].get("period_summary", {}).items()
         if values.get("missing_branches")}
    )
    diagnostics["prescale_candidates"] = sorted({name for values in merged["data"]["report"].get("branch_diagnostics", {}).values()
                                                 for name in values.get("prescale_candidates", [])})
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="measure Data/MC efficiency and SF")
    measure.add_argument("--data-files", action="append")
    measure.add_argument("--mc-files", action="append")
    measure.add_argument("--skip-data", action="store_true")
    measure.add_argument("--skip-mc", action="store_true")
    measure.add_argument("--data-manifest", default=str(DEFAULT_DATA_MANIFEST))
    measure.add_argument("--mc-manifest", help="manifest containing the requested ttbar dataset")
    measure.add_argument("--mc-sample", default=DEFAULT_MC_SAMPLE, help="manifest nickname filter")
    measure.add_argument("--mc-dataset", default=DEFAULT_MC_DATASET,
                         help="DAS dataset used when --mc-files and a matching manifest are absent")
    measure.add_argument("--golden-json", default=DEFAULT_GOLDEN)
    measure.add_argument("--skip-golden", action="store_true")
    measure.add_argument("--config", default=str(DEFAULT_CONFIG))
    measure.add_argument("--tree-name", default="Events")
    measure.add_argument("--max-events", type=int, default=-1)
    measure.add_argument("--max-files", type=int, default=-1)
    measure.add_argument("--file-start", type=int, default=0,
                         help="zero-based offset applied to both resolved file lists")
    measure.add_argument("--output-dir", default=str(DERIVE_DIR / "results"))
    measure.add_argument("--no-plots", action="store_true")
    merge = sub.add_parser("merge", help="merge JSON reports from chunked measurements")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output-dir", required=True)
    plot = sub.add_parser("plot", help="recreate plots from a JSON report")
    plot.add_argument("--report", required=True)
    plot.add_argument("--output-dir", required=True)
    inspect = sub.add_parser("inspect", help="inspect required branches in input files")
    inspect.add_argument("--files", action="append", required=True)
    inspect.add_argument("--config", default=str(DEFAULT_CONFIG))
    inspect.add_argument("--tree-name", default="Events")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        config = load_config(args.config)
        required = BASE_BRANCHES | set(trigger_branches(config).values()) | set(mass_definition(config)["branches"])
        for path in flatten(args.files):
            names = branch_names(path, args.tree_name)
            print(json.dumps({
                "file": path,
                "entries": entry_count(path, args.tree_name),
                "mass_variable": config.get("mass_variable", "msoftdrop"),
                "missing": sorted(required - names),
                "prescale_candidates": sorted(name for name in names if "prescale" in name.lower()),
            }, sort_keys=True))
        return 0
    if args.command == "merge":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = merge_reports(args.inputs)
        with open(output_dir / "trigger_sf.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        write_metadata(output_dir, report)
        write_csv(output_dir / "trigger_sf.csv", report["bins"])
        data = Stats.from_payload(report["data"]["stats"], report["bin_edges"]["pt"], report["bin_edges"]["mass"])
        mc = Stats.from_payload(report["mc"]["stats"], report["bin_edges"]["pt"], report["bin_edges"]["mass"])
        definition = mass_definition(report["config"])
        write_root(output_dir / "trigger_sf.root", data, mc, report["bins"], report["data"]["report"],
                   report["mc"]["report"], definition["root_label"])
        write_plots(output_dir, data, mc, mass_axis_label=definition["axis_label"])
        write_diagnostics(output_dir, report["data"]["report"], report["mc"]["report"])
        report["outputs"] = {"correctionlib": write_correctionlib(output_dir, data, mc, report["config"])}
        with open(output_dir / "trigger_sf.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        return 0
    if args.command == "plot":
        with open(args.report, encoding="utf-8") as handle:
            report = json.load(handle)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = Stats.from_payload(report["data"]["stats"], report["bin_edges"]["pt"], report["bin_edges"]["mass"])
        mc = Stats.from_payload(report["mc"]["stats"], report["bin_edges"]["pt"], report["bin_edges"]["mass"])
        write_plots(output_dir, data, mc, mass_axis_label=mass_definition(report["config"])["axis_label"])
        write_diagnostics(output_dir, report["data"]["report"], report["mc"]["report"])
        return 0
    if args.max_events == 0 or args.max_events < -1:
        raise ValueError("--max-events must be -1 or positive")
    if args.max_files == 0 or args.max_files < -1:
        raise ValueError("--max-files must be -1 or positive")
    if args.file_start < 0:
        raise ValueError("--file-start must be non-negative")
    config = load_config(args.config)
    discovery_limit = None if args.max_files < 0 else args.file_start + args.max_files
    data_files = [] if args.skip_data else resolve_files(args.data_files, args.data_manifest, None)
    mc_files = [] if args.skip_mc else resolve_files(args.mc_files, args.mc_manifest, args.mc_sample,
                                                     args.mc_dataset, discovery_limit)
    file_stop = None if args.max_files < 0 else args.file_start + args.max_files
    data_files = data_files[args.file_start:file_stop]
    mc_files = mc_files[args.file_start:file_stop]
    if not data_files and not mc_files:
        raise ValueError("no Data or MC inputs were selected")
    if any(path.startswith("root://") for path in data_files + mc_files):
        os.environ.setdefault("X509_USER_PROXY", DEFAULT_PROXY)
        if not os.path.isfile(os.environ["X509_USER_PROXY"]):
            raise RuntimeError(f"remote inputs require proxy {os.environ['X509_USER_PROXY']}")
    data = process_sample(data_files, is_data=True, config=config, tree_name=args.tree_name, max_events=args.max_events,
                          golden_json=args.golden_json, skip_golden=args.skip_golden)
    mc = process_sample(mc_files, is_data=False, config=config, tree_name=args.tree_name, max_events=args.max_events,
                         golden_json=None, skip_golden=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = report_from_results(config, data, mc, data_files, mc_files, args.mc_dataset)
    report["processing"] = {
        "tree": args.tree_name,
        "max_events_per_sample": args.max_events,
        "max_files_per_sample": args.max_files,
        "file_start": args.file_start,
        "data_manifest": args.data_manifest,
        "mc_manifest": args.mc_manifest,
        "mc_sample": args.mc_sample,
        "golden_json": None if args.skip_golden else args.golden_json,
        "data_deduplicated_by": "(run, luminosityBlock, event)",
    }
    data_records_path = output_dir / "data_records.npz"
    mc_records_path = output_dir / "mc_records.npz"
    save_records(data_records_path, data.records)
    save_records(mc_records_path, mc.records)
    report["processing"]["data_records_file"] = str(data_records_path.resolve())
    report["processing"]["mc_records_file"] = str(mc_records_path.resolve())
    with open(output_dir / "trigger_sf.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_metadata(output_dir, report)
    write_csv(output_dir / "trigger_sf.csv", report["bins"])
    definition = mass_definition(config)
    write_root(output_dir / "trigger_sf.root", data.stats, mc.stats, report["bins"], data.report, mc.report,
               definition["root_label"])
    if not args.no_plots:
        write_plots(output_dir, data.stats, mc.stats, mass_axis_label=definition["axis_label"])
    report["outputs"] = {"correctionlib": write_correctionlib(output_dir, data.stats, mc.stats, config)}
    with open(output_dir / "trigger_sf.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({"output_dir": str(output_dir), "data": data.report, "mc": mc.report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
