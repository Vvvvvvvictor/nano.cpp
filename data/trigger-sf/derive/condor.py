#!/usr/bin/env python3
"""Prepare, recover and merge trigger-SF jobs using Condor and EOS xrdcp."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import measure_trigger_sf as measurement

DERIVE_DIR = Path(__file__).resolve().parent
SAMPLES = ("data", "mc")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_paths(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def write_paths(path, paths):
    Path(path).write_text("".join(f"{item}\n" for item in paths), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eos_path(value):
    # Do not stat EOS paths: workers use xrdcp and may have a stalled FUSE mount.
    path = Path(os.path.abspath(value))
    if not str(path).startswith("/eos/cms/"):
        raise ValueError("Condor output must be under /eos/cms; use local measurement for local output")
    return path


def remote(path):
    return "root://eoscms.cern.ch//" + str(path).lstrip("/")


def submission(base, override=None):
    metadata = read_json(base / "submission.json")
    config = Path(override or metadata["config"]).resolve()
    if digest(config) != metadata["config_sha256"]:
        raise ValueError(f"configuration digest differs from submission: {config}")
    return metadata, config


def reports(base, results, allow_missing=False):
    metadata = read_json(base / "submission.json")
    found, missing = [], []
    for index in range(int(metadata["tasks"])):
        path = results / f"task_{index:04d}/result/trigger_sf.json"
        if path.is_file():
            found.append(path)
        else:
            missing.append(index)
    if missing and not allow_missing:
        raise RuntimeError(f"missing task reports: {missing[:20]}")
    return found, missing


def validate_reports(paths, config):
    expected_config = measurement.load_config(str(config))
    expected_definition = None
    observed = {sample: [] for sample in SAMPLES}
    for path in paths:
        payload = read_json(path)
        definition = payload.get("mass_definition")
        if (payload.get("config") != expected_config or not definition or
                definition.get("variable") != expected_config["mass_variable"]):
            raise ValueError(f"incompatible configuration or mass definition: {path}")
        if expected_definition is None:
            expected_definition = definition
        elif definition != expected_definition:
            raise ValueError(f"mixed mass definitions: {path}")
        for sample in SAMPLES:
            if not payload[sample]["report"].get("coverage", {}).get("complete", False):
                raise RuntimeError(f"incomplete {sample} input coverage: {path}")
            cache = payload.get("processing", {}).get(f"{sample}_records_file")
            if not cache or not Path(cache).is_file():
                raise RuntimeError(f"missing {sample} selected-event cache: {path}")
            observed[sample].extend(payload["inputs"][sample])
    return observed


def validate_inputs(base, observed):
    for sample in SAMPLES:
        expected = read_paths(base / f"{sample}_files.txt")
        identities = [measurement.input_identity(path) for path in expected]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate {sample} files in original input list")
        if Counter(identities) != Counter(map(measurement.input_identity, observed[sample])):
            raise RuntimeError(f"reports do not cover original {sample} inputs exactly once")


def missing_inputs(base, indices):
    return {sample: [path for index in indices
                     for path in read_paths(base / f"task_{index:04d}/{sample}.txt")]
            for sample in SAMPLES}


def write_job(path, arguments, memory="3000 MB", disk="3000 MB", retry=False):
    # Arguments are serialized with HTCondor's quoted argument syntax.
    def quoted(value):
        return "'" + str(value).replace("'", "''").replace('"', '""') + "'"

    rows = [" ".join(quoted(value) for value in (DERIVE_DIR / "run.sh", *row)) for row in arguments]
    logs = path.parent / "logs"
    logs.mkdir(exist_ok=True)
    text = f"""universe = vanilla
executable = {DERIVE_DIR / 'worker.sh'}
initialdir = /tmp
getenv = True
request_cpus = 1
request_memory = {memory}
request_disk = {disk}
should_transfer_files = NO
transfer_executable = False
output = {logs}/{path.stem}.$(ClusterId).$(ProcId).out
error = {logs}/{path.stem}.$(ClusterId).$(ProcId).err
log = {path.with_suffix('.log')}
"""
    if retry:
        text += "on_exit_remove = (ExitCode == 0) || (NumJobStarts >= 4)\nmax_retries = 3\n"
    text += 'arguments = "$(job_args)"\nqueue job_args from (\n' + "\n".join(rows) + "\n)\n"
    path.write_text(text, encoding="utf-8")
    return path


def finish_job(path, submit):
    print(json.dumps({"submit_file": str(path)}, indent=2))
    if submit:
        subprocess.run(["condor_submit", str(path)], check=True)


def prepare_submission(output, results, data, mc, config, max_events, chunk_size, dataset, manifest=None):
    if chunk_size <= 0 or (max_events != -1 and max_events <= 0):
        raise ValueError("chunk size and event limit must be positive (or max-events=-1)")
    if output.exists():
        raise FileExistsError(f"submission directory must be new: {output}")
    measurement.load_config(str(config))
    inputs = {sample: list(dict.fromkeys(paths)) for sample, paths in zip(SAMPLES, (data, mc))}
    for sample, paths in inputs.items():
        keys = [measurement.input_identity(path) for path in paths]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate logical {sample} files across endpoints")
    data, mc = inputs["data"], inputs["mc"]
    count = max(len(mc), (len(data) + chunk_size - 1) // chunk_size)
    if not count:
        raise ValueError("no Data or MC inputs")
    output.mkdir(parents=True)
    for sample, paths in inputs.items():
        write_paths(output / f"{sample}_files.txt", paths)
    chunks = [{"data": [], "mc": []} for _ in range(count)]
    for index, path in enumerate(data):
        chunks[index * count // len(data)]["data"].append(path)
    for index, path in enumerate(mc):
        chunks[index]["mc"].append(path)
    for index, chunk in enumerate(chunks):
        task = output / f"task_{index:04d}"
        task.mkdir()
        for sample in SAMPLES:
            write_paths(task / f"{sample}.txt", chunk[sample])
    metadata = {"config": str(config), "config_sha256": digest(config),
                "data_manifest": str(manifest) if manifest else None, "mc_dataset": dataset,
                "tasks": count, "data_files": len(data), "mc_files": len(mc),
                "max_events": max_events, "chunk_data_files_target": chunk_size,
                "results_dir": str(results), "transport": "xrdcp"}
    for sample in SAMPLES:
        metadata[f"{sample}_files_sha256"] = digest(output / f"{sample}_files.txt")
    write_json(output / "submission.json", metadata)
    return write_job(output / "submit.sub",
                     [("measure", output, results, index, max_events, config) for index in range(count)],
                     retry=True)


def submit(args):
    data = (read_paths(args.data_files_list) if args.data_files_list else
            measurement.manifest_files(args.data_manifest, None))
    mc = read_paths(args.mc_files_list) if args.mc_files_list else measurement.das_files(args.mc_dataset)
    path = prepare_submission(Path(args.output_dir).resolve(), eos_path(args.results_dir), data, mc,
                              Path(args.config).resolve(), args.max_events, args.chunk_data_files,
                              args.mc_dataset, args.data_manifest)
    finish_job(path, args.submit)


def resume(args):
    base, results = Path(args.submission_dir).resolve(), eos_path(args.results_dir)
    metadata, config = submission(base, args.config)
    paths, missing = reports(base, results, allow_missing=True)
    observed = validate_reports(paths, config)
    pending = missing_inputs(base, missing)
    validate_inputs(base, {sample: observed[sample] + pending[sample] for sample in SAMPLES})
    if not missing:
        print('{"missing_tasks": 0, "status": "complete"}')
        return
    path = write_job(base / "resume.sub",
                     [("measure", base, results, index, metadata["max_events"], config) for index in missing],
                     retry=True)
    finish_job(path, args.submit)


def recover(args):
    base, results = Path(args.submission_dir).resolve(), Path(args.results_dir).resolve()
    metadata, config = submission(base, args.config)
    paths, missing_tasks = reports(base, results, allow_missing=True)
    covered = validate_reports(paths, config)
    missing = missing_inputs(base, missing_tasks)
    validate_inputs(base, {sample: covered[sample] + missing[sample] for sample in SAMPLES})
    if not missing_tasks:
        print('{"missing_tasks": 0, "status": "complete"}')
        return
    output = Path(args.recovery_submission_dir).resolve()
    path = prepare_submission(output, eos_path(args.recovery_results_dir), missing["data"], missing["mc"],
                              config, metadata["max_events"], args.data_files_per_task, metadata["mc_dataset"])
    write_json(output / "recovery_plan.json", {"source_submission": str(base), "source_results": str(results),
                                             "missing_tasks": missing_tasks})
    finish_job(path, args.submit)


def build_merge(args):
    base, recovery = Path(args.submission_dir).resolve(), Path(args.recovery_submission_dir).resolve()
    metadata, config = submission(base, args.config)
    submission(recovery, str(config))
    paths, _ = reports(base, Path(args.results_dir).resolve(), allow_missing=True)
    extra, _ = reports(recovery, Path(args.recovery_results_dir).resolve())
    paths.extend(extra)
    validate_inputs(base, validate_reports(paths, config))
    output, results = Path(args.merged_submission_dir).resolve(), Path(args.merged_results_dir).resolve()
    if output.exists() or results.exists():
        raise FileExistsError("merged submission and result directories must be new")
    output.mkdir(parents=True)
    results.mkdir(parents=True)
    for sample in SAMPLES:
        shutil.copyfile(base / f"{sample}_files.txt", output / f"{sample}_files.txt")
    metadata.update({"config": str(config), "tasks": len(paths), "results_dir": str(results),
                     "merged_submissions": [str(base), str(recovery)]})
    write_json(output / "submission.json", metadata)
    for index, source in enumerate(paths):
        result = results / f"task_{index:04d}/result"
        result.mkdir(parents=True)
        (result / "trigger_sf.json").symlink_to(source.resolve())
    print(json.dumps({"reports": len(paths), "merged_submission": str(output)}, indent=2))


def merge_local(base, results, output, config, paths=None):
    metadata, config = submission(base, str(config))
    if paths is None:
        paths, _ = reports(base, results)
    validate_inputs(base, validate_reports(paths, config))
    measurement.main(["merge", "--output-dir", str(output), "--inputs", *map(str, paths)])
    path = output / "trigger_sf.json"
    payload = read_json(path)
    if not payload["processing"].get("global_data_deduplication"):
        raise RuntimeError("merge did not perform global Data de-duplication")
    payload["processing"].update({"full_submission": str(base), "full_results": str(results),
                                  "full_task_count": metadata["tasks"], "full_input_coverage": True})
    payload.setdefault("metadata", {})["statistics_scope"] = (
        "Complete selected input files from the submission; see processing.max_events_per_sample for event limits"
    )
    payload["metadata"]["prescale_verification"] = (
        "not performed from NanoAOD; per-period trigger reference/target prescale check remains required"
    )
    write_json(path, payload)


def merge(args):
    base, results, output = (Path(value).resolve() for value in
                             (args.submission_dir, args.results_dir, args.output_dir))
    _, config = submission(base, args.config)
    if args.local:
        merge_local(base, results, output, config)
        return
    paths, _ = reports(base, results)
    validate_inputs(base, validate_reports(paths, config))
    eos_path(output)
    path = write_job(base / "merge.sub", [("merge", base, results, output, config)],
                     args.request_memory, args.request_disk)
    finish_job(path, args.submit)


def download(source, destination):
    if str(source).startswith("/eos/cms/"):
        subprocess.run(["xrdcp", "-f", remote(source), str(destination)], check=True)
    else:
        shutil.copyfile(source, destination)


def publish(directory, destination, replacements):
    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            if value in replacements:
                return replacements[value]
            if value.startswith(str(directory) + "/"):
                return str(destination) + value[len(str(directory)):]
        return value

    subprocess.run(["xrdfs", "eoscms.cern.ch", "mkdir", "-p", str(destination)], check=True)
    for path in sorted(directory.iterdir(), key=lambda item: (item.name == "trigger_sf.json", item.name)):
        if path.suffix == ".json":
            payload = read_json(path)
            rewritten = rewrite(payload)
            if rewritten != payload:
                write_json(path, rewritten)
        subprocess.run(["xrdcp", "-f", str(path), remote(destination / path.name)], check=True)


def worker(args):
    kind, base, results, *values = args.arguments
    base, results = Path(base), Path(results)
    config = Path(values[-1])
    submission(base, str(config))
    destination = (results / f"task_{int(values[0]):04d}/result" if kind == "measure" else
                   eos_path(values[0]))
    with tempfile.TemporaryDirectory(prefix="trigger_sf_", dir=os.environ.get("_CONDOR_SCRATCH_DIR")) as scratch:
        scratch = Path(scratch)
        output = scratch / "result"
        output.mkdir()
        log = output / "worker.log"
        replacements = {}
        status = 0
        # Keep measurement output bounded and upload failure logs before exiting.
        with log.open("w") as stream:
            sys.stdout.flush()
            sys.stderr.flush()
            saved = [os.dup(fd) for fd in (1, 2)]
            try:
                os.dup2(stream.fileno(), 1)
                os.dup2(stream.fileno(), 2)
                if kind == "measure":
                    index, max_events, _ = values
                    command = ["measure", "--config", str(config), "--max-events", max_events,
                               "--max-files", "-1", "--output-dir", str(output), "--no-plots"]
                    for sample in SAMPLES:
                        files = read_paths(base / f"task_{int(index):04d}/{sample}.txt")
                        if files:
                            for path in files:
                                command.extend([f"--{sample}-files", path])
                        else:
                            command.append(f"--skip-{sample}")
                    measurement.main(command)
                elif kind == "merge":
                    metadata = read_json(base / "submission.json")
                    staged = []
                    for index in range(metadata["tasks"]):
                        task = scratch / f"task_{index:04d}"
                        task.mkdir()
                        report = task / "trigger_sf.json"
                        download(results / f"task_{index:04d}/result/trigger_sf.json", report)
                        payload = read_json(report)
                        for sample in SAMPLES:
                            source = payload["processing"][f"{sample}_records_file"]
                            cache = task / f"{sample}_records.npz"
                            download(source, cache)
                            replacements[str(cache)] = source
                            payload["processing"][f"{sample}_records_file"] = str(cache)
                        write_json(report, payload)
                        staged.append(report)
                    merge_local(base, results, output, config, staged)
                else:
                    raise ValueError(f"unknown worker action: {kind}")
            except Exception:
                import traceback
                traceback.print_exc()
                status = 1
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                for fd, original in zip((1, 2), saved):
                    os.dup2(original, fd)
                    os.close(original)
        if status:
            (output / "trigger_sf.json").unlink(missing_ok=True)
        # Upload the report last so recovery never observes an unfinished cache upload.
        publish(output, destination, replacements)
        if status:
            raise RuntimeError(f"worker failed; see {destination / 'worker.log'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("submit", help="prepare measurement tasks; --submit launches them")
    new.add_argument("--output-dir", required=True)
    new.add_argument("--results-dir", required=True)
    new.add_argument("--data-manifest", default=str(measurement.DEFAULT_DATA_MANIFEST))
    new.add_argument("--data-files-list")
    new.add_argument("--mc-files-list")
    new.add_argument("--mc-dataset", default=measurement.DEFAULT_MC_DATASET)
    new.add_argument("--config", default=str(measurement.DEFAULT_CONFIG))
    new.add_argument("--chunk-data-files", type=int, default=6)
    new.add_argument("--max-events", type=int, default=-1)
    new.add_argument("--submit", action="store_true")
    for name in ("resume", "recover", "build-merge", "merge"):
        command = commands.add_parser(name)
        command.add_argument("--submission-dir", required=True)
        command.add_argument("--results-dir", required=True)
        command.add_argument("--config", help="explicit replacement for the recorded config path; digest must match")
        if name != "build-merge":
            command.add_argument("--submit", action="store_true")
        if name in ("recover", "build-merge"):
            command.add_argument("--recovery-submission-dir", required=True)
            command.add_argument("--recovery-results-dir", required=True)
        if name == "recover":
            command.add_argument("--data-files-per-task", type=int, default=2)
        if name == "build-merge":
            command.add_argument("--merged-submission-dir", required=True)
            command.add_argument("--merged-results-dir", required=True)
        if name == "merge":
            command.add_argument("--output-dir", required=True)
            command.add_argument("--request-memory", default="8000 MB")
            command.add_argument("--request-disk", default="6000 MB")
            command.add_argument("--local", action="store_true", help="execute a small merge locally")
    return parser


def main():
    if sys.argv[1:2] == ["_worker"]:
        worker(argparse.Namespace(arguments=sys.argv[2:]))
        return
    args = build_parser().parse_args()
    if args.command == "merge" and args.local and args.submit:
        raise ValueError("--local and --submit are mutually exclusive")
    {"submit": submit, "resume": resume, "recover": recover, "build-merge": build_merge,
     "merge": merge}[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
