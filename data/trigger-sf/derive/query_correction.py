#!/usr/bin/env python3
"""Safely query the 2024 event-level OR trigger correction."""

from __future__ import annotations

import argparse
from pathlib import Path


CORRECTION_NAMES = (
    "zbb_2024_or_trigger_sf",
    "zbb_2024_or_trigger_sf_gptmass_x2p",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("pt", type=float)
    parser.add_argument("mass", type=float)
    parser.add_argument("--variation", choices=("nominal", "stat_up", "stat_down"), default="nominal")
    args = parser.parse_args()
    import correctionlib

    corrections = correctionlib.CorrectionSet.from_file(str(args.payload))
    sf_name = None
    for name in CORRECTION_NAMES:
        try:
            sf_correction = corrections[name]
            sf_name = name
            break
        except (KeyError, IndexError):
            continue
    if sf_name is None:
        raise KeyError(f"payload has none of the supported corrections: {', '.join(CORRECTION_NAMES)}")
    mass_input = sf_correction.inputs[1].name
    valid_correction = corrections[f"{sf_name}_valid"]
    status_correction = corrections[f"{sf_name}_status"]
    valid = valid_correction.evaluate(args.pt, args.mass)
    status = status_correction.evaluate(args.pt, args.mass)
    if valid < 0.5 or status < 0.5:
        raise RuntimeError(f"invalid trigger-SF bin at pt={args.pt}, {mass_input}={args.mass} (status={status})")
    value = sf_correction.evaluate(args.pt, args.mass, args.variation)
    if value < 0.0:
        raise RuntimeError("correction payload returned its invalid-bin sentinel")
    print(f"{value:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
