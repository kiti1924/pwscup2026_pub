"""Run selected historical operations on explicitly supplied local data.

This interface neither reads competition tokens nor creates/submits archives.
It does not replace the full historical generation and model-selection runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .kernels import priority, reassign, transform


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def write_new(frame, path):
    """Create a fresh output; never overwrite a historical result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")


def checked_matrix(frame, name):
    if not frame.columns.is_unique or "record_id" not in frame or frame.record_id.duplicated().any():
        raise ValueError(f"{name}: missing or duplicate record keys")
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"{name}: empty or missing values")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: nonfinite values")


def calibrate_mia(baseline, evidence, masks):
    """FA114 numerical operation; evidence is FA106 blend_e0, not raw distances.

    Masks must supply own-data rare/home and known-private excluded flags.
    Target membership labels must never be supplied as these flags.
    """
    checked_matrix(baseline, "baseline")
    checked_matrix(evidence, "evidence")
    checked_matrix(masks, "masks")
    if set(baseline.columns) != set(evidence.columns):
        raise ValueError("Evidence targets differ from baseline")
    if set(masks.columns) != {"record_id", "rare", "excluded", "home"}:
        raise ValueError("Expected record_id, rare, excluded, home masks only")
    if set(evidence.record_id) != set(baseline.record_id) or set(masks.record_id) != set(baseline.record_id):
        raise ValueError("Missing or additional record keys")
    for col in ["rare", "excluded", "home"]:
        if not masks[col].isin([0, 1]).all():
            raise ValueError(f"{col}: expected binary flags")
    ids = baseline.record_id.to_numpy()
    columns = baseline.columns.drop("record_id").tolist()
    if not columns:
        raise ValueError("No target columns")
    e = evidence.set_index("record_id").loc[ids, columns].to_numpy(float)
    m = masks.set_index("record_id").loc[ids]
    b = baseline[columns].to_numpy(float)
    pred, reports = reassign(b, priority(e, m.home.to_numpy(bool)),
                             m.rare.to_numpy(bool), m.excluded.to_numpy(bool))
    out = pd.DataFrame(pred, columns=columns)
    out.insert(0, "record_id", ids)
    return out, reports


def integrate_aia(sources, routing, expected_rows=None):
    """Select complete answer vectors per target; preserve time_hat strings."""
    mapping = {int(k): str(v) for k, v in routing.items()}
    if len(mapping) != len(routing) or not mapping:
        raise ValueError("Empty or duplicate canonical target IDs")
    parts = []
    for target, source in sorted(mapping.items()):
        if source not in sources:
            raise ValueError(f"Missing source submission {source}")
        frame = sources[source]
        if set(frame.columns) != {"target_k", "challenge_row_id", "time_hat"}:
            raise ValueError("Expected only AIA answer columns")
        if frame.isna().any().any():
            raise ValueError("Missing AIA values")
        keys = pd.to_numeric(frame.target_k, errors="raise").to_numpy(float)
        if not np.isfinite(keys).all() or not np.equal(keys, np.floor(keys)).all():
            raise ValueError("Nonintegral target ID")
        selected = frame.loc[keys == target, ["target_k", "challenge_row_id", "time_hat"]].copy()
        if selected.empty or selected.challenge_row_id.duplicated().any():
            raise ValueError(f"Target {target}: missing or duplicate challenge keys")
        if expected_rows is not None and len(selected) != expected_rows:
            raise ValueError(f"Target {target}: wrong row count")
        times = pd.to_numeric(selected.time_hat, errors="raise").to_numpy(float)
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("Invalid predicted time")
        parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    jitter = commands.add_parser("jitter", help="G22 event-time transform on the pre-jitter C")
    jitter.add_argument("--input", type=Path, required=True)
    jitter.add_argument("--output", type=Path, required=True)
    jitter.add_argument("--width", type=float, default=1.0)
    calibration = commands.add_parser("calibrate-mia", help="FA114 operation using saved FA106 evidence")
    calibration.add_argument("--baseline", type=Path, required=True)
    calibration.add_argument("--evidence", type=Path, required=True)
    calibration.add_argument("--masks", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    integration = commands.add_parser("integrate-aia", help="Select complete measured AIA target vectors")
    integration.add_argument("--sources", type=Path, required=True,
                             help="JSON object mapping submission ID to local F_aia.csv, relative to the JSON")
    integration.add_argument("--routing", type=Path,
                             default=Path(__file__).resolve().parents[1]/"provenance/final_aia_routing.json")
    integration.add_argument("--rows-per-target", type=int, default=326)
    integration.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new path")
    if args.command == "jitter":
        frame = read(args.input)
        if not np.isfinite(args.width) or args.width < 0:
            raise ValueError("Width must be finite and nonnegative")
        if not {"time", "onset", "death"}.issubset(frame):
            raise ValueError("Expected the pre-jitter C with event columns")
        values = frame[["time", "onset", "death"]].to_numpy(float)
        if frame.empty or not np.isfinite(values).all() or (frame.time < 0).any():
            raise ValueError("Invalid input events")
        if not frame.onset.isin([0, 1]).all() or not frame.death.isin([0, 1]).all():
            raise ValueError("Event flags must be binary")
        if (frame.onset + frame.death > 1).any():
            raise ValueError("Mutually exclusive events required")
        out = transform(frame, args.width)
    elif args.command == "calibrate-mia":
        out, _ = calibrate_mia(read(args.baseline), read(args.evidence), read(args.masks))
    else:
        if args.rows_per_target <= 0:
            raise ValueError("Positive row count required")
        paths = json.loads(args.sources.read_text(encoding="utf-8"))
        sources = {str(k): pd.read_csv(args.sources.parent/Path(v), dtype=str, keep_default_na=False)
                   for k, v in paths.items()}
        out = integrate_aia(sources, json.loads(args.routing.read_text(encoding="utf-8")), args.rows_per_target)
    write_new(out, args.output)
    print(f"Wrote {len(out)} rows to {args.output}")
