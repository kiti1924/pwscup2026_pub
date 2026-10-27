"""Apply a shrinkage-only true-rare marginal tilt to synthetic rare rows.

The input is already fully synthetic.  Only rows detected as rare by the public
Mahalanobis rule are adjusted, and only through monotone marginal quantile
shrinkage toward B_self's true-rare distribution.  Joint row combinations and
all outcomes remain synthetic.  A vectorized line search keeps every adjusted
row inside the public rare region, so the detected rare mass is unchanged.

This is an experiment, not a privacy claim.  Because the transform uses a small
24-record reference group, every selected candidate must pass membership and
attribute-inference audits before packaging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_STARTER_ROOT = _WORKSPACE_ROOT / "pwscup2026-kit" / "starter"
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
if str(_STARTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT))
if str(_STARTER_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT / "src"))

from experiments.run_aligned_synth import (  # noqa: E402
    _detector_mask,
    _format_continuous,
    _load_detector,
    _score_one,
)
from reference._common import (  # noqa: E402
    coerce_to_submission,
    dist_paths,
    load_cfg,
    read_b,
    write_c,
)
from pwscup2026.common import schema, submission_schema  # noqa: E402


def _truth_mask_for_b(b: pd.DataFrame, b_self: pd.DataFrame) -> np.ndarray:
    required = {"record_id", "is_rare"}
    if not required.issubset(b_self.columns):
        raise ValueError("B_self.csv must contain record_id and is_rare")
    truth = b[["record_id"]].merge(
        b_self[["record_id", "is_rare"]], on="record_id", how="left", validate="one_to_one"
    )
    if truth["is_rare"].isna().any() or len(truth) != len(b):
        raise ValueError("B_self.csv does not align one-to-one with B")
    return truth["is_rare"].astype(bool).to_numpy()


def _rank_quantiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(dtype=float)
    return np.clip((ranks - 0.5) / len(ranks), 1e-6, 1.0 - 1e-6)


def _quantile(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    try:
        return np.quantile(values, q, method="linear")
    except TypeError:
        return np.quantile(values, q, interpolation="linear")


def _tilt_target(
    rare_continuous: pd.DataFrame,
    truth_rare: pd.DataFrame,
    strength: float,
) -> pd.DataFrame:
    out = rare_continuous.copy(deep=True)
    for col in schema.CONTINUOUS:
        current = rare_continuous[col].to_numpy(dtype=float)
        q = _rank_quantiles(current)
        target = _quantile(truth_rare[col].to_numpy(dtype=float), q)
        out[col] = (1.0 - float(strength)) * current + float(strength) * target
    return out


def _keep_inside_rare_region(
    original: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict[str, Any],
    detector: tuple[np.ndarray, np.ndarray, float],
    steps: int = 24,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Move as far toward target as possible without crossing the detector band."""
    mu, inv_sigma, threshold = detector
    cols = list(schema.CONTINUOUS)
    x0 = original[cols].to_numpy(dtype=float)
    x1 = target[cols].to_numpy(dtype=float)
    low = np.zeros(len(original), dtype=float)
    high = np.ones(len(original), dtype=float)

    for _ in range(int(steps)):
        mid = (low + high) / 2.0
        proposal = pd.DataFrame(x0 + mid[:, None] * (x1 - x0), columns=cols)
        proposal = _format_continuous(proposal, cfg)
        inside = _detector_mask(proposal, mu, inv_sigma, threshold)
        low = np.where(inside, mid, low)
        high = np.where(inside, high, mid)

    final = pd.DataFrame(x0 + low[:, None] * (x1 - x0), columns=cols)
    final = _format_continuous(final, cfg)
    if not bool(_detector_mask(final, mu, inv_sigma, threshold).all()):
        raise RuntimeError("rare-region line search failed")
    return final, low


def _apply_tilt(
    base: pd.DataFrame,
    truth_rare: pd.DataFrame,
    strength: float,
    cfg: dict[str, Any],
    detector: tuple[np.ndarray, np.ndarray, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    mu, inv_sigma, threshold = detector
    continuous = _format_continuous(base, cfg)
    rare_mask = _detector_mask(continuous, mu, inv_sigma, threshold)
    rare_original = continuous.loc[rare_mask].reset_index(drop=True)
    target = _tilt_target(rare_original, truth_rare, strength)
    rare_final, realized = _keep_inside_rare_region(rare_original, target, cfg, detector)

    out = base.copy(deep=True)
    out.loc[rare_mask, list(schema.CONTINUOUS)] = rare_final.to_numpy(dtype=float)
    out = coerce_to_submission(out, cfg)
    check = _detector_mask(_format_continuous(out, cfg), mu, inv_sigma, threshold)
    if int(check.sum()) != int(rare_mask.sum()):
        raise RuntimeError("detected rare mass changed after tilt")
    diagnostics = {
        "rare_detected": int(check.sum()),
        "mean_realized_fraction": float(np.mean(realized)),
        "min_realized_fraction": float(np.min(realized)),
    }
    return out, diagnostics


def _parse_strengths(value: str) -> list[float]:
    strengths = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not strengths or any(not 0.0 <= value <= 1.0 for value in strengths):
        raise ValueError("strengths must be non-empty values in [0, 1]")
    return strengths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="True-rare marginal shrinkage experiment")
    ap.add_argument("--dist", required=True)
    ap.add_argument("--base-csv", required=True, help="Fully synthetic 13-column C candidate")
    ap.add_argument("--strengths", default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    try:
        dist = dist_paths(args.dist)
        cfg = load_cfg()
        b = read_b(dist)
        b_self_path = dist / "B_self.csv"
        if not b_self_path.exists():
            raise FileNotFoundError(f"missing {b_self_path.name}")
        b_self = pd.read_csv(b_self_path)
        truth_mask = _truth_mask_for_b(b, b_self)
        truth_rare = b.loc[truth_mask, list(schema.CONTINUOUS)].reset_index(drop=True)
        if len(truth_rare) < 3:
            raise ValueError("too few true-rare rows for aggregate shrinkage")
        base_submitted = pd.read_csv(args.base_csv)
        expected_submitted = list(submission_schema.SUBMITTED_C_COLUMNS)
        if list(base_submitted.columns) != expected_submitted or len(base_submitted) != len(b):
            raise ValueError("base C has an unexpected shape or columns")
        base = base_submitted.copy()
        base.insert(0, "record_id", np.arange(len(base), dtype=int))
        detector = _load_detector(dist)
        strengths = _parse_strengths(args.strengths)
        out_dir = Path(args.out_dir) if args.out_dir else dist / "rare_tilt"
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_dir = out_dir / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary: dict[str, Any] = {
        "base_csv": Path(args.base_csv).name,
        "true_rare_count": int(truth_mask.sum()),
        "privacy_note": "Uses true-rare aggregate marginals; every candidate requires privacy audit.",
        "results": [],
    }
    print(
        "strength\tU\tU_gen\tU_spec\tU_rare\tU_valid\tprof\tc2st\t"
        "rare_n\tmean_realized"
    )
    failures = 0
    for strength in strengths:
        label = f"{strength:.3f}".replace(".", "p")
        csv_path = out_dir / f"C_truth_tilt_{label}.csv"
        score_path = scores_dir / f"truth_tilt_{label}.json"
        try:
            candidate, diagnostics = _apply_tilt(
                base, truth_rare, strength, cfg, detector
            )
            write_c(candidate, csv_path)
            payload, error = _score_one(csv_path, dist, score_path)
            if payload is None or error is not None:
                raise RuntimeError(error or "selfscore returned no payload")
            facets = payload["facets"]
            rare = payload["detail"]["U_rare"]
            row = {
                "strength": float(strength),
                "csv": csv_path.name,
                "score_json": score_path.name,
                "utility": payload["utility"],
                **facets,
                "prof_cal": rare.get("prof_cal"),
                "c2st_cal": rare.get("c2st_cal"),
                **diagnostics,
            }
            summary["results"].append(row)
            print(
                f"{strength:.3f}\t{row['utility']:.6f}\t{row['U_gen']:.6f}\t"
                f"{row['U_spec']:.6f}\t{row['U_rare']:.6f}\t{row['U_valid']:.6f}\t"
                f"{row['prof_cal']:.6f}\t{row['c2st_cal']:.6f}\t"
                f"{row['rare_detected']}\t{row['mean_realized_fraction']:.6f}"
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{strength:.3f}\tERROR\t{type(exc).__name__}: {exc}")

    summary_path = out_dir / "rare_truth_tilt_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
