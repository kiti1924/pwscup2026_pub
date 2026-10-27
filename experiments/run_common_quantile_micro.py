"""Run common-only marginal quantile micro-tilts and official selfscores.

The public Mahalanobis detector is applied to both B and the submitted base C.
Only C rows outside the detector region are changed.  For one selected
continuous column, each common C value is blended toward the matching empirical
quantile of the public-detected common B values.  Rare rows are checked for
bitwise equality and the detected rare count must remain unchanged.
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
for _import_root in (_WORKSPACE_ROOT, _STARTER_ROOT, _STARTER_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from experiments.run_aligned_synth import _detector_mask, _load_detector, _score_one
from reference._common import dist_paths, read_b, write_c
from pwscup2026.common import schema, submission_schema


def _rank_quantiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(dtype=float)
    return np.clip((ranks - 0.5) / len(ranks), 1e-6, 1.0 - 1e-6)


def _quantile(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    try:
        return np.quantile(values, q, method="linear")
    except TypeError:
        return np.quantile(values, q, interpolation="linear")


def _parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out or any(not 0.0 < alpha <= 1.0 for alpha in out):
        raise ValueError("alphas must be non-empty values in (0, 1]")
    return out


def _parse_columns(value: str) -> list[str]:
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    allowed = set(schema.CONTINUOUS)
    if not out or any(col not in allowed for col in out):
        raise ValueError(f"columns must be in {sorted(allowed)}")
    return out


def _bitwise_same(base: pd.DataFrame, candidate: pd.DataFrame, mask: np.ndarray) -> bool:
    for col in submission_schema.SUBMITTED_C_COLUMNS:
        a = base.loc[mask, col].to_numpy()
        b = candidate.loc[mask, col].to_numpy()
        if not np.array_equal(a, b):
            return False
    return True


def _apply_one(
    base: pd.DataFrame,
    c_rare: np.ndarray,
    b_common: pd.DataFrame,
    col: str,
    alpha: float,
    detector: tuple[np.ndarray, np.ndarray, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    common = ~c_rare
    current = base.loc[common, col].to_numpy(dtype=float)
    q = _rank_quantiles(current)
    target = _quantile(b_common[col].to_numpy(dtype=float), q)
    candidate = base.copy(deep=True)
    candidate.loc[common, col] = (1.0 - float(alpha)) * current + float(alpha) * target

    mu, inv_sigma, threshold = detector
    after_rare = _detector_mask(candidate, mu, inv_sigma, threshold)
    rare_n = int(after_rare.sum())
    diagnostics = {
        "column": col,
        "alpha": float(alpha),
        "base_rare_n": int(c_rare.sum()),
        "candidate_rare_n": rare_n,
        "rare_bitwise_unchanged": _bitwise_same(base, candidate, c_rare),
        "n_common_changed": int(np.count_nonzero(candidate.loc[common, col].to_numpy(dtype=float) != current)),
        "mean_abs_change": float(np.mean(np.abs(candidate.loc[common, col].to_numpy(dtype=float) - current))),
        "max_abs_change": float(np.max(np.abs(candidate.loc[common, col].to_numpy(dtype=float) - current))),
    }
    if rare_n != int(c_rare.sum()):
        raise RuntimeError(f"rare count changed: {int(c_rare.sum())} -> {rare_n}")
    if not diagnostics["rare_bitwise_unchanged"]:
        raise RuntimeError("rare rows changed before serialization")
    return candidate, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Common-only empirical quantile micro-tilt")
    parser.add_argument("--dist", required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alphas", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--columns", default="ALT,TG,FPG")
    args = parser.parse_args()

    dist = dist_paths(args.dist)
    base_path = Path(args.base_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(base_path)
    expected = list(submission_schema.SUBMITTED_C_COLUMNS)
    if list(base.columns) != expected:
        raise ValueError(f"base columns differ from submitted schema: {list(base.columns)}")
    b = read_b(dist)
    if len(base) != len(b):
        raise ValueError(f"base rows {len(base)} != B rows {len(b)}")

    detector = _load_detector(dist)
    mu, inv_sigma, threshold = detector
    c_rare = _detector_mask(base, mu, inv_sigma, threshold)
    b_rare = _detector_mask(b, mu, inv_sigma, threshold)
    b_common = b.loc[~b_rare].reset_index(drop=True)
    if int(c_rare.sum()) != 61 or int(b_rare.sum()) != 61:
        raise ValueError(f"unexpected public rare counts C={int(c_rare.sum())}, B={int(b_rare.sum())}")

    alphas = _parse_float_list(args.alphas)
    columns = _parse_columns(args.columns)
    summary: dict[str, Any] = {
        "base_csv": str(base_path),
        "base_public_rare_n": int(c_rare.sum()),
        "b_public_rare_n": int(b_rare.sum()),
        "operation": "common-only x'=(1-alpha)x+alpha*Q_B_common(rank(x))",
        "columns": columns,
        "alphas": alphas,
        "results": [],
    }

    for col in columns:
        for alpha in alphas:
            tag = f"common_q_{col.lower()}_a{alpha:.2f}".replace(".", "p")
            csv_path = out_dir / f"C_{tag}.csv"
            score_path = scores_dir / f"{tag}.json"
            candidate, diagnostics = _apply_one(base, c_rare, b_common, col, alpha, detector)
            write_c(candidate, csv_path)
            payload, error = _score_one(csv_path, dist, score_path)
            if payload is None or error is not None:
                raise RuntimeError(error or f"selfscore failed for {csv_path.name}")
            row = {
                "candidate": csv_path.name,
                "score_json": score_path.name,
                "utility": float(payload["utility"]),
                **{key: float(value) for key, value in payload["facets"].items()},
                **diagnostics,
            }
            summary["results"].append(row)
            print(
                f"{col}\t{alpha:.2f}\tU={row['utility']:.9f}\t"
                f"Ugen={row['U_gen']:.9f}\tUspec={row['U_spec']:.9f}\t"
                f"Urare={row['U_rare']:.9f}\tUvalid={row['U_valid']:.9f}\t"
                f"rare={row['candidate_rare_n']}\tbitwise={row['rare_bitwise_unchanged']}"
            )

    summary["results"].sort(key=lambda row: row["utility"], reverse=True)
    summary_path = out_dir / "common_quantile_micro_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
