"""Match the prefecture marginal while breaking row-level associations.

The official utility scorer uses ``prefecture`` only through its marginal TVD
and the no-fabrication group checks.  Reference AIA, in contrast, uses it as a
blocking key.  This experiment therefore copies only the aggregate category
counts from B, randomly permutes that multiset, and assigns it independently to
an already synthetic C.  No record identifiers or row correspondence are used.
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

from experiments.run_aligned_synth import _score_one  # noqa: E402
from reference._common import dist_paths, read_b, write_c  # noqa: E402
from pwscup2026.common import submission_schema  # noqa: E402


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Independent prefecture-marginal rebalance")
    ap.add_argument("--dist", required=True)
    ap.add_argument("--base-csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    try:
        dist = dist_paths(args.dist)
        b = read_b(dist)
        base = pd.read_csv(args.base_csv)
        expected = list(submission_schema.SUBMITTED_C_COLUMNS)
        if list(base.columns) != expected or len(base) != len(b):
            raise ValueError("base C has an unexpected shape or columns")
        if not args.tag.replace("_", "").replace("-", "").isalnum():
            raise ValueError("tag must contain only letters, digits, underscore, or hyphen")
        seeds = _parse_seeds(args.seeds)
        out_dir = Path(args.out_dir) if args.out_dir else dist / "prefecture_rebalance"
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_dir = out_dir / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    prefecture_multiset = b["prefecture"].to_numpy(copy=True)
    source_counts = pd.Series(prefecture_multiset).value_counts(dropna=False).sort_index()
    summary: dict[str, Any] = {
        "base_csv": Path(args.base_csv).name,
        "tag": args.tag,
        "privacy_note": (
            "Only aggregate prefecture counts are retained; their assignment to synthetic rows "
            "is an independent seeded permutation."
        ),
        "results": [],
    }
    print("seed\tU\tU_gen\tU_spec\tU_rare\tU_valid\tno_fab")
    failures = 0
    for seed in seeds:
        csv_path = out_dir / f"C_pref_{args.tag}_s{seed}.csv"
        score_path = scores_dir / f"pref_{args.tag}_s{seed}.json"
        try:
            rng = np.random.default_rng(int(seed))
            candidate = base.copy(deep=True)
            candidate["prefecture"] = rng.permutation(prefecture_multiset)
            candidate_counts = candidate["prefecture"].value_counts(dropna=False).sort_index()
            if not source_counts.equals(candidate_counts):
                raise RuntimeError("prefecture marginal mismatch after permutation")
            write_c(candidate, csv_path)
            payload, error = _score_one(csv_path, dist, score_path)
            if payload is None or error is not None:
                raise RuntimeError(error or "selfscore returned no payload")
            facets = payload["facets"]
            no_fab = float(payload["detail"]["U_valid"]["no_fab"])
            row = {
                "seed": int(seed),
                "csv": csv_path.name,
                "score_json": score_path.name,
                "utility": float(payload["utility"]),
                **{key: float(value) for key, value in facets.items()},
                "no_fab_score": no_fab,
            }
            summary["results"].append(row)
            print(
                f"{seed}\t{row['utility']:.6f}\t{row['U_gen']:.6f}\t"
                f"{row['U_spec']:.6f}\t{row['U_rare']:.6f}\t{row['U_valid']:.6f}\t"
                f"{row['no_fab_score']:.6f}"
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{seed}\tERROR\t{type(exc).__name__}: {exc}")

    summary["results"].sort(key=lambda row: (row["utility"], row["U_gen"]), reverse=True)
    summary_path = out_dir / f"prefecture_{args.tag}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary={summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
