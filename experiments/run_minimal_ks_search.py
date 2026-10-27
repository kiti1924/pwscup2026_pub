"""Search and validate minimal common-only KS corrections on round5 final C.

The seven corrections change one continuous value in a public-common row and
cross every tied maximum of that column's empirical CDF discrepancy.  The
search evaluates every subset of size 0..5 with the cached Cox FastSpec
(cache cleared because these are static-column changes), then officially
scores the top three subsets at each size and applies the privacy audits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_WORKSPACE = Path(__file__).resolve().parents[1]
_STARTER = _WORKSPACE / "pwscup2026-kit" / "starter"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))
if str(_STARTER / "src") not in sys.path:
    sys.path.insert(0, str(_STARTER / "src"))

from experiments.run_aligned_synth import _detector_mask, _load_detector, _score_one  # noqa: E402
from experiments.run_outcome_swap_hillclimb import FastSpec  # noqa: E402


CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
STATIC = CONTINUOUS + ["sex", "smoking", "prefecture"]
N = 1049
STEP = 1.0 / N
STATIC_AUC_MAX = 0.578820811686
STATIC_TPR_MAX = 0.033365109628
STATIC_RARE_TPR_MAX = 1.0 / 24.0

# (column, 0-based row, expected old value, target value).  Every move is a
# nearest B order-statistic crossing all tied KS maxima for that column.
MOVES = [
    ("ALT", 149, 16.509825008358018, 16.50917323973691),
    ("SBP", 648, 109.23071528179348, 109.22733543558076),
    ("FPG", 867, 105.49332290936464, 105.51982871942418),
    ("HDL", 719, 51.91617117029884, 51.989687135507324),
    ("BMI", 522, 27.93555512094564, 27.797930800484338),
    ("age", 410, 55.0, 54.0),
    ("TG", 378, 139.90869260453323, 140.43459058622685),
]


def _ks(c: pd.DataFrame, b: pd.DataFrame, col: str) -> float:
    x = np.sort(c[col].to_numpy(dtype=float))
    y = np.sort(b[col].to_numpy(dtype=float))
    values = np.sort(np.unique(np.concatenate([x, y])))
    d = np.searchsorted(x, values, side="right") / len(x)
    d -= np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(d)))


def _rare_mask(c: pd.DataFrame, detector: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
    return _detector_mask(c[CONTINUOUS], *detector)


def _apply(base: pd.DataFrame, subset: tuple[int, ...]) -> pd.DataFrame:
    out = base.copy(deep=True)
    for j in subset:
        col, row, old, target = MOVES[j]
        if not np.isclose(float(out.at[row, col]), old, rtol=0.0, atol=1e-12):
            raise AssertionError(f"source changed before move {col}/{row}")
        out.at[row, col] = target
    return out


def _validate_subset(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    b: pd.DataFrame,
    detector: tuple[np.ndarray, np.ndarray, float],
    base_rare: np.ndarray,
    base_ks: dict[str, float],
    subset: tuple[int, ...],
) -> dict[str, Any]:
    rare = _rare_mask(candidate, detector)
    assert np.array_equal(rare, base_rare), f"rare mask changed for {subset}"
    for col in CONTINUOUS:
        observed = _ks(candidate, b, col)
        expected = base_ks[col] - (STEP if any(MOVES[j][0] == col for j in subset) else 0.0)
        assert np.isclose(observed, expected, rtol=0.0, atol=2e-12), (
            f"KS mismatch {subset}/{col}: observed={observed}, expected={expected}"
        )
    assert (candidate["age"].to_numpy(dtype=float) % 1 == 0).all()
    displacement = float(sum(abs(MOVES[j][3] - MOVES[j][2]) for j in subset))
    return {
        "subset": list(subset),
        "moves": [MOVES[j][0] for j in subset],
        "k": len(subset),
        "displacement": displacement,
        "rare_count": int(rare.sum()),
        "ks": {col: _ks(candidate, b, col) for col in CONTINUOUS},
    }


def _official(
    path: Path, dist: Path, scores_dir: Path, label: str
) -> tuple[dict[str, Any] | None, str | None]:
    score_path = scores_dir / f"{label}.json"
    payload, error = _score_one(path, dist, score_path)
    return payload, error


def _audit(
    script: Path, dist: Path, paths: list[Path], out_path: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--dist",
        str(dist),
        "--c",
        *[str(p) for p in paths],
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(
        command,
        cwd=str(_STARTER),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
    return json.loads(out_path.read_text(encoding="utf-8"))


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    facets = payload.get("facets", {})
    detail = payload.get("detail", {}) or {}
    return {
        "utility": payload.get("utility"),
        "U_gen": facets.get("U_gen"),
        "U_spec": facets.get("U_spec"),
        "U_rare": facets.get("U_rare"),
        "U_valid": facets.get("U_valid"),
        "U_spec_detail": detail.get("U_spec", {}),
        "U_rare_detail": detail.get("U_rare", {}),
    }


def _passes_static(row: dict[str, Any]) -> bool:
    return bool(
        float(row["all_tpr_at_fpr_0.01"]) <= STATIC_TPR_MAX + 1e-12
        and float(row["true_rare_tpr_at_fpr_0.01"]) <= STATIC_RARE_TPR_MAX + 1e-12
        and float(row["roc_auc"]) <= STATIC_AUC_MAX + 1e-12
        and int(row["member_d_le_0.5"]) <= 4
        and int(row["control_d_le_0.5"]) == 0
        and int(row["member_exact_match_13col"]) == 0
        and int(row["control_exact_match_13col"]) == 0
    )


def _passes_privacy(row: dict[str, Any]) -> bool:
    mia = row["mia"]
    aia = row["aia"]
    return bool(
        float(mia["exposure"]) <= 0.3333334 + 1e-12
        and abs(float(aia["attack_score"])) <= 1e-12
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=str(_WORKSPACE / "_private_team10" / "outcome_swap_hillclimb_round5" / "final_best.csv"),
    )
    parser.add_argument("--dist", default=str(_WORKSPACE / "_private_team10"))
    parser.add_argument(
        "--out-dir", default=str(_WORKSPACE / "_private_team10" / "minimal_ks_after_round5")
    )
    args = parser.parse_args(argv)
    started = time.monotonic()

    base_path = Path(args.base).resolve()
    dist = Path(args.dist).resolve()
    out_dir = Path(args.out_dir).resolve()
    scores_dir = out_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(base_path)
    b = pd.read_csv(dist / "B.csv")
    assert len(base) == len(b) == N
    assert list(base.columns) == [
        "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
        "smoking", "FPG", "time", "onset", "death",
    ]
    detector = _load_detector(dist)
    base_rare = _rare_mask(base, detector)
    assert int(base_rare.sum()) == 61
    base_ks = {col: _ks(base, b, col) for col in CONTINUOUS}
    base_rare_count = int(base_rare.sum())

    # Check all source values, schema constraints, and each one-step correction.
    for col, row, old, target in MOVES:
        assert np.isclose(float(base.at[row, col]), old, rtol=0.0, atol=1e-12)
        if col == "age":
            assert float(target).is_integer() and 40 <= target <= 74
        q = _apply(base, (MOVES.index((col, row, old, target)),))
        _validate_subset(base, q, b, detector, base_rare, base_ks, (MOVES.index((col, row, old, target)),))

    # FastSpec is exact for U_spec, but its original cache key is outcomes-only;
    # clear it for every static candidate so different subsets are not aliased.
    fast = FastSpec(b)
    base_recorded = base.copy(deep=True)
    fast.cache.clear()
    base_u_spec, base_spec_detail = fast.score(base_recorded)
    base_official, base_error = _official(base_path, dist, scores_dir, "base")
    if base_error or base_official is None:
        raise RuntimeError(base_error or "base official score failed")
    base_u_gen = float(base_official["facets"]["U_gen"])
    base_u_rare = float(base_official["facets"]["U_rare"])
    base_u_valid = float(base_official["facets"]["U_valid"])

    all_rows: list[dict[str, Any]] = []
    by_size: dict[int, list[dict[str, Any]]] = {}
    for k in range(6):
        for subset in itertools.combinations(range(len(MOVES)), k):
            q = _apply(base, subset)
            invariant = _validate_subset(base, q, b, detector, base_rare, base_ks, subset)
            fast.cache.clear()
            u_spec, spec_detail = fast.score(q)
            u_gen = base_u_gen + k / (N * 10.0)
            objective = min(u_gen, float(u_spec), base_u_rare, base_u_valid)
            row = {
                **invariant,
                "u_gen_fast": u_gen,
                "u_spec_fast": float(u_spec),
                "objective_fast": float(objective),
                "fast_spec_detail": spec_detail,
                "base_rare_count": base_rare_count,
            }
            all_rows.append(row)
            by_size.setdefault(k, []).append(row)

    top: list[dict[str, Any]] = []
    for k in range(6):
        ranked = sorted(
            by_size[k],
            key=lambda r: (-r["objective_fast"], r["displacement"], r["k"], r["subset"]),
        )
        for rank, row in enumerate(ranked[:3], start=1):
            row = dict(row)
            row["rank_within_size"] = rank
            subset = tuple(row["subset"])
            label = f"k{k:02d}_r{rank:02d}_" + ("none" if not subset else "_".join(MOVES[j][0] for j in subset))
            csv_path = out_dir / f"{label}.csv"
            _apply(base, subset).to_csv(csv_path, index=False)
            row["csv"] = csv_path.name
            payload, error = _official(csv_path, dist, scores_dir, label)
            row["official_error"] = error
            row["official"] = _compact(payload) if payload is not None else None
            top.append(row)

    top_paths = [out_dir / row["csv"] for row in top]
    full_static = _audit(
        _WORKSPACE / "experiments" / "full_static_nearest_audit.py",
        dist,
        top_paths,
        out_dir / "full_static_top3.json",
    )
    privacy = _audit(
        _WORKSPACE / "experiments" / "local_privacy_audit.py",
        dist,
        top_paths,
        out_dir / "privacy_top3.json",
    )
    full_by_name = {item["candidate"]: item for item in full_static["candidates"]}
    privacy_by_name = {item["candidate"]: item for item in privacy["candidates"]}
    for row in top:
        row["full_static"] = full_by_name[row["csv"]]
        row["privacy"] = privacy_by_name[row["csv"]]
        row["static_pass"] = _passes_static(row["full_static"])
        row["privacy_pass"] = _passes_privacy(row["privacy"])
        official = row.get("official") or {}
        facets = official
        row["pass"] = bool(row["static_pass"] and row["privacy_pass"] and facets)
        if facets:
            row["objective_official"] = float(
                min(facets["U_gen"], facets["U_spec"], facets["U_rare"], facets["U_valid"])
            )
        else:
            row["objective_official"] = -1.0

    passing = [r for r in top if r["pass"]]
    if not passing:
        raise RuntimeError("No officially scored top-3 candidate passed all gates")
    passing.sort(
        key=lambda r: (
            -r["objective_official"],
            r["full_static"]["roc_auc"],
            r["displacement"],
            r["k"],
        )
    )
    best = passing[0]
    best_path = out_dir / "best.csv"
    best_path.write_bytes((out_dir / best["csv"]).read_bytes())
    best_score_path = scores_dir / "best.json"
    source_score_path = scores_dir / f"{Path(best['csv']).stem}.json"
    best_score_path.write_bytes(source_score_path.read_bytes())

    summary = {
        "base": str(base_path),
        "dist": str(dist),
        "out_dir": str(out_dir),
        "elapsed_seconds": time.monotonic() - started,
        "subset_count": len(all_rows),
        "subsets_by_size": {str(k): len(by_size[k]) for k in by_size},
        "base_fast_U_spec": float(base_u_spec),
        "base_official": _compact(base_official),
        "base_spec_detail": base_spec_detail,
        "moves": [
            {"index": i, "column": m[0], "row": m[1], "old": m[2], "target": m[3]}
            for i, m in enumerate(MOVES)
        ],
        "all_fast_results": all_rows,
        "top3_official_and_audited": top,
        "best": best,
        "best_csv": str(best_path),
        "best_score_json": str(best_score_path),
        "full_static_file": str(out_dir / "full_static_top3.json"),
        "privacy_file": str(out_dir / "privacy_top3.json"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best": best["csv"],
                "k": best["k"],
                "moves": best["moves"],
                "official": best["official"],
                "full_static": best["full_static"],
                "privacy": best["privacy"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
