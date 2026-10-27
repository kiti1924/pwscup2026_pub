"""Probe a few privacy-screened marginal-preserving static swaps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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

from experiments.run_aligned_synth import _load_detector  # noqa: E402
from experiments.run_minimal_ks_search import (  # noqa: E402
    CONTINUOUS,
    N,
    _audit,
    _compact,
    _official,
    _passes_privacy,
    _passes_static,
    _rare_mask,
)
from experiments.run_outcome_swap_hillclimb import FastSpec  # noqa: E402


SWAPS = {
    "smoke_309_702": ("smoking", 309, 702),
    "ALT_394_886": ("ALT", 394, 886),
    "ALT_601_756": ("ALT", 601, 756),
    "ALT_66_741": ("ALT", 66, 741),
}
AGE_MOVE = ("age", 410, 55.0, 54.0)


def _apply(base: pd.DataFrame, swap_name: str | None, add_age: bool) -> pd.DataFrame:
    out = base.copy(deep=True)
    if swap_name is not None:
        col, i, j = SWAPS[swap_name]
        left, right = out.at[i, col], out.at[j, col]
        assert left != right
        out.at[i, col], out.at[j, col] = right, left
    if add_age:
        col, row, old, target = AGE_MOVE
        assert np.isclose(float(out.at[row, col]), old, rtol=0.0, atol=1e-12)
        out.at[row, col] = target
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", default=str(_WORKSPACE / "_private_team10" / "rare_ks_after_round6" / "best.csv")
    )
    parser.add_argument("--dist", default=str(_WORKSPACE / "_private_team10"))
    parser.add_argument(
        "--out-dir", default=str(_WORKSPACE / "_private_team10" / "static_swap_after_rare")
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    base_path, dist, out_dir = Path(args.base).resolve(), Path(args.dist).resolve(), Path(args.out_dir).resolve()
    scores_dir = out_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    base, b = pd.read_csv(base_path), pd.read_csv(dist / "B.csv")
    assert len(base) == len(b) == N
    detector = _load_detector(dist)
    base_rare = _rare_mask(base, detector)
    assert int(base_rare.sum()) == 61
    base_official, error = _official(base_path, dist, scores_dir, "base")
    if error or base_official is None:
        raise RuntimeError(error or "base score failed")
    base_ugen = float(base_official["facets"]["U_gen"])
    base_urare = float(base_official["facets"]["U_rare"])
    fast = FastSpec(b)
    rows = []
    for swap_name in [None, *SWAPS]:
        for add_age in (False, True):
            if swap_name is None and not add_age:
                continue
            q = _apply(base, swap_name, add_age)
            assert np.array_equal(_rare_mask(q, detector), base_rare)
            # A pure swap preserves its complete column multiset exactly.
            if swap_name is not None:
                col = SWAPS[swap_name][0]
                assert np.array_equal(np.sort(q[col].to_numpy()), np.sort(base[col].to_numpy()))
            fast.cache.clear()
            uspec, detail = fast.score(q)
            ugen = base_ugen + ((1.0 / (N * 10.0)) if add_age else 0.0)
            rows.append(
                {
                    "swap": swap_name,
                    "add_age": add_age,
                    "ugen_fast": float(ugen),
                    "uspec_fast": float(uspec),
                    "objective_fast": float(min(ugen, uspec, base_urare)),
                    "fast_spec_detail": detail,
                }
            )
    rows.sort(key=lambda r: (-r["objective_fast"], r["swap"] or "", r["add_age"]))
    top = rows[:5]
    for rank, row in enumerate(top, start=1):
        label = f"r{rank:02d}_{row['swap'] or 'none'}_age{int(row['add_age'])}"
        path = out_dir / f"{label}.csv"
        _apply(base, row["swap"], row["add_age"]).to_csv(path, index=False)
        payload, error = _official(path, dist, scores_dir, label)
        row["csv"] = path.name
        row["official_error"] = error
        row["official"] = _compact(payload) if payload else None
    paths = [out_dir / row["csv"] for row in top]
    full = _audit(
        _WORKSPACE / "experiments" / "full_static_nearest_audit.py",
        dist,
        paths,
        out_dir / "full_static.json",
    )
    privacy = _audit(
        _WORKSPACE / "experiments" / "local_privacy_audit.py",
        dist,
        paths,
        out_dir / "privacy.json",
    )
    full_by = {row["candidate"]: row for row in full["candidates"]}
    privacy_by = {row["candidate"]: row for row in privacy["candidates"]}
    for row in top:
        row["full_static"] = full_by[row["csv"]]
        row["privacy"] = privacy_by[row["csv"]]
        row["pass"] = bool(
            row["official"]
            and _passes_static(row["full_static"])
            and _passes_privacy(row["privacy"])
        )
        row["objective_official"] = (
            min(
                row["official"]["U_gen"],
                row["official"]["U_spec"],
                row["official"]["U_rare"],
                row["official"]["U_valid"],
            )
            if row["official"]
            else -1.0
        )
    passing = [row for row in top if row["pass"]]
    if not passing:
        raise RuntimeError("no static swap candidate passed")
    passing.sort(key=lambda r: (-r["objective_official"], r["full_static"]["roc_auc"]))
    best = passing[0]
    (out_dir / "best.csv").write_bytes((out_dir / best["csv"]).read_bytes())
    (scores_dir / "best.json").write_bytes(
        (scores_dir / f"{Path(best['csv']).stem}.json").read_bytes()
    )
    summary = {
        "base": str(base_path),
        "base_official": _compact(base_official),
        "elapsed_seconds": time.monotonic() - started,
        "all_fast": rows,
        "top": top,
        "best": best,
        "best_csv": str(out_dir / "best.csv"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best": best, "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
