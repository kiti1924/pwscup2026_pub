"""Search tiny rare-profile corrections on the validated round6 candidate."""

from __future__ import annotations

import argparse
import itertools
import json
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

from experiments.run_aligned_synth import _load_detector  # noqa: E402
from experiments.run_minimal_ks_search import (  # noqa: E402
    CONTINUOUS,
    N,
    _audit,
    _compact,
    _ks,
    _official,
    _passes_privacy,
    _passes_static,
    _rare_mask,
)
from experiments.run_outcome_swap_hillclimb import FastSpec  # noqa: E402


# General-population corrections each improve one full-data KS statistic by 1/N.
GENERAL = {
    "gen_ALT": ("ALT", 149, 16.509825008358018, 16.50917323973691),
    "gen_age": ("age", 410, 55.0, 54.0),
}

# One-cell corrections crossing a maximum of the rare-profile empirical CDF.
RARE = {
    "rare_age": ("age", 790, 45.0, 47.0),
    "rare_BMI": ("BMI", 67, 21.18155124941159, 21.035130964203056),
    "rare_SBP": ("SBP", 161, 132.70139494066584, 123.73106120379802),
    "rare_TG": ("TG", 285, 95.2165632183394, 91.12107119059311),
    "rare_HDL": ("HDL", 847, 48.55467080716003, 48.923507230655865),
    "rare_ALT": ("ALT", 802, 52.29923361388465, 37.58303354063965),
    "rare_FPG": ("FPG", 259, 97.3668416115016, 100.0),
}


def _apply(base: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    out = base.copy(deep=True)
    for name in names:
        col, row, old, target = (GENERAL | RARE)[name]
        observed = float(out.at[row, col])
        if not np.isclose(observed, old, rtol=0.0, atol=1e-12):
            raise AssertionError(f"source mismatch {name}: {observed} != {old}")
        out.at[row, col] = target
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=str(_WORKSPACE / "_private_team10" / "outcome_swap_hillclimb_round6b" / "final_best.csv"),
    )
    parser.add_argument("--dist", default=str(_WORKSPACE / "_private_team10"))
    parser.add_argument(
        "--out-dir", default=str(_WORKSPACE / "_private_team10" / "rare_ks_after_round6")
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
    detector = _load_detector(dist)
    base_rare = _rare_mask(base, detector)
    assert int(base_rare.sum()) == 61
    base_ks = {col: _ks(base, b, col) for col in CONTINUOUS}
    base_official, error = _official(base_path, dist, scores_dir, "base")
    if error or base_official is None:
        raise RuntimeError(error or "base scoring failed")
    base_ugen = float(base_official["facets"]["U_gen"])

    # Test each rare correction with no extra general move, ALT only, and ALT+age.
    name_sets: list[tuple[str, ...]] = []
    for rare_name in RARE:
        for general_names in ((), ("gen_ALT",), ("gen_ALT", "gen_age")):
            name_sets.append((rare_name, *general_names))
    # Also retain the previously verified three-rare profile correction.
    for general_names in ((), ("gen_ALT",), ("gen_ALT", "gen_age")):
        name_sets.append(("rare_age", "rare_BMI", "rare_TG", *general_names))

    fast = FastSpec(b)
    rows: list[dict[str, Any]] = []
    for names in name_sets:
        candidate = _apply(base, names)
        rare_mask = _rare_mask(candidate, detector)
        assert np.array_equal(rare_mask, base_rare), f"rare mask changed: {names}"
        assert (candidate["age"].to_numpy(dtype=float) % 1 == 0).all()
        cand_ks = {col: _ks(candidate, b, col) for col in CONTINUOUS}
        delta_ugen = sum(base_ks[col] - cand_ks[col] for col in CONTINUOUS) / 10.0
        ugen_fast = base_ugen + delta_ugen
        fast.cache.clear()
        uspec_fast, detail = fast.score(candidate)
        # Every single rare move was already measured above 0.9795, so Urare
        # cannot bind while ranking candidates near 0.9783.
        objective_fast = min(float(ugen_fast), float(uspec_fast))
        rows.append(
            {
                "names": list(names),
                "ugen_fast": float(ugen_fast),
                "uspec_fast": float(uspec_fast),
                "objective_fast": float(objective_fast),
                "rare_count": int(rare_mask.sum()),
                "ks": cand_ks,
                "fast_spec_detail": detail,
            }
        )

    rows.sort(key=lambda r: (-r["objective_fast"], len(r["names"]), r["names"]))
    top = rows[:7]
    for rank, row in enumerate(top, start=1):
        names = tuple(row["names"])
        label = f"r{rank:02d}_" + "_".join(names)
        path = out_dir / f"{label}.csv"
        _apply(base, names).to_csv(path, index=False)
        payload, error = _official(path, dist, scores_dir, label)
        row["csv"] = path.name
        row["official_error"] = error
        row["official"] = _compact(payload) if payload is not None else None

    paths = [out_dir / row["csv"] for row in top]
    full_static = _audit(
        _WORKSPACE / "experiments" / "full_static_nearest_audit.py",
        dist,
        paths,
        out_dir / "full_static_top.json",
    )
    privacy = _audit(
        _WORKSPACE / "experiments" / "local_privacy_audit.py",
        dist,
        paths,
        out_dir / "privacy_top.json",
    )
    static_by_name = {row["candidate"]: row for row in full_static["candidates"]}
    privacy_by_name = {row["candidate"]: row for row in privacy["candidates"]}
    for row in top:
        row["full_static"] = static_by_name[row["csv"]]
        row["privacy"] = privacy_by_name[row["csv"]]
        row["static_pass"] = _passes_static(row["full_static"])
        row["privacy_pass"] = _passes_privacy(row["privacy"])
        official = row.get("official") or {}
        row["pass"] = bool(row["static_pass"] and row["privacy_pass"] and official)
        row["objective_official"] = (
            min(official["U_gen"], official["U_spec"], official["U_rare"], official["U_valid"])
            if official
            else -1.0
        )

    passing = [row for row in top if row["pass"]]
    if not passing:
        raise RuntimeError("no rare correction passed all gates")
    passing.sort(
        key=lambda r: (
            -r["objective_official"],
            r["full_static"]["roc_auc"],
            len(r["names"]),
        )
    )
    best = passing[0]
    (out_dir / "best.csv").write_bytes((out_dir / best["csv"]).read_bytes())
    (scores_dir / "best.json").write_bytes(
        (scores_dir / f"{Path(best['csv']).stem}.json").read_bytes()
    )
    summary = {
        "base": str(base_path),
        "dist": str(dist),
        "elapsed_seconds": time.monotonic() - started,
        "base_official": _compact(base_official),
        "all_fast": rows,
        "top_official_audited": top,
        "best": best,
        "best_csv": str(out_dir / "best.csv"),
        "best_score": str(scores_dir / "best.json"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best": best, "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
