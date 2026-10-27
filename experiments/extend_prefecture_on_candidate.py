"""Extend the deterministic prefecture overlay from prefix 10 to prefix 62.

This script keeps the current best candidate's non-prefecture columns and
replaces only its prefecture column with the deterministic prefix overlay
derived from the original prefix-0 candidate.  It writes checkpoint files,
runs selfscore plus the fixed full-static and local privacy audits, and picks
the highest-U_gen checkpoint that passes every requested gate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "pwscup2026-kit" / "starter"
SRC = STARTER / "src"
if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(SRC), str(STARTER), os.environ.get("PYTHONPATH", "")]
)

from pwscup2026.common import schema
from pwscup2026.rare import outlier


DIST = ROOT / "_private_team10"
BASE = DIST / "calibrated_tilt030_final_fine" / (
    "C_cal_tilt030_finalfine_u15_h1.000_lr0.350_i7.csv"
)
CURRENT = DIST / "outcome_search_rare_0350" / "micro" / "M_01_FPG_log_TG_+1+1.csv"
OUT = DIST / "prefecture_extended_best0350"
OLD_MOVES = DIST / "minimal_prefecture_sweep" / "moves.json"
CHECKPOINTS = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 62]
NUM = list(schema.CONTINUOUS)


def _std_dist(x: pd.DataFrame, y: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    scale = b[NUM].std(ddof=0).to_numpy(float)
    scale[scale <= 1e-9] = 1.0
    dx = (x[NUM].to_numpy(float)[:, None, :] - y[NUM].to_numpy(float)[None, :, :]) / scale
    return np.sqrt((dx * dx).sum(axis=2)).min(axis=1)


def _load_rare(c: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ref = json.loads((DIST / "utility_ref.json").read_text(encoding="utf-8"))
    c_d = outlier.mahalanobis(
        c,
        np.asarray(ref["mu"], float),
        np.asarray(ref["sigma"], float),
        NUM,
    )
    c_rare = c_d >= float(ref["band"]["lo"])
    b_self = pd.read_csv(DIST / "B_self.csv")
    values = b_self.set_index("record_id").reindex(b["record_id"])["is_rare"]
    if values.isna().any():
        raise ValueError("B/B_self record_id alignment failed")
    return c_rare, values.to_numpy(bool)


def _build_moves(c0: pd.DataFrame, b: pd.DataFrame, max_moves: int = 62) -> list[dict[str, Any]]:
    b_static = b.drop(columns=["record_id"]) if "record_id" in b else b
    c_rare, b_rare = _load_rare(c0, b)
    all_d = _std_dist(c0, b_static, b)
    rare_d = _std_dist(c0, b_static.loc[b_rare], b) if b_rare.any() else all_d
    delta = c0["prefecture"].value_counts().sub(b["prefecture"].value_counts(), fill_value=0)
    surplus = {str(k): int(v) for k, v in delta.items() if v > 0}
    deficit = {str(k): int(-v) for k, v in delta.items() if v < 0}
    eligible = (~c_rare) & c0["prefecture"].isin(surplus)
    rank_score = all_d + 0.25 * rare_d
    order = sorted(np.flatnonzero(eligible), key=lambda i: (-rank_score[i], -all_d[i], int(i)))
    preferred = 17
    if preferred in order and c0.at[preferred, "prefecture"] == "広島県":
        order = [preferred] + [i for i in order if i != preferred]

    remaining = dict(surplus)
    target_order = sorted(deficit, key=lambda k: (-deficit[k], k))
    moves: list[dict[str, Any]] = []
    used: set[int] = set()
    ti = 0
    for i in order:
        src = str(c0.at[int(i), "prefecture"])
        if int(i) in used or remaining.get(src, 0) <= 0:
            continue
        while ti < len(target_order) and deficit[target_order[ti]] <= 0:
            ti += 1
        if ti >= len(target_order):
            break
        dst = "奈良県" if not moves else target_order[ti]
        deficit[dst] -= 1
        if deficit[dst] == 0:
            target_order = [x for x in target_order if x != dst]
            ti = 0
        remaining[src] -= 1
        used.add(int(i))
        moves.append({
            "row_index": int(i),
            "from": src,
            "to": dst,
            "distance_all_B": float(all_d[i]),
            "distance_true_rare_B": float(rare_d[i]),
            "rank_score": float(rank_score[i]),
        })
        if len(moves) >= max_moves:
            break
    if len(moves) != max_moves:
        raise ValueError(f"deterministic move list has {len(moves)} rows, expected {max_moves}")
    return moves


def _assert_first12(moves: list[dict[str, Any]]) -> None:
    old = json.loads(OLD_MOVES.read_text(encoding="utf-8"))
    if moves[:12] != old[:12]:
        raise AssertionError("first 12 deterministic moves differ from existing moves.json")


def _assert_overlay_inputs(base: pd.DataFrame, current: pd.DataFrame) -> None:
    if list(base.columns) != list(current.columns):
        raise AssertionError("base/current column topology differs")
    if len(base) != 1049 or len(current) != 1049:
        raise AssertionError("candidate row count is not 1049")
    # The current best intentionally contains non-prefecture improvements
    # (e.g. FPG/TG).  Those 12 columns are retained; only prefecture is
    # replaced by the deterministic overlay derived from prefix-0.


def _score_one(path: Path) -> dict[str, Any]:
    py = STARTER / ".venv" / "Scripts" / "python.exe"
    proc = subprocess.run(
        [str(py), "-m", "pwscup2026.kit.selfscore", str(path), "--dist", str(DIST), "--json"],
        cwd=str(STARTER), capture_output=True, text=True, check=False, env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"selfscore failed for {path.name}: {proc.stderr[-1000:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"selfscore JSON parse failed for {path.name}: {exc}") from exc


def _run_full_static(paths: list[Path], out: Path) -> dict[str, Any]:
    py = STARTER / ".venv" / "Scripts" / "python.exe"
    script = ROOT / "experiments" / "full_static_nearest_audit.py"
    proc = subprocess.run(
        [str(py), str(script), "--dist", str(DIST), "--c", *map(str, paths), "--out", str(out)],
        cwd=str(STARTER), capture_output=True, text=True, check=False, env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"full-static failed: {proc.stderr[-1000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def _run_local(paths: list[Path], out: Path) -> dict[str, Any]:
    py = STARTER / ".venv" / "Scripts" / "python.exe"
    script = ROOT / "experiments" / "local_privacy_audit.py"
    proc = subprocess.run(
        [str(py), str(script), "--dist", str(DIST), "--c", *map(str, paths), "--out", str(out)],
        cwd=str(STARTER), capture_output=True, text=True, check=False, env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"local privacy failed: {proc.stderr[-1000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def _passes(row: dict[str, Any], score: dict[str, Any], local: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    eps = 1e-12
    if float(row["all_tpr_at_fpr_0.01"]) > 0.033365109628 + eps:
        reasons.append("full_static_allTPR")
    if float(row["true_rare_tpr_at_fpr_0.01"]) > (1.0 / 24.0) + eps:
        reasons.append("full_static_rareTPR")
    if float(row["roc_auc"]) > 0.578820811686 + eps:
        reasons.append("full_static_AUC")
    if int(row["member_d_le_0.5"]) > 4:
        reasons.append("full_static_member_d")
    if int(row["control_d_le_0.5"]) != 0:
        reasons.append("full_static_control_d")
    if int(row.get("member_exact_match_13col", 0)) != 0 or int(row.get("control_exact_match_13col", 0)) != 0:
        reasons.append("full_static_exact13")
    if float(score["facets"]["U_rare"]) < 0.9780967702 - eps:
        reasons.append("U_rare")
    if float(score["facets"]["U_valid"]) != 1.0:
        reasons.append("U_valid")
    if int(score.get("exact_matches", 0)) != 0:
        reasons.append("selfscore_exact")
    if float(local["mia"]["exposure"]) > 0.3333334 + eps:
        reasons.append("MIA_exposure")
    if abs(float(local["aia"]["attack_score"])) > eps:
        reasons.append("AIA")
    return not reasons, reasons


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c0 = pd.read_csv(BASE)
    current = pd.read_csv(CURRENT)
    _assert_overlay_inputs(c0, current)
    b = pd.read_csv(DIST / "B.csv")
    moves = _build_moves(c0, b, 62)
    _assert_first12(moves)
    (OUT / "moves.json").write_text(json.dumps(moves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    paths: dict[int, Path] = {}
    scores: dict[int, dict[str, Any]] = {}
    for k in CHECKPOINTS:
        c = current.copy(deep=True)
        pref = c0["prefecture"].copy()
        for m in moves[:k]:
            if str(pref.iloc[m["row_index"]]) != m["from"]:
                raise AssertionError(f"move source mismatch at row {m['row_index']} for k={k}")
            pref.iloc[m["row_index"]] = m["to"]
        c["prefecture"] = pref.to_numpy()
        if k == 10 and not c["prefecture"].equals(current["prefecture"]):
            raise AssertionError("current best is not the claimed prefix10 prefecture overlay")
        if not np.array_equal(c.drop(columns=["prefecture"]).astype(str).to_numpy(), current.drop(columns=["prefecture"]).astype(str).to_numpy()):
            raise AssertionError(f"non-prefecture columns changed at k={k}")
        path = OUT / f"C_prefext_prefix{k:02d}.csv"
        c.to_csv(path, index=False)
        paths[k] = path
        scores[k] = _score_one(path)
        score_path = OUT / "scores"
        score_path.mkdir(exist_ok=True)
        (score_path / f"C_prefext_prefix{k:02d}.json").write_text(json.dumps(scores[k], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"k={k} Ugen={scores[k]['facets']['U_gen']:.12f} Urare={scores[k]['facets']['U_rare']:.12f}")

    ordered_paths = [paths[k] for k in CHECKPOINTS]
    static = _run_full_static(ordered_paths, OUT / "full_static_checkpoints.json")
    local = _run_local(ordered_paths, OUT / "privacy_checkpoints.json")
    static_by = {r["candidate"]: r for r in static["candidates"]}
    local_by = {r["candidate"]: r for r in local["candidates"]}
    results: list[dict[str, Any]] = []
    for k in CHECKPOINTS:
        name = paths[k].name
        ok, reasons = _passes(static_by[name], scores[k], local_by[name])
        results.append({"k": k, "csv": name, "passes": ok, "fail_reasons": reasons, "score": scores[k], "full_static": static_by[name], "privacy": local_by[name]})
        print(f"gate k={k}: {'PASS' if ok else 'FAIL ' + ','.join(reasons)}")
    safe = [r for r in results if r["passes"]]
    if not safe:
        raise RuntimeError("no checkpoint passed all requested gates")
    best = max(safe, key=lambda r: float(r["score"]["facets"]["U_gen"]))
    report = {"base": str(BASE), "current_best": str(CURRENT), "checkpoints": CHECKPOINTS, "moves": moves, "results": results, "best": best, "gates": {"allTPR_le": 0.033365109628, "rare_le": 1.0 / 24.0, "AUC_le": 0.578820811686, "member_d_le_0.5_le": 4, "control_d_le_0.5_eq": 0, "exact13_eq": 0, "MIA_exposure_le": 0.3333334, "AIA_eq": 0, "Urare_ge": 0.9780967702, "Uvalid_eq": 1}}
    (OUT / "extension_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"best={best['csv']} Ugen={best['score']['facets']['U_gen']:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
