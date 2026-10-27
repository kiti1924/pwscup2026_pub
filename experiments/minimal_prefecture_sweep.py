"""Minimal one-row prefecture rebalancing sweep for the current final C.

Only ``prefecture`` is changed.  Rows are selected from prefecture-surplus
groups, excluding rows flagged by the disclosed rare detector and ranking by
large standardized static distance from B (with an extra distance guard for
the true-rare B members).  Prefix candidates k=1..12 are written and scored.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "pwscup2026-kit" / "starter"
SRC = STARTER / "src"
if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ``_score_one`` launches a fresh interpreter; make the starter package
# visible there as well (the repository is not installed system-wide).
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(SRC), str(STARTER), os.environ.get("PYTHONPATH", "")]
)

from pwscup2026.common import schema
from pwscup2026.rare import outlier
from pwscup2026.scoring.attack_score import _tpr_at_fpr


BASE = ROOT / "_private_team10" / "calibrated_tilt030_final_fine" / (
    "C_cal_tilt030_finalfine_u15_h1.000_lr0.350_i7.csv"
)
DIST = ROOT / "_private_team10"
OUT = DIST / "minimal_prefecture_sweep"


def _score_one(csv_path: Path, dist: Path, score_path: Path):
    """Run official selfscore with the starter environment's dependencies."""
    py = STARTER / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)
    proc = subprocess.run(
        [str(py), "-m", "pwscup2026.kit.selfscore", str(csv_path), "--dist", str(dist), "--json"],
        cwd=str(STARTER), capture_output=True, text=True, check=False,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "selfscore failed"
        score_path.write_text(json.dumps({"error": msg}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return None, msg
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        msg = f"selfscore JSON parse failed: {exc}"
        score_path.write_text(json.dumps({"error": msg}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return None, msg
    score_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, None


def _std_dist(x: pd.DataFrame, y: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Minimum standardized static distance from each x row to y rows."""
    cols = list(schema.CONTINUOUS)
    scale = b[cols].std(ddof=0).to_numpy(float).copy()
    scale[scale <= 1e-9] = 1.0
    dx = (x[cols].to_numpy(float)[:, None, :] - y[cols].to_numpy(float)[None, :, :]) / scale
    return np.sqrt((dx * dx).sum(axis=2)).min(axis=1)


def _exact_count(c: pd.DataFrame, b: pd.DataFrame) -> int:
    cols = list(c.columns)
    rows = set(b[cols].itertuples(index=False, name=None))
    return int(sum(row in rows for row in c.itertuples(index=False, name=None)))


def _load_rare(dist: Path, c: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ref = json.loads((dist / "utility_ref.json").read_text(encoding="utf-8"))
    c_d = outlier.mahalanobis(c, np.asarray(ref["mu"], float), np.asarray(ref["sigma"], float), schema.CONTINUOUS)
    c_rare = c_d >= float(ref["band"]["lo"])
    b_self = pd.read_csv(dist / "B_self.csv")
    if "record_id" in b.columns:
        b_rare = b_self.set_index("record_id").reindex(b["record_id"])["is_rare"].to_numpy(bool)
    else:
        b_rare = b_self["is_rare"].to_numpy(bool)
    return c_rare, b_rare


def _local_privacy(candidates: list[Path], dist: Path) -> dict:
    """Reuse the disclosed local proxy implementation; aggregate-only output."""
    py = STARTER / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)
    out_path = OUT / "privacy_audit.json"
    proc = subprocess.run(
        [str(py), str(ROOT / "experiments" / "local_privacy_audit.py"),
         "--dist", str(dist), "--c", *[str(p) for p in candidates], "--out", str(out_path)],
        cwd=str(STARTER), capture_output=True, text=True, check=False, env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"local privacy audit failed: {proc.stderr[-1000:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def _static_proxy(c: pd.DataFrame, b: pd.DataFrame, dist: Path, c_rare: np.ndarray, b_rare: np.ndarray) -> dict:
    """Full-static nearest-distance proxy at FPR=.01.

    This follows the prior audit's fixed A_bg sample (seed 0), standardizes
    continuous columns on the combined static pool, and one-hot encodes the
    categorical static columns.  The rare TPR is evaluated at the same
    overall operating threshold, not on a separately re-fit rare ROC.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    a_bg = pd.read_csv(dist / "A_bg.csv")
    controls = a_bg.iloc[np.random.default_rng(0).choice(len(a_bg), len(b), replace=False)].reset_index(drop=True)
    num = list(schema.CONTINUOUS)
    cat = ["sex", "smoking", "prefecture"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    all_x = prep.fit_transform(pd.concat([c, b, controls], ignore_index=True))
    xc = all_x[: len(c)]
    xm = all_x[len(c) : len(c) + len(b)]
    xk = all_x[len(c) + len(b) :]
    member_score = -np.sqrt(((xm[:, None, :] - xc[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    control_score = -np.sqrt(((xk[:, None, :] - xc[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    scores = np.r_[member_score, control_score]
    truth = np.r_[np.ones(len(b), dtype=int), np.zeros(len(controls), dtype=int)]
    tpr, auc = _tpr_at_fpr(scores, truth, 0.01)
    fpr, _, thresholds = roc_curve(truth, scores)
    j = int(np.searchsorted(fpr, 0.01, side="right"))
    j = max(1, min(j, len(fpr) - 1))
    if fpr[j] > fpr[j - 1]:
        frac = (0.01 - fpr[j - 1]) / (fpr[j] - fpr[j - 1])
        threshold = float(thresholds[j - 1] + frac * (thresholds[j] - thresholds[j - 1]))
    else:
        threshold = float(thresholds[j])
    rare_member_tpr = float((member_score[b_rare] >= threshold).mean()) if b_rare.any() else None
    return {
        "all_tpr_at_fpr_0_01": float(tpr),
        "true_rare_tpr_at_fpr_0_01": rare_member_tpr,
        "auc": float(auc),
        "threshold": threshold,
        "definition": "nearest full-static distance to candidate C; combined-pool z-score + one-hot; controls=A_bg seed0",
        "c_rare_detected": int(c_rare.sum()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = OUT / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    c0 = pd.read_csv(BASE)
    b = pd.read_csv(DIST / "B.csv")
    if "record_id" in b:
        b_static = b.drop(columns=["record_id"])
    else:
        b_static = b
    c_rare, b_rare = _load_rare(DIST, c0, b)
    delta = c0["prefecture"].value_counts().sub(b["prefecture"].value_counts(), fill_value=0)
    surplus = {str(k): int(v) for k, v in delta.items() if v > 0}
    deficit = {str(k): int(-v) for k, v in delta.items() if v < 0}

    # Rank eligible source rows by distance from all B and from true-rare B.
    all_d = _std_dist(c0, b_static, b)
    rare_d = _std_dist(c0, b_static.loc[b_rare], b) if b_rare.any() else all_d
    eligible = (~c_rare) & c0["prefecture"].isin(surplus)
    rank_score = all_d + 0.25 * rare_d
    order = sorted(np.flatnonzero(eligible), key=lambda i: (-rank_score[i], -all_d[i], int(i)))
    # Lowest-risk first move identified by the parent audit.
    preferred = 17
    if preferred in order and c0.at[preferred, "prefecture"] == "広島県":
        order = [preferred] + [i for i in order if i != preferred]
    remaining = dict(surplus)
    target_order = sorted(deficit, key=lambda k: (-deficit[k], k))
    moves = []
    used = set()
    ti = 0
    for i in order:
        src = str(c0.at[int(i), "prefecture"])
        if i in used or remaining.get(src, 0) <= 0:
            continue
        while ti < len(target_order) and deficit[target_order[ti]] <= 0:
            ti += 1
        if ti >= len(target_order):
            break
        dst = "奈良県" if len(moves) == 0 else target_order[ti]
        if len(moves) == 0:
            deficit[dst] -= 1
            if deficit[dst] == 0:
                target_order = [x for x in target_order if x != dst]
                ti = 0
        remaining[src] -= 1
        if len(moves) > 0:
            deficit[dst] -= 1
        if len(moves) > 0 and deficit[dst] == 0:
            ti += 1
        used.add(int(i))
        moves.append({"row_index": int(i), "from": src, "to": dst, "distance_all_B": float(all_d[i]), "distance_true_rare_B": float(rare_d[i]), "rank_score": float(rank_score[i])})
        if len(moves) >= 12:
            break

    candidates = []
    for k in range(1, len(moves) + 1):
        c = c0.copy(deep=True)
        for m in moves[:k]:
            c.at[m["row_index"], "prefecture"] = m["to"]
        path = OUT / f"C_minpref_prefix{k:02d}.csv"
        c.to_csv(path, index=False)
        score_path = scores / f"C_minpref_prefix{k:02d}.json"
        if score_path.exists():
            try:
                payload = json.loads(score_path.read_text(encoding="utf-8"))
                err = None if "utility" in payload else "cached score has no utility"
            except json.JSONDecodeError:
                payload, err = _score_one(path, DIST, score_path)
        else:
            payload, err = _score_one(path, DIST, score_path)
        if err or payload is None:
            raise RuntimeError(f"selfscore failed for {path.name}: {err}")
        candidates.append((path, c, payload))
        print(f"k={k}\tU={payload['utility']:.12f}\tU_gen={payload['facets']['U_gen']:.12f}\tU_spec={payload['facets']['U_spec']:.12f}\tU_rare={payload['facets']['U_rare']:.12f}")

    privacy = _local_privacy([p for p, _, _ in candidates], DIST)
    privacy_by = {x["candidate"]: x for x in privacy["candidates"]}
    results = []
    for path, c, payload in candidates:
        static = _static_proxy(c, b_static, DIST, c_rare, b_rare)
        mia = privacy_by[path.name]["mia"]
        aia = privacy_by[path.name]["aia"]
        facets = payload["facets"]
        results.append({
            "k": int(path.stem[-2:]),
            "csv": path.name,
            "score_json": str((scores / f"{path.stem}.json").name),
            "utility": float(payload["utility"]),
            **{k: float(v) for k, v in facets.items()},
            "MIA_exposure": mia["exposure"],
            "AIA": aia["attack_score"],
            "exact_matches": _exact_count(c, b),
            "static_full": static,
        })
    report = {
        "baseline": {"csv": BASE.name, "U": 0.9712106768350809, "U_spec": 0.9719340199677773, "U_rare": 0.9721569112733572, "U_valid": 1.0},
        "base_csv": str(BASE),
        "B_prefecture_counts": {str(k): int(v) for k, v in b["prefecture"].value_counts().sort_index().items()},
        "C_prefecture_counts": {str(k): int(v) for k, v in c0["prefecture"].value_counts().sort_index().items()},
        "delta_C_minus_B": {str(k): int(v) for k, v in delta.items() if v},
        "c_rare_detected_baseline": int(c_rare.sum()),
        "b_true_rare": int(b_rare.sum()),
        "moves": moves,
        "results": results,
        "privacy_audit": privacy,
        "selection_criteria": {"U": ">0.9712106768", "U_spec": ">=0.9718", "U_rare": ">=0.9720", "U_valid": "=1", "MIA_exposure": "<=0.3333334", "AIA": "=0", "exact_matches": "=0", "static_true_rare_tpr": "<=0.041667"},
    }
    (OUT / "minimal_prefecture_sweep_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "moves.json").write_text(json.dumps(moves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary={OUT / 'minimal_prefecture_sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
