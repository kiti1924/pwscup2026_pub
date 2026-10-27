"""Frozen, explicitly non-official B14 privacy diagnostics (G03).

Reference/background fitting rows and evaluated controls are disjoint.
No background rows may be used to train the candidate for this protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "pwscup2026-kit/starter/src")]

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pwscup2026.attack.mia import overlap_mia
from pwscup2026.attack.aia import aia_predict
from pwscup2026.scoring.attack_score import _tpr_at_fpr
from experiments.local_privacy_audit import _load_inputs, _member_rare_mask, _rare_mask_from_reference
from experiments.full_static_nearest_audit import _nearest_distance, _exact_match_count, NUMERIC

QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
CORR = ["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_background(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= 2500:
        raise ValueError("Need more than 2500 background rows")
    order = np.random.default_rng(seed).permutation(n)
    return order[:2500], order[2500:]


def modal_window(times: np.ndarray) -> float:
    """Maximize a closed one-year window; deterministic earliest-window tie."""
    ordered = np.sort(np.asarray(times, dtype=float))
    if not len(ordered):
        raise ValueError("Empty times")
    ends = np.searchsorted(ordered, ordered + 1.0, side="right")
    counts = ends - np.arange(len(ordered))
    start = int(np.argmax(counts))
    return float((ordered[start] + ordered[ends[start] - 1]) / 2.0)


def log_qi_predict(c: pd.DataFrame, query: pd.DataFrame, bg: pd.DataFrame, modal: bool) -> np.ndarray:
    if list(query.columns) != QI:
        raise ValueError("Prediction query must contain only public QI, in frozen order")
    cont = QI[:5]
    def transform(frame: pd.DataFrame) -> np.ndarray:
        x = frame[cont].to_numpy(dtype=float).copy()
        x[:, [2, 4]] = np.log1p(x[:, [2, 4]])
        return x
    scale = transform(bg).std(axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    cx, qx = transform(c) / scale, transform(query) / scale
    csex, cpref = c.sex.to_numpy(), c.prefecture.to_numpy()
    ct = c.time.to_numpy(dtype=float)
    result = []
    for i, row in enumerate(query.itertuples(index=False)):
        idx = np.flatnonzero((csex == row.sex) & (cpref == row.prefecture))
        if not len(idx):
            idx = np.flatnonzero(csex == row.sex)
        if not len(idx):
            idx = np.arange(len(c))
        d = np.mean((cx[idx] - qx[i]) ** 2, axis=1)
        kth = np.partition(d, min(20, len(d)) - 1)[min(20, len(d)) - 1] if modal else d.min()
        ts = ct[idx[d <= kth]]
        result.append(modal_window(ts) if modal else float(np.sort(ts).mean()))
    return np.asarray(result)


def success_metrics(member_hit: np.ndarray, control_hit: np.ndarray) -> dict:
    if not len(member_hit) or not len(control_hit):
        return {"n_member": len(member_hit), "n_control": len(control_hit), "p_m": None, "p_c": None, "r_raw": None, "r_se_adjusted": None}
    pm, pc = float(member_hit.mean()), float(control_hit.mean())
    se = np.sqrt(pm * (1 - pm) / len(member_hit) + pc * (1 - pc) / len(control_hit))
    den = max(1 - pc, 1e-9)
    return {"n_member": len(member_hit), "n_control": len(control_hit), "p_m": pm, "p_c": pc,
            "r_raw": float(np.clip((pm - pc) / den, 0, 1)),
            "r_se_adjusted": float(np.clip((pm - pc - 1.64 * se) / den, 0, 1))}


def standardized_aia(b: pd.DataFrame, controls: pd.DataFrame, mh: np.ndarray, ch: np.ndarray) -> dict:
    keys = ["sex", "prefecture", "onset", "age_bin"]
    def table(frame, hit):
        t = frame[["sex", "prefecture", "onset", "age"]].copy()
        t["age_bin"] = (t.age // 10).astype(int)
        t["hit"] = hit.astype(float)
        return t.groupby(keys).hit.agg(["mean", "size"])
    mt, ct = table(b, mh), table(controls, ch)
    common = mt.index.intersection(ct.index)
    if not len(common):
        return {"member_coverage": 0.0, "p_m": None, "p_c": None, "r_raw": None}
    weights = mt.loc[common, "size"].to_numpy(dtype=float)
    pm = float(np.average(mt.loc[common, "mean"], weights=weights))
    pc = float(np.average(ct.loc[common, "mean"], weights=weights))
    return {"member_coverage": float(weights.sum() / len(b)), "n_common_strata": len(common),
            "min_control_stratum": int(ct.loc[common, "size"].min()), "p_m": pm, "p_c": pc,
            "r_raw": float(np.clip((pm - pc) / max(1 - pc, 1e-9), 0, 1))}


def mia_metrics(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    y, s = labels[mask], scores[mask]
    if len(np.unique(y)) != 2:
        return {"tpr_at_fpr_0.01": None, "auc": None}
    tpr, auc = _tpr_at_fpr(s, y, 0.01)
    return {"tpr_at_fpr_0.01": float(tpr), "auc": float(auc),
            "n_member": int(y.sum()), "n_control": int(len(y) - y.sum())}


def evaluate(candidate: Path, seed: int, out: Path) -> dict:
    tag = f"{candidate.stem}_fold{seed}"
    result_path = out / f"{tag}.json"
    if result_path.exists():
        raise FileExistsError(f"Completed fold exists: {result_path}")
    start = time.monotonic()
    dist = ROOT / "_private_team14"
    b, bs, bg, cfg = _load_inputs(dist)
    pd.testing.assert_frame_equal(b, bs.drop(columns="is_rare"))
    c = pd.read_csv(candidate)
    tr, te = split_background(len(bg), seed)
    ref, controls = bg.iloc[tr].reset_index(drop=True), bg.iloc[te].reset_index(drop=True)
    rare = _member_rare_mask(b, bs)
    dr, cr = _rare_mask_from_reference(b, dist), _rare_mask_from_reference(controls, dist)
    query = pd.concat([b, controls], ignore_index=True)
    labels = np.r_[np.ones(len(b), dtype=int), np.zeros(len(controls), dtype=int)]
    mean, std = ref[NUMERIC].mean().to_numpy(), ref[NUMERIC].std(ddof=0).to_numpy()
    std = np.where(std > 0, std, 1.0)
    dn = cdist((query[NUMERIC] - mean) / std, (c[NUMERIC] - mean) / std).min(axis=1)
    ds = _nearest_distance(query, c, mean, std)
    scores = {"kde_ratio": overlap_mia(c, ref, query, cfg), "continuous_nn": -dn, "full_static_nn": -ds}
    masks = {"all": np.ones(len(query), dtype=bool), "true_member_rare_detected_control": np.r_[rare, cr], "detected_both": np.r_[dr, cr]}
    result = {"candidate": candidate.relative_to(ROOT).as_posix(), "candidate_sha256": digest(candidate), "seed": seed,
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "label": "local privacy proxies, not official Anon; phi not applied; control rare labels are detector estimates",
              "n_reference": len(ref), "n_control": len(controls), "mia": {}, "aia": {}}
    predictions = pd.DataFrame({"source": np.where(labels == 1, "B14_member", "background_control"),
                                "record_id": query.record_id, "rare_evaluation": np.r_[rare, cr].astype(int)})
    for name, values in scores.items():
        result["mia"][name] = {kind: mia_metrics(values, labels, mask) for kind, mask in masks.items()}
        predictions[f"mia_{name}"] = values
    q = query[QI].copy()
    predictions_aia = {"official_1nn": aia_predict(c, q, ref, cfg)[1],
                       "log_qi_1nn": log_qi_predict(c, q, ref, False),
                       "log_qi_window20": log_qi_predict(c, q, ref, True)}
    for name, values in predictions_aia.items():
        hit = np.abs(values - query.time.to_numpy()) <= 0.5
        mh, ch = hit[:len(b)], hit[len(b):]
        result["aia"][name] = {"all": success_metrics(mh, ch), "rare_proxy": success_metrics(mh[rare], ch[cr]),
                                "stratified_all": standardized_aia(b, controls, mh, ch)}
        predictions[f"time_hat_{name}"] = values
    dif = (b[CORR].corr(method="spearman") - c[CORR].corr(method="spearman")).abs()
    pairs = [(CORR[i], CORR[j], float(dif.iloc[i, j])) for i in range(len(CORR)) for j in range(i + 1, len(CORR))]
    result["diagnostics"] = {"n_true_rare": int(rare.sum()), "n_detected_rare": int(dr.sum()),
        "member_distance_quantiles": np.quantile(ds[:len(b)], [0, .1, .5, .9, 1]).tolist(),
        "true_rare_distance_quantiles": np.quantile(ds[:len(b)][rare], [0, .1, .5, .9, 1]).tolist(),
        "detected_rare_distance_quantiles": np.quantile(ds[:len(b)][dr], [0, .1, .5, .9, 1]).tolist(),
        "exact_B_rows": _exact_match_count(c, b), "exact_background_rows": _exact_match_count(c, bg),
        "duplicate_C_rows": int(c.duplicated().sum()), "correlation_pairs": sorted(pairs, key=lambda x: -x[2])}
    split = pd.DataFrame({"background_row": np.r_[tr, te], "partition": ["reference"] * len(tr) + ["control"] * len(te)})
    split.to_csv(out / f"{tag}_split.csv", index=False)
    predictions.to_csv(out / f"{tag}_predictions.csv", index=False)
    result["seconds"] = time.monotonic() - start
    result["input_sha256"] = {p.name: digest(p) for p in [dist / "B.csv", dist / "B_self.csv", dist / "A_bg.csv", dist / "utility_ref.json", dist / "scoring_config.yaml"]}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in map(int, args.seeds.split(",")):
        r = evaluate(args.candidate.resolve(), seed, args.out)
        print(json.dumps({"candidate": args.candidate.name, "seed": seed, "seconds": r["seconds"],
                          "mia": {k: v["all"]["tpr_at_fpr_0.01"] for k, v in r["mia"].items()},
                          "aia": {k: v["stratified_all"]["r_raw"] for k, v in r["aia"].items()}}), flush=True)


if __name__ == "__main__":
    main()
