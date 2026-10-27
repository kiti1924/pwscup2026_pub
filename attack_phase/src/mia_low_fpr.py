"""Inspectable low-FPR membership-inference experts.

This module intentionally keeps each expert observable.  It does not infer
hidden membership labels from the public targets; calibration uses only a
held-out slice of ``A_bg`` and experiment evaluation uses explicit proxies or
pseudo targets with known construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde


CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
BLOCK = ["sex", "prefecture"]
STATIC = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG"]
EXPERTS = ["kde", "nn_relative", "partial_match", "static_exact", "rare_interaction"]
AUX_EXPERTS = ["exact_tiebreak_nn"]
ROUTE_WEIGHTS: dict[str, dict[str, float]] = {
    "smooth": {"kde": .40, "nn_relative": .35, "partial_match": .15, "static_exact": .05, "rare_interaction": .05},
    "partial": {"kde": .10, "nn_relative": .35, "partial_match": .35, "static_exact": .15, "rare_interaction": .05},
    "rare": {"kde": .10, "nn_relative": .25, "partial_match": .15, "static_exact": .05, "rare_interaction": .45},
    "cluster": {"kde": .30, "nn_relative": .40, "partial_match": .20, "static_exact": .05, "rare_interaction": .05},
    "background_control": {"kde": .60, "nn_relative": .20, "partial_match": .05, "static_exact": 0.0, "rare_interaction": .15},
}
TARGET_ROUTES = {
    2: "rare", 6: "partial", 13: "rare", 17: "rare", 18: "background_control",
    19: "partial", 24: "cluster", 29: "background_control",
}


def route_for_target(target: int) -> str:
    return TARGET_ROUTES.get(int(target), "smooth")


def grouped_tpr_at_fpr(scores: Sequence[float], truth: Sequence[int], target_fpr: float = .01) -> tuple[float, float]:
    """Official-compatible grouped-tie ROC interpolation and trapezoid AUC."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(truth, dtype=int)
    if len(s) != len(y) or not np.isfinite(s).all():
        raise ValueError("scores/truth shape mismatch or non-finite score")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.5
    endpoints = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tpr = np.r_[0.0, np.cumsum(y)[endpoints] / n_pos]
    fpr = np.r_[0.0, np.cumsum(1 - y)[endpoints] / n_neg]
    return float(np.interp(target_fpr, fpr, tpr)), float(np.trapezoid(tpr, fpr))


def _keys(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame[list(columns)].astype(str).agg("\x1f".join, axis=1)


def background_overlap_mask(frame: pd.DataFrame, background: pd.DataFrame, columns: Sequence[str] = STATIC) -> np.ndarray:
    """Rows whose disclosed static attributes exactly equal a known A_bg row."""
    cols = [c for c in columns if c in frame and c in background]
    return _keys(frame, cols).isin(set(_keys(background, cols))).to_numpy(bool)


def filter_background_derived_target(target_frame: pd.DataFrame, background: pd.DataFrame, min_rows: int = 25) -> tuple[pd.DataFrame, dict]:
    """Remove A_bg-derived static copies from a target evidence set.

    The original target is returned if filtering would make density/distance
    estimation degenerate, but the diagnostic still records that fallback.
    """
    overlap = background_overlap_mask(target_frame, background)
    residual = target_frame.loc[~overlap].reset_index(drop=True)
    fallback = len(residual) < min_rows
    return (target_frame.reset_index(drop=True) if fallback else residual), {
        "target_rows": int(len(target_frame)), "background_overlap_rows": int(overlap.sum()),
        "residual_rows": int(len(residual)), "fallback_to_full_target": bool(fallback),
    }


def mahalanobis_rarity(frame: pd.DataFrame, rare_detector: Mapping[str, object]) -> np.ndarray:
    cols = list(rare_detector["columns"])
    x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    mu = np.asarray(rare_detector["mu"], dtype=float)
    inv = np.linalg.pinv(np.asarray(rare_detector["sigma"], dtype=float))
    x = np.where(np.isfinite(x), x, mu)
    delta = x - mu
    return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, inv, delta)))


def _fit_kde(x: np.ndarray) -> gaussian_kde | None:
    try:
        return gaussian_kde(x.T)
    except (np.linalg.LinAlgError, ValueError):
        return None


@dataclass
class LowFPRFeatureModel:
    target: pd.DataFrame
    reference: pd.DataFrame
    rare_detector: Mapping[str, object]

    def __post_init__(self) -> None:
        self.target = self.target.reset_index(drop=True)
        self.reference = self.reference.reset_index(drop=True)
        self.mean = self.reference[CONTINUOUS].mean().to_numpy(float)
        self.std = self.reference[CONTINUOUS].std(ddof=0).replace(0, 1).fillna(1).to_numpy(float)
        self.target_z = (self.target[CONTINUOUS].to_numpy(float) - self.mean) / self.std
        self.reference_z = (self.reference[CONTINUOUS].to_numpy(float) - self.mean) / self.std
        self.kde_target = _fit_kde(self.target_z)
        self.kde_reference = _fit_kde(self.reference_z)
        self.target_static_keys = set(_keys(self.target, STATIC))
        self.target_blocks = {k: np.asarray(v, dtype=int) for k, v in self.target.groupby(BLOCK, dropna=False).indices.items()}
        self.reference_blocks = {k: np.asarray(v, dtype=int) for k, v in self.reference.groupby(BLOCK, dropna=False).indices.items()}

    def _kde_ratio(self, z: np.ndarray) -> np.ndarray:
        if self.kde_target is None or self.kde_reference is None:
            return np.zeros(len(z), dtype=float)
        a = np.log(np.clip(self.kde_target(z.T), 1e-300, None))
        b = np.log(np.clip(self.kde_reference(z.T), 1e-300, None))
        return np.nan_to_num(a - b, nan=0.0, posinf=700.0, neginf=-700.0)

    @staticmethod
    def _block_key(row: pd.Series) -> tuple:
        return tuple(row[c] for c in BLOCK)

    def _nearest(self, candidates: pd.DataFrame, z: np.ndarray, pool_z: np.ndarray, blocks: Mapping[tuple, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        distance = np.full(len(candidates), np.inf)
        nearest = np.full(len(candidates), -1, dtype=int)
        candidate_groups = candidates.groupby(BLOCK, dropna=False).indices
        for key, candidate_idx in candidate_groups.items():
            key = key if isinstance(key, tuple) else (key,)
            pool_idx = blocks.get(key)
            if pool_idx is None or not len(pool_idx):
                continue
            ci = np.asarray(candidate_idx, dtype=int)
            dm = cdist(z[ci], pool_z[pool_idx]) / math.sqrt(len(CONTINUOUS))
            arg = np.argmin(dm, axis=1)
            distance[ci] = dm[np.arange(len(ci)), arg]
            nearest[ci] = pool_idx[arg]
        return distance, nearest

    def raw_features(self, candidates: pd.DataFrame) -> pd.DataFrame:
        candidates = candidates.reset_index(drop=True)
        z = (candidates[CONTINUOUS].to_numpy(float) - self.mean) / self.std
        d_c, nearest = self._nearest(candidates, z, self.target_z, self.target_blocks)
        d_a, _ = self._nearest(candidates, z, self.reference_z, self.reference_blocks)
        finite_c = np.where(np.isfinite(d_c), d_c, 1e6)
        finite_a = np.where(np.isfinite(d_a), d_a, 1e6)
        nn_relative = np.log(finite_a + 1e-9) - np.log(finite_c + 1e-9)

        near_count = np.zeros(len(candidates), dtype=float)
        nearest_smoking = np.zeros(len(candidates), dtype=float)
        valid = nearest >= 0
        if valid.any():
            diffs = np.abs(z[valid] - self.target_z[nearest[valid]])
            near_count[valid] = (diffs <= .05).sum(axis=1) / len(CONTINUOUS)
            nearest_smoking[valid] = (
                candidates.loc[valid, "smoking"].to_numpy() == self.target.iloc[nearest[valid]]["smoking"].to_numpy()
            ).astype(float)
        exact = _keys(candidates, STATIC).isin(self.target_static_keys).to_numpy(float)
        partial = .82 * near_count + .18 * nearest_smoking + .005 * np.tanh(nn_relative)
        rarity = mahalanobis_rarity(candidates, self.rare_detector)
        gate = float(self.rare_detector.get("gate", 8.0))
        rare_interaction = np.maximum(0.0, rarity / max(gate, 1e-9) - .5) * np.maximum(0.0, nn_relative)
        kde = self._kde_ratio(z)
        result = pd.DataFrame({
            "kde": kde,
            "nn_relative": nn_relative,
            "partial_match": partial,
            "static_exact": exact,
            "exact_tiebreak_nn": exact + 1e-6 * (np.tanh(nn_relative) + 1.0) / 2.0,
            "rare_mahalanobis": rarity,
            "rare_interaction": rare_interaction,
            "distance_c": finite_c,
            "distance_a": finite_a,
            "near_feature_fraction": near_count,
        })
        return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def empirical_percentile(values: Sequence[float], null: Sequence[float]) -> np.ndarray:
    """Upper-tail empirical percentile in [0,1], with ties kept tied."""
    v = np.asarray(values, dtype=float)
    n = np.sort(np.asarray(null, dtype=float))
    if len(n) == 0:
        raise ValueError("empty calibration null")
    below = np.searchsorted(n, v, side="left")
    return np.clip(below / len(n), 0.0, 1.0)


def calibrated_scores(raw: pd.DataFrame, null_raw: pd.DataFrame, route: str = "smooth") -> pd.DataFrame:
    if route not in ROUTE_WEIGHTS:
        raise ValueError(f"unknown route: {route}")
    out = pd.DataFrame(index=raw.index)
    null_pct = pd.DataFrame(index=null_raw.index)
    for col in EXPERTS + AUX_EXPERTS:
        out[col] = empirical_percentile(raw[col], null_raw[col])
        null_pct[col] = empirical_percentile(null_raw[col], null_raw[col])
    out["rare_mahalanobis"] = empirical_percentile(raw["rare_mahalanobis"], null_raw["rare_mahalanobis"])
    weights = ROUTE_WEIGHTS[route]
    equal = out[EXPERTS].mean(axis=1)
    null_equal = null_pct[EXPERTS].mean(axis=1)
    route_score = sum(weights[c] * out[c] for c in EXPERTS)
    null_route = sum(weights[c] * null_pct[c] for c in EXPERTS)
    out["ensemble_rank_equal"] = empirical_percentile(equal, null_equal)
    out["ensemble_rank_route"] = empirical_percentile(route_score, null_route)
    # Fisher-style combination is calibrated back against the null, retaining
    # a bounded output and avoiding a probability-calibration claim.
    p = np.clip(1.0 - out[EXPERTS].to_numpy(float), 1.0 / (len(null_raw) + 1), 1.0)
    null_p = np.clip(1.0 - null_pct[EXPERTS].to_numpy(float), 1.0 / (len(null_raw) + 1), 1.0)
    fisher = -np.log(p).mean(axis=1)
    fisher_null = -np.log(null_p).mean(axis=1)
    out["ensemble_p_fisher"] = empirical_percentile(fisher, fisher_null)
    return out.clip(0.0, 1.0)


def apply_background_veto(scores: pd.DataFrame, candidates: pd.DataFrame, full_background: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    veto = background_overlap_mask(candidates, full_background)
    result = scores.copy()
    result.loc[veto, :] = 0.0
    return result, veto


def evaluate_score_frame(member_scores: pd.DataFrame, control_scores: pd.DataFrame, rare_member: np.ndarray | None = None, rare_control: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    truth = np.r_[np.ones(len(member_scores), dtype=int), np.zeros(len(control_scores), dtype=int)]
    for col in member_scores.columns:
        scores = np.r_[member_scores[col].to_numpy(float), control_scores[col].to_numpy(float)]
        tpr, auc = grouped_tpr_at_fpr(scores, truth)
        rare_tpr = float("nan")
        rare_auc = float("nan")
        rare_n_member = rare_n_control = 0
        if rare_member is not None and rare_control is not None:
            mask = np.r_[rare_member, rare_control].astype(bool)
            rare_n_member, rare_n_control = int(np.sum(rare_member)), int(np.sum(rare_control))
            if rare_n_member and rare_n_control:
                rare_tpr, rare_auc = grouped_tpr_at_fpr(scores[mask], truth[mask])
        rows.append({"model": col, "tpr_at_fpr_0_01": tpr, "auc": auc, "rare_tpr_at_fpr_0_01": rare_tpr,
                     "rare_auc": rare_auc, "n_member": len(member_scores), "n_control": len(control_scores),
                     "rare_n_member": rare_n_member, "rare_n_control": rare_n_control})
    return pd.DataFrame(rows)


def bootstrap_metric(scores: Sequence[float], truth: Sequence[int], n_boot: int = 400, seed: int = 20260828) -> dict:
    """Stratified bootstrap stability for TPR@1% FPR and AUC."""
    scores, truth = np.asarray(scores, float), np.asarray(truth, int)
    pos, neg = np.flatnonzero(truth == 1), np.flatnonzero(truth == 0)
    rng = np.random.default_rng(seed)
    tprs, aucs = [], []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        tpr, auc = grouped_tpr_at_fpr(scores[idx], truth[idx])
        tprs.append(tpr); aucs.append(auc)
    q = lambda x: [float(np.quantile(x, .025)), float(np.median(x)), float(np.quantile(x, .975))]
    return {"n_boot": n_boot, "seed": seed, "tpr_ci95_median": q(tprs), "auc_ci95_median": q(aucs)}


def make_pseudo_target(members: pd.DataFrame, reference: pd.DataFrame, mechanism: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = members.iloc[rng.integers(0, len(members), size=len(members))].reset_index(drop=True).copy()
    if mechanism == "raw_resample":
        return sampled.drop(columns=["record_id", "is_rare"], errors="ignore")
    if mechanism != "perturbed_rounded":
        raise ValueError(mechanism)
    std = reference[CONTINUOUS].std(ddof=0).replace(0, 1).to_numpy(float)
    noise = rng.normal(0.0, .035, size=(len(sampled), len(CONTINUOUS))) * std
    perturbed = sampled[CONTINUOUS].to_numpy(float) + noise
    for j, col in enumerate(CONTINUOUS):
        sampled[col] = perturbed[:, j]
    sampled["age"] = sampled["age"].round().clip(40, 74)
    for col in ["BMI", "SBP", "TG", "HDL", "ALT", "FPG"]:
        sampled[col] = sampled[col].round(1)
    return sampled.drop(columns=["record_id", "is_rare"], errors="ignore")


def validate_release_targets(target_dir, expected: Sequence[int]) -> None:
    present = sorted(int(p.stem.split("_")[1]) for p in target_dir.glob("C_*.csv"))
    if present != sorted(int(x) for x in expected):
        raise ValueError(f"target mismatch expected={sorted(expected)} present={present}")
