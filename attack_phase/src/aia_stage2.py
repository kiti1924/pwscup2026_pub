"""Stage-2 AIA: metric-aware abstention and risk-stratified donor sets.

This module deliberately does not use a population predictor on a gate miss.
The abstention sentinel is finite and non-negative but outside the observed
time range, so it cannot accidentally count as a ±0.5-year success.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from aia_stage1 import AIAEngine, BLOCK_COLUMNS, QI_COLUMNS, OUTCOME_COLUMNS, evaluate_time_predictions, make_pseudo_target


STAGE2_METHODS = ("gated_abstain_1nn", "gated_abstain_donor_modal")
RISK_COLUMNS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
RISK_WEIGHT_VARIANTS = {
    "uniform": {column: 1.0 for column in RISK_COLUMNS},
    "age_risk": {"age": 2.0, "BMI": 1.0, "SBP": 1.2, "TG": 1.2, "HDL": 1.0, "ALT": 1.0, "FPG": 1.2},
}


@dataclass(frozen=True)
class Stage2Config:
    gate_log_contrast: float = 1.0
    risk_weights: str = "uniform"
    donor_k: int = 15
    sentinel_margin: float = 2.0

    def validate(self) -> None:
        if self.risk_weights not in RISK_WEIGHT_VARIANTS:
            raise ValueError(f"unknown risk weight variant: {self.risk_weights}")
        if self.gate_log_contrast < 0 or self.donor_k < 1 or self.sentinel_margin <= 0:
            raise ValueError("invalid stage2 configuration")


def stage2_config_grid() -> list[Stage2Config]:
    return [
        Stage2Config(threshold, weights, donor_k)
        for threshold in (0.5, 1.0, 1.5)
        for weights in ("uniform", "age_risk")
        for donor_k in (5, 15)
    ]


def risk_stratified_triplet_swap(members: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Swap outcome triplets within block and adjacent clinical-risk order."""
    rng = np.random.default_rng(seed)
    result = members.iloc[rng.permutation(len(members))].reset_index(drop=True).copy()
    std = result[list(RISK_COLUMNS)].std(ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(float)
    values = result[list(RISK_COLUMNS)].to_numpy(float)
    z = (values - values.mean(axis=0)) / std
    # A signed, transparent clinical-risk proxy.  HDL is protective; the
    # remaining columns increase the proxy when above their cohort mean.
    signs = np.asarray([1.5, 0.8, 1.0, 0.8, -0.8, 0.5, 1.0], float)
    risk = np.average(z * np.sign(signs), axis=1, weights=np.abs(signs))
    for _, group in result.groupby(list(BLOCK_COLUMNS), dropna=False).groups.items():
        idx = np.asarray(group, dtype=int)
        if len(idx) < 2:
            continue
        ordered = idx[np.argsort(risk[idx], kind="mergesort")]
        # Adjacent cyclic donors simulate a utility-preserving local swap.
        shift = 1 if int(rng.integers(0, 2)) == 0 else -1
        donor = np.roll(ordered, shift)
        result.loc[ordered, list(OUTCOME_COLUMNS)] = result.loc[donor, list(OUTCOME_COLUMNS)].to_numpy()
    return result


def make_stage2_pseudo_target(members: pd.DataFrame, mechanism: str, seed: int) -> pd.DataFrame:
    if mechanism == "risk_stratified_triplet_swap":
        return risk_stratified_triplet_swap(members, seed)
    if mechanism in {"near_copy", "perturbed", "triplet_swap", "bootstrap_copy"}:
        return make_pseudo_target(members, mechanism, seed)
    raise ValueError(mechanism)


class AIAStage2Engine:
    """Gate C-specific evidence, then abstain or use a risk-stratified donor set."""

    def __init__(self, target: pd.DataFrame, background: pd.DataFrame, config: Stage2Config) -> None:
        config.validate()
        self.config = config
        self.base = AIAEngine(target, background, gate_log_contrast=config.gate_log_contrast)
        self.target = self.base.target
        self.background = self.base.background
        self._risk_std = self.background[list(RISK_COLUMNS)].std(ddof=1).replace(0.0, 1.0).fillna(1.0).to_numpy(float)
        self._target_risk_z = self.target[list(RISK_COLUMNS)].to_numpy(float) / self._risk_std
        self._risk_weights = np.asarray([RISK_WEIGHT_VARIANTS[config.risk_weights][col] for col in RISK_COLUMNS], float)
        observed = pd.concat([self.target["time"], self.background["time"]], ignore_index=True).to_numpy(float)
        self.sentinel = float(np.nanmax(observed) + config.sentinel_margin)

    def _gate(self, row: pd.Series) -> tuple[bool, float, float, float]:
        _, c_dist = self.base._neighbors(row, "target", 1)
        _, a_dist = self.base._neighbors(row, "background", 1)
        d_c = float(c_dist.min())
        d_a = float(a_dist.min())
        contrast = float(np.log(d_a + 1e-9) - np.log(d_c + 1e-9))
        return contrast >= self.config.gate_log_contrast, d_c, d_a, contrast

    def _donors(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(row[col] for col in BLOCK_COLUMNS)
        groups = self.target.groupby(list(BLOCK_COLUMNS), dropna=False).groups
        idx = groups.get(key)
        if idx is None or len(idx) == 0:
            sex_groups = self.target.groupby([BLOCK_COLUMNS[0]], dropna=False).groups
            idx = sex_groups.get((row[BLOCK_COLUMNS[0]],), sex_groups.get(row[BLOCK_COLUMNS[0]], np.arange(len(self.target))))
        idx = np.asarray(idx, dtype=int)
        # Public challenge rows expose age and four clinical QIs, but not BMI/FPG.
        # Use the intersection rather than fabricating unavailable attributes.
        available = [column for column in RISK_COLUMNS if column in row.index]
        positions = [RISK_COLUMNS.index(column) for column in available]
        std = self._risk_std[positions]
        query = row[available].to_numpy(float) / std
        delta = self.target.iloc[idx][available].to_numpy(float) / std - query
        weights = self._risk_weights[positions]
        distances = np.sqrt(np.sum(delta * delta * weights, axis=1) / weights.sum())
        order = np.argsort(distances, kind="mergesort")
        take = min(self.config.donor_k, len(order))
        kth = distances[order[take - 1]]
        keep = order[distances[order] <= kth]
        return idx[keep], distances[keep]

    def _predict_one(self, row: pd.Series, method: str) -> tuple[float, bool, float, float, float, int, bool]:
        if method not in STAGE2_METHODS:
            raise ValueError(f"unknown stage2 method: {method}")
        hit, d_c, d_a, contrast = self._gate(row)
        if not hit:
            return self.sentinel, False, d_c, d_a, contrast, 0, True
        if method == "gated_abstain_1nn":
            idx, distances = self.base._neighbors(row, "target", 1)
            nearest = distances == distances.min()
            value = float(np.mean(np.sort(self.target.iloc[idx[nearest]]["time"].to_numpy(float))))
            return value, True, d_c, d_a, contrast, int(nearest.sum()), False
        idx, distances = self._donors(row)
        value = self.base._modal_time(self.target.iloc[idx]["time"].to_numpy(float), distances)
        return value, True, d_c, d_a, contrast, len(idx), False

    def predict(self, query: pd.DataFrame, method: str) -> tuple[np.ndarray, pd.DataFrame]:
        rows = [self._predict_one(row, method) for _, row in query.reset_index(drop=True).iterrows()]
        values = np.asarray([row[0] for row in rows], float)
        diagnostics = pd.DataFrame(rows, columns=["time_hat", "gate_hit", "distance_c", "distance_a", "log_distance_contrast", "donor_count", "sentinel_hit"])
        return values, diagnostics


__all__ = [
    "AIAStage2Engine", "RISK_COLUMNS", "RISK_WEIGHT_VARIANTS", "STAGE2_METHODS", "Stage2Config",
    "evaluate_time_predictions", "make_stage2_pseudo_target", "risk_stratified_triplet_swap", "stage2_config_grid",
]
