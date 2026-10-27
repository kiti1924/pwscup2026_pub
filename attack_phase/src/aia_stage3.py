"""Stage-3 AIA robust-distance experiment.

Stage 2's abstention and donor-modal objective are frozen.  This module only
changes the distance geometry used for the C-vs-background membership gate
and for the donor ranking.  The baseline geometry is the Stage 2 normalized
L2 calculation, which is kept here as an explicit rollback configuration so
that every trial can be compared with the frozen candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from aia_stage1 import AIAEngine, BLOCK_COLUMNS, QI_COLUMNS, OUTCOME_COLUMNS
from aia_stage2 import RISK_COLUMNS, RISK_WEIGHT_VARIANTS


GEOMETRIES = ("l2", "l1", "huber_clip15")
STAGE3_METHODS = ("gated_abstain_1nn", "gated_abstain_donor_modal")


def robust_distance(
    delta: np.ndarray,
    geometry: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return a normalized distance for rows of standardized deltas.

    ``l2`` exactly matches the frozen Stage-2 geometry.  ``l1`` is a
    normalized absolute deviation, and ``huber_clip15`` caps each coordinate
    at 1.5 standard deviations before forming an L2 norm.  The fixed cap is
    intentionally not tuned on the test or Team 10 proxy cohorts.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown distance geometry: {geometry}")
    values = np.asarray(delta, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if weights is None:
        weight = np.ones(values.shape[1], dtype=float)
    else:
        weight = np.asarray(weights, dtype=float)
    if len(weight) != values.shape[1] or (weight < 0).any() or not weight.sum() > 0:
        raise ValueError("weights must be non-negative with positive sum")
    if geometry == "l1":
        return np.sum(np.abs(values) * weight, axis=1) / weight.sum()
    if geometry == "huber_clip15":
        clipped = np.minimum(np.abs(values), 1.5)
        return np.sqrt(np.sum(clipped * clipped * weight, axis=1) / weight.sum())
    return np.sqrt(np.sum(values * values * weight, axis=1) / weight.sum())


@dataclass(frozen=True)
class Stage3Config:
    """Fixed Stage-3 configuration; geometry is the only changed axis."""

    gate_geometry: str = "l2"
    donor_geometry: str = "l2"
    gate_log_contrast: float = 0.5
    risk_weights: str = "uniform"
    donor_k: int = 15
    sentinel_margin: float = 2.0

    def validate(self) -> None:
        if self.gate_geometry not in GEOMETRIES or self.donor_geometry not in GEOMETRIES:
            raise ValueError("unknown distance geometry")
        if self.risk_weights not in RISK_WEIGHT_VARIANTS:
            raise ValueError(f"unknown risk weight variant: {self.risk_weights}")
        if self.gate_log_contrast < 0 or self.donor_k < 1 or self.sentinel_margin <= 0:
            raise ValueError("invalid stage3 configuration")


def stage3_config_grid() -> list[Stage3Config]:
    """The preregistered three coupled metrics plus one decoupled control."""
    return [
        Stage3Config("l2", "l2"),
        Stage3Config("l1", "l1"),
        Stage3Config("huber_clip15", "huber_clip15"),
        # A single decoupled variant tests whether robust gating alone is
        # useful while retaining the frozen donor ranking.
        Stage3Config("l1", "l2"),
    ]


class AIAStage3Engine:
    """Stage-2 abstention/donor modal with explicit robust distance geometry."""

    def __init__(self, target: pd.DataFrame, background: pd.DataFrame, config: Stage3Config) -> None:
        config.validate()
        self.config = config
        self.base = AIAEngine(target, background, gate_log_contrast=config.gate_log_contrast)
        self.target = self.base.target
        self.background = self.base.background
        self._risk_std = self.background[list(RISK_COLUMNS)].std(ddof=1).replace(0.0, 1.0).fillna(1.0).to_numpy(float)
        self._risk_weights = np.asarray(
            [RISK_WEIGHT_VARIANTS[config.risk_weights][column] for column in RISK_COLUMNS], dtype=float
        )
        observed = pd.concat([self.target["time"], self.background["time"]], ignore_index=True).to_numpy(float)
        self.sentinel = float(np.nanmax(observed) + config.sentinel_margin)

    def _neighbors_geometry(
        self, row: pd.Series, source: str, k: int, geometry: str, weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if source == "target":
            source_z = self.base._target_z
        elif source == "background":
            source_z = self.base._background_z
        else:
            raise ValueError(source)
        query = row[list(QI_COLUMNS)].to_numpy(float) / self.base._std
        idx = self.base._pool_indices(row, source)
        distances = robust_distance(source_z[idx] - query, geometry, weights)
        order = np.argsort(distances, kind="mergesort")
        take = min(max(int(k), 1), len(order))
        kth = distances[order[take - 1]]
        keep = distances[order] <= kth
        return idx[order[keep]], distances[order[keep]]

    def _gate(self, row: pd.Series) -> tuple[bool, float, float, float]:
        _, c_dist = self._neighbors_geometry(row, "target", 1, self.config.gate_geometry)
        _, a_dist = self._neighbors_geometry(row, "background", 1, self.config.gate_geometry)
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
            idx = sex_groups.get(
                (row[BLOCK_COLUMNS[0]],),
                sex_groups.get(row[BLOCK_COLUMNS[0]], np.arange(len(self.target))),
            )
        idx = np.asarray(idx, dtype=int)
        available = [column for column in RISK_COLUMNS if column in row.index]
        positions = [RISK_COLUMNS.index(column) for column in available]
        std = self._risk_std[positions]
        query = row[available].to_numpy(float) / std
        delta = self.target.iloc[idx][available].to_numpy(float) / std - query
        weights = self._risk_weights[positions]
        distances = robust_distance(delta, self.config.donor_geometry, weights)
        order = np.argsort(distances, kind="mergesort")
        take = min(self.config.donor_k, len(order))
        kth = distances[order[take - 1]]
        keep = distances[order] <= kth
        return idx[order[keep]], distances[order[keep]]

    def _predict_one(self, row: pd.Series, method: str) -> tuple[float, bool, float, float, float, int, bool]:
        if method not in STAGE3_METHODS:
            raise ValueError(f"unknown stage3 method: {method}")
        hit, d_c, d_a, contrast = self._gate(row)
        if not hit:
            return self.sentinel, False, d_c, d_a, contrast, 0, True
        if method == "gated_abstain_1nn":
            idx, distances = self._neighbors_geometry(row, "target", 1, self.config.gate_geometry)
            nearest = distances == distances.min()
            value = float(np.mean(np.sort(self.target.iloc[idx[nearest]]["time"].to_numpy(float))))
            return value, True, d_c, d_a, contrast, int(nearest.sum()), False
        idx, distances = self._donors(row)
        value = self.base._modal_time(self.target.iloc[idx]["time"].to_numpy(float), distances)
        return value, True, d_c, d_a, contrast, len(idx), False

    def predict(self, query: pd.DataFrame, method: str) -> tuple[np.ndarray, pd.DataFrame]:
        rows = [self._predict_one(row, method) for _, row in query.reset_index(drop=True).iterrows()]
        values = np.asarray([row[0] for row in rows], dtype=float)
        diagnostics = pd.DataFrame(
            rows,
            columns=[
                "time_hat", "gate_hit", "distance_c", "distance_a",
                "log_distance_contrast", "donor_count", "sentinel_hit",
            ],
        )
        return values, diagnostics


__all__ = [
    "AIAStage3Engine", "GEOMETRIES", "STAGE3_METHODS", "Stage3Config",
    "robust_distance", "stage3_config_grid",
]
