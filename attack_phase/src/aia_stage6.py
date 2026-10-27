"""Corrective H3 AIA estimator: single-event Cox and true window mass.

Stage 4's gate, sentinel, donor pools, and rollback path are preserved.  This
module changes only the in-gate time estimate.  It fixes two semantic issues:
``onset`` and ``death`` are collapsed to one ``any_event`` indicator, and the
window center is selected from a support-complete candidate set rather than
only from the observed time grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from aia_stage1 import BLOCK_COLUMNS, QI_COLUMNS
from aia_stage3 import AIAStage3Engine, Stage3Config
from aia_stage4 import COX_COLUMNS, CauseSpecificCox


STAGE6_METHODS = ("frozen_donor_modal", "any_event_cox_window_mode")
FROZEN_BASE = Stage3Config("l2", "l2")
RIDGE = 1.0
WINDOW = 0.5


def any_event(onset: Sequence[int], death: Sequence[int]) -> np.ndarray:
    """Collapse overlapping onset/death indicators without double counting."""
    left = np.asarray(onset, dtype=int)
    right = np.asarray(death, dtype=int)
    if left.shape != right.shape:
        raise ValueError("onset and death must have the same shape")
    return ((left != 0) | (right != 0)).astype(int)


def support_centers(
    support: Sequence[float], tau: float = WINDOW, censor_time: float | None = None
) -> np.ndarray:
    """Return a deterministic candidate set containing every sliding-window edge.

    For a discrete event support, window mass changes only when a left or right
    edge crosses a support point.  Including ``support +/- tau`` therefore
    avoids the observed-grid approximation that misses a center such as 1.5
    for support ``[1, 2]`` and ``tau=.5``.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    values = np.asarray(support, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("support must be a non-empty finite vector")
    candidates = [values, values - tau, values + tau]
    if censor_time is not None:
        if not np.isfinite(censor_time):
            raise ValueError("censor_time must be finite")
        candidates.extend(([float(censor_time)], [float(censor_time) - tau], [float(censor_time) + tau]))
    out = np.unique(np.concatenate(candidates))
    return out[out >= 0.0]


def window_mass_from_support(
    support: Sequence[float], masses: Sequence[float], centers: Sequence[float],
    tau: float = WINDOW,
) -> np.ndarray:
    """Sum point masses in each inclusive ``center +/- tau`` window."""
    points = np.asarray(support, dtype=float)
    weights = np.asarray(masses, dtype=float)
    query = np.asarray(centers, dtype=float)
    if points.shape != weights.shape or query.ndim != 1:
        raise ValueError("support, masses and centers must be aligned vectors")
    if not np.isfinite(np.r_[points, weights, query]).all():
        raise ValueError("window inputs must be finite")
    return ((np.abs(points[:, None] - query[None, :]) <= tau) * weights[:, None]).sum(axis=0)


class AnyEventRidgeCox(CauseSpecificCox):
    """The Stage-4 ridge Cox engine fitted once to ``onset OR death``."""

    def __init__(
        self,
        durations: Sequence[float],
        onset: Sequence[int],
        death: Sequence[int],
        covariates: np.ndarray,
        ridge: float = RIDGE,
    ) -> None:
        self.events = any_event(onset, death)
        super().__init__(durations, self.events, covariates, ridge)


@dataclass(frozen=True)
class Stage6Config:
    """Only the in-gate conditional-time estimator varies."""

    method: str = "frozen_donor_modal"
    ridge: float = RIDGE
    tau: float = WINDOW

    def validate(self) -> None:
        if self.method not in STAGE6_METHODS:
            raise ValueError(f"unknown stage6 method: {self.method}")
        if self.ridge <= 0 or self.tau <= 0:
            raise ValueError("invalid stage6 configuration")


def stage6_config_grid() -> list[Stage6Config]:
    """Rollback plus the one preregistered H3 corrective challenger."""
    return [Stage6Config(method) for method in STAGE6_METHODS]


class AIAStage6Engine:
    """Stage-2 frozen gate/sentinel/pools with the corrected H3 estimator."""

    def __init__(self, target: pd.DataFrame, background: pd.DataFrame, config: Stage6Config) -> None:
        config.validate()
        self.config = config
        self.base = AIAStage3Engine(target, background, FROZEN_BASE)
        self.target = self.base.target
        self.sentinel = self.base.sentinel
        self._columns = [column for column in COX_COLUMNS if column in self.target.columns]
        self.censor_time = float(np.nanmax(self.target["time"].to_numpy(float)))
        covariates = self.target[self._columns].to_numpy(float)
        self.model = AnyEventRidgeCox(
            self.target["time"].to_numpy(float),
            self.target["onset"].to_numpy(int),
            self.target["death"].to_numpy(int),
            covariates,
            config.ridge,
        )
        support = self.model.baseline_times
        self.event_support = support if len(support) else np.array([self.censor_time])
        self._centers = support_centers(self.event_support, config.tau, self.censor_time)

    def _query_covariates(self, row: pd.Series) -> np.ndarray:
        return np.asarray([[float(row[column]) for column in self._columns]], dtype=float)

    def _event_point_masses(self, covariates: np.ndarray) -> np.ndarray:
        support = self.event_support
        if len(self.model.baseline_times) == 0:
            return np.zeros(len(support), dtype=float)
        risk = np.exp(self.model.linear_predictor(covariates))[0]
        before = np.nextafter(support, -np.inf)
        survival_before = np.exp(-risk * self.model.baseline_at(before))
        survival_after = np.exp(-risk * self.model.baseline_at(support))
        return np.maximum(0.0, survival_before - survival_after)

    def cox_window_mass(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return support-complete centers and event+plateau masses."""
        covariates = self._query_covariates(row)
        event_masses = self._event_point_masses(covariates)
        event_mass = window_mass_from_support(
            self.event_support, event_masses, self._centers, self.config.tau
        )
        risk = np.exp(self.model.linear_predictor(covariates))[0]
        plateau = float(np.exp(-risk * self.model.baseline_at([self.censor_time])[0]))
        total = event_mass + np.where(
            np.abs(self._centers - self.censor_time) <= self.config.tau,
            plateau,
            0.0,
        )
        return self._centers.copy(), total

    def any_event_cox_window_mode(self, row: pd.Series) -> float:
        centers, masses = self.cox_window_mass(row)
        return float(centers[int(np.argmax(masses))])

    def _predict_one(self, row: pd.Series) -> tuple[float, bool, float, float, float, int, bool]:
        hit, d_c, d_a, contrast = self.base._gate(row)
        if not hit:
            return self.sentinel, False, d_c, d_a, contrast, 0, True
        if self.config.method == "frozen_donor_modal":
            idx, distances = self.base._donors(row)
            value = self.base.base._modal_time(
                self.target.iloc[idx]["time"].to_numpy(float), distances
            )
            return value, True, d_c, d_a, contrast, len(idx), False
        return self.any_event_cox_window_mode(row), True, d_c, d_a, contrast, 0, False

    def predict(self, query: pd.DataFrame, method: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        if method is not None and method != self.config.method:
            raise ValueError("stage6 engines are constructed per method")
        rows = [self._predict_one(row) for _, row in query.reset_index(drop=True).iterrows()]
        values = np.asarray([row[0] for row in rows], dtype=float)
        diagnostics = pd.DataFrame(rows, columns=[
            "time_hat", "gate_hit", "distance_c", "distance_a",
            "log_distance_contrast", "donor_count", "sentinel_hit",
        ])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("time_hat must be finite and non-negative")
        return values, diagnostics


__all__ = [
    "AIAStage6Engine", "AnyEventRidgeCox", "FROZEN_BASE", "RIDGE",
    "STAGE6_METHODS", "Stage6Config", "any_event", "stage6_config_grid",
    "support_centers", "window_mass_from_support",
]
