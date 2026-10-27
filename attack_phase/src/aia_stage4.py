"""Stage-4 AIA: survival-aware conditional time attacks (hypothesis H3).

Stage 2's ``gated_abstain_donor_modal`` stays the frozen rollback.  The
membership gate, the abstention sentinel, the block/donor pools, and the
official +/-0.5 objective are all unchanged.  The single axis this stage varies
is *how a time is chosen inside the gate*: an empirical donor modal window is
replaced by a conditional distribution from a cause-specific Cox model fit on
the target ``C``.

The defense calibrated ``time``/``onset``/``death`` with cause-specific Cox
calibration and outcome-triplet swaps, so ``C`` may reproduce the conditional
law ``P(time | QI)`` of ``B`` much more faithfully than it reproduces any
individual record.  ``time`` is administratively censored (most rows sit on a
single follow-up plateau), so the observed-time law is a continuous event part
plus an atom at the plateau; both parts are modelled explicitly here.

No external survival package is required: the Cox fit is a small, auditable
Newton-Raphson with Breslow tie handling and a fixed ridge penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from aia_stage1 import BLOCK_COLUMNS, QI_COLUMNS
from aia_stage3 import AIAStage3Engine, Stage3Config


COX_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT", "sex")
CAUSES = ("onset", "death")
STAGE4_METHODS = (
    "frozen_donor_modal",
    "cox_window_mode",
    "cox_risk_donor_modal",
    "cox_union_donor_modal",
)
FROZEN_BASE = Stage3Config("l2", "l2")

RIDGE = 1.0
MAX_ITER = 50
TOLERANCE = 1e-7
WINDOW = 0.5


class CauseSpecificCox:
    """Breslow-tie Cox proportional hazards with a fixed ridge penalty.

    ``events`` marks this cause only; every other outcome is censoring, which
    is the cause-specific convention under competing risks.
    """

    def __init__(
        self,
        durations: Sequence[float],
        events: Sequence[int],
        covariates: np.ndarray,
        ridge: float = RIDGE,
    ) -> None:
        time = np.asarray(durations, dtype=float)
        event = np.asarray(events, dtype=int)
        x = np.asarray(covariates, dtype=float)
        if len(time) != len(event) or len(time) != len(x):
            raise ValueError("durations, events and covariates must align")
        if len(time) == 0:
            raise ValueError("cox fit needs a non-empty cohort")
        self.mean = x.mean(axis=0)
        self.scale = np.where(x.std(axis=0, ddof=0) > 0, x.std(axis=0, ddof=0), 1.0)
        z = (x - self.mean) / self.scale
        self.ridge = float(ridge)
        self.beta = np.zeros(z.shape[1], dtype=float)
        self._fit(time, event, z)
        self._breslow(time, event, z)

    # -- fitting --------------------------------------------------------------

    def _fit(self, time: np.ndarray, event: np.ndarray, z: np.ndarray) -> None:
        event_times = np.unique(time[event == 1])
        if len(event_times) == 0:
            # No event of this cause: the hazard is identically zero.
            self.converged = True
            return
        for _ in range(MAX_ITER):
            eta = np.clip(z @ self.beta, -30.0, 30.0)
            w = np.exp(eta)
            gradient = np.zeros_like(self.beta)
            hessian = np.zeros((len(self.beta), len(self.beta)), dtype=float)
            for t in event_times:
                at_risk = time >= t
                deaths = (time == t) & (event == 1)
                s0 = float(w[at_risk].sum())
                if s0 <= 0:
                    continue
                s1 = w[at_risk] @ z[at_risk]
                s2 = (z[at_risk] * w[at_risk][:, None]).T @ z[at_risk]
                mean = s1 / s0
                d = int(deaths.sum())
                gradient += z[deaths].sum(axis=0) - d * mean
                hessian -= d * (s2 / s0 - np.outer(mean, mean))
            gradient -= 2.0 * self.ridge * self.beta
            hessian -= 2.0 * self.ridge * np.eye(len(self.beta))
            step = np.linalg.solve(hessian, gradient)
            self.beta = self.beta - step
            if np.max(np.abs(step)) < TOLERANCE:
                self.converged = True
                return
        self.converged = False

    def _breslow(self, time: np.ndarray, event: np.ndarray, z: np.ndarray) -> None:
        eta = np.clip(z @ self.beta, -30.0, 30.0)
        w = np.exp(eta)
        event_times = np.unique(time[event == 1])
        increments = []
        for t in event_times:
            s0 = float(w[time >= t].sum())
            d = int(((time == t) & (event == 1)).sum())
            increments.append(d / s0 if s0 > 0 else 0.0)
        self.baseline_times = np.asarray(event_times, dtype=float)
        self.baseline_cumhaz = np.cumsum(np.asarray(increments, dtype=float)) if len(increments) else np.array([])

    # -- prediction -----------------------------------------------------------

    def linear_predictor(self, covariates: np.ndarray) -> np.ndarray:
        z = (np.asarray(covariates, dtype=float) - self.mean) / self.scale
        return np.clip(z @ self.beta, -30.0, 30.0)

    def baseline_at(self, times: Sequence[float]) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        if len(self.baseline_times) == 0:
            return np.zeros(len(t), dtype=float)
        position = np.searchsorted(self.baseline_times, t, side="right")
        padded = np.r_[0.0, self.baseline_cumhaz]
        return padded[position]

    def cumulative_hazard(self, times: Sequence[float], covariates: np.ndarray) -> np.ndarray:
        """Return an ``(n_rows, n_times)`` cause-specific cumulative hazard."""
        base = self.baseline_at(times)
        risk = np.exp(self.linear_predictor(covariates))
        return np.outer(risk, base)


@dataclass(frozen=True)
class Stage4Config:
    """Only the in-gate time estimator varies; everything else is frozen."""

    method: str = "frozen_donor_modal"
    donor_k: int = 15
    ridge: float = RIDGE

    def validate(self) -> None:
        if self.method not in STAGE4_METHODS:
            raise ValueError(f"unknown stage4 method: {self.method}")
        if self.donor_k < 1 or self.ridge <= 0:
            raise ValueError("invalid stage4 configuration")


def stage4_config_grid() -> list[Stage4Config]:
    """Rollback plus the three preregistered survival-aware estimators."""
    return [Stage4Config(method) for method in STAGE4_METHODS]


class AIAStage4Engine:
    """Frozen gate/sentinel with a survival-aware in-gate time estimator."""

    def __init__(self, target: pd.DataFrame, background: pd.DataFrame, config: Stage4Config) -> None:
        config.validate()
        self.config = config
        # The frozen Stage-2/3 engine supplies the gate, the QI donor pool and
        # the abstention sentinel so the rollback path stays bit-identical.
        self.base = AIAStage3Engine(target, background, FROZEN_BASE)
        self.target = self.base.target
        self.sentinel = self.base.sentinel
        self._columns = [column for column in COX_COLUMNS if column in self.target.columns]
        self._fit_causes()

    def _fit_causes(self) -> None:
        covariates = self.target[self._columns].to_numpy(float)
        duration = self.target["time"].to_numpy(float)
        self.censor_time = float(np.nanmax(duration))
        self.models: dict[str, CauseSpecificCox] = {}
        for cause in CAUSES:
            events = self.target[cause].to_numpy(int) if cause in self.target else np.zeros(len(self.target), int)
            self.models[cause] = CauseSpecificCox(duration, events, covariates, self.config.ridge)
        # A common-scale scalar risk: total predicted cumulative hazard at the
        # end of follow-up under both competing causes.
        total = np.zeros(len(self.target), dtype=float)
        for model in self.models.values():
            total += model.cumulative_hazard([self.censor_time], covariates)[:, 0]
        self._target_risk = np.log(total + 1e-12)
        self._time_grid = np.unique(self.target["time"].to_numpy(float))

    # -- survival-aware estimators -------------------------------------------

    def _query_covariates(self, row: pd.Series) -> np.ndarray:
        return np.asarray([[float(row[column]) for column in self._columns]], dtype=float)

    def _total_cumhaz(self, times: Sequence[float], covariates: np.ndarray) -> np.ndarray:
        total = np.zeros((len(covariates), len(np.asarray(times))), dtype=float)
        for model in self.models.values():
            total += model.cumulative_hazard(times, covariates)
        return total

    def _survival(self, times: Sequence[float], covariates: np.ndarray) -> np.ndarray:
        return np.exp(-self._total_cumhaz(times, covariates))

    def cox_risk(self, row: pd.Series) -> float:
        covariates = self._query_covariates(row)
        return float(np.log(self._total_cumhaz([self.censor_time], covariates)[0, 0] + 1e-12))

    def cox_window_mode(self, row: pd.Series) -> float:
        """Maximize observed-time mass in the official +/-0.5 window.

        The observed time is ``min(event time, end of follow-up)``, so its law
        is the event density plus an atom of size ``S(censor_time)`` at the
        follow-up plateau.  Both parts are scored on the same grid.
        """
        covariates = self._query_covariates(row)
        grid = self._time_grid
        lower = np.clip(grid - WINDOW, 0.0, None)
        upper = np.minimum(grid + WINDOW, self.censor_time)
        survival_lower = self._survival(lower, covariates)[0]
        survival_upper = self._survival(upper, covariates)[0]
        mass = survival_lower - survival_upper
        atom = float(self._survival([self.censor_time], covariates)[0, 0])
        mass = mass + np.where(np.abs(grid - self.censor_time) <= WINDOW, atom, 0.0)
        best = int(np.argmax(mass))
        return float(grid[best])

    def _block_indices(self, row: pd.Series) -> np.ndarray:
        groups = self.target.groupby(list(BLOCK_COLUMNS), dropna=False).groups
        idx = groups.get(tuple(row[column] for column in BLOCK_COLUMNS))
        if idx is None or len(idx) == 0:
            sex_groups = self.target.groupby([BLOCK_COLUMNS[0]], dropna=False).groups
            idx = sex_groups.get(
                (row[BLOCK_COLUMNS[0]],),
                sex_groups.get(row[BLOCK_COLUMNS[0]], np.arange(len(self.target))),
            )
        return np.asarray(idx, dtype=int)

    def cox_risk_donors(self, row: pd.Series) -> np.ndarray:
        """Block-local donors ranked by predicted cumulative-hazard similarity."""
        idx = self._block_indices(row)
        distance = np.abs(self._target_risk[idx] - self.cox_risk(row))
        order = np.argsort(distance, kind="mergesort")
        take = min(self.config.donor_k, len(order))
        kth = distance[order[take - 1]]
        return idx[order[distance[order] <= kth]]

    def _modal_over(self, indices: np.ndarray, row: pd.Series) -> float:
        """Official +/-0.5 modal window with the frozen inverse-QI weighting."""
        qi = [column for column in QI_COLUMNS if column in row.index]
        std = self.base.base._std[[QI_COLUMNS.index(column) for column in qi]]
        delta = self.target.iloc[indices][qi].to_numpy(float) / std - row[qi].to_numpy(float) / std
        distances = np.sqrt(np.mean(delta * delta, axis=1))
        return self.base.base._modal_time(
            self.target.iloc[indices]["time"].to_numpy(float), distances
        )

    # -- prediction -----------------------------------------------------------

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
        if self.config.method == "cox_window_mode":
            return self.cox_window_mode(row), True, d_c, d_a, contrast, 0, False
        cox_idx = self.cox_risk_donors(row)
        if self.config.method == "cox_union_donor_modal":
            qi_idx, _ = self.base._donors(row)
            cox_idx = np.unique(np.r_[qi_idx, cox_idx])
        return self._modal_over(cox_idx, row), True, d_c, d_a, contrast, len(cox_idx), False

    def predict(self, query: pd.DataFrame, method: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        if method is not None and method != self.config.method:
            raise ValueError("stage4 engines are constructed per method")
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
    "AIAStage4Engine", "CAUSES", "COX_COLUMNS", "CauseSpecificCox",
    "FROZEN_BASE", "STAGE4_METHODS", "Stage4Config", "stage4_config_grid",
]
