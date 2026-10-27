"""Stage-5 AIA: aggregate / reconstruction-style feasibility (hypothesis H4).

The defense matched marginals, categorical profiles, rare profiles and Cox
coefficients to ``B`` very closely, so ``C`` may over-describe ``B`` at the
*aggregate* level even where no individual near-copy survives.  ``time`` is
continuous, so a binary reconstruction attack cannot be ported directly.  This
module therefore asks the feasibility question first, exactly as the Goal
requires, instead of building a large constrained optimisation.

The decisive quantity is an **oracle cell ceiling**.  Any attack whose
prediction is constant inside an aggregation cell — every group-count,
group-mean, group-quantile, histogram, constrained-least-squares or
nonnegative-reconstruction variant of H4 — cannot beat the ceiling obtained by
choosing each cell's single best time with the true member labels revealed.
If that ceiling does not clear the frozen Stage-2 donor attack, no member of
that family can, and H4 is closed by construction rather than by search.

Two concrete aggregate estimators are also evaluated so the ceiling is not the
only evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from aia_stage1 import BLOCK_COLUMNS
from aia_stage3 import AIAStage3Engine, Stage3Config


FROZEN_BASE = Stage3Config("l2", "l2")
FROZEN_METHOD = "gated_abstain_donor_modal"
STAGE5_METHODS = (
    "frozen_donor_modal",
    "cell_modal_aggregate",
    "group_mean_corrected_donor",
)
WINDOW = 0.5
MIN_CELL_ROWS = 20
AGE_QUANTILES = (1 / 3, 2 / 3)


def age_cuts(target: pd.DataFrame) -> tuple[float, ...]:
    """Fixed attacker-side age terciles; no membership label is involved."""
    age = target["age"].to_numpy(float)
    return tuple(float(np.quantile(age, q)) for q in AGE_QUANTILES)


def cell_keys(frame: pd.DataFrame, cuts: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the fine ``sex|prefecture|age-bin`` key and a coarse ``sex|age-bin``."""
    bins = np.searchsorted(np.asarray(cuts, dtype=float), frame["age"].to_numpy(float), side="left")
    sex = frame[BLOCK_COLUMNS[0]].astype(str).to_numpy()
    prefecture = frame[BLOCK_COLUMNS[1]].astype(str).to_numpy()
    fine = np.array([f"{s}|{p}|a{int(b)}" for s, p, b in zip(sex, prefecture, bins)], dtype=object)
    coarse = np.array([f"{s}|a{int(b)}" for s, b in zip(sex, bins)], dtype=object)
    return fine, coarse


def window_success(prediction: float, truth: Sequence[float]) -> np.ndarray:
    return np.abs(np.asarray(truth, dtype=float) - float(prediction)) <= WINDOW


def modal_window_value(times: Sequence[float]) -> float:
    """The +/-0.5 window mode of an empirical time sample."""
    values = np.asarray(times, dtype=float)
    if len(values) == 0:
        raise ValueError("modal window needs a non-empty sample")
    candidates = np.unique(values)
    counts = np.asarray([np.sum(np.abs(values - c) <= WINDOW) for c in candidates])
    return float(candidates[int(np.argmax(counts))])


def _r_time(member: np.ndarray, control: np.ndarray) -> float:
    if not len(member) or not len(control):
        return float("nan")
    p_member = float(member.mean())
    p_control = float(control.mean())
    return float(np.clip((p_member - p_control) / max(1.0 - p_control, 1e-9), 0.0, 1.0))


class AggregateCells:
    """Deterministic aggregation cells derived from the target only."""

    def __init__(self, target: pd.DataFrame, min_rows: int = MIN_CELL_ROWS) -> None:
        self.target = target.reset_index(drop=True)
        self.min_rows = int(min_rows)
        self.cuts = age_cuts(self.target)
        fine, coarse = cell_keys(self.target, self.cuts)
        self._fine = {str(k): np.flatnonzero(fine == k) for k in pd.unique(fine)}
        self._coarse = {str(k): np.flatnonzero(coarse == k) for k in pd.unique(coarse)}
        self._all = np.arange(len(self.target), dtype=int)

    def resolve(self, fine: str, coarse: str) -> tuple[str, np.ndarray]:
        """Finest cell with enough target rows; deterministic fallback chain."""
        for level, key in (("fine", fine), ("coarse", coarse)):
            source = self._fine if level == "fine" else self._coarse
            idx = source.get(key)
            if idx is not None and len(idx) >= self.min_rows:
                return f"{level}:{key}", idx
        return "global:*", self._all

    def assign(self, frame: pd.DataFrame) -> np.ndarray:
        """Return one resolved cell label per row of ``frame``."""
        fine, coarse = cell_keys(frame, self.cuts)
        return np.array(
            [self.resolve(str(f), str(c))[0] for f, c in zip(fine, coarse)], dtype=object
        )

    def target_times(self, label: str) -> np.ndarray:
        level, key = label.split(":", 1)
        if level == "fine":
            idx = self._fine[key]
        elif level == "coarse":
            idx = self._coarse[key]
        else:
            idx = self._all
        return self.target.iloc[idx]["time"].to_numpy(float)


def oracle_cell_ceiling(
    cells: AggregateCells,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    member_active: Sequence[bool] | None = None,
    control_active: Sequence[bool] | None = None,
) -> dict[str, float | int]:
    """Best possible score for any attack constant within an aggregation cell.

    The member labels are revealed here on purpose: this is an upper bound, not
    an attack.  One time is chosen per cell to maximise member success, and the
    *same* time is then charged to the controls of that cell, because a real
    attacker cannot separate them inside the cell.  ``*_active`` restricts the
    bound to the rows a frozen gate would answer; inactive rows abstain and can
    never succeed.
    """
    member_labels = cells.assign(members)
    control_labels = cells.assign(controls)
    member_truth = members["time"].to_numpy(float)
    control_truth = controls["time"].to_numpy(float)
    member_hit = np.zeros(len(members), dtype=bool)
    control_hit = np.zeros(len(controls), dtype=bool)
    m_active = np.ones(len(members), bool) if member_active is None else np.asarray(member_active, bool)
    c_active = np.ones(len(controls), bool) if control_active is None else np.asarray(control_active, bool)
    for label in pd.unique(member_labels):
        rows = (member_labels == label) & m_active
        if not rows.any():
            continue
        grid = np.unique(cells.target_times(str(label)))
        scores = np.asarray([window_success(t, member_truth[rows]).sum() for t in grid])
        best = float(grid[int(np.argmax(scores))])
        member_hit[rows] = window_success(best, member_truth[rows])
        control_rows = (control_labels == label) & c_active
        if control_rows.any():
            control_hit[control_rows] = window_success(best, control_truth[control_rows])
    return {
        "oracle_success_member": float(member_hit.mean()),
        "oracle_success_control": float(control_hit.mean()),
        "oracle_r_time_all": _r_time(member_hit, control_hit),
        "active_member_rows": int(m_active.sum()),
        "active_control_rows": int(c_active.sum()),
        "cells": int(len(pd.unique(member_labels))),
    }


@dataclass(frozen=True)
class Stage5Config:
    method: str = "frozen_donor_modal"

    def validate(self) -> None:
        if self.method not in STAGE5_METHODS:
            raise ValueError(f"unknown stage5 method: {self.method}")


def stage5_config_grid() -> list[Stage5Config]:
    return [Stage5Config(method) for method in STAGE5_METHODS]


class AIAStage5Engine:
    """Frozen gate/sentinel with an aggregate-constraint in-gate estimator."""

    def __init__(self, target: pd.DataFrame, background: pd.DataFrame, config: Stage5Config) -> None:
        config.validate()
        self.config = config
        self.base = AIAStage3Engine(target, background, FROZEN_BASE)
        self.target = self.base.target
        self.sentinel = self.base.sentinel
        self.cells = AggregateCells(self.target)

    def frozen_predictions(self, query: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        return self.base.predict(query, FROZEN_METHOD)

    def predict(self, query: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        query = query.reset_index(drop=True)
        values, diagnostics = self.frozen_predictions(query)
        if self.config.method == "frozen_donor_modal":
            return values, diagnostics
        active = ~diagnostics["sentinel_hit"].to_numpy(bool)
        labels = self.cells.assign(query)
        result = values.copy()
        if self.config.method == "cell_modal_aggregate":
            for label in pd.unique(labels):
                rows = (labels == label) & active
                if rows.any():
                    result[rows] = modal_window_value(self.cells.target_times(str(label)))
        else:
            # One linear constraint per cell: shift the frozen predictions so the
            # in-gate cell mean matches the target cell mean.
            for label in pd.unique(labels):
                rows = (labels == label) & active
                if not rows.any():
                    continue
                shift = float(np.mean(self.cells.target_times(str(label)))) - float(values[rows].mean())
                result[rows] = np.maximum(values[rows] + shift, 0.0)
        result[~active] = self.sentinel
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError("time_hat must be finite and non-negative")
        return result, diagnostics


__all__ = [
    "AIAStage5Engine", "AggregateCells", "FROZEN_BASE", "FROZEN_METHOD",
    "MIN_CELL_ROWS", "STAGE5_METHODS", "Stage5Config", "age_cuts", "cell_keys",
    "modal_window_value", "oracle_cell_ceiling", "stage5_config_grid",
    "window_success",
]
