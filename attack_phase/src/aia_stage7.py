"""Corrective H4 feasibility bounds for cell-constant AIA attacks.

Stage 5 is retained as historical evidence, but its observed-grid and
member-only ceiling were not a true oracle for the official time objective.
This module keeps the Stage-2/3 gate, sentinel, and cell fallback fixed while
enumerating all relevant continuous sliding-window centers.  The feasible
family is deliberately limited to one constant prediction per resolved cell;
per-row constrained reconstruction is outside this bound.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from aia_stage1 import _r_time
from aia_stage3 import AIAStage3Engine, Stage3Config
from aia_stage5 import AggregateCells


FROZEN_BASE = Stage3Config("l2", "l2")
FROZEN_METHOD = "frozen_donor_modal"
STAGE7_METHODS = (FROZEN_METHOD, "cell_modal_aggregate")
WINDOW = 0.5
MIN_CELL_ROWS = 20


def support_centers(support: Sequence[float], tau: float = WINDOW) -> np.ndarray:
    """Enumerate every relevant non-negative center for a discrete support.

    A window count changes only when a left or right edge crosses a support
    point.  Therefore support, support-tau, and support+tau (plus zero for the
    non-negative prediction domain) contain an optimizer; no observed-grid
    restriction is imposed.
    """
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be positive and finite")
    values = np.asarray(support, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("support must be a non-empty finite vector")
    # Keep observed support first as a deterministic tie breaker.  Boundary
    # points remain included because inclusive windows can attain their
    # maximum at an unobserved ``x +/- tau`` center.
    candidates = np.concatenate((values, values - tau, values + tau, np.array([0.0])))
    _, first = np.unique(candidates, return_index=True)
    candidates = candidates[np.sort(first)]
    return candidates[candidates >= 0.0]


def sliding_hits(center: float, truth: Sequence[float], tau: float = WINDOW) -> np.ndarray:
    """Return official inclusive +/-tau window hits."""
    values = np.asarray(truth, dtype=float)
    return np.abs(values - float(center)) <= tau


def max_sliding_hits(truth: Sequence[float], tau: float = WINDOW) -> tuple[int, float]:
    """Return the exact maximum hit count and deterministic optimizing center."""
    values = np.asarray(truth, dtype=float)
    candidates = support_centers(values, tau)
    counts = np.asarray([int(sliding_hits(c, values, tau).sum()) for c in candidates])
    best = int(np.argmax(counts))
    return int(counts[best]), float(candidates[best])


def _resolved_labels(cells: AggregateCells, members: pd.DataFrame, controls: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    member_labels = cells.assign(members)
    control_labels = cells.assign(controls)
    labels = list(pd.unique(np.concatenate((member_labels, control_labels))))
    return member_labels, control_labels, [str(label) for label in labels]


def oracle_cell_constant(
    cells: AggregateCells,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    member_active: Sequence[bool] | None = None,
    control_active: Sequence[bool] | None = None,
    tau: float = WINDOW,
) -> dict[str, object]:
    """Compute the exact cell-constant oracle for official ``R_time``.

    For a fixed cell ``c``, its contribution is
    ``hits_member(c)/N_member - hits_control(c)/N_control``.  Global member
    and control rates are sums of these independent cell contributions, so
    maximizing each cell separately maximizes the complete cell-constant
    objective.  The selected predictions are also reported through the
    official normalized ``r_time_all`` score.  Cell options are selected by
    Dinkelbach's fractional-programming update because the control rate also
    appears in the official denominator.
    """
    member_labels, control_labels, labels = _resolved_labels(cells, members, controls)
    member_truth = members["time"].to_numpy(float)
    control_truth = controls["time"].to_numpy(float)
    m_active = np.ones(len(members), bool) if member_active is None else np.asarray(member_active, bool)
    c_active = np.ones(len(controls), bool) if control_active is None else np.asarray(control_active, bool)
    if m_active.shape != (len(members),) or c_active.shape != (len(controls),):
        raise ValueError("active masks must match their frame lengths")
    member_hit = np.zeros(len(members), dtype=bool)
    control_hit = np.zeros(len(controls), dtype=bool)
    centers: dict[str, float] = {}
    contributions: dict[str, float] = {}
    options: dict[str, list[tuple[float, int, int, np.ndarray, np.ndarray]]] = {}
    n_member = max(len(members), 1)
    n_control = max(len(controls), 1)
    for label in labels:
        mr = (member_labels == label) & m_active
        cr = (control_labels == label) & c_active
        support = np.concatenate((member_truth[mr], control_truth[cr]))
        if len(support) == 0:
            continue
        candidates = support_centers(support, tau)
        options[label] = []
        for center in candidates:
            mh = sliding_hits(center, member_truth[mr], tau)
            ch = sliding_hits(center, control_truth[cr], tau)
            options[label].append((float(center), int(mh.sum()), int(ch.sum()), mh, ch))

    # For a fixed trial ratio r, maximize the additive transform
    # m/Nm + (r-1)c/Nc in each cell.  Updating r from the resulting global hit
    # counts is Dinkelbach's exact finite-program algorithm (up to tolerance).
    ratio = 0.0
    selected: dict[str, tuple[float, int, int, np.ndarray, np.ndarray]] = {}
    for _ in range(100):
        selected = {}
        for label, cell_options in options.items():
            scores = np.asarray([
                m / n_member + (ratio - 1.0) * c / n_control
                for _, m, c, _, _ in cell_options
            ])
            best_score = float(scores.max())
            tied = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-12))
            # Prefer the largest denominator (fewest control hits) among
            # transformed ties.  This avoids a zero-denominator plateau
            # cycling with an equally scoring positive candidate.
            best = min(tied, key=lambda i: (cell_options[int(i)][2], -cell_options[int(i)][1], int(i)))
            selected[label] = cell_options[int(best)]
        total_m = sum(item[1] for item in selected.values())
        total_c = sum(item[2] for item in selected.values())
        p_member = total_m / n_member
        p_control = total_c / n_control
        candidate_ratio = float(np.clip(
            (p_member - p_control) / max(1.0 - p_control, 1e-9), 0.0, 1.0
        ))
        if abs(candidate_ratio - ratio) <= 1e-12:
            ratio = candidate_ratio
            break
        ratio = candidate_ratio

    for label, item in selected.items():
        center, member_count, control_count, mh, ch = item
        centers[label] = center
        contributions[label] = float(member_count / n_member - control_count / n_control)
        mr = (member_labels == label) & m_active
        cr = (control_labels == label) & c_active
        member_hit[mr] = mh
        control_hit[cr] = ch
    return {
        "oracle_success_member": float(member_hit.mean()) if len(member_hit) else float("nan"),
        "oracle_success_control": float(control_hit.mean()) if len(control_hit) else float("nan"),
        "oracle_r_time_contribution": float(member_hit.mean() - control_hit.mean()),
        "oracle_r_time_all": _r_time(member_hit, control_hit),
        "active_member_rows": int(m_active.sum()),
        "active_control_rows": int(c_active.sum()),
        "cells": len(labels),
        "centers": centers,
        "cell_contributions": contributions,
    }


def member_success_upper_bound(
    cells: AggregateCells,
    members: pd.DataFrame,
    member_active: Sequence[bool] | None = None,
    tau: float = WINDOW,
) -> dict[str, object]:
    """Return the conservative member-success upper bound.

    Controls are ignored and every cell gets its own best member-only center.
    Since official ``R_time = (p_member-p_control)/(1-p_control)`` is never
    greater than ``p_member`` for rates in [0,1], this is a conservative bound
    on any cell-constant attack's official score.
    """
    labels = cells.assign(members)
    truth = members["time"].to_numpy(float)
    active = np.ones(len(members), bool) if member_active is None else np.asarray(member_active, bool)
    if active.shape != (len(members),):
        raise ValueError("active mask must match member frame length")
    hits = np.zeros(len(members), dtype=bool)
    centers: dict[str, float] = {}
    for label in [str(x) for x in pd.unique(labels)]:
        rows = (labels == label) & active
        if not rows.any():
            continue
        count, center = max_sliding_hits(truth[rows], tau)
        centers[label] = center
        hits[rows] = sliding_hits(center, truth[rows], tau)
        assert int(hits[rows].sum()) == count
    return {
        "member_success_upper": float(hits.mean()) if len(hits) else float("nan"),
        "active_member_rows": int(active.sum()),
        "cells": int(len(pd.unique(labels))),
        "centers": centers,
    }


class AIAStage7Engine:
    """Stage-2/3 frozen gate and sentinel with one fixed aggregate challenger."""

    def __init__(self, target: pd.DataFrame, background: pd.DataFrame, method: str = FROZEN_METHOD) -> None:
        if method not in STAGE7_METHODS:
            raise ValueError(f"unknown stage7 method: {method}")
        self.method = method
        self.base = AIAStage3Engine(target, background, FROZEN_BASE)
        self.cells = AggregateCells(self.base.target, min_rows=MIN_CELL_ROWS)

    def predict(self, query: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        values, diagnostics = self.base.predict(query, "gated_abstain_donor_modal")
        if self.method == FROZEN_METHOD:
            return values, diagnostics
        result = values.copy()
        labels = self.cells.assign(query.reset_index(drop=True))
        active = ~diagnostics["sentinel_hit"].to_numpy(bool)
        for label in [str(x) for x in pd.unique(labels)]:
            rows = (labels == label) & active
            if not rows.any():
                continue
            _, center = max_sliding_hits(self.cells.target_times(label))
            result[rows] = center
        result[~active] = self.base.sentinel
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError("time_hat must be finite and non-negative")
        return result, diagnostics


__all__ = [
    "AIAStage7Engine", "AggregateCells", "FROZEN_BASE", "FROZEN_METHOD",
    "MIN_CELL_ROWS", "STAGE7_METHODS", "WINDOW",
    "max_sliding_hits", "member_success_upper_bound", "oracle_cell_constant",
    "sliding_hits", "support_centers",
]
