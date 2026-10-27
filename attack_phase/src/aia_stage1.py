"""Stage-1 AIA attacks: block nearest-neighbour and time-window variants.

The official attack uses a sex-by-prefecture nearest neighbour over five
continuous QI columns and copies one ``time`` value.  This module keeps that
baseline frozen and adds deterministic, metric-aware variants.  It contains
no hidden membership labels; evaluation helpers operate only on supplied
member/control proxy cohorts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


QI_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT")
BLOCK_COLUMNS = ("sex", "prefecture")
OUTCOME_COLUMNS = ("time", "onset", "death")
METHODS = (
    "official_1nn",
    "weighted_knn",
    "weighted_modal_window",
    "membership_gated",
    "weighted_modal_k25",
)


def _groups(frame: pd.DataFrame, keys: Sequence[str]) -> dict[tuple, np.ndarray]:
    result: dict[tuple, np.ndarray] = {}
    for key, idx in frame.groupby(list(keys), dropna=False).indices.items():
        result[key if isinstance(key, tuple) else (key,)] = np.asarray(idx, dtype=int)
    return result


def _r_time(success_member: Sequence[bool], success_control: Sequence[bool]) -> float:
    member = np.asarray(success_member, dtype=float)
    control = np.asarray(success_control, dtype=float)
    if not len(member) or not len(control):
        return float("nan")
    p_member = float(member.mean())
    p_control = float(control.mean())
    return float(np.clip((p_member - p_control) / max(1.0 - p_control, 1e-9), 0.0, 1.0))


def r_time_deadzone(
    success_member: Sequence[bool], success_control: Sequence[bool], z: float = 1.64
) -> float:
    member = np.asarray(success_member, dtype=float)
    control = np.asarray(success_control, dtype=float)
    if not len(member) or not len(control):
        return float("nan")
    if z <= 0:
        return _r_time(member, control)
    p_member = float(member.mean())
    p_control = float(control.mean())
    se = np.sqrt(
        p_member * (1.0 - p_member) / len(member)
        + p_control * (1.0 - p_control) / len(control)
    )
    return float(np.clip((p_member - p_control - z * se) / max(1.0 - p_control, 1e-9), 0.0, 1.0))


def evaluate_time_predictions(
    member_hat: Sequence[float], member_truth: Sequence[float],
    control_hat: Sequence[float], control_truth: Sequence[float],
    tau: float = 0.5, member_rare: Sequence[bool] | None = None,
    control_rare: Sequence[bool] | None = None, rare_z: float = 1.64,
    min_rare_n: int = 1,
) -> dict[str, float | int | bool]:
    member_success = np.abs(np.asarray(member_hat, float) - np.asarray(member_truth, float)) <= tau
    control_success = np.abs(np.asarray(control_hat, float) - np.asarray(control_truth, float)) <= tau
    result: dict[str, float | int | bool] = {
        "success_member": float(member_success.mean()) if len(member_success) else float("nan"),
        "success_control": float(control_success.mean()) if len(control_success) else float("nan"),
        "r_time_all": _r_time(member_success, control_success),
        "tau_time": float(tau),
        "rare_evaluable": False,
        "rare_member_n": 0,
        "rare_control_n": 0,
    }
    if member_rare is not None and control_rare is not None:
        mm = np.asarray(member_rare, bool)
        cm = np.asarray(control_rare, bool)
        result["rare_member_n"] = int(mm.sum())
        result["rare_control_n"] = int(cm.sum())
        if int(mm.sum()) >= min_rare_n and int(cm.sum()) >= min_rare_n:
            result["rare_evaluable"] = True
            result["success_member_rare"] = float(member_success[mm].mean())
            result["success_control_rare"] = float(control_success[cm].mean())
            result["r_time_rare_raw"] = _r_time(member_success[mm], control_success[cm])
            result["r_time_rare"] = r_time_deadzone(member_success[mm], control_success[cm], rare_z)
            result["r_time_score"] = max(float(result["r_time_all"]), float(result["r_time_rare"]))
    if "r_time_score" not in result:
        result["r_time_score"] = float(result["r_time_all"])
    return result


def rare_mask(frame: pd.DataFrame, detector: Mapping[str, object], gate: float | None = None) -> np.ndarray:
    cols = list(detector["columns"])
    x = frame[cols].to_numpy(float)
    mu = np.asarray(detector["mu"], float)
    sigma = np.linalg.pinv(np.asarray(detector["sigma"], float))
    delta = x - mu
    distance = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, sigma, delta)))
    return distance >= float(detector.get("gate", 8.0) if gate is None else gate)


@dataclass
class AIAEngine:
    target: pd.DataFrame
    background: pd.DataFrame
    weights: Mapping[str, float] | None = None
    gate_log_contrast: float = 1.0
    modal_k: int = 15
    swap_k: int = 25

    def __post_init__(self) -> None:
        self.target = self.target.reset_index(drop=True).copy()
        self.background = self.background.reset_index(drop=True).copy()
        self.weights = dict(self.weights or {col: 1.0 for col in QI_COLUMNS})
        # Match the official reference exactly: pandas' sample SD (ddof=1).
        self._std = self.background[list(QI_COLUMNS)].std(ddof=1).replace(0.0, 1.0).fillna(1.0).to_numpy(float)
        self._target_z = self.target[list(QI_COLUMNS)].to_numpy(float) / self._std
        self._background_z = self.background[list(QI_COLUMNS)].to_numpy(float) / self._std
        self._target_blocks = _groups(self.target, BLOCK_COLUMNS)
        self._background_blocks = _groups(self.background, BLOCK_COLUMNS)
        self._target_sex = _groups(self.target, [BLOCK_COLUMNS[0]])
        self._background_sex = _groups(self.background, [BLOCK_COLUMNS[0]])
        self._weights = np.asarray([float(self.weights.get(col, 1.0)) for col in QI_COLUMNS], float)
        if (self._weights < 0).any() or not self._weights.sum() > 0:
            raise ValueError("weights must be non-negative with positive sum")

    def _pool_indices(self, row: pd.Series, source: str) -> np.ndarray:
        if source == "target":
            blocks, sex, n = self._target_blocks, self._target_sex, len(self.target)
        else:
            blocks, sex, n = self._background_blocks, self._background_sex, len(self.background)
        key = tuple(row[col] for col in BLOCK_COLUMNS)
        idx = blocks.get(key)
        if idx is None or not len(idx):
            idx = sex.get((row[BLOCK_COLUMNS[0]],))
        if idx is None or not len(idx):
            idx = np.arange(n, dtype=int)
        return idx

    def _distance(self, query: np.ndarray, source_z: np.ndarray, idx: np.ndarray) -> np.ndarray:
        delta = source_z[idx] - query
        return np.sqrt(np.sum(delta * delta * self._weights, axis=1) / self._weights.sum())

    def _neighbors(self, row: pd.Series, source: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = row[list(QI_COLUMNS)].to_numpy(float) / self._std
        source_z = self._target_z if source == "target" else self._background_z
        idx = self._pool_indices(row, source)
        distances = self._distance(query, source_z, idx)
        order = np.argsort(distances, kind="mergesort")
        take = min(max(int(k), 1), len(order))
        # Keep all records tied at the kth distance so predictions do not
        # depend on target CSV row order.  This also preserves every exact
        # nearest-neighbour tie used by the frozen official baseline.
        kth = distances[order[take - 1]]
        chosen = order[distances[order] <= kth]
        return idx[chosen], distances[chosen]

    @staticmethod
    def _modal_time(times: np.ndarray, distances: np.ndarray) -> float:
        weights = 1.0 / np.maximum(distances, 1e-3)
        candidates = np.unique(times)
        window_scores = np.asarray([weights[np.abs(times - value) <= 0.5].sum() for value in candidates])
        best = np.flatnonzero(window_scores == window_scores.max())
        return float(candidates[best[0]])

    @staticmethod
    def _weighted_mean(times: np.ndarray, distances: np.ndarray) -> float:
        weights = 1.0 / np.maximum(distances, 1e-3)
        return float(np.average(times, weights=weights))

    @staticmethod
    def _prefix_with_ties(
        indices: np.ndarray, distances: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        take = min(max(int(k), 1), len(indices))
        kth = distances[take - 1]
        keep = distances <= kth
        return indices[keep], distances[keep]

    def _predict_one(self, row: pd.Series, method: str) -> tuple[float, float, float, bool, int]:
        if method not in METHODS:
            raise ValueError(f"unknown AIA method: {method}")
        nn_idx, nn_dist = self._neighbors(row, "target", max(self.modal_k, self.swap_k, 5))
        nearest = nn_dist == nn_dist.min()
        nearest_time = float(np.mean(np.sort(self.target.iloc[nn_idx[nearest]]["time"].to_numpy(float))))
        d_c = float(nn_dist.min())
        a_idx, a_dist = self._neighbors(row, "background", max(self.modal_k, self.swap_k, 5))
        d_a = float(a_dist.min())
        contrast = float(np.log(d_a + 1e-9) - np.log(d_c + 1e-9))
        if method == "official_1nn":
            prediction = nearest_time
            count = int(nearest.sum())
            gated = False
        elif method == "weighted_knn":
            chosen, chosen_dist = self._prefix_with_ties(nn_idx, nn_dist, 5)
            prediction = self._weighted_mean(self.target.iloc[chosen]["time"].to_numpy(float), chosen_dist)
            count = len(chosen)
            gated = False
        elif method == "weighted_modal_window":
            chosen, chosen_dist = self._prefix_with_ties(nn_idx, nn_dist, self.modal_k)
            prediction = self._modal_time(self.target.iloc[chosen]["time"].to_numpy(float), chosen_dist)
            count = len(chosen)
            gated = False
        elif method == "weighted_modal_k25":
            chosen, chosen_dist = self._prefix_with_ties(nn_idx, nn_dist, self.swap_k)
            prediction = self._modal_time(self.target.iloc[chosen]["time"].to_numpy(float), chosen_dist)
            count = len(chosen)
            gated = False
        else:
            gated = contrast >= self.gate_log_contrast
            if gated:
                chosen, chosen_dist = self._prefix_with_ties(nn_idx, nn_dist, self.modal_k)
                prediction = self._modal_time(self.target.iloc[chosen]["time"].to_numpy(float), chosen_dist)
                count = len(chosen)
            else:
                chosen, chosen_dist = self._prefix_with_ties(a_idx, a_dist, 5)
                prediction = self._weighted_mean(self.background.iloc[chosen]["time"].to_numpy(float), chosen_dist)
                count = len(chosen)
        return prediction, d_c, contrast, gated, count

    def predict(self, query: pd.DataFrame, method: str) -> tuple[np.ndarray, pd.DataFrame]:
        query = query.reset_index(drop=True)
        rows = [self._predict_one(row, method) for _, row in query.iterrows()]
        values = np.asarray([row[0] for row in rows], float)
        diagnostics = pd.DataFrame(rows, columns=["time_hat", "distance_c", "log_distance_contrast", "gate_hit", "neighbor_count"])
        diagnostics["distance_a"] = diagnostics["distance_c"] * np.exp(diagnostics["log_distance_contrast"])
        return np.maximum(values, 0.0), diagnostics


def make_pseudo_target(members: pd.DataFrame, mechanism: str, seed: int, noise_fraction: float = 0.035) -> pd.DataFrame:
    if mechanism not in {"near_copy", "perturbed", "triplet_swap", "bootstrap_copy"}:
        raise ValueError(mechanism)
    rng = np.random.default_rng(seed)
    if mechanism == "bootstrap_copy":
        indices = rng.integers(0, len(members), size=len(members))
    else:
        indices = rng.permutation(len(members))
    result = members.iloc[indices].reset_index(drop=True).copy()
    if mechanism == "perturbed":
        for col in QI_COLUMNS:
            scale = float(members[col].std(ddof=0)) or 1.0
            result[col] = result[col].to_numpy(float) + rng.normal(0.0, noise_fraction * scale, len(result))
        result["age"] = result["age"].round().clip(40, 74)
        for col in ("SBP", "TG", "HDL", "ALT"):
            result[col] = result[col].round(1)
    elif mechanism == "triplet_swap":
        for _, indices in result.groupby(list(BLOCK_COLUMNS), dropna=False).groups.items():
            idx = np.asarray(indices, int)
            shuffled = idx[rng.permutation(len(idx))]
            result.loc[idx, list(OUTCOME_COLUMNS)] = result.loc[shuffled, list(OUTCOME_COLUMNS)].to_numpy()
    return result


def validate_f_aia_candidate(
    frame: pd.DataFrame, challenge_ids: Mapping[int, Sequence[int]]
) -> None:
    expected = {"target_k", "challenge_row_id", "time_hat"}
    if set(frame.columns) != expected:
        raise ValueError("F_aia columns must be target_k, challenge_row_id, time_hat")
    expected_count = sum(len(ids) for ids in challenge_ids.values())
    if len(frame) != expected_count:
        raise ValueError("F_aia row count mismatch")
    for target, row_ids in challenge_ids.items():
        part = frame[frame["target_k"] == int(target)]
        if len(part) != len(row_ids) or sorted(part["challenge_row_id"].tolist()) != sorted(int(x) for x in row_ids):
            raise ValueError(f"missing challenge rows for target {target}")
    values = frame["time_hat"].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("time_hat must be finite and non-negative")


__all__ = [
    "AIAEngine", "BLOCK_COLUMNS", "METHODS", "QI_COLUMNS", "evaluate_time_predictions",
    "make_pseudo_target", "rare_mask", "r_time_deadzone", "validate_f_aia_candidate",
]
