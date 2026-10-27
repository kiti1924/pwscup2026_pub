"""Distinct-distance-shell MIA experts for the S4 nearest-decoy probe.

The frozen MIA pipeline uses the same block-wise standardized geometry as
``LowFPRFeatureModel``.  This module changes only which distinct distance
shell is used for the relative score.  Shell membership is defined by unique
finite distances, never by the second row returned by an argsort, so exact
ties cannot make the result depend on CSV order.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mia_low_fpr import (
    BLOCK,
    CONTINUOUS,
    LowFPRFeatureModel,
    empirical_percentile,
)


FORMULAS = ("h1a_target_only", "h1b_bilateral")
EPS = 1e-9
MISSING_DISTANCE = 1e6


def _key(value: object) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def distinct_shell_values(distances: Sequence[float], max_shells: int = 3) -> tuple[np.ndarray, int]:
    """Return the first distinct finite distances, with conservative repeats.

    If fewer than ``max_shells`` exist, the last available shell is repeated.
    The repeat is explicit in the fallback count and makes the score
    deterministic for singleton/duplicate pools.
    """
    if max_shells < 1:
        raise ValueError("max_shells must be positive")
    values = np.asarray(distances, dtype=float)
    finite = np.sort(np.unique(values[np.isfinite(values)]))
    if len(finite) == 0:
        return np.full(max_shells, MISSING_DISTANCE, dtype=float), max_shells
    take = min(len(finite), max_shells)
    shells = np.empty(max_shells, dtype=float)
    shells[:take] = finite[:take]
    shells[take:] = finite[take - 1]
    return shells, max_shells - take


class DistinctShellFeatureModel(LowFPRFeatureModel):
    """Low-FPR feature model with H1a or H1b distinct-shell geometry."""

    def __init__(
        self,
        target: pd.DataFrame,
        reference: pd.DataFrame,
        rare_detector: Mapping[str, object],
        formula: str,
    ) -> None:
        if formula not in FORMULAS:
            raise ValueError(f"unknown skip1 formula: {formula}")
        self.formula = formula
        super().__init__(target, reference, rare_detector)

    def _shells(
        self,
        candidates: pd.DataFrame,
        z: np.ndarray,
        pool_z: np.ndarray,
        blocks: Mapping[tuple, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        result = np.full((len(candidates), 3), MISSING_DISTANCE, dtype=float)
        fallback_count = np.full(len(candidates), 3, dtype=int)
        groups = candidates.groupby(BLOCK, dropna=False).indices
        all_indices = np.arange(len(pool_z), dtype=int)
        for raw_key, candidate_indices in groups.items():
            key = _key(raw_key)
            pool_indices = blocks.get(key)
            # This is only an edge-case fallback.  On the official data the
            # block is populated, and the normal path is identical to the
            # frozen block geometry.
            if pool_indices is None or not len(pool_indices):
                pool_indices = all_indices
            ci = np.asarray(candidate_indices, dtype=int)
            matrix = cdist(z[ci], pool_z[pool_indices]) / np.sqrt(len(CONTINUOUS))
            for local, row_distances in enumerate(matrix):
                shells, missing = distinct_shell_values(row_distances, 3)
                result[ci[local]] = shells
                fallback_count[ci[local]] = missing
        return result, fallback_count

    def raw_features(self, candidates: pd.DataFrame) -> pd.DataFrame:
        candidates = candidates.reset_index(drop=True)
        result = super().raw_features(candidates)
        z = (candidates[CONTINUOUS].to_numpy(float) - self.mean) / self.std
        c_shells, c_missing = self._shells(candidates, z, self.target_z, self.target_blocks)
        a_shells, a_missing = self._shells(candidates, z, self.reference_z, self.reference_blocks)

        if self.formula == "h1a_target_only":
            relative = np.log(a_shells[:, 0] + EPS) - np.log(c_shells[:, 1] + EPS)
        else:
            relative = np.log(a_shells[:, 1] + EPS) - np.log(c_shells[:, 1] + EPS)

        result["nn_relative"] = relative
        result["distance_c"] = c_shells[:, 1]
        result["distance_a"] = a_shells[:, 1] if self.formula == "h1b_bilateral" else a_shells[:, 0]
        for side, shells in (("c", c_shells), ("a", a_shells)):
            for index in range(3):
                result[f"distance_{side}{index + 1}"] = shells[:, index]
        result["gap_c21"] = np.log((c_shells[:, 1] + EPS) / (c_shells[:, 0] + EPS))
        result["gap_a21"] = np.log((a_shells[:, 1] + EPS) / (a_shells[:, 0] + EPS))
        result["gap_contrast"] = result["gap_c21"] - result["gap_a21"]
        result["c_shell_fallback_count"] = c_missing
        result["a_shell_fallback_count"] = a_missing
        return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)


class Skip1Scorer:
    """Frozen-stage-style percentile calibration for one skip1 formula."""

    def __init__(
        self,
        target: pd.DataFrame,
        reference: pd.DataFrame,
        calibration: pd.DataFrame,
        rare_detector: Mapping[str, object],
        formula: str,
    ) -> None:
        self.formula = formula
        self.model = DistinctShellFeatureModel(target, reference, rare_detector, formula)
        self.null_raw = self.model.raw_features(calibration)

    def score(self, candidates: pd.DataFrame) -> np.ndarray:
        raw = self.model.raw_features(candidates)
        return empirical_percentile(raw["nn_relative"], self.null_raw["nn_relative"])

    def diagnostics(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return self.model.raw_features(candidates)


__all__ = [
    "DistinctShellFeatureModel",
    "EPS",
    "FORMULAS",
    "MISSING_DISTANCE",
    "Skip1Scorer",
    "distinct_shell_values",
]
