"""Stage-3 MIA record-difficulty normalization.

Stage 2's frozen Block-NN score remains the rollback baseline.  This module
adds a deliberately small, inspectable challenger: score each candidate
against target-independent shadow targets made from a disjoint A_bg pool and
subtract the candidate's typical shadow score (its record difficulty) from the
target score.  The shadows are fixed before pseudo tune/test evaluation, so
the method cannot learn from membership labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mia_low_fpr import grouped_tpr_at_fpr, mahalanobis_rarity
from mia_stage2 import FrozenStage1NNScorer, make_stage2_pseudo_target


STAGE3_METHODS = (
    "frozen_stage2_nn",
    "difficulty_mean_a025",
    "difficulty_mean_a050",
    "difficulty_median_a025",
    "difficulty_q75_a025",
)


@dataclass(frozen=True)
class DifficultyConfig:
    """A fixed shadow mechanism/seed grid; no evaluation labels are used."""

    mechanisms: tuple[str, ...] = (
        "raw_resample",
        "perturbed_rounded",
        "block_microaggregate",
    )
    seeds: tuple[int, ...] = (6201, 6202, 6203)

    def validate(self) -> None:
        if not self.mechanisms or not self.seeds:
            raise ValueError("shadow grid must be non-empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("shadow seeds must be unique")


class RecordDifficultyNormalizer:
    """Estimate target-independent record difficulty from fixed shadow C's."""

    def __init__(
        self,
        core: pd.DataFrame,
        shadow_source: pd.DataFrame,
        rare_detector: Mapping[str, object],
        config: DifficultyConfig = DifficultyConfig(),
    ) -> None:
        config.validate()
        if set(core["record_id"]).intersection(shadow_source["record_id"]):
            raise ValueError("shadow source overlaps crossfit core")
        self.config = config
        self.core = core.reset_index(drop=True)
        self.shadow_source = shadow_source.reset_index(drop=True)
        self.rare_detector = rare_detector
        self.shadow_scorers: list[FrozenStage1NNScorer] = []
        for mechanism in config.mechanisms:
            for seed in config.seeds:
                target = make_stage2_pseudo_target(
                    shadow_source,
                    core,
                    mechanism,
                    int(seed),
                )
                self.shadow_scorers.append(
                    FrozenStage1NNScorer(target, core, rare_detector)
                )

    @property
    def shadow_count(self) -> int:
        return len(self.shadow_scorers)

    def difficulty(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Return mean/median/q75 shadow score for every candidate row."""
        if not len(candidates):
            return pd.DataFrame({name: np.array([], dtype=float) for name in
                                 ("difficulty_mean", "difficulty_median", "difficulty_q75")})
        matrix = np.column_stack([scorer.score(candidates) for scorer in self.shadow_scorers])
        return pd.DataFrame({
            "difficulty_mean": matrix.mean(axis=1),
            "difficulty_median": np.median(matrix, axis=1),
            "difficulty_q75": np.quantile(matrix, .75, axis=1),
        })

    def normalize(
        self,
        baseline: Sequence[float],
        candidates: pd.DataFrame,
        method: str,
    ) -> np.ndarray:
        """Return an uncalibrated ranking score; the caller may rank it."""
        base = np.asarray(baseline, dtype=float)
        if method == "frozen_stage2_nn":
            return base.copy()
        if method not in STAGE3_METHODS:
            raise ValueError(method)
        difficulty = self.difficulty(candidates)
        if method == "difficulty_mean_a025":
            return base - .25 * difficulty["difficulty_mean"].to_numpy(float)
        if method == "difficulty_mean_a050":
            return base - .50 * difficulty["difficulty_mean"].to_numpy(float)
        if method == "difficulty_median_a025":
            return base - .25 * difficulty["difficulty_median"].to_numpy(float)
        return base - .25 * difficulty["difficulty_q75"].to_numpy(float)

    def score_methods(self, baseline: Sequence[float], candidates: pd.DataFrame) -> dict[str, np.ndarray]:
        """Normalize one cohort once and return every preregistered method."""
        base = np.asarray(baseline, dtype=float)
        difficulty = self.difficulty(candidates)
        return {
            "frozen_stage2_nn": base,
            "difficulty_mean_a025": base - .25 * difficulty["difficulty_mean"].to_numpy(float),
            "difficulty_mean_a050": base - .50 * difficulty["difficulty_mean"].to_numpy(float),
            "difficulty_median_a025": base - .25 * difficulty["difficulty_median"].to_numpy(float),
            "difficulty_q75_a025": base - .25 * difficulty["difficulty_q75"].to_numpy(float),
        }


def rank_bounded(values: Sequence[float]) -> np.ndarray:
    """Convert an arbitrary ranking score to bounded confidence values."""
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("non-finite ranking score")
    if len(x) <= 1:
        return np.full(len(x), .5, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 / (len(x) - 1)
        start = stop
    return ranks


def _strict_tpr_at_fpr(scores: Sequence[float], truth: Sequence[int], target_fpr: float) -> float:
    """A tie-sensitive upper bound with stable row-order tie breaking."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(truth, dtype=int)
    order = np.argsort(-s, kind="mergesort")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if not n_pos or not n_neg:
        return 0.0
    allowed = max(0, int(np.floor(target_fpr * n_neg)))
    selected = order[:allowed] if allowed else np.array([], dtype=int)
    return float(y[selected].sum() / n_pos)


def evaluate_methods(
    member_scores: Mapping[str, Sequence[float]],
    control_scores: Mapping[str, Sequence[float]],
    member: pd.DataFrame,
    control: pd.DataFrame,
    rare_detector: Mapping[str, object],
) -> pd.DataFrame:
    """Evaluate 0.1% and 1% FPR, rare strata, and tie sensitivity."""
    truth = np.r_[np.ones(len(member), dtype=int), np.zeros(len(control), dtype=int)]
    rare_member = mahalanobis_rarity(member, rare_detector) >= float(rare_detector.get("gate", 8.0))
    rare_control = mahalanobis_rarity(control, rare_detector) >= float(rare_detector.get("gate", 8.0))
    rows: list[dict[str, object]] = []
    for method in member_scores:
        scores = np.r_[np.asarray(member_scores[method], float), np.asarray(control_scores[method], float)]
        tpr001, auc = grouped_tpr_at_fpr(scores, truth, target_fpr=.001)
        tpr01, _ = grouped_tpr_at_fpr(scores, truth, target_fpr=.01)
        rare_mask = np.r_[rare_member, rare_control]
        if rare_member.any() and rare_control.any():
            rare_tpr001, rare_auc = grouped_tpr_at_fpr(scores[rare_mask], truth[rare_mask], target_fpr=.001)
            rare_tpr01, _ = grouped_tpr_at_fpr(scores[rare_mask], truth[rare_mask], target_fpr=.01)
        else:
            rare_tpr001 = rare_tpr01 = rare_auc = float("nan")
        strict = _strict_tpr_at_fpr(scores, truth, .001)
        rows.append({
            "model": method,
            "tpr_at_fpr_0_001": tpr001,
            "tpr_at_fpr_0_01": tpr01,
            "auc": auc,
            "rare_tpr_at_fpr_0_001": rare_tpr001,
            "rare_tpr_at_fpr_0_01": rare_tpr01,
            "rare_auc": rare_auc,
            "rare_n_member": int(rare_member.sum()),
            "rare_n_control": int(rare_control.sum()),
            "tie_sensitive_tpr_0_001": strict,
            "tie_delta_0_001": float(strict - tpr001),
            "n_member": len(member),
            "n_control": len(control),
        })
    return pd.DataFrame(rows)


def select_with_guardrails(
    metrics: pd.DataFrame,
    baseline: str = "frozen_stage2_nn",
    min_mean_gain: float = .005,
    worst_tolerance: float = .005,
    auc_tolerance: float = .01,
) -> tuple[str, pd.DataFrame]:
    """Tune-only selection using mean/worst 0.1% TPR and AUC."""
    summary = metrics.groupby("model", as_index=False).agg(
        mean_tpr_001=("tpr_at_fpr_0_001", "mean"),
        worst_run_tpr_001=("tpr_at_fpr_0_001", "min"),
        mean_tpr_01=("tpr_at_fpr_0_01", "mean"),
        mean_auc=("auc", "mean"),
        mean_rare_tpr_001=("rare_tpr_at_fpr_0_001", "mean"),
        worst_tie_delta=("tie_delta_0_001", "min"),
    )
    base = summary.loc[summary.model == baseline].iloc[0]
    summary["mean_gain_vs_baseline"] = summary.mean_tpr_001 - float(base.mean_tpr_001)
    summary["eligible"] = (
        (summary.model != baseline)
        & (summary.mean_gain_vs_baseline >= min_mean_gain)
        & (summary.worst_run_tpr_001 >= float(base.worst_run_tpr_001) - worst_tolerance)
        & (summary.mean_auc >= float(base.mean_auc) - auc_tolerance)
        & (summary.mean_tpr_01 >= float(base.mean_tpr_01) - worst_tolerance)
    )
    eligible = summary[summary.eligible].sort_values(
        ["mean_tpr_001", "worst_run_tpr_001", "mean_auc", "model"],
        ascending=[False, False, False, True],
    )
    winner = baseline if eligible.empty else str(eligible.iloc[0].model)
    return winner, summary.sort_values(
        ["mean_tpr_001", "worst_run_tpr_001", "mean_auc", "model"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
