"""Stage-4 categorical-aware classifier density-ratio MIA.

The Stage-2 single-split Block-NN score is deliberately kept as the frozen
rollback baseline.  This module adds two small, inspectable classifiers that
estimate the density ratio between a pseudo target ``C`` (positive class) and
an A_bg reference (negative class).  Continuous columns are standardized and
categorical columns are one-hot encoded for logistic regression; ExtraTrees
uses deterministic ordinal encoding.  Classifier outputs are calibrated only
against an independent null frame and are never fit on pseudo tune/test rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from mia_low_fpr import CONTINUOUS, empirical_percentile, mahalanobis_rarity
from mia_stage3 import rank_bounded


STATIC = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG"]
CATEGORICAL = ["sex", "prefecture", "smoking"]
CLASSIFIER_METHODS = ("classifier_logistic", "classifier_extratrees")
COMBINATION_METHODS = (
    "blend_nn_logistic_25",
    "blend_nn_extratrees_25",
    "gate_logistic_p995_nn_p995",
    "gate_extratrees_p995_nn_p995",
)
STAGE4_METHODS = ("frozen_stage2_nn", *CLASSIFIER_METHODS, *COMBINATION_METHODS)


def static_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the disclosed static columns and drop identifiers/outcomes."""
    missing = [column for column in STATIC if column not in frame]
    if missing:
        raise ValueError(f"missing static columns: {missing}")
    return frame.drop(columns=["record_id", "is_rare"], errors="ignore")[STATIC].copy()


def _safe_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    out = static_frame(frame)
    for column in CONTINUOUS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in CATEGORICAL:
        out[column] = out[column].astype("string").fillna("__MISSING__")
    return out


def _logistic_pipeline(seed: int) -> Pipeline:
    # The fixed preprocessing is intentionally simple enough to audit in a
    # run metadata file and robust to an unseen category in public C.
    numerical = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformer = ColumnTransformer([
        ("continuous", numerical, CONTINUOUS),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", transformer),
        ("model", LogisticRegression(
            C=0.5, penalty="l2", solver="liblinear", max_iter=1000,
            class_weight="balanced", random_state=int(seed),
        )),
    ])


def _extra_trees_pipeline(seed: int) -> Pipeline:
    transformer = ColumnTransformer([
        ("continuous", SimpleImputer(strategy="median"), CONTINUOUS),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", transformer),
        ("model", ExtraTreesClassifier(
            n_estimators=160, max_depth=12, min_samples_leaf=3,
            max_features=0.7, class_weight="balanced", random_state=int(seed),
            n_jobs=1,
        )),
    ])


@dataclass
class CategoricalDensityRatioClassifier:
    """Fit fixed logistic/ExtraTrees density-ratio estimators."""

    target: pd.DataFrame
    reference: pd.DataFrame
    seed: int = 20260828

    def __post_init__(self) -> None:
        if len(self.target) == 0 or len(self.reference) == 0:
            raise ValueError("target and reference must be non-empty")
        self.target = _safe_matrix(self.target).reset_index(drop=True)
        self.reference = _safe_matrix(self.reference).reset_index(drop=True)
        x = pd.concat([self.target, self.reference], ignore_index=True)
        y = np.r_[np.ones(len(self.target), dtype=int), np.zeros(len(self.reference), dtype=int)]
        self.models = {
            "classifier_logistic": _logistic_pipeline(self.seed),
            "classifier_extratrees": _extra_trees_pipeline(self.seed + 17),
        }
        for model in self.models.values():
            model.fit(x, y)

    def raw_scores(self, candidates: pd.DataFrame) -> dict[str, np.ndarray]:
        x = _safe_matrix(candidates)
        result: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            estimator = model.named_steps["model"]
            if hasattr(estimator, "decision_function"):
                value = model.decision_function(x)
            else:
                value = model.predict_proba(x)[:, 1]
            value = np.asarray(value, dtype=float)
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite {name} score")
            result[name] = value
        return result

    def calibrated_scores(
        self, candidates: pd.DataFrame, null_frame: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Map raw classifier scores to empirical upper-tail null percentiles."""
        candidate_raw = self.raw_scores(candidates)
        null_raw = self.raw_scores(null_frame)
        return {
            name: empirical_percentile(candidate_raw[name], null_raw[name])
            for name in CLASSIFIER_METHODS
        }


def compose_methods(
    baseline: Sequence[float], classifier_scores: Mapping[str, Sequence[float]]
) -> dict[str, np.ndarray]:
    """Return baseline, standalone classifiers, and four pre-registered tails."""
    base = np.asarray(baseline, dtype=float)
    if not np.isfinite(base).all():
        raise ValueError("non-finite baseline score")
    logistic = np.asarray(classifier_scores["classifier_logistic"], dtype=float)
    extra = np.asarray(classifier_scores["classifier_extratrees"], dtype=float)
    if len(base) != len(logistic) or len(base) != len(extra):
        raise ValueError("baseline/classifier length mismatch")
    result = {
        "frozen_stage2_nn": base.copy(),
        "classifier_logistic": logistic.copy(),
        "classifier_extratrees": extra.copy(),
        "blend_nn_logistic_25": .75 * base + .25 * logistic,
        "blend_nn_extratrees_25": .75 * base + .25 * extra,
        # A gate only promotes an already extreme NN row when the independent
        # classifier is also extreme.  Non-qualified rows are bit-identical.
        "gate_logistic_p995_nn_p995": np.where(
            (logistic >= .995) & (base >= .995), np.maximum(base, .995), base
        ),
        "gate_extratrees_p995_nn_p995": np.where(
            (extra >= .995) & (base >= .995), np.maximum(base, .995), base
        ),
    }
    return {name: np.clip(value, 0.0, 1.0) for name, value in result.items()}


def evaluate_stage4(
    member_scores: Mapping[str, Sequence[float]],
    control_scores: Mapping[str, Sequence[float]],
    member: pd.DataFrame,
    control: pd.DataFrame,
    rare_detector: Mapping[str, object],
) -> pd.DataFrame:
    """Evaluate primary .1% FPR, secondary 1% FPR, AUC, rare and tie metrics."""
    from mia_stage3 import evaluate_methods

    # Stage 3's evaluator uses the established grouped ROC and tie diagnostics.
    return evaluate_methods(member_scores, control_scores, member, control, rare_detector)


def mechanism_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["mechanism", "model"], as_index=False).agg(
        mean_tpr_001=("tpr_at_fpr_0_001", "mean"),
        worst_run_tpr_001=("tpr_at_fpr_0_001", "min"),
        mean_tpr_01=("tpr_at_fpr_0_01", "mean"),
        mean_auc=("auc", "mean"),
        mean_rare_tpr_001=("rare_tpr_at_fpr_0_001", "mean"),
        min_rare_tpr_001=("rare_tpr_at_fpr_0_001", "min"),
        tie_delta_min=("tie_delta_0_001", "min"),
        tie_delta_max=("tie_delta_0_001", "max"),
        runs=("seed", "count"),
    )


def select_with_guardrails(
    metrics: pd.DataFrame,
    baseline: str = "frozen_stage2_nn",
    min_mean_gain: float = .005,
    worst_tolerance: float = .005,
    auc_tolerance: float = .01,
    tpr01_tolerance: float = .005,
) -> tuple[str, pd.DataFrame]:
    """Tune-only selection; all failed challengers fall back to frozen NN."""
    summary = metrics.groupby("model", as_index=False).agg(
        mean_tpr_001=("tpr_at_fpr_0_001", "mean"),
        worst_run_tpr_001=("tpr_at_fpr_0_001", "min"),
        mean_tpr_01=("tpr_at_fpr_0_01", "mean"),
        mean_auc=("auc", "mean"),
        mean_rare_tpr_001=("rare_tpr_at_fpr_0_001", "mean"),
        tie_delta_min=("tie_delta_0_001", "min"),
    )
    base = summary.loc[summary.model == baseline]
    if len(base) != 1:
        raise ValueError(f"missing unique baseline {baseline}")
    base = base.iloc[0]
    summary["mean_gain_vs_baseline"] = summary.mean_tpr_001 - float(base.mean_tpr_001)
    summary["eligible"] = (
        (summary.model != baseline)
        & (summary.mean_gain_vs_baseline >= min_mean_gain)
        & (summary.worst_run_tpr_001 >= float(base.worst_run_tpr_001) - worst_tolerance)
        & (summary.mean_auc >= float(base.mean_auc) - auc_tolerance)
        & (summary.mean_tpr_01 >= float(base.mean_tpr_01) - tpr01_tolerance)
    )
    winners = summary[summary.eligible].sort_values(
        ["mean_tpr_001", "worst_run_tpr_001", "mean_auc", "model"],
        ascending=[False, False, False, True],
    )
    winner = baseline if winners.empty else str(winners.iloc[0].model)
    return winner, summary.sort_values(
        ["mean_tpr_001", "worst_run_tpr_001", "mean_auc", "model"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def test_only_directional_confirmation(
    summary: pd.DataFrame, candidate: str, baseline: str = "frozen_stage2_nn"
) -> bool:
    """Independent confirmation gate mirroring preregistered selection rules."""
    base = summary.loc[summary.model == baseline]
    cand = summary.loc[summary.model == candidate]
    if len(base) != 1 or len(cand) != 1:
        raise ValueError("missing confirmation model")
    b, c = base.iloc[0], cand.iloc[0]
    return bool(
        float(c.mean_tpr_001) > float(b.mean_tpr_001)
        and float(c.worst_run_tpr_001) >= float(b.worst_run_tpr_001) - .005
        and float(c.mean_auc) >= float(b.mean_auc) - .01
        and float(c.mean_tpr_01) >= float(b.mean_tpr_01) - .005
    )

