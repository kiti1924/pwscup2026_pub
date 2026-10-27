"""Stage-2 low-FPR MIA calibration and narrowly gated needle experts.

The stage-1 block-nearest-neighbour score remains available as a frozen
baseline.  Stage 2 repeats disjoint reference/calibration fits, aggregates
their evidence, and calibrates the aggregate on a separate meta-null.  Public
targets never provide labels or tuning feedback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mia_low_fpr import (
    CONTINUOUS,
    LowFPRFeatureModel,
    background_overlap_mask,
    calibrated_scores,
    empirical_percentile,
)


PARTITION_SIZES = {
    "crossfit_core": 2600,
    "meta_calibration": 500,
    "proxy_control": 700,
    "pseudo_tune": 671,
    "pseudo_test": 672,
}
DEFAULT_FOLD_SEEDS = (3101, 3102, 3103, 3104, 3105)
STAGE2_BASE_MODELS = (
    "crossfit_raw_median",
    "crossfit_raw_mean",
    "crossfit_raw_max",
    "crossfit_rank_mean",
    "crossfit_fisher",
)


def static_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["record_id", "is_rare"], errors="ignore").copy()


def record_id_digest(frame: pd.DataFrame) -> str:
    payload = ",".join(map(str, frame["record_id"].tolist())).encode()
    return hashlib.sha256(payload).hexdigest()


def partition_background(
    background: pd.DataFrame,
    seed: int,
    sizes: Mapping[str, int] = PARTITION_SIZES,
) -> dict[str, pd.DataFrame]:
    """Deterministically create mutually disjoint outer evaluation partitions."""
    if sum(sizes.values()) != len(background):
        raise ValueError(f"partition sizes {sum(sizes.values())} != A_bg rows {len(background)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(background))
    result: dict[str, pd.DataFrame] = {}
    start = 0
    for name, size in sizes.items():
        result[name] = background.iloc[order[start:start + size]].reset_index(drop=True)
        start += size
    sets = [set(frame["record_id"]) for frame in result.values()]
    if sum(map(len, sets)) != len(set().union(*sets)):
        raise AssertionError("outer A_bg partitions overlap")
    return result


def deterministic_fold_indices(
    core: pd.DataFrame,
    fold_seeds: Sequence[int] = DEFAULT_FOLD_SEEDS,
    reference_size: int = 1800,
) -> list[dict[str, np.ndarray]]:
    """Repeated disjoint reference/calibration partitions of one model core."""
    if reference_size <= 0 or reference_size >= len(core):
        raise ValueError("reference_size must leave a non-empty calibration fold")
    folds = []
    for seed in fold_seeds:
        order = np.random.default_rng(int(seed)).permutation(len(core))
        reference = order[:reference_size]
        calibration = order[reference_size:]
        if set(reference).intersection(calibration):
            raise AssertionError("reference/calibration overlap")
        folds.append({"seed": int(seed), "reference": reference, "calibration": calibration})
    return folds


def assert_evaluation_disjoint(parts: Mapping[str, pd.DataFrame]) -> None:
    """Reject any evaluation row that appears in a model or meta-calibration set."""
    model_ids = set(parts["crossfit_core"]["record_id"]) | set(parts["meta_calibration"]["record_id"])
    for name in ("proxy_control", "pseudo_tune", "pseudo_test"):
        overlap = model_ids.intersection(parts[name]["record_id"])
        if overlap:
            raise AssertionError(f"{name} has {len(overlap)} rows in reference/calibration pools")


@dataclass(frozen=True)
class NeedleGate:
    name: str
    exact: bool = False
    partial_min_fraction: float | None = None
    partial_max_distance: float | None = None
    promotion_floor: float = 0.995

    def validate(self) -> None:
        if not 0.0 <= self.promotion_floor <= 1.0:
            raise ValueError("promotion_floor outside [0,1]")
        if (self.partial_min_fraction is None) != (self.partial_max_distance is None):
            raise ValueError("partial fraction and distance must be supplied together")


GATE_VARIANTS = (
    NeedleGate("no_gate"),
    NeedleGate("exact_floor_0995", exact=True, promotion_floor=.995),
    NeedleGate("exact_floor_0999", exact=True, promotion_floor=.999),
    NeedleGate("exact_or_partial_6of7_d005", exact=True, partial_min_fraction=6 / 7, partial_max_distance=.05, promotion_floor=.995),
    NeedleGate("exact_or_partial_7of7_d005", exact=True, partial_min_fraction=1.0, partial_max_distance=.05, promotion_floor=.995),
    NeedleGate("exact_or_partial_6of7_d003", exact=True, partial_min_fraction=6 / 7, partial_max_distance=.03, promotion_floor=.995),
)


def gate_mask(features: pd.DataFrame, gate: NeedleGate) -> np.ndarray:
    gate.validate()
    mask = np.zeros(len(features), dtype=bool)
    if gate.exact:
        mask |= features["static_exact"].to_numpy(float) >= 1.0
    if gate.partial_min_fraction is not None:
        mask |= (
            features["near_feature_fraction"].to_numpy(float) >= gate.partial_min_fraction
        ) & (
            features["distance_c"].to_numpy(float) <= float(gate.partial_max_distance)
        )
    return mask


def apply_needle_gate(
    baseline: Sequence[float],
    features: pd.DataFrame,
    gate: NeedleGate,
) -> tuple[np.ndarray, np.ndarray]:
    """Promote only qualified rows; every non-qualified score stays bit-identical."""
    score = np.asarray(baseline, dtype=float).copy()
    mask = gate_mask(features, gate)
    score[mask] = np.maximum(score[mask], gate.promotion_floor)
    return np.clip(score, 0.0, 1.0), mask


def _aggregate(raw: np.ndarray, rank: np.ndarray) -> dict[str, np.ndarray]:
    p = np.clip(1.0 - rank, 1e-6, 1.0)
    return {
        "crossfit_raw_median": np.median(raw, axis=1),
        "crossfit_raw_mean": np.mean(raw, axis=1),
        "crossfit_raw_max": np.max(raw, axis=1),
        "crossfit_rank_mean": np.mean(rank, axis=1),
        "crossfit_fisher": -np.log(p).mean(axis=1),
    }


class CrossFitNNScorer:
    """Repeated block-NN fits with an independent aggregate null."""

    def __init__(
        self,
        target: pd.DataFrame,
        core: pd.DataFrame,
        meta_calibration: pd.DataFrame,
        rare_detector: Mapping[str, object],
        fold_seeds: Sequence[int] = DEFAULT_FOLD_SEEDS,
        reference_size: int = 1800,
    ) -> None:
        self.target = static_frame(target).reset_index(drop=True)
        self.core = core.reset_index(drop=True)
        self.meta = meta_calibration.reset_index(drop=True)
        core_ids = set(self.core["record_id"])
        meta_ids = set(self.meta["record_id"])
        if core_ids.intersection(meta_ids):
            raise ValueError("meta calibration overlaps crossfit core")
        self.fold_specs = deterministic_fold_indices(self.core, fold_seeds, reference_size)
        self.models: list[LowFPRFeatureModel] = []
        self.null_raw: list[pd.DataFrame] = []
        for spec in self.fold_specs:
            reference = static_frame(self.core.iloc[spec["reference"]])
            calibration = static_frame(self.core.iloc[spec["calibration"]])
            model = LowFPRFeatureModel(self.target, reference, rare_detector)
            self.models.append(model)
            self.null_raw.append(model.raw_features(calibration))
        meta_raw, meta_rank, _ = self._fold_evidence(self.meta)
        self.meta_aggregates = _aggregate(meta_raw, meta_rank)

    @property
    def fold_metadata(self) -> list[dict[str, object]]:
        rows = []
        for spec in self.fold_specs:
            ref = self.core.iloc[spec["reference"]]
            cal = self.core.iloc[spec["calibration"]]
            rows.append({
                "seed": spec["seed"],
                "reference_rows": len(ref),
                "calibration_rows": len(cal),
                "reference_record_id_sha256": record_id_digest(ref),
                "calibration_record_id_sha256": record_id_digest(cal),
            })
        return rows

    def _fold_evidence(self, candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[pd.DataFrame]]:
        static = static_frame(candidates)
        raw_columns, rank_columns, raw_frames = [], [], []
        for model, null in zip(self.models, self.null_raw):
            raw = model.raw_features(static)
            raw_frames.append(raw)
            raw_columns.append(raw["nn_relative"].to_numpy(float))
            rank_columns.append(empirical_percentile(raw["nn_relative"], null["nn_relative"]))
        return np.column_stack(raw_columns), np.column_stack(rank_columns), raw_frames

    def score(self, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw, rank, raw_frames = self._fold_evidence(candidates)
        aggregates = _aggregate(raw, rank)
        scores = pd.DataFrame({
            name: empirical_percentile(value, self.meta_aggregates[name])
            for name, value in aggregates.items()
        })
        features = pd.DataFrame({
            "static_exact": raw_frames[0]["static_exact"].to_numpy(float),
            "near_feature_fraction": np.median(
                np.column_stack([x["near_feature_fraction"] for x in raw_frames]), axis=1
            ),
            "distance_c": np.median(
                np.column_stack([x["distance_c"] for x in raw_frames]), axis=1
            ),
        })
        return scores.clip(0.0, 1.0), features


class FrozenStage1NNScorer:
    """The stage-1 1,800/800 single-split block-NN baseline."""

    def __init__(self, target: pd.DataFrame, core: pd.DataFrame, rare_detector: Mapping[str, object]) -> None:
        if len(core) != 2600:
            raise ValueError("frozen stage-1 core must contain 2,600 rows")
        reference = static_frame(core.iloc[:1800])
        calibration = static_frame(core.iloc[1800:])
        self.model = LowFPRFeatureModel(static_frame(target), reference, rare_detector)
        self.null_raw = self.model.raw_features(calibration)

    def score(self, candidates: pd.DataFrame) -> np.ndarray:
        raw = self.model.raw_features(static_frame(candidates))
        return calibrated_scores(raw, self.null_raw, "smooth")["nn_relative"].to_numpy(float)


def make_block_microaggregate(
    members: pd.DataFrame,
    seed: int,
    group_size: int = 3,
) -> pd.DataFrame:
    """Block-aware microaggregation pseudo defense using member rows only."""
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    rng = np.random.default_rng(seed)
    output = static_frame(members).reset_index(drop=True).copy()
    output[CONTINUOUS] = output[CONTINUOUS].astype(float)
    for _, indices in output.groupby(["sex", "prefecture"], dropna=False).indices.items():
        idx = np.asarray(indices, dtype=int)
        idx = idx[rng.permutation(len(idx))]
        for start in range(0, len(idx), group_size):
            group = idx[start:start + group_size]
            if len(group) == 1 and start > 0:
                group = np.r_[idx[max(0, start - group_size):start], group]
            centroid = output.loc[group, CONTINUOUS].mean().to_numpy(float)
            output.loc[group, CONTINUOUS] = centroid
            mode = output.loc[group, "smoking"].mode(dropna=False)
            if len(mode):
                output.loc[group, "smoking"] = mode.iloc[0]
    output["age"] = output["age"].round().clip(40, 74)
    for col in ["BMI", "SBP", "TG", "HDL", "ALT", "FPG"]:
        output[col] = output[col].round(1)
    return output


def make_stage2_pseudo_target(
    members: pd.DataFrame,
    reference: pd.DataFrame,
    mechanism: str,
    seed: int,
) -> pd.DataFrame:
    """Pseudo target mechanisms used by the preregistered stage-2 benchmark."""
    from mia_low_fpr import make_pseudo_target

    if mechanism == "block_microaggregate":
        return make_block_microaggregate(members, seed=seed, group_size=3)
    return make_pseudo_target(members, reference, mechanism, seed)


def select_with_guardrails(
    metrics: pd.DataFrame,
    baseline_model: str,
    candidate_models: Sequence[str],
    min_mean_gain: float = .005,
    worst_mechanism_tolerance: float = .005,
    auc_tolerance: float = .01,
) -> tuple[str, pd.DataFrame]:
    """Tune-only selection; ties and failed guardrails retain the baseline."""
    per_mechanism = metrics.groupby(["model", "mechanism"], as_index=False).agg(
        mean_tpr=("tpr_at_fpr_0_01", "mean"),
        min_tpr=("tpr_at_fpr_0_01", "min"),
        mean_auc=("auc", "mean"),
    )
    summary = per_mechanism.groupby("model", as_index=False).agg(
        mean_tpr=("mean_tpr", "mean"),
        worst_mechanism_tpr=("mean_tpr", "min"),
        min_run_tpr=("min_tpr", "min"),
        mean_auc=("mean_auc", "mean"),
    )
    base = summary.loc[summary["model"] == baseline_model]
    if len(base) != 1:
        raise ValueError(f"missing unique baseline {baseline_model}")
    base = base.iloc[0]
    eligible = summary[summary["model"].isin(candidate_models)].copy()
    eligible["mean_gain_vs_baseline"] = eligible["mean_tpr"] - base["mean_tpr"]
    eligible["eligible"] = (
        (eligible["mean_gain_vs_baseline"] >= min_mean_gain)
        & (eligible["worst_mechanism_tpr"] >= base["worst_mechanism_tpr"] - worst_mechanism_tolerance)
        & (eligible["mean_auc"] >= base["mean_auc"] - auc_tolerance)
    )
    summary = summary.merge(
        eligible[["model", "mean_gain_vs_baseline", "eligible"]], on="model", how="left"
    )
    summary["mean_gain_vs_baseline"] = summary["mean_gain_vs_baseline"].fillna(0.0)
    summary["eligible"] = summary["eligible"].fillna(False).astype(bool)
    winners = summary[summary["eligible"]].sort_values(
        ["mean_tpr", "worst_mechanism_tpr", "min_run_tpr", "mean_auc", "model"],
        ascending=[False, False, False, False, True],
    )
    winner = baseline_model if winners.empty else str(winners.iloc[0]["model"])
    return winner, summary.sort_values(
        ["mean_tpr", "worst_mechanism_tpr", "mean_auc", "model"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def validate_f_mia_candidate(
    frame: pd.DataFrame,
    record_ids: Sequence[int],
    target_columns: Sequence[str],
) -> None:
    expected_columns = ["record_id", *map(str, target_columns)]
    if frame.shape != (len(record_ids), len(expected_columns)):
        raise ValueError(f"candidate shape {frame.shape} != {(len(record_ids), len(expected_columns))}")
    if list(frame.columns) != expected_columns:
        raise ValueError("candidate columns do not match official target order")
    if frame["record_id"].duplicated().any() or frame["record_id"].tolist() != list(record_ids):
        raise ValueError("record_id order/uniqueness mismatch")
    values = frame.iloc[:, 1:].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("candidate scores must be finite and bounded [0,1]")


def route_public_target(target_id: int, full_target: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the adopted evidence frame; 18/29 deliberately retain full C."""
    target_id = int(target_id)
    return full_target.reset_index(drop=True).copy(), {
        "target": target_id,
        "full_c_used": True,
        "background_candidate_veto_applied": False,
        "filtered_c_score_adopted": False,
    }


def gate_asdict(gate: NeedleGate) -> dict[str, object]:
    return asdict(gate)
