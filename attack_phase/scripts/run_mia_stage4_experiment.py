#!/usr/bin/env python
"""Run Stage-4 categorical-aware classifier MIA diagnostics.

No token, ZIP, or final candidate is read or written.  The runner evaluates a
fixed classifier/gate set on disjoint pseudo tune/test cohorts and writes only
ignored diagnostics under ``attack_phase/outputs``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mia_stage2 import (  # noqa: E402
    FrozenStage1NNScorer,
    assert_evaluation_disjoint,
    make_stage2_pseudo_target,
    partition_background,
    record_id_digest,
)
from mia_stage3 import rank_bounded  # noqa: E402
from mia_stage4 import (  # noqa: E402
    CLASSIFIER_METHODS,
    STAGE4_METHODS,
    CategoricalDensityRatioClassifier,
    compose_methods,
    evaluate_stage4,
    mechanism_summary,
    select_with_guardrails,
    test_only_directional_confirmation,
)


PRIMARY_SEED = 20260828
TUNE_SEEDS = (9101, 9102, 9103)
TEST_SEEDS = (9201, 9202, 9203, 9204, 9205)
MECHANISMS = ("raw_resample", "perturbed_rounded", "block_microaggregate")


def _split_member_control(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = len(pool) // 2
    return pool.iloc[:cut].reset_index(drop=True), pool.iloc[cut:].reset_index(drop=True)


def _run_pool(
    parts: dict[str, pd.DataFrame],
    rare_detector: dict,
    pool_name: str,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    members, controls = _split_member_control(parts[pool_name])
    rows: list[pd.DataFrame] = []
    for mechanism in MECHANISMS:
        for seed in seeds:
            target = make_stage2_pseudo_target(
                members, parts["crossfit_core"], mechanism, int(seed)
            )
            baseline_scorer = FrozenStage1NNScorer(
                target, parts["crossfit_core"], rare_detector
            )
            classifier = CategoricalDensityRatioClassifier(
                target, parts["crossfit_core"], seed=int(seed)
            )
            # meta_calibration is not used in fitting either classifier and is
            # disjoint from all outer evaluation pools.
            member_cls = classifier.calibrated_scores(members, parts["meta_calibration"])
            control_cls = classifier.calibrated_scores(controls, parts["meta_calibration"])
            member_scores = compose_methods(baseline_scorer.score(members), member_cls)
            control_scores = compose_methods(baseline_scorer.score(controls), control_cls)
            metrics = evaluate_stage4(
                member_scores, control_scores, members, controls, rare_detector
            )
            metrics.insert(0, "seed", int(seed))
            metrics.insert(0, "mechanism", mechanism)
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby("model", as_index=False).agg(
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


def run(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.target_dir / "mia_columns.json").read_text(encoding="utf-8"))
    targets = [int(x) for x in config["target_columns"]]
    schema = json.loads((args.team_dir / "schema.json").read_text(encoding="utf-8"))
    background = pd.read_csv(args.team_dir / "A_bg.csv")
    parts = partition_background(background, PRIMARY_SEED)
    assert_evaluation_disjoint(parts)
    rare_detector = schema["rare_detector"]
    partition_meta = {
        name: {"rows": len(frame), "record_id_sha256": record_id_digest(frame)}
        for name, frame in parts.items()
    }
    (output / "partition_metadata.json").write_text(
        json.dumps(partition_meta, indent=2), encoding="utf-8"
    )
    config_meta = {
        "primary_fpr": .001,
        "secondary_fpr": .01,
        "tune_seeds": list(TUNE_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "mechanisms": list(MECHANISMS),
        "classifier_methods": list(CLASSIFIER_METHODS),
        "stage4_methods": list(STAGE4_METHODS),
        "preprocessing": {
            "continuous": ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"],
            "categorical": ["sex", "prefecture", "smoking"],
            "logistic": "l2 C=0.5 balanced liblinear",
            "extratrees": "160 trees depth=12 min_leaf=3 max_features=0.7",
        },
        "guardrails": {
            "min_mean_tpr_001_gain": .005,
            "worst_run_tpr_001_tolerance": .005,
            "mean_auc_tolerance": .01,
            "mean_tpr_01_tolerance": .005,
        },
    }
    (output / "stage4_config.json").write_text(
        json.dumps(config_meta, indent=2), encoding="utf-8"
    )

    tune = _run_pool(parts, rare_detector, "pseudo_tune", TUNE_SEEDS)
    tune.to_csv(output / "pseudo_tune_metrics.csv", index=False)
    mechanism_summary(tune).to_csv(output / "pseudo_tune_mechanism_summary.csv", index=False)
    selected, selection = select_with_guardrails(tune)
    selection.to_csv(output / "pseudo_tune_selection.csv", index=False)

    # Always carry the highest non-baseline diagnostic to an untouched test so
    # a tune failure cannot be hidden by reporting only the baseline.
    challengers = selection.loc[selection.model != "frozen_stage2_nn"]
    diagnostic = "classifier_logistic" if challengers.empty else str(challengers.iloc[0].model)
    test = _run_pool(parts, rare_detector, "pseudo_test", TEST_SEEDS)
    test = test[test.model.isin(["frozen_stage2_nn", diagnostic])].copy()
    test.to_csv(output / "pseudo_test_metrics.csv", index=False)
    mechanism_summary(test).to_csv(output / "pseudo_test_mechanism_summary.csv", index=False)
    test_summary = _summary(test)
    test_summary.to_csv(output / "pseudo_test_summary.csv", index=False)
    confirmed = test_only_directional_confirmation(test_summary, diagnostic)
    adopted = bool(selected != "frozen_stage2_nn" and diagnostic == selected and confirmed)
    decision = {
        "selected_on_tune": selected,
        "diagnostic_test_model": diagnostic,
        "tune_guardrails_passed": bool(selected != "frozen_stage2_nn"),
        "independent_test_confirmation_passed": confirmed,
        "adopted": adopted,
        "final_method": selected if adopted else "frozen_stage2_nn",
        "primary_fpr": .001,
        "secondary_fpr": .01,
        "rare_gate": rare_detector.get("gate", 8.0),
        "no_token_read": True,
        "no_zip_created": True,
        "no_submission_created": True,
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Public rows are diagnostics only; no score is copied into the frozen
    # candidate builder here.
    practice = pd.read_csv(args.team_dir / "B_practice.csv")
    public_rows = []
    for target in targets:
        c = pd.read_csv(args.target_dir / f"C_{target}.csv")
        baseline_scorer = FrozenStage1NNScorer(c, parts["crossfit_core"], rare_detector)
        classifier = CategoricalDensityRatioClassifier(c, parts["crossfit_core"], seed=PRIMARY_SEED)
        cls = classifier.calibrated_scores(practice, parts["meta_calibration"])
        methods = compose_methods(baseline_scorer.score(practice), cls)
        for method, scores in methods.items():
            public_rows.append(pd.DataFrame({
                "record_id": practice.record_id,
                "target": target,
                "method": method,
                "score": np.clip(scores, 0.0, 1.0),
                "rank": rank_bounded(scores),
            }))
    public = pd.concat(public_rows, ignore_index=True)
    public.to_csv(output / "public_rank_diagnostics.csv", index=False)
    (output / "run.json").write_text(json.dumps({
        "rows": int(len(public)),
        "targets": targets,
        "candidate_shape": [int(len(practice)), len(targets) + 1],
        "finite_bounded": bool(np.isfinite(public["score"]).all() and (public["score"].between(0, 1)).all()),
        "selected": selected,
        "adopted": adopted,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected": selected, "diagnostic": diagnostic, "adopted": adopted}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--team-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

