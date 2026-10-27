#!/usr/bin/env python
"""Run stage-2 cross-fit/null-calibrated MIA experiments and build a candidate.

This script never reads a token and never creates a ZIP/submission.  Public C
files are used only after method selection has been completed on pseudo tune.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mia_low_fpr import (  # noqa: E402
    bootstrap_metric,
    evaluate_score_frame,
    grouped_tpr_at_fpr,
    validate_release_targets,
)
from mia_stage2 import (  # noqa: E402
    DEFAULT_FOLD_SEEDS,
    GATE_VARIANTS,
    STAGE2_BASE_MODELS,
    CrossFitNNScorer,
    FrozenStage1NNScorer,
    NeedleGate,
    apply_needle_gate,
    assert_evaluation_disjoint,
    background_overlap_mask,
    gate_asdict,
    gate_mask,
    make_stage2_pseudo_target,
    partition_background,
    record_id_digest,
    route_public_target,
    select_with_guardrails,
    static_frame,
    validate_f_mia_candidate,
    wilson_interval,
)


PRIMARY_SEED = 20260828
TUNE_SEEDS = (4101, 4102, 4103)
TEST_SEEDS = (5101, 5102, 5103, 5104, 5105)
SPLIT_SEEDS = (20260828, 20260829, 20260830, 20260831, 20260832)
MECHANISMS = ("raw_resample", "perturbed_rounded", "block_microaggregate")
MIN_MEAN_GAIN = .005
WORST_MECHANISM_TOLERANCE = .005
AUC_TOLERANCE = .01


def _split_member_control(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = len(pool) // 2
    return pool.iloc[:cut].reset_index(drop=True), pool.iloc[cut:].reset_index(drop=True)


def _gate_by_name(name: str) -> NeedleGate:
    matches = [gate for gate in GATE_VARIANTS if gate.name == name]
    if len(matches) != 1:
        raise ValueError(name)
    return matches[0]


def _split_method(method: str) -> tuple[str, NeedleGate]:
    if "__" not in method:
        return method, _gate_by_name("no_gate")
    base, gate_name = method.split("__", 1)
    return base, _gate_by_name(gate_name)


def _score_models(
    target: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
    rare_detector: dict,
    candidates: pd.DataFrame,
    include_all_gates: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, CrossFitNNScorer]:
    frozen = FrozenStage1NNScorer(target, parts["crossfit_core"], rare_detector)
    crossfit = CrossFitNNScorer(
        target,
        parts["crossfit_core"],
        parts["meta_calibration"],
        rare_detector,
        fold_seeds=DEFAULT_FOLD_SEEDS,
    )
    cf_scores, features = crossfit.score(candidates)
    result = pd.DataFrame({"frozen_stage1_nn": frozen.score(candidates)})
    for model in STAGE2_BASE_MODELS:
        result[model] = cf_scores[model]
        if include_all_gates:
            for gate in GATE_VARIANTS:
                if gate.name == "no_gate":
                    continue
                gated, _ = apply_needle_gate(cf_scores[model], features, gate)
                result[f"{model}__{gate.name}"] = gated
    return result, features, crossfit


def _score_selected(
    scorer: CrossFitNNScorer,
    candidates: pd.DataFrame,
    method: str,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    base, gate = _split_method(method)
    scores, features = scorer.score(candidates)
    if base not in scores:
        raise ValueError(f"unknown stage2 base: {base}")
    selected, mask = apply_needle_gate(scores[base], features, gate)
    return selected, features, mask


def _pseudo_runs(
    parts: dict[str, pd.DataFrame],
    rare_detector: dict,
    pool_name: str,
    seeds: tuple[int, ...],
    include_all_gates: bool,
) -> pd.DataFrame:
    members, controls = _split_member_control(parts[pool_name])
    rows = []
    for mechanism in MECHANISMS:
        for seed in seeds:
            target = make_stage2_pseudo_target(members, parts["crossfit_core"], mechanism, seed)
            member_scores, member_features, scorer = _score_models(
                target, parts, rare_detector, members, include_all_gates
            )
            frozen = FrozenStage1NNScorer(target, parts["crossfit_core"], rare_detector)
            control_cf, control_features = scorer.score(controls)
            control_scores = pd.DataFrame({"frozen_stage1_nn": frozen.score(controls)})
            for model in STAGE2_BASE_MODELS:
                control_scores[model] = control_cf[model]
                if include_all_gates:
                    for gate in GATE_VARIANTS:
                        if gate.name == "no_gate":
                            continue
                        gated, _ = apply_needle_gate(control_cf[model], control_features, gate)
                        control_scores[f"{model}__{gate.name}"] = gated
            metrics = evaluate_score_frame(member_scores, control_scores)
            metrics.insert(0, "seed", seed)
            metrics.insert(0, "mechanism", mechanism)
            for model in metrics["model"]:
                base, gate = _split_method(model) if model != "frozen_stage1_nn" else (model, _gate_by_name("no_gate"))
                if model == "frozen_stage1_nn" or gate.name == "no_gate":
                    member_hits = control_hits = 0
                else:
                    member_hits = int(gate_mask(member_features, gate).sum())
                    control_hits = int(gate_mask(control_features, gate).sum())
                mask = metrics["model"] == model
                metrics.loc[mask, "gate_member_hits"] = member_hits
                metrics.loc[mask, "gate_control_hits"] = control_hits
                metrics.loc[mask, "base_model"] = base
                metrics.loc[mask, "gate"] = gate.name
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def _mechanism_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["mechanism", "model"], as_index=False).agg(
        mean_tpr=("tpr_at_fpr_0_01", "mean"),
        min_tpr=("tpr_at_fpr_0_01", "min"),
        mean_auc=("auc", "mean"),
        sd_tpr=("tpr_at_fpr_0_01", "std"),
        runs=("seed", "count"),
        mean_member_gate_hits=("gate_member_hits", "mean"),
        mean_control_gate_hits=("gate_control_hits", "mean"),
    )


def _confirm_test(
    test: pd.DataFrame,
    selected_stage2: str,
    tune_adopted: bool,
) -> tuple[bool, dict[str, float | bool | str]]:
    summary = test.groupby("model", as_index=False).agg(
        mean_tpr=("tpr_at_fpr_0_01", "mean"),
        min_tpr=("tpr_at_fpr_0_01", "min"),
        mean_auc=("auc", "mean"),
    ).set_index("model")
    base = summary.loc["frozen_stage1_nn"]
    candidate = summary.loc[selected_stage2]
    directional = bool(
        candidate["mean_tpr"] > base["mean_tpr"]
        and candidate["min_tpr"] >= base["min_tpr"] - WORST_MECHANISM_TOLERANCE
    )
    adopted = bool(tune_adopted and directional)
    return adopted, {
        "selected_stage2": selected_stage2,
        "tune_guardrails_passed": bool(tune_adopted),
        "test_directional_confirmation_passed": directional,
        "stage2_test_mean_tpr": float(candidate["mean_tpr"]),
        "frozen_test_mean_tpr": float(base["mean_tpr"]),
        "stage2_test_min_tpr": float(candidate["min_tpr"]),
        "frozen_test_min_tpr": float(base["min_tpr"]),
        "stage2_test_mean_auc": float(candidate["mean_auc"]),
        "frozen_test_mean_auc": float(base["mean_auc"]),
        "final_method": selected_stage2 if adopted else "frozen_stage1_nn",
    }


def _team10_scores(
    target: pd.DataFrame,
    practice: pd.DataFrame,
    control: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
    rare_detector: dict,
    selected_stage2: str,
) -> tuple[pd.DataFrame, dict[str, dict], pd.DataFrame]:
    frozen = FrozenStage1NNScorer(target, parts["crossfit_core"], rare_detector)
    scorer = CrossFitNNScorer(target, parts["crossfit_core"], parts["meta_calibration"], rare_detector)
    member_selected, member_features, member_gate = _score_selected(scorer, practice, selected_stage2)
    control_selected, control_features, control_gate = _score_selected(scorer, control, selected_stage2)
    member = pd.DataFrame({
        "frozen_stage1_nn": frozen.score(practice),
        selected_stage2: member_selected,
    })
    negative = pd.DataFrame({
        "frozen_stage1_nn": frozen.score(control),
        selected_stage2: control_selected,
    })
    if selected_stage2 == "frozen_stage1_nn":
        member = member[["frozen_stage1_nn"]]
        negative = negative[["frozen_stage1_nn"]]
    metrics = evaluate_score_frame(member, negative)
    truth = np.r_[np.ones(len(member), int), np.zeros(len(negative), int)]
    boot = {}
    for col in member:
        values = np.r_[member[col].to_numpy(float), negative[col].to_numpy(float)]
        boot[col] = bootstrap_metric(values, truth, n_boot=400, seed=PRIMARY_SEED + sum(map(ord, col)))
    gates = pd.DataFrame([
        {"cohort": "B_practice", "rows": len(member_features), "gate_hits": int(member_gate.sum())},
        {"cohort": "A_bg_proxy_control", "rows": len(control_features), "gate_hits": int(control_gate.sum())},
    ])
    return metrics, boot, gates


def _split_stability(
    background: pd.DataFrame,
    c10: pd.DataFrame,
    practice: pd.DataFrame,
    rare_detector: dict,
    selected_stage2: str,
) -> pd.DataFrame:
    rows = []
    for split_seed in SPLIT_SEEDS:
        parts = partition_background(background, split_seed)
        assert_evaluation_disjoint(parts)
        control = parts["proxy_control"]
        frozen = FrozenStage1NNScorer(c10, parts["crossfit_core"], rare_detector)
        scorer = CrossFitNNScorer(c10, parts["crossfit_core"], parts["meta_calibration"], rare_detector)
        selected_member, _, _ = _score_selected(scorer, practice, selected_stage2)
        selected_control, _, _ = _score_selected(scorer, control, selected_stage2)
        candidates = {
            "frozen_stage1_nn": (frozen.score(practice), frozen.score(control)),
            selected_stage2: (selected_member, selected_control),
        }
        for model, (member_score, control_score) in candidates.items():
            truth = np.r_[np.ones(len(member_score), int), np.zeros(len(control_score), int)]
            tpr, auc = grouped_tpr_at_fpr(np.r_[member_score, control_score], truth)
            hits = int(np.sum(control_score >= .99))
            lo, hi = wilson_interval(hits, len(control_score))
            rows.append({
                "split_seed": split_seed,
                "model": model,
                "tpr_at_fpr_0_01": tpr,
                "auc": auc,
                "null_rows": len(control_score),
                "null_hits_ge_0_99": hits,
                "null_rate_ge_0_99": hits / len(control_score),
                "null_rate_wilson95_low": lo,
                "null_rate_wilson95_high": hi,
            })
    return pd.DataFrame(rows).drop_duplicates(["split_seed", "model"])


def _public_outputs(
    target_dir: Path,
    targets: list[str],
    practice: pd.DataFrame,
    background: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
    rare_detector: dict,
    selected_stage2: str,
    final_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    wide = pd.DataFrame({"record_id": practice["record_id"].tolist()})
    long_rows, needle_rows, overlap_rows, fold_metadata = [], [], [], []
    null = parts["proxy_control"]
    for target_name in targets:
        target_id = int(target_name)
        c_loaded = pd.read_csv(target_dir / f"C_{target_name}.csv")
        c_full, routing = route_public_target(target_id, c_loaded)
        # Stage-1 decision is binding: target 18/29 use the full C.  Background
        # overlap remains a negative/control diagnostic and never a candidate veto.
        frozen = FrozenStage1NNScorer(c_full, parts["crossfit_core"], rare_detector)
        scorer = CrossFitNNScorer(c_full, parts["crossfit_core"], parts["meta_calibration"], rare_detector)
        selected_score, features, selected_mask = _score_selected(scorer, practice, selected_stage2)
        frozen_score = frozen.score(practice)
        final_score = frozen_score if final_method == "frozen_stage1_nn" else selected_score
        wide[target_name] = final_score
        for i, record_id in enumerate(practice["record_id"]):
            long_rows.append({
                "record_id": record_id,
                "target": target_id,
                "frozen_stage1_nn": float(frozen_score[i]),
                "selected_stage2": float(selected_score[i]),
                "final_score": float(final_score[i]),
                "selected_gate_hit": bool(selected_mask[i]),
                "full_c_used": bool(routing["full_c_used"]),
            })
        if target_id in (6, 19):
            _, candidate_features = scorer.score(practice)
            _, null_features = scorer.score(null)
            for gate in GATE_VARIANTS:
                if gate.name == "no_gate":
                    continue
                candidate_hits = int(gate_mask(candidate_features, gate).sum())
                null_hits = int(gate_mask(null_features, gate).sum())
                null_rate = null_hits / len(null_features)
                needle_rows.append({
                    "target": target_id,
                    "gate": gate.name,
                    "candidate_rows": len(candidate_features),
                    "candidate_hits": candidate_hits,
                    "candidate_hit_rate": candidate_hits / len(candidate_features),
                    "heldout_null_rows": len(null_features),
                    "heldout_null_hits": null_hits,
                    "heldout_null_hit_rate": null_rate,
                    "null_expected_candidate_hits": null_rate * len(candidate_features),
                    "interpretation": "unlabeled diagnostic; not membership truth",
                })
        if target_id in (18, 29):
            static_overlap = background_overlap_mask(static_frame(c_full), static_frame(background))
            overlap_rows.append({
                "target": target_id,
                "c_rows": len(c_full),
                "background_static_overlap_rows": int(static_overlap.sum()),
                "background_static_overlap_rate": float(static_overlap.mean()),
                "adopted_c_evidence": "full_C",
                "candidate_veto_applied": False,
                "filtered_c_score_adopted": False,
            })
        fold_metadata.append({"target": target_id, "folds": scorer.fold_metadata})
    validate_f_mia_candidate(wide, practice["record_id"].tolist(), targets)
    return (
        wide,
        pd.DataFrame(long_rows),
        pd.DataFrame(needle_rows),
        pd.DataFrame(overlap_rows),
        fold_metadata,
    )


def run(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    target_config = json.loads((args.target_dir / "mia_columns.json").read_text(encoding="utf-8"))
    targets = [str(x) for x in target_config["target_columns"]]
    validate_release_targets(args.target_dir, [int(x) for x in targets])
    schema = json.loads((args.team_dir / "schema.json").read_text(encoding="utf-8"))
    rare_detector = schema["rare_detector"]
    background = pd.read_csv(args.team_dir / "A_bg.csv")
    practice = pd.read_csv(args.team_dir / "B_practice.csv")
    parts = partition_background(background, args.seed)
    assert_evaluation_disjoint(parts)

    partition_meta = {
        name: {"rows": len(frame), "record_id_sha256": record_id_digest(frame)}
        for name, frame in parts.items()
    }
    (output / "partition_metadata.json").write_text(
        json.dumps(partition_meta, indent=2), encoding="utf-8"
    )

    tune = _pseudo_runs(parts, rare_detector, "pseudo_tune", TUNE_SEEDS, include_all_gates=True)
    tune.to_csv(output / "pseudo_tune_metrics.csv", index=False)
    tune_mechanism = _mechanism_summary(tune)
    tune_mechanism.to_csv(output / "pseudo_tune_mechanism_summary.csv", index=False)

    base_metrics = tune[tune["model"].isin(["frozen_stage1_nn", *STAGE2_BASE_MODELS])]
    selected_base, base_selection = select_with_guardrails(
        base_metrics,
        "frozen_stage1_nn",
        STAGE2_BASE_MODELS,
        MIN_MEAN_GAIN,
        WORST_MECHANISM_TOLERANCE,
        AUC_TOLERANCE,
    )
    base_selection.to_csv(output / "pseudo_tune_base_selection.csv", index=False)
    if selected_base == "frozen_stage1_nn":
        # No challenger cleared the adoption guardrails.  For one diagnostic
        # confirmatory read, carry forward the highest-ranked cross-fit model
        # from tune rather than choosing an arbitrary aggregation.
        selected_stage2 = str(
            base_selection.loc[base_selection["model"] != "frozen_stage1_nn", "model"].iloc[0]
        )
        tune_guardrails_passed = False
        gate_selection = pd.DataFrame()
    else:
        gate_candidates = [
            f"{selected_base}__{gate.name}" for gate in GATE_VARIANTS if gate.name != "no_gate"
        ]
        gate_metrics = tune[tune["model"].isin([selected_base, *gate_candidates])].copy()
        selected_stage2, gate_selection = select_with_guardrails(
            gate_metrics,
            selected_base,
            gate_candidates,
            MIN_MEAN_GAIN,
            WORST_MECHANISM_TOLERANCE,
            AUC_TOLERANCE,
        )
        tune_guardrails_passed = True
    gate_selection.to_csv(output / "pseudo_tune_gate_selection.csv", index=False)

    test_all = _pseudo_runs(parts, rare_detector, "pseudo_test", TEST_SEEDS, include_all_gates=True)
    test = test_all[test_all["model"].isin(["frozen_stage1_nn", selected_stage2])].copy()
    test.to_csv(output / "pseudo_test_metrics.csv", index=False)
    _mechanism_summary(test).to_csv(output / "pseudo_test_mechanism_summary.csv", index=False)
    stage2_adopted, confirmation = _confirm_test(test, selected_stage2, tune_guardrails_passed)
    final_method = str(confirmation["final_method"])

    c10 = pd.read_csv(args.target_dir / "C_10.csv")
    proxy_metrics, proxy_bootstrap, proxy_gate = _team10_scores(
        c10, practice, parts["proxy_control"], parts, rare_detector, selected_stage2
    )
    proxy_metrics.to_csv(output / "team10_proxy_metrics.csv", index=False)
    proxy_gate.to_csv(output / "team10_proxy_gate_hits.csv", index=False)
    (output / "team10_proxy_bootstrap.json").write_text(
        json.dumps(proxy_bootstrap, indent=2), encoding="utf-8"
    )

    stability = _split_stability(background, c10, practice, rare_detector, selected_stage2)
    stability.to_csv(output / "split_seed_stability.csv", index=False)
    stability_summary = stability.groupby("model", as_index=False).agg(
        mean_tpr=("tpr_at_fpr_0_01", "mean"),
        min_tpr=("tpr_at_fpr_0_01", "min"),
        max_tpr=("tpr_at_fpr_0_01", "max"),
        mean_auc=("auc", "mean"),
        mean_null_rate=("null_rate_ge_0_99", "mean"),
        min_null_rate=("null_rate_ge_0_99", "min"),
        max_null_rate=("null_rate_ge_0_99", "max"),
        min_wilson_low=("null_rate_wilson95_low", "min"),
        max_wilson_high=("null_rate_wilson95_high", "max"),
        split_seeds=("split_seed", "count"),
    )
    stability_summary.to_csv(output / "split_seed_stability_summary.csv", index=False)

    wide, public_long, needle_diag, overlap_diag, fold_meta = _public_outputs(
        args.target_dir,
        targets,
        practice,
        background,
        parts,
        rare_detector,
        selected_stage2,
        final_method,
    )
    wide.to_csv(output / "F_mia_candidate.csv", index=False)
    candidate_sha256 = hashlib.sha256((output / "F_mia_candidate.csv").read_bytes()).hexdigest()
    public_long.to_csv(output / "public_candidate_long.csv", index=False)
    needle_diag.to_csv(output / "public_needle_diagnostics_targets6_19.csv", index=False)
    overlap_diag.to_csv(output / "public_background_control_targets18_29.csv", index=False)
    (output / "public_fold_metadata.json").write_text(json.dumps(fold_meta, indent=2), encoding="utf-8")

    run_meta = {
        "run_id": output.name,
        "seed": args.seed,
        "target_columns": targets,
        "target_count": len(targets),
        "fold_seeds": list(DEFAULT_FOLD_SEEDS),
        "split_stability_seeds": list(SPLIT_SEEDS),
        "pseudo_design": {
            "mechanisms": list(MECHANISMS),
            "tune_seeds": list(TUNE_SEEDS),
            "test_seeds": list(TEST_SEEDS),
            "test_used_for_model_selection": False,
        },
        "selection_rule": {
            "sequence": "select crossfit aggregate on pseudo tune; test gate variants on that base; independent test is directional veto only",
            "min_mean_tpr_gain": MIN_MEAN_GAIN,
            "worst_mechanism_tolerance": WORST_MECHANISM_TOLERANCE,
            "mean_auc_tolerance": AUC_TOLERANCE,
            "team10_or_public_used_for_tuning": False,
        },
        "selected_base_on_tune": selected_base,
        "selected_stage2_on_tune": selected_stage2,
        "stage2_adopted": stage2_adopted,
        "confirmation": confirmation,
        "final_candidate_method": final_method,
        "gate_variants": [gate_asdict(gate) for gate in GATE_VARIANTS],
        "full_c_routing": {"18": True, "29": True},
        "background_candidate_veto_applied": False,
        "filtered_c_score_adopted": False,
        "candidate_validation": {
            "rows": len(wide),
            "columns": len(wide.columns),
            "unique_record_ids": int(wide["record_id"].nunique()),
            "finite_bounded": True,
            "sha256": candidate_sha256,
        },
        "team10_role": "confirmation-only source-cohort proxy; not official membership truth",
        "public_role": "unlabeled score generation and false-alarm/needle diagnostics only",
        "no_token_read": True,
        "no_zip_created": True,
        "no_submission_created": True,
    }
    (output / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "selected_stage2": selected_stage2,
        "stage2_adopted": stage2_adopted,
        "final_method": final_method,
        "candidate_shape": list(wide.shape),
    }, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--team-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
