"""Build the bounded S4 nearest-decoy MIA + multiplex AIA probe.

This script performs the local H1a/H1b sanity check, freezes one formula
before public scoring, creates one deterministic AIA assignment, and writes
candidate CSVs for the target router.  It never reads the Team 10 token and
never creates a ZIP or submits to Codabench.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "attack_phase/src"))

from aia_stage1 import AIAEngine, validate_f_aia_candidate  # noqa: E402
from aia_stage3 import AIAStage3Engine, Stage3Config  # noqa: E402
from aia_stage5 import AIAStage5Engine, Stage5Config  # noqa: E402
from aia_stage6 import AIAStage6Engine, Stage6Config  # noqa: E402
from mia_low_fpr import CONTINUOUS, grouped_tpr_at_fpr  # noqa: E402
from mia_stage2 import FrozenStage1NNScorer, make_stage2_pseudo_target, partition_background  # noqa: E402
from mia_stage9_skip1 import FORMULAS, Skip1Scorer  # noqa: E402


TARGETS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 24, 26, 29)
MIA_PRIMARY_SEED = 20260907
MIA_TUNE_SEEDS = (10301, 10302, 10303)
MIA_TEST_SEEDS = (10401, 10402, 10403, 10404, 10405)
MIA_MECHANISMS = ("c_only_nearest_decoy", "raw_resample", "perturbed_rounded", "block_microaggregate")
MIA_C_ONLY_MIN_GAIN = 0.02
MIA_ORDINARY_WORST_TOLERANCE = 0.10
FALLBACK_RESTORE_TARGETS = (3, 5, 7, 9, 13, 18, 20, 23, 26, 29)
ASSIGNMENT_SEED = 20260907
TESTED_AIA_EXPERTS = ("official_1nn", "gated_abstain_donor_modal", "cell_modal_aggregate")

AIA_EXPERT_POOL = (
    "any_event_cox_window_mode", "group_mean_corrected_donor", "weighted_knn",
    "weighted_modal_window", "weighted_modal_k25", "membership_gated",
    "stage3_l1_l1", "stage3_huber_huber", "stage3_l1_l2",
)

# Each first tier is a fingerprint-prioritized tie set.  The fixed RNG only
# chooses within that tier; no server output participates in this assignment.
AIA_PRIORITY_TIERS = {
    1: (("any_event_cox_window_mode", "stage3_l1_l2"),),
    2: (("weighted_knn", "stage3_l1_l1"),),
    3: (("weighted_modal_window", "weighted_modal_k25"),),
    4: (("stage3_l1_l1", "weighted_modal_window"),),
    5: (("stage3_huber_huber", "weighted_modal_k25"),),
    6: (("group_mean_corrected_donor", "stage3_l1_l2"),),
    7: (("stage3_huber_huber", "weighted_modal_window"),),
    8: (("membership_gated", "stage3_l1_l1"),),
    9: (("weighted_modal_k25", "stage3_l1_l1"),),
    10: (("weighted_knn", "stage3_l1_l2"),),
    12: (("group_mean_corrected_donor", "stage3_l1_l2"),),
    13: (("weighted_modal_window", "stage3_l1_l1"),),
    17: (("any_event_cox_window_mode", "stage3_l1_l2"),),
    18: (("weighted_knn", "membership_gated"),),
    19: (("stage3_l1_l2", "weighted_knn"),),
    20: (("weighted_modal_k25", "stage3_l1_l1"),),
    22: (("membership_gated", "stage3_huber_huber"),),
    23: (("weighted_knn", "weighted_modal_window"),),
    24: (("group_mean_corrected_donor", "stage3_huber_huber"),),
    26: (("any_event_cox_window_mode", "stage3_l1_l1"),),
    29: (("membership_gated", "weighted_knn"),),
}


def _split(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = len(pool) // 2
    return pool.iloc[:cut].reset_index(drop=True), pool.iloc[cut:].reset_index(drop=True)


def make_c_only_nearest_decoy(
    members: pd.DataFrame, controls: pd.DataFrame, seed: int, fraction: float = 0.5
) -> pd.DataFrame:
    """Create a C-only closest-shell nuisance around held-out controls.

    The member part is an ordinary perturbed release.  Additional rows near a
    deterministic control subset create the first shell without changing the
    member/control truth used for evaluation.  The mechanism is intentionally
    local and is never claimed to reproduce the hidden defense.
    """
    rng = np.random.default_rng(seed)
    base = make_stage2_pseudo_target(members, members, "perturbed_rounded", seed)
    count = max(1, min(len(controls), int(round(len(members) * fraction))))
    chosen = controls.iloc[rng.permutation(len(controls))[:count]].reset_index(drop=True).copy()
    scales = members[list(CONTINUOUS)].std(ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(float)
    noise = rng.normal(0.0, 0.001, size=(count, len(CONTINUOUS))) * scales
    continuous = chosen[list(CONTINUOUS)].to_numpy(float) + noise
    chosen = chosen.drop(columns=list(CONTINUOUS))
    for index, column in enumerate(CONTINUOUS):
        chosen[column] = continuous[:, index]
    chosen["record_id"] = -1 - np.arange(count, dtype=int)
    return pd.concat([base, chosen], ignore_index=True, sort=False)


def _mia_metrics(
    target: pd.DataFrame,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    core: pd.DataFrame,
    formula: str,
    rare_detector: dict,
    seed: int,
    mechanism: str,
) -> dict[str, object]:
    reference = core.iloc[:1800].reset_index(drop=True)
    calibration = core.iloc[1800:].reset_index(drop=True)
    baseline = FrozenStage1NNScorer(target, core, rare_detector)
    challenger = Skip1Scorer(target, reference, calibration, rare_detector, formula)
    member_base = baseline.score(members)
    control_base = baseline.score(controls)
    member_new = challenger.score(members)
    control_new = challenger.score(controls)
    truth = np.r_[np.ones(len(members), dtype=int), np.zeros(len(controls), dtype=int)]
    base_tpr, base_auc = grouped_tpr_at_fpr(np.r_[member_base, control_base], truth)
    new_tpr, new_auc = grouped_tpr_at_fpr(np.r_[member_new, control_new], truth)
    return {
        "seed": int(seed), "mechanism": mechanism, "formula": formula,
        "baseline_tpr_at_fpr_0_01": float(base_tpr), "skip1_tpr_at_fpr_0_01": float(new_tpr),
        "baseline_auc": float(base_auc), "skip1_auc": float(new_auc),
        "delta_tpr": float(new_tpr - base_tpr), "delta_auc": float(new_auc - base_auc),
        "c_shell_fallback_rate": float(challenger.diagnostics(members)["c_shell_fallback_count"].gt(0).mean()),
        "a_shell_fallback_rate": float(challenger.diagnostics(members)["a_shell_fallback_count"].gt(0).mean()),
    }


def _run_mia_pool(
    background: pd.DataFrame,
    practice: pd.DataFrame,
    rare_detector: dict,
    seeds: tuple[int, ...],
    pool_name: str,
) -> pd.DataFrame:
    parts = partition_background(background, MIA_PRIMARY_SEED)
    members, controls = _split(parts[pool_name])
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for mechanism in MIA_MECHANISMS:
            if mechanism == "c_only_nearest_decoy":
                target = make_c_only_nearest_decoy(members, controls, seed)
            else:
                target = make_stage2_pseudo_target(members, parts["crossfit_core"], mechanism, seed)
            for formula in FORMULAS:
                rows.append(_mia_metrics(target, members, controls, parts["crossfit_core"], formula, rare_detector, seed, mechanism))
    return pd.DataFrame(rows)


def _select_mia(tune: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    summary = tune.groupby(["formula", "mechanism"], as_index=False).agg(
        mean_delta_tpr=("delta_tpr", "mean"), worst_delta_tpr=("delta_tpr", "min"),
        mean_delta_auc=("delta_auc", "mean"), runs=("seed", "count"),
    )
    records: list[dict[str, object]] = []
    for formula in FORMULAS:
        c_only = summary[(summary.formula == formula) & (summary.mechanism == "c_only_nearest_decoy")].iloc[0]
        ordinary = summary[(summary.formula == formula) & (summary.mechanism != "c_only_nearest_decoy")]
        record = {
            "formula": formula,
            "c_only_mean_delta_tpr": float(c_only.mean_delta_tpr),
            "ordinary_mean_delta_tpr": float(ordinary.mean_delta_tpr.mean()),
            "ordinary_worst_delta_tpr": float(ordinary.worst_delta_tpr.min()),
            "ordinary_mean_delta_auc": float(ordinary.mean_delta_auc.mean()),
            "c_only_gate": bool(c_only.mean_delta_tpr >= MIA_C_ONLY_MIN_GAIN),
            "ordinary_gate": bool(ordinary.worst_delta_tpr.min() >= -MIA_ORDINARY_WORST_TOLERANCE),
        }
        record["eligible"] = bool(record["c_only_gate"] and record["ordinary_gate"])
        records.append(record)
    decision = pd.DataFrame(records)
    eligible = decision[decision.eligible]
    if len(eligible) and bool((eligible.formula == "h1a_target_only").any()):
        selected = "h1a_target_only"
    elif len(eligible):
        selected = "h1b_bilateral"
    else:
        selected = "none"
    decision["selected_formula"] = selected
    decision["c_only_min_gain"] = MIA_C_ONLY_MIN_GAIN
    decision["ordinary_worst_tolerance"] = MIA_ORDINARY_WORST_TOLERANCE
    return selected, decision


def _crowding_score(candidates: pd.DataFrame, target: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    """Fixed-radius reverse-mass diagnostic, separate from smooth density ratios."""
    columns = list(CONTINUOUS)
    std = reference[columns].std(ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(float)
    query = candidates[columns].to_numpy(float) / std
    target_z = target[columns].to_numpy(float) / std
    reference_z = reference[columns].to_numpy(float) / std
    radius = 0.25
    target_mass = (cdist(query, target_z) <= radius).sum(axis=1) / max(len(target), 1)
    reference_mass = (cdist(query, reference_z) <= radius).sum(axis=1) / max(len(reference), 1)
    return np.log(target_mass + 1e-6) - np.log(reference_mass + 1e-6)


def _crowding_sanity(members: pd.DataFrame, controls: pd.DataFrame, background: pd.DataFrame, seed: int = 20260907) -> dict[str, object]:
    """Check member-crowd versus nonmember-crowd on a bounded held-out fixture."""
    rng = np.random.default_rng(seed)
    member_rows = members.iloc[:200].reset_index(drop=True)
    control_rows = controls.iloc[:200].reset_index(drop=True)
    scales = member_rows[list(CONTINUOUS)].std(ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(float)
    def add_crowd(base: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
        selected = anchors.iloc[rng.permutation(len(anchors))[:100]].reset_index(drop=True).copy()
        values = selected[list(CONTINUOUS)].to_numpy(float) + rng.normal(0.0, 0.01, (len(selected), len(CONTINUOUS))) * scales
        selected = selected.drop(columns=list(CONTINUOUS))
        for index, column in enumerate(CONTINUOUS):
            selected[column] = values[:, index]
        return pd.concat([base, selected], ignore_index=True, sort=False)
    truth = np.r_[np.ones(len(member_rows), dtype=int), np.zeros(len(control_rows), dtype=int)]
    member_case = _crowding_score(pd.concat([member_rows, control_rows], ignore_index=True), add_crowd(member_rows, member_rows), background)
    control_case = _crowding_score(pd.concat([member_rows, control_rows], ignore_index=True), add_crowd(member_rows, control_rows), background)
    member_tpr, member_auc = grouped_tpr_at_fpr(member_case, truth)
    control_tpr, control_auc = grouped_tpr_at_fpr(control_case, truth)
    passed = bool(member_auc >= 0.60 and member_tpr > 0.0 and control_tpr <= 0.10)
    return {
        "radius": 0.25, "member_crowd_tpr_at_fpr_0_01": float(member_tpr),
        "member_crowd_auc": float(member_auc), "nonmember_crowd_tpr_at_fpr_0_01": float(control_tpr),
        "nonmember_crowd_auc": float(control_auc), "passed": passed,
        "decision": "candidate_eligible" if passed else "diagnostic_failed_negative_control",
    }


def make_aia_assignment(targets: tuple[int, ...] = TARGETS) -> dict[int, str]:
    rng = np.random.default_rng(ASSIGNMENT_SEED)
    assignment: dict[int, str] = {}
    for target in targets:
        tiers = AIA_PRIORITY_TIERS.get(target, (AIA_EXPERT_POOL,))
        for tier in tiers:
            choices = sorted(set(tier).intersection(AIA_EXPERT_POOL).difference(TESTED_AIA_EXPERTS))
            if choices:
                assignment[target] = str(choices[int(rng.integers(0, len(choices)))])
                break
        if target not in assignment:
            raise ValueError(f"no untested AIA expert available for target {target}")
    return assignment


def _aia_predict(target: pd.DataFrame, challenge: pd.DataFrame, background: pd.DataFrame, expert: str) -> np.ndarray:
    if expert in {"weighted_knn", "weighted_modal_window", "weighted_modal_k25", "membership_gated"}:
        return AIAEngine(target, background).predict(challenge, expert)[0]
    if expert == "group_mean_corrected_donor":
        return AIAStage5Engine(target, background, Stage5Config(expert)).predict(challenge)[0]
    if expert == "any_event_cox_window_mode":
        return AIAStage6Engine(target, background, Stage6Config(expert)).predict(challenge)[0]
    configs = {
        "stage3_l1_l1": Stage3Config("l1", "l1"),
        "stage3_huber_huber": Stage3Config("huber_clip15", "huber_clip15"),
        "stage3_l1_l2": Stage3Config("l1", "l2"),
    }
    if expert not in configs:
        raise ValueError(f"unknown AIA expert: {expert}")
    return AIAStage3Engine(target, background, configs[expert]).predict(
        challenge, "gated_abstain_donor_modal"
    )[0]


def _build_public_candidates(
    target_dir: Path,
    team_dir: Path,
    base_mia_path: Path,
    output: Path,
    selected_formula: str,
    mia_override: pd.DataFrame | None = None,
) -> dict[str, object]:
    target_config = json.loads((target_dir / "mia_columns.json").read_text(encoding="utf-8"))
    targets = tuple(int(x) for x in target_config["target_columns"])
    background = pd.read_csv(team_dir / "A_bg.csv")
    practice = pd.read_csv(team_dir / "B_practice.csv")
    schema = json.loads((team_dir / "schema.json").read_text(encoding="utf-8"))
    parts = partition_background(background, MIA_PRIMARY_SEED)
    ref = parts["crossfit_core"].iloc[:1800].reset_index(drop=True)
    cal = parts["crossfit_core"].iloc[1800:].reset_index(drop=True)
    base = pd.read_csv(base_mia_path)
    if list(base["record_id"]) != list(practice["record_id"]):
        raise ValueError("frozen MIA source record_id order differs from B_practice")
    mia = pd.DataFrame({"record_id": base["record_id"]})
    for target in targets:
        if mia_override is not None:
            mia[str(target)] = mia_override[str(target)].to_numpy()
        elif target in (1, 2):
            mia[str(target)] = base[str(target)].to_numpy()
        else:
            c = pd.read_csv(target_dir / f"C_{target}.csv")
            mia[str(target)] = Skip1Scorer(c, ref, cal, schema["rare_detector"], selected_formula).score(practice)
    mia_path = output / "F_mia_skip1_selected.csv"
    mia.to_csv(mia_path, index=False)

    assignment = make_aia_assignment(targets)
    aia_rows: list[dict[str, object]] = []
    for target in targets:
        c = pd.read_csv(target_dir / f"C_{target}.csv")
        challenge = pd.read_csv(target_dir / f"aia_challenge_{target}.csv")
        expert = assignment[target]
        values = _aia_predict(c, challenge, background, expert)
        for row_id, value in zip(challenge["challenge_row_id"], values):
            aia_rows.append({"target_k": target, "challenge_row_id": int(row_id), "time_hat": float(value)})
    aia = pd.DataFrame(aia_rows)
    challenge_ids = {
        target: [int(x) for x in pd.read_csv(target_dir / f"aia_challenge_{target}.csv")["challenge_row_id"]]
        for target in targets
    }
    validate_f_aia_candidate(aia, challenge_ids)
    aia_path = output / "F_aia_multiplex.csv"
    aia.to_csv(aia_path, index=False)
    assignment_payload = {
        "assignment_seed": ASSIGNMENT_SEED,
        "rng": "numpy.default_rng/PCG64",
        "targets": list(targets),
        "candidate_expert_pool": list(AIA_EXPERT_POOL),
        "excluded_already_server_tested": list(TESTED_AIA_EXPERTS),
        "target_to_expert": {str(k): v for k, v in assignment.items()},
        "assignment_is_fixed_before_server": True,
        "server_result_used_for_assignment": False,
        "fingerprint_prior_is_not_label": True,
    }
    (output / "aia_assignment.json").write_text(json.dumps(assignment_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"mia_path": str(mia_path), "aia_path": str(aia_path), "assignment": assignment_payload}


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    background = pd.read_csv(args.team_dir / "A_bg.csv")
    practice = pd.read_csv(args.team_dir / "B_practice.csv")
    schema = json.loads((args.team_dir / "schema.json").read_text(encoding="utf-8"))
    tune = _run_mia_pool(background, practice, schema["rare_detector"], MIA_TUNE_SEEDS, "pseudo_tune")
    test = _run_mia_pool(background, practice, schema["rare_detector"], MIA_TEST_SEEDS, "pseudo_test")
    selected, selection = _select_mia(tune)
    parts = partition_background(background, MIA_PRIMARY_SEED)
    _, controls = _split(parts["pseudo_tune"])
    crowding = _crowding_sanity(_split(parts["pseudo_tune"])[0], controls, background)
    test_selected = test[test.formula == selected].copy()
    test_selected["selection_role"] = "untouched_independent_test_after_tune_selection"
    tune.to_csv(args.output / "mia_pseudo_tune_metrics.csv", index=False)
    test.to_csv(args.output / "mia_pseudo_test_metrics.csv", index=False)
    selection.to_csv(args.output / "mia_formula_selection.csv", index=False)
    test_selected.to_csv(args.output / "mia_selected_test_metrics.csv", index=False)
    fallback_mia = None
    fallback_decision = "not_used_skip1_passed"
    if selected == "none":
        s3 = pd.read_csv(args.s3_mia)
        base = pd.read_csv(args.base_mia)
        fallback_mia = pd.DataFrame({"record_id": base["record_id"]})
        for target in TARGETS:
            if target in (1, 2) or target in FALLBACK_RESTORE_TARGETS:
                fallback_mia[str(target)] = base[str(target)].to_numpy()
            else:
                fallback_mia[str(target)] = s3[str(target)].to_numpy()
        fallback_mia.to_csv(args.output / "F_mia_s3_rare_mask_group_ablation.csv", index=False)
        fallback_decision = "s3_rare_kde_mask_with_preregistered_balanced_restore"
    public = _build_public_candidates(args.target_dir, args.team_dir, args.base_mia, args.output, selected, fallback_mia)
    metadata = {
        "run_id": args.output.name,
        "purpose": "S4 exploratory nearest-decoy MIA plus AIA target-wise multiplex",
        "mia": {
            "protected_targets": [1, 2], "selected_formula": selected,
            "formula_candidates": list(FORMULAS), "local_sanity_gate": {
                "c_only_min_mean_delta_tpr": MIA_C_ONLY_MIN_GAIN,
                "ordinary_worst_delta_tpr_tolerance": MIA_ORDINARY_WORST_TOLERANCE,
            },
            "tune_seeds": list(MIA_TUNE_SEEDS), "untouched_test_seeds": list(MIA_TEST_SEEDS),
            "mechanisms": list(MIA_MECHANISMS), "test_used_for_selection": False,
            "fallback": fallback_decision,
            "fallback_restore_targets": list(FALLBACK_RESTORE_TARGETS) if selected == "none" else [],
            "crowding_sanity": crowding,
        },
        "aia": public["assignment"],
        "inputs": ["public/prelim_attack_20260825/C_<target>.csv", "public/prelim_attack_20260825/aia_challenge_<target>.csv", "team10/A_bg.csv", "team10/B_practice.csv", "team10/schema.json", "validated/frozen_stage2_mia/F_mia_candidate.csv"],
        "candidate_files": {"mia": "F_mia_skip1_selected.csv", "aia": "F_aia_multiplex.csv"},
        "no_token_read": True, "no_zip_created": True, "no_submission_created": True,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_mia_formula": selected, "assignment": public["assignment"]["target_to_expert"]}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--team-dir", type=Path, required=True)
    parser.add_argument("--base-mia", type=Path, required=True)
    parser.add_argument("--s3-mia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
