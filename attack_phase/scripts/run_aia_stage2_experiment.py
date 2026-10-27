#!/usr/bin/env python
"""Run AIA stage-2 abstention/donor-set experiments and public candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aia_stage1 import AIAEngine, evaluate_time_predictions, rare_mask, validate_f_aia_candidate  # noqa: E402
from aia_stage2 import (  # noqa: E402
    AIAStage2Engine,
    STAGE2_METHODS,
    Stage2Config,
    make_stage2_pseudo_target,
    stage2_config_grid,
)


PRIMARY_SEED = 20260828
TUNE_SEEDS = (8101, 8102, 8103)
TEST_SEEDS = (9101, 9102, 9103, 9104, 9105)
PROXY_SEEDS = (20260828, 20260829, 20260830, 20260831, 20260832)
MECHANISMS = ("near_copy", "perturbed", "triplet_swap", "bootstrap_copy", "risk_stratified_triplet_swap")
TAU = 0.5
RARE_Z = 1.64
MIN_RARE_N = 1


def _master_split(background: pd.DataFrame, practice: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bg = rng.permutation(len(background))
    members = rng.permutation(len(practice))
    return {
        "tune_background": background.iloc[bg[:1700]].reset_index(drop=True),
        "tune_controls": background.iloc[bg[1700:2400]].reset_index(drop=True),
        "test_background": background.iloc[bg[2400:4100]].reset_index(drop=True),
        "test_controls": background.iloc[bg[4100:4800]].reset_index(drop=True),
        "tune_members": practice.iloc[members[:500]].reset_index(drop=True),
        "test_members": practice.iloc[members[500:]].reset_index(drop=True),
    }


def _one_metrics(target: pd.DataFrame, members: pd.DataFrame, controls: pd.DataFrame, background: pd.DataFrame, method: str, config: Stage2Config | None, seed: int, mechanism: str) -> dict:
    if method == "official_1nn":
        engine = AIAEngine(target, background)
    else:
        engine = AIAStage2Engine(target, background, config or Stage2Config())
    member_hat, member_diag = engine.predict(members, method)
    control_hat, control_diag = engine.predict(controls, method)
    metrics = evaluate_time_predictions(
        member_hat, members["time"], control_hat, controls["time"], TAU,
        rare_mask(members, RARE_DETECTOR), rare_mask(controls, RARE_DETECTOR), RARE_Z, MIN_RARE_N,
    )
    member_hits = int(member_diag["gate_hit"].sum())
    control_hits = int(control_diag["gate_hit"].sum())
    total_hits = member_hits + control_hits
    return {
        "seed": seed, "mechanism": mechanism, "method": method,
        "config": "baseline" if config is None else config_name(config), **metrics,
        "member_gate_hits": member_hits, "control_gate_hits": control_hits,
        "gate_member_precision": member_hits / total_hits if total_hits else float("nan"),
        "member_sentinel_rate": float(member_diag["sentinel_hit"].mean()) if "sentinel_hit" in member_diag else 0.0,
        "control_sentinel_rate": float(control_diag["sentinel_hit"].mean()) if "sentinel_hit" in control_diag else 0.0,
    }


def config_name(config: Stage2Config) -> str:
    return f"g{config.gate_log_contrast:g}_{config.risk_weights}_k{config.donor_k}"


def _pseudo_runs(
    parts: dict[str, pd.DataFrame], seeds: tuple[int, ...], tune: bool,
    configs: list[Stage2Config] | None = None,
) -> pd.DataFrame:
    background = parts["tune_background" if tune else "test_background"]
    controls = parts["tune_controls" if tune else "test_controls"]
    members = parts["tune_members" if tune else "test_members"]
    rows = []
    configs = stage2_config_grid() if configs is None and tune else (configs or [Stage2Config()])
    for seed in seeds:
        for mechanism in MECHANISMS:
            pseudo = make_stage2_pseudo_target(members, mechanism, seed)
            rows.append(_one_metrics(pseudo, members, controls, background, "official_1nn", None, seed, mechanism))
            for config in configs:
                for method in STAGE2_METHODS:
                    rows.append(_one_metrics(pseudo, members, controls, background, method, config, seed, mechanism))
    return pd.DataFrame(rows)


def _select(tune: pd.DataFrame) -> tuple[str, Stage2Config, pd.DataFrame]:
    summary = tune.groupby(["config", "method"], as_index=False).agg(
        mean_r_time=("r_time_score", "mean"), worst_r_time=("r_time_score", "min"),
        mean_success_member=("success_member", "mean"), mean_success_control=("success_control", "mean"),
    )
    baseline = summary.loc[summary.method == "official_1nn"].iloc[0]
    summary["gain_vs_baseline"] = summary.mean_r_time - float(baseline.mean_r_time)
    summary["eligible"] = (summary.method != "official_1nn") & (summary.gain_vs_baseline >= 0.02) & (summary.worst_r_time >= float(baseline.worst_r_time) - 0.02)
    candidates = summary[summary.eligible].sort_values(["mean_r_time", "worst_r_time", "config"], ascending=[False, False, True])
    if len(candidates):
        chosen_name = str(candidates.iloc[0].config)
        chosen_method = str(candidates.iloc[0].method)
    else:
        chosen_name, chosen_method = "g1_uniform_k15", STAGE2_METHODS[0]
    configs = {config_name(config): config for config in stage2_config_grid()}
    chosen_config = configs[chosen_name]
    return chosen_method, chosen_config, summary.sort_values(["mean_r_time", "worst_r_time", "method"], ascending=[False, False, True]).reset_index(drop=True)


def _proxy(c10: pd.DataFrame, practice: pd.DataFrame, background: pd.DataFrame, config: Stage2Config, selected_method: str) -> pd.DataFrame:
    rows = []
    for seed in PROXY_SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(background))
        model_background = background.iloc[order[:3000]].reset_index(drop=True)
        controls = background.iloc[order[3000:3500]].reset_index(drop=True)
        for method in ("official_1nn", *STAGE2_METHODS):
            cfg = None if method == "official_1nn" else config
            rows.append(_one_metrics(c10, practice, controls, model_background, method, cfg, seed, "team10_proxy"))
    return pd.DataFrame(rows)


def _public(target_dir: Path, targets: list[int], background: pd.DataFrame, config: Stage2Config, challenger: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_rows, challenger_rows, diagnostics = [], [], []
    challenge_ids: dict[int, list[int]] = {}
    for target in targets:
        c = pd.read_csv(target_dir / f"C_{target}.csv")
        challenge = pd.read_csv(target_dir / f"aia_challenge_{target}.csv")
        ids = [int(x) for x in challenge["challenge_row_id"]]
        challenge_ids[target] = ids
        base_engine = AIAEngine(c, background)
        stage_engine = AIAStage2Engine(c, background, config)
        base, base_diag = base_engine.predict(challenge, "official_1nn")
        selected, selected_diag = stage_engine.predict(challenge, challenger)
        for i, row_id in enumerate(ids):
            base_rows.append({"target_k": target, "challenge_row_id": row_id, "time_hat": float(base[i])})
            challenger_rows.append({"target_k": target, "challenge_row_id": row_id, "time_hat": float(selected[i])})
            diagnostics.extend([
                {"target_k": target, "challenge_row_id": row_id, "method": "official_1nn", "time_hat": float(base[i]), "distance_c": float(base_diag.iloc[i].distance_c), "gate_hit": False, "sentinel_hit": False},
                {"target_k": target, "challenge_row_id": row_id, "method": challenger, "time_hat": float(selected[i]), "distance_c": float(selected_diag.iloc[i].distance_c), "gate_hit": bool(selected_diag.iloc[i].gate_hit), "sentinel_hit": bool(selected_diag.iloc[i].sentinel_hit)},
            ])
    base = pd.DataFrame(base_rows)
    challenger_frame = pd.DataFrame(challenger_rows)
    validate_f_aia_candidate(base, challenge_ids)
    validate_f_aia_candidate(challenger_frame, challenge_ids)
    return base, challenger_frame, pd.DataFrame(diagnostics)


def run(args: argparse.Namespace) -> None:
    global RARE_DETECTOR
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    targets = [int(x) for x in json.loads((args.target_dir / "mia_columns.json").read_text(encoding="utf-8"))["target_columns"]]
    background = pd.read_csv(args.team_dir / "A_bg.csv")
    practice = pd.read_csv(args.team_dir / "B_practice.csv")
    RARE_DETECTOR = json.loads((args.team_dir / "schema.json").read_text(encoding="utf-8"))["rare_detector"]
    parts = _master_split(background, practice, args.seed)
    tune = _pseudo_runs(parts, TUNE_SEEDS, True)
    pseudo_selected, config, selection = _select(tune)
    test_all = _pseudo_runs(parts, TEST_SEEDS, False, [config])
    test_summary = test_all.groupby(["config", "method"], as_index=False).agg(
        test_mean_r_time=("r_time_score", "mean"),
        test_worst_r_time=("r_time_score", "min"),
    )
    selection = selection.merge(test_summary, on=["config", "method"], how="left")
    test = test_all[(test_all.method == "official_1nn") | ((test_all.method == pseudo_selected) & (test_all.config == config_name(config)))].copy()
    base = test[test.method == "official_1nn"].r_time_score.mean()
    challenger = test[test.method == pseudo_selected].r_time_score.mean()
    base_worst = test[test.method == "official_1nn"].r_time_score.min()
    challenger_worst = test[test.method == pseudo_selected].r_time_score.min()
    test_passed = bool(challenger > base and challenger_worst >= base_worst - 0.02)
    c10 = pd.read_csv(args.target_dir / "C_10.csv")
    proxy = _proxy(c10, practice, background, config, pseudo_selected)
    proxy_base = float(proxy[proxy.method == "official_1nn"].r_time_score.mean())
    proxy_challenger = float(proxy[proxy.method == pseudo_selected].r_time_score.mean())
    proxy_passed = bool(proxy_challenger > proxy_base)
    final_method = pseudo_selected if test_passed and proxy_passed else "official_1nn"
    tune.to_csv(output / "pseudo_tune_metrics.csv", index=False)
    test_all.to_csv(output / "pseudo_test_metrics.csv", index=False)
    selection["pseudo_selected_method"] = pseudo_selected
    selection["selected_config"] = config_name(config)
    selection["test_directional_passed"] = test_passed
    selection["proxy_guard_passed"] = proxy_passed
    selection["final_method"] = final_method
    selection.to_csv(output / "method_selection.csv", index=False)
    proxy.to_csv(output / "team10_proxy_metrics.csv", index=False)
    base_frame, challenger_frame, diagnostics = _public(args.target_dir, targets, background, config, pseudo_selected)
    base_frame.to_csv(output / "F_aia_baseline.csv", index=False)
    challenger_frame.to_csv(output / "F_aia_challenger.csv", index=False)
    final_frame = base_frame if final_method == "official_1nn" else challenger_frame
    final_frame.to_csv(output / "F_aia_candidate.csv", index=False)
    diagnostics.to_csv(output / "public_method_diagnostics.csv", index=False)
    meta = {
        "run_id": output.name, "seed": args.seed, "targets": targets,
        "methods": ["official_1nn", *STAGE2_METHODS], "pseudo_mechanisms": list(MECHANISMS),
        "tune_seeds": list(TUNE_SEEDS), "test_seeds": list(TEST_SEEDS), "proxy_seeds": list(PROXY_SEEDS),
        "selected_config": config_name(config), "pseudo_selected_method": pseudo_selected,
        "test_directional_passed": test_passed, "proxy_guard_passed": proxy_passed, "final_method": final_method,
        "sentinel": "max(target/background observed time)+sentinel_margin; finite nonnegative and outside observed range",
        "sentinel_margin": config.sentinel_margin, "tau_time": TAU, "rare_deadzone_z": RARE_Z, "min_rare_n": MIN_RARE_N,
        "background_fallback_used": False, "public_role": "unlabeled candidate generation and diagnostics only",
        "candidate_validation": {"rows": len(final_frame), "targets": len(targets), "finite_nonnegative": True},
        "no_token_read": True, "no_zip_created": True, "no_submission_created": True,
    }
    (output / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "final_method": final_method, "challenger": pseudo_selected, "candidate_rows": len(final_frame)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--team-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
