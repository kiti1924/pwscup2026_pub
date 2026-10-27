# 各フェーズで何を提出したか

提出したのは加工CSVまたはMIA/AIA回答を含むZIPです。表のコードはその生成に使ったコードであり、コード自体を競技へ提出した意味ではありません。最終採用はフェーズごとに1件。得点のexact/表示丸め、取消、精密結果未回収を区別しています。

## 予備戦・加工（6件）

| ID | 呼称 / 状態 | 内容 | コード | 得点 | 最終採用 |
|---:|---|---|---|---:|---|
| 896537 | 中間 / Finished | 公式copula seed0 | 公式キット / 対応未確定（注記参照） | 0.4867（表示丸め） | — |
| 896550 | 中間 / Finished | 公式copula seed4 | 公式キット / 対応未確定（注記参照） | 0.6285（表示丸め） | — |
| 896581 | 中間 / Finished | 初期改善候補（生成系列未確定） | 公式キット / 対応未確定（注記参照） | 0.9083（表示丸め） | — |
| 896595 | 中間 / Finished | 希少群対応合成 covariate seed4/outcome seed56 | [run_aligned_synth.py](../experiments/run_aligned_synth.py)<br>[run_outcome_sweep.py](../experiments/run_outcome_sweep.py) | 0.9194（表示丸め） | — |
| 896618 | 中間 / Finished | tilt後の転帰再較正 | [run_rare_truth_tilt.py](../experiments/run_rare_truth_tilt.py)<br>[run_calibrated_outcomes.py](../experiments/run_calibrated_outcomes.py) | 0.9710（表示丸め） | — |
| 896713 | 最終 / Finished | 静的微修正までの最終加工系列 | [run_aligned_synth.py](../experiments/run_aligned_synth.py)<br>[run_calibrated_outcomes.py](../experiments/run_calibrated_outcomes.py)<br>[run_outcome_swap_hillclimb.py](../experiments/run_outcome_swap_hillclimb.py)<br>[run_static_swap_probe.py](../experiments/run_static_swap_probe.py) | 0.9784（表示丸め） | 採用 |

## 予備戦・攻撃（5件）

| ID | 呼称 / 状態 | 内容 | コード | 得点 | 最終採用 |
|---:|---|---|---|---:|---|
| 916932 | S0 / Finished | frozen_stage2_nn + official_1nn | [run_mia_stage2_experiment.py](../attack_phase/scripts/run_mia_stage2_experiment.py)<br>[run_aia_stage2_experiment.py](../attack_phase/scripts/run_aia_stage2_experiment.py)<br>[build_server_probe.py](../attack_phase/scripts/build_server_probe.py) | 0.2314（表示丸め） | 採用 |
| 916951 | S1-dual / Finished | NN75%/ExtraTrees25% + gated donor modal | [run_mia_stage4_experiment.py](../attack_phase/scripts/run_mia_stage4_experiment.py)<br>[build_server_probe.py](../attack_phase/scripts/build_server_probe.py) | 0.1909（表示丸め） | — |
| 916998 | S3 / Finished | frozen1/2 + rare-KDE other19 / cell-modal | [make_s3_mia_candidate.py](../attack_phase/scripts/make_s3_mia_candidate.py)<br>[build_s3_orthogonal_inputs.py](../attack_phase/scripts/build_s3_orthogonal_inputs.py) | 0.1762（表示丸め） | — |
| 917054 | S4 / Finished | frozen12 + KDE9 / AIA multiplex | [build_s4_exploratory_probe.py](../attack_phase/scripts/build_s4_exploratory_probe.py) | 0.1980（表示丸め） | — |
| 917072 | S5 / Finished | KDE4/19/24 + frozen18 / AIA multiplex | [build_s5_server_probe.py](../attack_phase/scripts/build_s5_server_probe.py) | 0.1594（表示丸め） | — |

## 本戦・加工（17件、取消4件を含む）

| ID | 呼称 / 状態 | 内容 | コード | 得点 | 最終採用 |
|---:|---|---|---|---:|---|
| 942802 | G02 / Finished | G02の加工候補 | [run_processing.py](../research/run_processing.py) | 0.6763119999 | — |
| 942917 | G18/dup_exact 初回 / Cancelled | Exact duplicate control | [probe_metric_rules.py](../research/probe_metric_rules.py) | 未回収 / 未採点 | — |
| 942918 | G18/dup_epsilon / finished | Sub-resolution jitter, exact duplicate metric contrast | [probe_metric_rules.py](../research/probe_metric_rules.py) | 0.6838227694 | — |
| 942922 | G18/mass73 初回 / Cancelled | Mass deadzone inner control, other rare facets may bind | [probe_metric_rules.py](../research/probe_metric_rules.py) | 未回収 / 未採点 | — |
| 942930 | G18/mass100 初回 / Cancelled | Mass outer control with mass binding U_rare | [probe_metric_rules.py](../research/probe_metric_rules.py) | 未回収 / 未採点 | — |
| 942931 | G18/censor_outside / finished | Censoring zero while U_valid mean stays above overall bottleneck | [probe_metric_rules.py](../research/probe_metric_rules.py) | 0.6763119999 | — |
| 942932 | G18/c2st_low / finished | Unsaturated C2ST diagnostic control from saved search | [probe_metric_rules.py](../research/probe_metric_rules.py) | 0.8552530528 | — |
| 942934 | G18/c2st_selected 初回 / Cancelled | Selected C2ST candidate; posthoc comparison, no internal seed proof | [probe_metric_rules.py](../research/probe_metric_rules.py) | 未回収 / 未採点 | — |
| 942936 | G18/cox_calibrated / finished | Cox calibration with unchanged static features | [probe_metric_rules.py](../research/probe_metric_rules.py) | 0.8445799975 | — |
| 942969 | G18/dup_exact / finished | Exact duplicate control | [probe_metric_rules.py](../research/probe_metric_rules.py) | 0.6838227694 | — |
| 942974 | G18/mass73 / finished | Mass deadzone inner control, other rare facets may bind | [probe_metric_rules.py](../research/probe_metric_rules.py) | 0.6614836279 | — |
| 942980 | G18/mass100 / Finished_observed_details_not_recovered | Mass outer control with mass binding U_rare | [probe_metric_rules.py](../research/probe_metric_rules.py) | 未回収 / 未採点 | — |
| 942981 | G18/c2st_selected / Finished_observed_details_not_recovered | Selected C2ST candidate; posthoc comparison, no internal seed proof | [probe_metric_rules.py](../research/probe_metric_rules.py) | 未回収 / 未採点 | — |
| 943366 | G20 / Finished | G20の加工候補 | [geographic_intervention.py](../research/geographic_intervention.py) | 0.9134756611 | — |
| 943396 | G21-001 / Finished | G21-001の加工候補 | [overnight_refinement.py](../research/overnight_refinement.py) | 0.9170339095 | — |
| 943413 | G21-002 / Finished | G21-002の加工候補 | [compose_group_outcomes.py](../research/compose_group_outcomes.py) | 0.9295762067 | — |
| 943786 | G22 / Finished | G22の加工候補 | [utility_slack.py](../research/utility_slack.py) | 0.9295762067 | 採用 |

## 本戦・攻撃（8件）

| ID | 呼称 / 状態 | 内容 | コード | 得点 | 最終採用 |
|---:|---|---|---|---:|---|
| 965017 | FA04-S1 / Finished | full_static_nn + official_1nn | [build_final_attack_baseline.py](../research/build_final_attack_baseline.py)<br>[final_attack_api.py](../research/final_attack_api.py) | 0.1138766319 | — |
| 971407 | FA15 / Finished | peer_bg_excluded (target9 full_static) + raw_soft_pref1 | [prepare_mechanism_comparison.py](../research/prepare_mechanism_comparison.py)<br>[final_attack_extensions.py](../research/final_attack_extensions.py)<br>[mechanism_components.py](../research/mechanism_components.py)<br>[prefecture_weight_attacks.py](../research/prefecture_weight_attacks.py) | 0.1406393926 | — |
| 972302 | CL01-S3 / Finished | structural MIA + 30% AIA abstention | [structural_attack.py](../research/claude/structural_attack.py)<br>[build_candidates.py](../research/claude/build_candidates.py) | 0.1203（表示丸め） | — |
| 974256 | FA95 / Finished | 972302 MIA + FA55 public API AIA | [fa55_public_api.py](../research/fa55_public_api.py)<br>[run_fa55_public_api.py](../research/run_fa55_public_api.py) | 0.0984（表示丸め） | — |
| 974917 | FA103/A / Finished | FA100 age10 AIA + FA98 rare-slot MIA | [run_main_public_contrasts.py](../research/run_main_public_contrasts.py)<br>[main_attack_contrasts.py](../research/main_attack_contrasts.py)<br>[mia_rare_slots.py](../research/mia_rare_slots.py)<br>[package_public_attack.py](../research/package_public_attack.py) | 0.0759903110 | — |
| 974934 | FA107/B / Finished | ungated rank2 AIA + FA106 calibration MIA | [run_main_public_contrasts.py](../research/run_main_public_contrasts.py)<br>[nearest_decoy.py](../research/nearest_decoy.py)<br>[conditional_null_mia.py](../research/conditional_null_mia.py)<br>[run_main_transfer.py](../research/runs/20261010-FA106-cl06-main-transfer-001/run_main_transfer.py)<br>[package_public_attack.py](../research/package_public_attack.py) | 0.0714232807 | — |
| 974967 | FA108 / Finished | six-source measured AIA integration + 972302 MIA | [build_final.py](../research/runs/20261011-FA108-measured-final-integration-001/build_final.py)<br>[package_public_attack.py](../research/package_public_attack.py) | 0.1615815185 | — |
| 975433 | FA114 / Finished | 974967 AIA + rare/excluded-protected nonrare calibration | [build_rare_fixed_main.py](../research/build_rare_fixed_main.py)<br>[rare_fixed_calibration.py](../research/rare_fixed_calibration.py)<br>[conditional_null_mia.py](../research/conditional_null_mia.py) | 0.1684450253 | 採用 |

## 対応の確度

- 予備戦の提出ID・表示値・選択状態はユーザー提供スクリーンショットの転記。初期加工はGit前に開始しており、896581の生成コードは未確定です。896595/896618は当時レポートの処理とスコアによる事後対応で、提出ZIPとのhash結合は未確認です。
- 予備戦S0の最終攻撃力は自己チームを除いた20標的で.2429347629。表の.2314とは集計の分母が違います。
- G18は採点実装の診断用で、最終加工へ順番に適用した操作ではありません。942980/942981はG22終了時にFinished化を観測していますが、精密結果を保存できていません。
- 本戦の972302/974256はこの台帳では表示4桁値を使います。精密値のないところを推定して埋めていません。
- 原記録のパス・hashは[証拠一覧](../provenance/evidence.json)、行ごとの注記は[JSON台帳](../provenance/submissions.json)。公開用の事後整理であり、全コードを提出時点の同一版と保証するものではありません。G22・FA106・FA114の主要コードは当時hashと別途照合しています。
