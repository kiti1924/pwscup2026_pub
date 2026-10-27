# コードの役割一覧

元の相対パスを保持しています。入口は提出物の生成・監査に用いたスクリプト、依存はそこから呼ばれる部品です。依存ファイル内の全実験・全手法を採用した意味ではありません。原コードは実験時の固定パスや保存中間物を要求するため、実行条件は[再現手順](REPRODUCE.md)を参照してください。

## 予備戦・加工

| コード | 意味 | 区分 |
|---|---|---|
| [experiments/extend_prefecture_on_candidate.py](../experiments/extend_prefecture_on_candidate.py) | 既存候補の県補正を追加し、静的・プライバシー制約を点検する。 | 入口 |
| [experiments/full_static_nearest_audit.py](../experiments/full_static_nearest_audit.py) | 静的10属性で元Bとの距離・完全一致・希少近傍リスクを点検する。 | 依存 |
| [experiments/local_privacy_audit.py](../experiments/local_privacy_audit.py) | 公式参照攻撃を用いる候補監査。未知の攻撃に対する安全保証ではない。 | 依存 |
| [experiments/minimal_prefecture_sweep.py](../experiments/minimal_prefecture_sweep.py) | 変更県数を限定したprefix探索を行う。 | 入口 |
| [experiments/prefecture_after_round2.py](../experiments/prefecture_after_round2.py) | 転帰交換第2段階後に県補正を適用する。 | 入口 |
| [experiments/prefecture_after_swap.py](../experiments/prefecture_after_swap.py) | 転帰交換後の候補で県分布補正を再評価する。 | 入口 |
| [experiments/run_aligned_synth.py](../experiments/run_aligned_synth.py) | 多数群・公開検出希少群を分け、相関を保った共変量と競合リスク転帰を合成する初期生成器。 | 入口 |
| [experiments/run_calibrated_outcomes.py](../experiments/run_calibrated_outcomes.py) | Cox係数とハザードを較正し、合成共変量へtime/onset/deathを付与する。 | 入口 |
| [experiments/run_common_quantile_micro.py](../experiments/run_common_quantile_micro.py) | 多数群ALT等の分位点を部分的に移動して周辺分布を改善する。 | 入口 |
| [experiments/run_minimal_ks_search.py](../experiments/run_minimal_ks_search.py) | KSの律速列だけを少数セル変更で補正し、監査条件を確認する。 | 入口 |
| [experiments/run_outcome_coordinate_search.py](../experiments/run_outcome_coordinate_search.py) | 転帰モデルの係数を座標探索してU_specを較正する。 | 入口 |
| [experiments/run_outcome_micro_refine.py](../experiments/run_outcome_micro_refine.py) | 転帰較正パラメータを小範囲で調整する。 | 入口 |
| [experiments/run_outcome_swap_hillclimb.py](../experiments/run_outcome_swap_hillclimb.py) | time/onset/deathの三つ組を交換し、転帰整合性を保ちながらCox指標を改善する。 | 入口 |
| [experiments/run_outcome_sweep.py](../experiments/run_outcome_sweep.py) | 合成静的属性を構築し、転帰生成seedを比較する。 | 入口 |
| [experiments/run_prefecture_rebalance.py](../experiments/run_prefecture_rebalance.py) | 県の周辺分布を調整し、U_genの律速を緩和する。 | 入口 |
| [experiments/run_rare_ks_search.py](../experiments/run_rare_ks_search.py) | 希少群SBP等の周辺分布を限定的に修正する。 | 入口 |
| [experiments/run_rare_tilt_micro_composite.py](../experiments/run_rare_tilt_micro_composite.py) | 希少属性の複合方向の微小補正でU_rareを改善する。 | 入口 |
| [experiments/run_rare_truth_tilt.py](../experiments/run_rare_truth_tilt.py) | 加工者に配布された自Bの希少ラベルを用い、希少領域の分布を補正する。 | 入口 |
| [experiments/run_static_swap_probe.py](../experiments/run_static_swap_probe.py) | 最終段階でsmoking交換とageの小変更を検査する。 | 入口 |

## 予備戦・攻撃

| コード | 意味 | 区分 |
|---|---|---|
| [attack_phase/scripts/build_attack_candidates.py](../attack_phase/scripts/build_attack_candidates.py) | MIA/AIAを組み合わせ、明示されたtokenファイルからZIPを組み立て検証する。 | 入口 |
| [attack_phase/scripts/build_s3_orthogonal_inputs.py](../attack_phase/scripts/build_s3_orthogonal_inputs.py) | S3のcell-modal AIA回答を生成する。 | 入口 |
| [attack_phase/scripts/build_s4_exploratory_probe.py](../attack_phase/scripts/build_s4_exploratory_probe.py) | S4で標的ごとに異なるAIA expertを割り当てる。局所実験と提出設定を区別する。 | 入口 |
| [attack_phase/scripts/build_s5_server_probe.py](../attack_phase/scripts/build_s5_server_probe.py) | S5の固定KDE対象3標的とAIA multiplexを構築する。 | 入口 |
| [attack_phase/scripts/build_server_probe.py](../attack_phase/scripts/build_server_probe.py) | 予備戦probe設定に従い標的別のMIA/AIAを選択して包装する。 | 入口 |
| [attack_phase/scripts/make_s3_mia_candidate.py](../attack_phase/scripts/make_s3_mia_candidate.py) | S3の希少KDE MIAと固定NNの標的別組合せを作る。 | 入口 |
| [attack_phase/scripts/run_aia_stage2_experiment.py](../attack_phase/scripts/run_aia_stage2_experiment.py) | 予備戦の公式1NN回答とgate付きdonor-modal回答を生成・比較する入口。 | 入口 |
| [attack_phase/scripts/run_mia_stage2_experiment.py](../attack_phase/scripts/run_mia_stage2_experiment.py) | 予備戦S0の固定Block NN MIAを生成・検証する入口。 | 入口 |
| [attack_phase/scripts/run_mia_stage4_experiment.py](../attack_phase/scripts/run_mia_stage4_experiment.py) | S1-dualで提出した固定NN/ExtraTreesブレンドを作成する。 | 入口 |
| [attack_phase/src/aia_stage1.py](../attack_phase/src/aia_stage1.py) | 公式1NN、重み付き近傍、modal-window等のAIA基礎実装。 | 依存 |
| [attack_phase/src/aia_stage2.py](../attack_phase/src/aia_stage2.py) | membership gateとdonor-modal、棄権sentinelを実装する。 | 依存 |
| [attack_phase/src/aia_stage3.py](../attack_phase/src/aia_stage3.py) | L1/Huber等の距離を用いるAIA expert群。S4/S5の依存。 | 依存 |
| [attack_phase/src/aia_stage4.py](../attack_phase/src/aia_stage4.py) | 生存時間を考慮したAIA expert群。独立実験全体の採用を意味しない。 | 依存 |
| [attack_phase/src/aia_stage5.py](../attack_phase/src/aia_stage5.py) | 集団・セル単位の時間推測を提供する依存実装。 | 依存 |
| [attack_phase/src/aia_stage6.py](../attack_phase/src/aia_stage6.py) | 時間窓と転帰の意味を補正したAIA expert群。 | 依存 |
| [attack_phase/src/aia_stage7.py](../attack_phase/src/aia_stage7.py) | S3で使ったcell-modal推測等を実装する。 | 依存 |
| [attack_phase/src/mia_low_fpr.py](../attack_phase/src/mia_low_fpr.py) | 予備戦MIAの距離特徴と低FPR評価を定義する。 | 依存 |
| [attack_phase/src/mia_stage2.py](../attack_phase/src/mia_stage2.py) | 性別×県のBlock NNと参照・較正分離を実装する。 | 依存 |
| [attack_phase/src/mia_stage3.py](../attack_phase/src/mia_stage3.py) | 難易度補正・shadow生成等を提供する依存モジュール。手法全体を採用した意味ではない。 | 依存 |
| [attack_phase/src/mia_stage4.py](../attack_phase/src/mia_stage4.py) | NNとExtraTreesの固定比率ブレンドを実装する。 | 依存 |
| [attack_phase/src/mia_stage9_skip1.py](../attack_phase/src/mia_stage9_skip1.py) | 近傍decoyを想定した距離式の比較を提供するS4ビルダー依存。 | 依存 |
| [attack_phase/src/target_router.py](../attack_phase/src/target_router.py) | MIAは標的列、AIAはtarget_kで回答を結合し、キー整合性を点検する。 | 依存 |

## 本戦・加工

| コード | 意味 | 区分 |
|---|---|---|
| [experiments/full_static_nearest_audit.py](../experiments/full_static_nearest_audit.py) | 静的10属性で元Bとの距離・完全一致・希少近傍リスクを点検する。 | 依存 |
| [experiments/local_privacy_audit.py](../experiments/local_privacy_audit.py) | 公式参照攻撃を用いる候補監査。未知の攻撃に対する安全保証ではない。 | 依存 |
| [experiments/run_aligned_synth.py](../experiments/run_aligned_synth.py) | 多数群・公開検出希少群を分け、相関を保った共変量と競合リスク転帰を合成する初期生成器。 | 依存 |
| [experiments/run_calibrated_outcomes.py](../experiments/run_calibrated_outcomes.py) | Cox係数とハザードを較正し、合成共変量へtime/onset/deathを付与する。 | 依存 |
| [experiments/run_outcome_sweep.py](../experiments/run_outcome_sweep.py) | 合成静的属性を構築し、転帰生成seedを比較する。 | 依存 |
| [research/adaptive_privacy.py](../research/adaptive_privacy.py) | 6MIA/6AIAの加工時自己攻撃と候補×攻撃行列を実装する。 | 依存 |
| [research/audit_utility_slack.py](../research/audit_utility_slack.py) | G22候補の決定的再生成、MIA不変性、陽性対照の検出能力を点検する。 | 入口 |
| [research/background_copula.py](../research/background_copula.py) | 背景の分布と群別copulaから新しい静的共変量を作り転帰を合成する。 | 入口 |
| [research/background_partition.py](../research/background_partition.py) | 生成・攻撃参照・較正に使う背景を分割し、情報の混用を防ぐ。 | 依存 |
| [research/background_transport.py](../research/background_transport.py) | 背景の数値属性をBの群別周辺へ順位輸送する共通生成部品。 | 依存 |
| [research/calibrated_copula.py](../research/calibrated_copula.py) | B集団相関とdetector群別発症率へ合成モデルを較正する。 | 入口 |
| [research/compose_group_outcomes.py](../research/compose_group_outcomes.py) | G21-002で修復後の静的列と群較正済みの合成転帰i2を組み合わせる。 | 入口 |
| [research/confirm_overnight.py](../research/confirm_overnight.py) | 凍結候補と基準を未使用の背景再分割で同じ攻撃版にかける。 | 入口 |
| [research/evaluate_background_copula.py](../research/evaluate_background_copula.py) | 背景copula候補の保存評価と比較を提供する。 | 依存 |
| [research/evaluate_background_transport.py](../research/evaluate_background_transport.py) | 背景輸送候補の基準・対照と指標最大値を集約する。 | 依存 |
| [research/geographic_intervention.py](../research/geographic_intervention.py) | G17g/G20の県再割当てを生成する。最終系列はseed2/gamma0/G。 | 入口 |
| [research/geography_free_attacks.py](../research/geography_free_attacks.py) | 県を使わない自己攻撃を加えて、県変更による見かけの防御を点検する。 | 依存 |
| [research/overnight_refinement.py](../research/overnight_refinement.py) | G21-001のalpha=.67/seed3301の性別相関修復を生成・監査する。 | 入口 |
| [research/privacy_suite.py](../research/privacy_suite.py) | MIA低FPR、AIA的中率差・希少補正等の診断指標を定義する。 | 依存 |
| [research/probe_metric_rules.py](../research/probe_metric_rules.py) | 本戦加工G18の重複・微小差・希少質量・転帰等の採点確認候補を作る。最終採用ではない。 | 入口 |
| [research/repair_sex_correlation.py](../research/repair_sex_correlation.py) | 県×detector群内の性別交換で集団相関を修復し、公式形式とUを検査する。 | 依存 |
| [research/replay_calibrated_privacy.py](../research/replay_calibrated_privacy.py) | 保存した攻撃予測・設定の同一性と再採点を点検する。 | 依存 |
| [research/residual_rare.py](../research/residual_rare.py) | 希少残差生成研究の共通生成・採点関数。最終系列ではimportされた部品のみ。 | 依存 |
| [research/run_processing.py](../research/run_processing.py) | 本戦B14に公式/独自生成器を適用する初期基準G02。 | 入口 |
| [research/summarize_tradeoff.py](../research/summarize_tradeoff.py) | Uと自己攻撃の複数指標を併記し、候補間トレードオフを集計する。 | 依存 |
| [research/utility_slack.py](../research/utility_slack.py) | G22で合成イベント時刻だけに幅1年/seed3401の反射付き摂動を加える。 | 入口 |

## 本戦・攻撃

| コード | 意味 | 区分 |
|---|---|---|
| [research/attack_ensembles.py](../research/attack_ensembles.py) | 固定MIA/AIAアンサンブルの共通演算。依存コードとして収録。 | 依存 |
| [research/audit_attack_fingerprints.py](../research/audit_attack_fingerprints.py) | 公開Cの分布・重複・時刻等の観測可能な特徴を監査する。 | 依存 |
| [research/build_final_attack_baseline.py](../research/build_final_attack_baseline.py) | 本戦全標的の初期予測を生成し、キー・形式・包装を検証する。 | 入口 |
| [research/build_rare_fixed_main.py](../research/build_rare_fixed_main.py) | 974967のAIAと972302の希少MIAを固定し、非希少MIAを補正して975433を包装する。 | 入口 |
| [research/claude/build_candidates.py](../research/claude/build_candidates.py) | CL01のS3/S4/S5候補を作る。実際に提出されたのはS3=972302だけ。 | 入口 |
| [research/claude/structural_attack.py](../research/claude/structural_attack.py) | 972302の基礎となる県事前確率と希少順位の構造層。自Bのis_rareを利用する。 | 依存 |
| [research/common_metric.py](../research/common_metric.py) | 共通の公開属性変換・近傍距離を提供する依存部品。 | 依存 |
| [research/compare_attack_portfolios.py](../research/compare_attack_portfolios.py) | 保存攻撃の相補性・固定mix・事後上限を比較する。最終提出を自動選択しない。 | 依存 |
| [research/compare_final_attacks.py](../research/compare_final_attacks.py) | 初期攻撃を予備戦・shadowで比較する。採点専用処理を含む。 | 入口 |
| [research/conditional_null_mia.py](../research/conditional_null_mia.py) | 性別・県・年齢帯/近傍の背景非memberを基準にMIA信号を較正する。 | 入口 |
| [research/direct_density_aia.py](../research/direct_density_aia.py) | 背景対比の選択性と性別内の回答枠割当てを提供する。 | 依存 |
| [research/distance_norms.py](../research/distance_norms.py) | L1等の近傍距離・時間ドナーの演算を提供する。 | 依存 |
| [research/fa55_public_api.py](../research/fa55_public_api.py) | 974256のL1背景比/trim距離他C比を組み合わせ、選択行へ近傍時刻を返す。 | 依存 |
| [research/final_attack_api.py](../research/final_attack_api.py) | 本戦初回965017のfull_static_nn MIA、公式1NN AIAを含む公開入力API。 | 依存 |
| [research/final_attack_extensions.py](../research/final_attack_extensions.py) | 背景一致行を除去した他C対比MIAと局所時間窓AIAを実装する。 | 依存 |
| [research/known_private_exclusion.py](../research/known_private_exclusion.py) | 既知の自己非共有行との照合による除外maskを作る。相手所属真値を使わない。 | 依存 |
| [research/main_attack_contrasts.py](../research/main_attack_contrasts.py) | FA100の年齢帯選択と公開Cのpeer距離を定義する。 | 依存 |
| [research/marginal_alignment.py](../research/marginal_alignment.py) | 公開分布の周辺順位整合を提供する依存部品。 | 依存 |
| [research/mechanism_components.py](../research/mechanism_components.py) | 県制約等を因子分解したMIA/AIAの実装。971407のraw_soft_pref1経路を含む。 | 入口 |
| [research/mechanism_study_io.py](../research/mechanism_study_io.py) | 実験計画、入力hash、コード版の読込・保存を共通化する。 | 依存 |
| [research/mia_rare_slots.py](../research/mia_rare_slots.py) | S3再生と、固定スコア枠の希少行内並替えを実装する。 | 依存 |
| [research/nearest_decoy.py](../research/nearest_decoy.py) | 第2近傍等のドナー選択を提供する。974934のungated rank2系列で使用。 | 依存 |
| [research/package_public_attack.py](../research/package_public_attack.py) | 凍結済みCSVをそのまま包装する。予測・送信を行わない。 | 入口 |
| [research/package_saved_final_attack.py](../research/package_saved_final_attack.py) | 保存予測からMIA/AIAをキー付きで選択し、本戦ZIPを決定的に作る。 | 依存 |
| [research/prefecture_weight_attacks.py](../research/prefecture_weight_attacks.py) | 県の一致をhard条件からsoft距離へ変えるAIA/MIAを提供する。 | 入口 |
| [research/prepare_mechanism_comparison.py](../research/prepare_mechanism_comparison.py) | 971407のpeer_bg_excluded/raw_soft_pref1回答を包装し、基準との差を検査する。 | 入口 |
| [research/rare_fixed_calibration.py](../research/rare_fixed_calibration.py) | 最終975433の非希少・非除外行だけを条件付き優先順位で並替える。 | 依存 |
| [research/run_fa55_public_api.py](../research/run_fa55_public_api.py) | FA55の予備戦再生、本戦推論、972302 MIAとの包装を行うFA95入口。 | 入口 |
| [research/run_main_public_contrasts.py](../research/run_main_public_contrasts.py) | 974917のage10 AIA/希少MIA並替えと974934のrank2 AIAを公開入力から作る。 | 入口 |
| [research/run_rare_fixed_calibration.py](../research/run_rare_fixed_calibration.py) | 非希少校正の分離推論・凍結・検証と共通IOを提供する。 | 依存 |
| [research/runs/20261010-FA106-cl06-main-transfer-001/run_main_transfer.py](../research/runs/20261010-FA106-cl06-main-transfer-001/run_main_transfer.py) | FA106として条件付き校正を本戦へ移し、974934用MIAと最終補正用配列を作る。 | 入口 |
| [research/runs/20261011-FA108-measured-final-integration-001/build_final.py](../research/runs/20261011-FA108-measured-final-integration-001/build_final.py) | FA108=974967で6提出から標的単位のAIA全回答を選び、972302 MIAと結合する。 | 入口 |
| [research/selection_ensemble.py](../research/selection_ensemble.py) | 二つの選択性信号を順位で組み合わせ、回答/棄権を決める。 | 依存 |
| [research/trimmed_distance.py](../research/trimmed_distance.py) | 外れ成分を除く距離と近傍を求める。FA55の他C対比に使用。 | 依存 |
| [research/verify_rare_fixed_calibration.py](../research/verify_rare_fixed_calibration.py) | 最終校正の順位計算を独立実装し、保護行・tie・多重集合を検算する。 | 依存 |
