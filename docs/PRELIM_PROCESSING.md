# 予備戦・加工：最小の有用性指標を順番に改善する

最終採用は **896713**。提出物は1049行×13列のCで、プログラム自体をサーバーに提出したわけではありません。以下のコードは、そのCを作るまでに使った処理です。

## 考え方

有用性は `U=min(U_gen,U_spec,U_rare,U_valid)` なので、既に高い成分を伸ばしても総合Uは上がりません。最初の公式copula候補では生存分析と希少群が律速でした。そのため「全面再生成」から、律速成分を小さく直す処理の積み重ねへ移りました。

## 採用系列

| 順 | コード | 意味と採用設定 |
|---:|---|---|
| 1 | [run_aligned_synth.py](../experiments/run_aligned_synth.py)、[run_outcome_sweep.py](../experiments/run_outcome_sweep.py) | 多数群と希少領域を意識した共変量を生成。共変量seed4、初期転帰seed56。公開detectorでの希少領域を61行に保った。真の希少24人とは別。 |
| 2 | [run_calibrated_outcomes.py](../experiments/run_calibrated_outcomes.py)、[run_outcome_coordinate_search.py](../experiments/run_outcome_coordinate_search.py) | cause-specific Coxの係数・ハザードを較正し、time/onset/deathを合成。転帰seed15を基準に再較正。 |
| 3 | [run_rare_truth_tilt.py](../experiments/run_rare_truth_tilt.py) | 自Bに配布された希少ラベルを加工用集団統計に利用し、希少領域を補正。初期tilt=.300。 |
| 4 | [run_outcome_micro_refine.py](../experiments/run_outcome_micro_refine.py)、[run_common_quantile_micro.py](../experiments/run_common_quantile_micro.py) | 転帰の微較正と多数群ALTの分位点blend（alpha=.20）。 |
| 5 | [run_rare_tilt_micro_composite.py](../experiments/run_rare_tilt_micro_composite.py) | 複合tilt=.350で希少群指標を改善。その後さらに転帰を較正。 |
| 6 | [minimal_prefecture_sweep.py](../experiments/minimal_prefecture_sweep.py)、[prefecture_after_swap.py](../experiments/prefecture_after_swap.py)、[prefecture_after_round2.py](../experiments/prefecture_after_round2.py)、[extend_prefecture_on_candidate.py](../experiments/extend_prefecture_on_candidate.py) | 県の分布を少数変更。prefix10/12、21変更、26変更と、転帰交換の間に調整。 |
| 7 | [run_outcome_swap_hillclimb.py](../experiments/run_outcome_swap_hillclimb.py) | 転帰三つ組を交換してCox指標を改善。複数roundを実施。三つ組を維持して不整合を避ける。 |
| 8 | [run_minimal_ks_search.py](../experiments/run_minimal_ks_search.py)、[run_rare_ks_search.py](../experiments/run_rare_ks_search.py) | 多数群KSの少数セル、希少SBP・一般ALTを補正。 |
| 9 | [run_static_swap_probe.py](../experiments/run_static_swap_probe.py) | smoking交換とageの一セル補正で最終候補へ。 |

[local_privacy_audit.py](../experiments/local_privacy_audit.py)と[full_static_nearest_audit.py](../experiments/full_static_nearest_audit.py)は、各候補の参照攻撃と元レコードへの近さを検査する部品です。これらの低い代理リスクを、未知の攻撃への安全保証とはしていません。

## 最終値と限界

最終Cは元記録の `_private_team10/static_swap_after_rare/best.csv`。公開版には収録していません。

| 指標 | 最終ローカル値 |
|---|---:|
| U / U_gen | 0.978360343 |
| U_spec | 0.978483158 |
| U_rare | 0.981066700 |
| U_valid | 1.000000000 |

公式最終集計のUは0.9783603432、Anon/総合は0.6791891346で10位。加工画面の参照攻撃protectionと最終Anonは別です。

このフェーズはGit導入前に開始しています。採用操作は当時の加工総合レポートと保存候補から事後整理しました。全操作の厳密な適用順・探索履歴・提出時のファイルハッシュを一つの凍結runとしては確定できません。896581（表示U=.9083）は提出を確認していますが、生成コードとの一意な対応は未確定です。収録コードを「全提出のバイト完全再現」とは表現しません。

根拠：元記録 `docs/PWSCUP2026_prelim_processing_comprehensive_report.md`、`analysis/prelim_review_20260924/REPORT.md`、同`outputs/submission_ledger.csv`。根拠ファイルの版とhashは[証拠一覧](../provenance/evidence.json)に記録しています。
