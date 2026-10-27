# 予備戦・攻撃：実採用はS0

5回の実提出のうち、最終採用は **916932（S0）**。MIAは`frozen_stage2_nn`、AIAは公式1NNでした。ローカルで良かったdonor-modalや、後から作った統合候補を最終提出済みと混同しないことが重要です。

## S0の生成経路

```text
run_mia_stage2_experiment.py → frozen_stage2_nn の F_mia
run_aia_stage2_experiment.py → official_1nn の F_aia_baseline
build_server_probe.py → S0_baseline の標的ルーティング
build_attack_candidates.py → ZIP構築・公式validator
```

- [mia_low_fpr.py](../attack_phase/src/mia_low_fpr.py) / [mia_stage2.py](../attack_phase/src/mia_stage2.py)：性別×県のブロック内でCへの近傍距離を参照距離と比較し、局所密度の差を補正するMIA。参照1800/較正800の単一分割による固定候補。
- [aia_stage1.py](../attack_phase/src/aia_stage1.py)：公開QIからCの近い行を見つけ、その時刻を返す公式1NN等を提供。
- [target_router.py](../attack_phase/src/target_router.py)：MIAを標的列単位、AIAを`target_k`と`challenge_row_id`で選択・結合する。

## 提出した他候補

| ID | 呼称 | 使ったコード・差分 | 表示総合 |
|---:|---|---|---:|
| 916932 | S0 | 固定Block NN＋公式1NN | .2314 |
| 916951 | S1-dual | [mia_stage4.py](../attack_phase/src/mia_stage4.py)のNN75%/ExtraTrees25%＋[aia_stage2.py](../attack_phase/src/aia_stage2.py)のgate付きdonor-modal | .1909 |
| 916998 | S3 | [make_s3_mia_candidate.py](../attack_phase/scripts/make_s3_mia_candidate.py)で標的1/2を固定NN、他19を希少KDE。 [build_s3_orthogonal_inputs.py](../attack_phase/scripts/build_s3_orthogonal_inputs.py)でcell-modal AIA | .1762 |
| 917054 | S4 | [build_s4_exploratory_probe.py](../attack_phase/scripts/build_s4_exploratory_probe.py)：固定NN12標的＋KDE9標的、AIA expertを標的別に割当て | .1980 |
| 917072 | S5 | [build_s5_server_probe.py](../attack_phase/scripts/build_s5_server_probe.py)：KDE対象を4/19/24に限定、別のAIA割当て | .1594 |

S1-current（固定MIA＋donor-modal）、S2の標的別統合、S6のcomplement/rare-overlayは候補として存在しましたが、実提出5件には含まれません。ビルダーに別候補の枝が残っている場合もあります。

## 結果の読み方

S0の画面値.2313664409は自己チームを含む21標的の当時集計。最終は自己除外の20標的で0.2429347629、攻撃13位です。事後に公開正解で再採点した未提出統合の.275642を公式成績として扱いません。

根拠：元記録`attack_phase/docs/SERVER_EVAL_LOG_20260906.md`、`attack_phase/config/server_probe_manifest.json`、`analysis/prelim_review_20260924/REPORT.md`、提出台帳。公式キットは`prelim-attack-20260825`。本戦の採点器で同じ値になるとは限りません。
