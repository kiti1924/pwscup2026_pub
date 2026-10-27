# 本戦・攻撃：8提出から最終975433へ

最終版は一つの新モデルではありません。**S3を基準に非希少MIAだけを補正した回答**と、**測定済み6提出から標的ごとに選んだAIA回答**を組み合わせています。

## 8件の提出とコード

| # | ID | 生成コード | 提出した内容 | 公式総合 |
|---:|---:|---|---|---:|
| 1 | 965017 | [build_final_attack_baseline.py](../research/build_final_attack_baseline.py)、[final_attack_api.py](../research/final_attack_api.py) | full_static_nn MIA＋公式1NN AIA | .1138766319 |
| 2 | 971407 | [prepare_mechanism_comparison.py](../research/prepare_mechanism_comparison.py)、[final_attack_extensions.py](../research/final_attack_extensions.py)、[mechanism_components.py](../research/mechanism_components.py)、[prefecture_weight_attacks.py](../research/prefecture_weight_attacks.py) | 背景一致除去＋他C対比MIA（標的9はfull_static_nn維持）、県をsoft条件にしたraw_soft_pref1 AIA | .1406393926 |
| 3 | 972302 | [structural_attack.py](../research/claude/structural_attack.py)、[build_candidates.py](../research/claude/build_candidates.py) | CL01のS3。県の事前確率と希少スコア枠内並替えを加えたMIA、下位30%を棄権に変えたAIA | .1203（丸め） |
| 4 | 974256 | [fa55_public_api.py](../research/fa55_public_api.py)、[run_fa55_public_api.py](../research/run_fa55_public_api.py) | 972302のMIAを固定、L1背景比＋trim距離他C比の選択アンサンブルAIA | .0984（丸め） |
| 5 | 974917 | [run_main_public_contrasts.py](../research/run_main_public_contrasts.py)、[main_attack_contrasts.py](../research/main_attack_contrasts.py)、[mia_rare_slots.py](../research/mia_rare_slots.py) | 探索A：age10選択AIA＋FA98希少枠並替えMIA | .0759903110 |
| 6 | 974934 | 同上＋[conditional_null_mia.py](../research/conditional_null_mia.py)、[run_main_transfer.py](../research/runs/20261010-FA106-cl06-main-transfer-001/run_main_transfer.py) | 探索B：ungated rank2 AIA＋FA106条件付き校正MIA | .0714232807 |
| 7 | 974967 | [build_final.py](../research/runs/20261011-FA108-measured-final-integration-001/build_final.py)、[package_public_attack.py](../research/package_public_attack.py) | 6提出の標的別最良AIA全回答を統合、MIAは972302全行列 | .1615815185 |
| 8 | **975433** | [build_rare_fixed_main.py](../research/build_rare_fixed_main.py)、[rare_fixed_calibration.py](../research/rare_fixed_calibration.py) | **974967のAIAを固定、非希少MIAだけ校正** | **.1684450253** |

`build_candidates.py`には未提出のS4/S5も残っています。表の#3で実際に提出したのはS3だけです。FA98の旧ローカルscreen不合格を、#5の構築・送信によって合格へ書き換えていません。

## MIAの意味

初回は静的10属性でCに近い人を高く評価します。971407では、背景と一致するC行を取り除き、標的Cへの距離を他Cへの距離と対比しました。背景再利用による偽陽性を抑える意図です。

S3は自身のBに配布された県・`is_rare`を使います。県の事前確率を`.087`のlog priorで加え、希少行の順序を別のpeer距離信号で並べ替えます。**自Bのis_rareを使うことと、相手Bへの所属真値を使うことは別です。** 公開版でも、最終処理に必要な希少maskを任意に推定・捏造していません。

FA106は、同じ性別・県・年齢帯や近傍の背景人を非member参照にし、距離信号を条件付きで較正します。最終FA114ではこの優先順位を**非希少かつ非除外行だけ**へ適用。972302の希少・除外行のスコア、各列のスコア値の多重集合、元のtieを保持します。tieを壊すならその標的全体を元に戻します。[verify_rare_fixed_calibration.py](../research/verify_rare_fixed_calibration.py)は別計算で順序を照合する検査です。

## 最終AIAの出所

各標的の326行全回答を丸ごと選択しました。行ごとに正解を見て都合の良い時刻へ直したものではありません。

| 出所提出 | 最終版で利用した標的 |
|---:|---|
| 965017 | 11 |
| 971407 | 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 25 |
| 972302 | 9, 17, 26, 27 |
| 974256 | 22, 28 |
| 974917 | 10 |
| 974934 | 0, 6, 13 |

機械可読版：[final_aia_routing.json](../provenance/final_aia_routing.json)。6提出の公式標的別AIA得点を比較し、同点では既存の出所を保持しました。これは**既に測った同じ標的へのフィードバック最適化**です。未知標的に一般化するrouterの検証結果とは異なります。単独で総合が低い#4〜6にも、特定標的では有効な回答がありました。

## 最終結果

975433はFinished・掲載を記録。974967との差は+0.0068635069。AIA全22標的と希少MIAは公式結果でも不変でした。MIA全体平均は.0456047656→.0507847708へ改善。最終攻撃力は.1684450253、21/23位です。

表示MIA全体・希少、AIA全体・希少を単純加算して総合にはしません。標的ごとの群の集約とfloorを経た公式得点を用います。公開版では当時の結果記録を整理しただけで、競技への新規送信は行いません。
