# 本戦・加工：G22、提出943786

## 最終Cの生成経路

```text
B14・分離した背景データ
  → background_copula.py / calibrated_copula.py
  → geographic_intervention.py       seed2 / gamma0 / G
  → overnight_refinement.py          性別相関修復 alpha=.67 / seed3301
  → compose_group_outcomes.py        合成転帰 i2
  → utility_slack.py                 幅1年 / seed3401
  → C_jitter_w1_s3401.csv             提出943786
```

G17g→G20では保存候補を単一提出のトレードオフとして再審査しました。G20は新しい生成器ではなく、同じ候補の採用判断の変更です。

## 各コードの意味

| コード | 何を変えるか |
|---|---|
| [background_partition.py](../research/background_partition.py) | 背景を生成用・攻撃参照用等に分離。確認条件を調整に混ぜないための土台。 |
| [background_copula.py](../research/background_copula.py) | 群ごとの分布と相関から新しい静的共変量を合成。 |
| [calibrated_copula.py](../research/calibrated_copula.py) | Bの集団相関とdetector群の発症率へ較正し、合成転帰を作る。 |
| [geographic_intervention.py](../research/geographic_intervention.py) | 県の割当てを変更。最終基底は`C_geo_s2_g0_G.csv`。 |
| [repair_sex_correlation.py](../research/repair_sex_correlation.py) / [overnight_refinement.py](../research/overnight_refinement.py) | 県×detector群内の性別人数を保ちつつ性別を交換し、sex/TGやsex/ALTの相関を修復。alpha=.67で49交換。 |
| [compose_group_outcomes.py](../research/compose_group_outcomes.py) | 修復後の静的列へ、別枝の`C_calcopula_s2_w1.0_g1_i2.csv`から合成転帰を組み合わせる。個人の真の転帰の転記ではない。 |
| [utility_slack.py](../research/utility_slack.py) | 既に合成されたイベント行だけにUniform(-1,1)年を加算。区間[.001,9.999]で反射し6桁丸め。 |

最終摂動は、`(onset==1 or death==1) and time<10`の198行だけです。静的10属性、onset/deathラベル、行政打切り時刻10は保持します。乱数は全1049行分をseed3401で引いてからイベント行に適用します。イベント行分だけ乱数を引くと同じ結果にはなりません。

## 提出マイルストーン

| ID | 段階 | 公式U |
|---:|---|---:|
| 942802 | [run_processing.py](../research/run_processing.py)のrare_aligned seed0基準 | .6763119999 |
| 943366 | G20：県割当て済み合成Cの採用 | .9134756611 |
| 943396 | G21-001：性別相関修復 | .9170339095 |
| 943413 | G21-002：合成転帰i2の組合せ | .9295762067 |
| **943786** | **G22：イベント時刻の幅1年摂動** | **.9295762067** |

別途G18で8種類の実装・サーバー診断候補と4件の取消履歴があります。[提出一覧](SUBMISSIONS.md)に分けて記録しています。G18は最終Cの前処理ではありません。

## 防御評価と限界

[adaptive_privacy.py](../research/adaptive_privacy.py)、[geography_free_attacks.py](../research/geography_free_attacks.py)、[confirm_overnight.py](../research/confirm_overnight.py)で候補と基準を同じ16攻撃で比較しました。探索2701..2703、凍結後のG22確認3101..3103。

G22はUを保ち、確認AIA全体最大.074931→.067461、標準化最大.036286→.027474。一方、希少raw最大は.163743→.178161と悪化しました。希少SE補正後の0を「被害なし」と解釈しません。背景再分割は独立した新しい個人による検証ではなく、最終Anonも未観測です。

最終CのSHA-256：`b37067cd9508864d7fb85ef5bf122235391b71a10b76c964328ef4f6cef6c4ba`。提出CSV/ZIPは公開版に含めていません。G22の生成コードhashは当時manifestと照合します。詳細は[証拠一覧](../provenance/evidence.json)。
