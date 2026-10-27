# 再現手順と検証範囲

## まず動かせる範囲

Python 3.12を基準に、リポジトリのルートで実行します。公開版入口はnumpy/pandas/scipyだけで動き、公式キットや競技トークンを読みません。

```sh
python -m venv .venv
# 作成した仮想環境を有効化してから
python -m pip install -r research/final_attack_requirements.txt
python -m unittest discover -s tests -v
python tools/verify_archive.py
python -m pwscup_public --help
```

`research/final_attack_requirements.txt`と`research/fa27_requirements.txt`は当時の本戦攻撃用の固定指定です。ルートの`requirements.txt`は加工用lifelines等も加えた実行環境案で、過去4フェーズすべてに共通する完全lockではありません。特に予備戦加工では公式キットの依存とAPIの版差を確認してください。

## G22の最終時刻摂動

入力は**G21-002で作成した摂動前の合成C**。元Bを入力する操作ではありません。

```sh
python -m pwscup_public jitter --input inputs/C_sex_group_i2.csv --output outputs/C_jitter_w1_s3401.csv
```

固定seed3401、既定幅1年で、イベント行だけを変えます。元関数[utility_slack.transform](../research/utility_slack.py)の本体を変更せず[公開kernel](../pwscup_public/kernels.py)に切り出しました。全行数分の乱数生成、反射処理、6桁丸めも同じです。

ただしこれは**最終段階だけ**の再現です。入力Cの生成には背景分割→copula較正→県変更→性別相関修復→合成転帰i2が必要で、[本戦加工](FINAL_PROCESSING.md)を参照してください。

## 最終MIAの非希少校正

```sh
python -m pwscup_public calibrate-mia --baseline inputs/972302_F_mia.csv --evidence inputs/FA106_blend_e0.csv --masks inputs/own_masks.csv --output outputs/FA114_F_mia.csv
```

入力の意味：

| ファイル | 列・内容 |
|---|---|
| baseline | `record_id`と各標的列。972302のMIA数値行列。 |
| evidence | 同じrecord_id・標的列。**FA106の条件付き校正済みblend_e0**。生距離・最終rank・別モデル出力を代入すると元の処理にはならない。 |
| masks | `record_id,rare,excluded,home`。全flagは0/1。rareは自Bの配布ラベル、excludedは既知自己非共有照合、homeは自Bの最多県。 |

record_idで整列してから、`priority`と`reassign`を当時と同じ本体で計算します。希少/除外行、スコア値の多重集合、元のtieを保持します。相手Bへの所属正解をmaskへ渡してはいけません。

FA106のevidenceを得るには、[conditional_null_mia.py](../research/conditional_null_mia.py)と[run_main_transfer.py](../research/runs/20261010-FA106-cl06-main-transfer-001/run_main_transfer.py)の処理が前段に必要です。公開入口はそのモデル探索や過去配列の復元を自動化していません。CSVの数値演算を再現する入口であり、当時と異なるpandasによるCSVのバイト列まで保証するものではありません。

## 最終AIAの標的別統合

入力は965017/971407/972302/974256/974917/974934の回答CSV。source指定JSONの値は、そのJSONがあるディレクトリからの相対パスです。

```json
{
  "965017": "965017_F_aia.csv",
  "971407": "971407_F_aia.csv",
  "972302": "972302_F_aia.csv",
  "974256": "974256_F_aia.csv",
  "974917": "974917_F_aia.csv",
  "974934": "974934_F_aia.csv"
}
```

```sh
python -m pwscup_public integrate-aia --sources inputs/sources.json --output outputs/final_F_aia.csv
```

既定の[routing](../provenance/final_aia_routing.json)で22標的を選択します。各標的326行、challengeキー重複なし、時刻が有限・非負であることを検査します。時刻は文字列として扱い、選択元の小数表記を保持します。任意の新しい標的への学習・選択ではありません。

上記3つの公開入口は既存出力を上書きせず、ZIP生成・トークン読込・競技送信をしません。

## 当時の研究用コードを読む・動かす場合

原コードの相対パスは保存しましたが、中間成果物は含めていません。`--help`だけでもimport先が当時のデータを参照するスクリプトがあります。入力や保存先の確認なしに一括実行しないでください。

公式キットは次の公開リポジトリから取得します。

```sh
git clone https://github.com/pwscup/pwscup2026-kit.git pwscup2026-kit
```

| 対象 | 対応する公式版 |
|---|---|
| 予備戦攻撃・予備戦結果再生 | `prelim-attack-20260825`（元記録4477277） |
| 本戦加工 | `main-process-20260912`（元記録3dd71bd。ローカルc67127eとはtree同一との当時確認） |
| 本戦攻撃 | `main-attack-20260928`（a490490） |

各版の公式ルール・scorer・設定を組にして使います。本戦版で予備戦値を再採点して同じになるとは限りません。公式キットの`starter`依存をその版の手順で導入し、必要に応じて`starter`と`starter/src`をPythonのモジュール探索パスに設定します。予備戦加工の全探索時点のキット版・環境は完全には確定していません。

非公開入力の元配置は予備戦`_private_team10/`、本戦`_private_team14/`、本戦公開攻撃配布`_private_team14/main_attack_20260928/files/`でした。これらは同梱していません。コード中のパス名は入力要件を説明するために残しています。別途許可された入力を持つ場合だけ用意し、ラウンドを混ぜないでください。

歴史的ビルダーは事前計画、入力hash、元の中間配列、元リポジトリのGit版、公式kitのHEAD、締切前であることを検査する場合があります。特に[build_rare_fixed_main.py](../research/build_rare_fixed_main.py)には当時の締切検査があります。これらは記録として保存し、日時を偽装して再実行する手順にはしていません。主要演算の再利用には上記の公開入口を使います。

## 公開版で検証すること

- 原コード97件の構文、収録ファイルのhash、import依存と解説リンク。
- G22・FA106・FA114の主要コード5件と当時hashの照合。
- 公開kernelと原関数の構文木一致。
- 人工データで、時刻摂動の静的/打切り保持、MIAの保護行/多重集合/tie保持、行順不変性、AIA標的全回答の選択と文字列保持。
- 配布ファイルや競技トークンを公開対象に入れていないこと。

全4フェーズの実データ採点、全過去環境の再構築、全提出ZIPのバイト完全再生成は、公開版単独の検証範囲ではありません。検査結果は[VALIDATION.md](VALIDATION.md)に記録します。
