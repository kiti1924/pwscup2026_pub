"""条件付きカテゴリ＋無正則化 Cox を使う fully synthetic 防御の実験CLI。

公式の防御実装は変更せず、このファイルだけで次を実行する。

* ``aligned``: B 全体の連続値を Gaussian copula から再合成する。
* ``rare_aligned``: 公開 Mahalanobis 検出器で分けた2群をそれぞれ再合成し、
  整形後の検出件数を rejection sampling で B と一致させる。
* sex/smoking は B で学習した標準化ロジスティック回帰から条件付きに生成する。
* time/onset/death は採点側と同じ予測子変換を使う無正則化 cause-specific Cox
  から条件付きに生成する。

既定の ``--mode all`` は、2モード×seed 0..4 を生成し、各結果を公式 selfscore
（``--json``）で採点する。入力 B の行や record_id は出力へコピーしない。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

# ``reference`` は配布物のスクリプト置き場で、Python package ではないため、
# starter root を実行時にだけ sys.path へ加える。絶対パスはソースへ埋め込まない。
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_STARTER_ROOT = _WORKSPACE_ROOT / "pwscup2026-kit" / "starter"
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
if str(_STARTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT))
if str(_STARTER_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT / "src"))

from reference._common import (  # noqa: E402  (実行時の path 設定後に import)
    coerce_to_submission,
    dist_paths,
    load_cfg,
    read_b,
    write_c,
)
from pwscup2026.common import schema, submission_schema  # noqa: E402


_TIME_FLOOR_YEARS = 1.0 / 365.25
_UNIFORM_EPS = 1e-6
_COX_COLUMNS = [
    "age",
    "sex",
    "BMI",
    "SBP",
    "log_TG",
    "HDL",
    "log_ALT",
    "FPG",
    "smoking",
]
_STATIC_CLINICAL = ("BMI", "SBP", "TG", "HDL", "ALT", "FPG")
_MODES = ("aligned", "rare_aligned")


def _nearest_correlation(corr: np.ndarray) -> np.ndarray:
    """相関行列を有限・半正定値へ丸める。"""
    corr = np.asarray(corr, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if corr.ndim != 2:
        corr = np.eye(len(schema.CONTINUOUS), dtype=float)
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    fixed = eigvecs @ np.diag(np.clip(eigvals, 1e-8, None)) @ eigvecs.T
    scale = np.sqrt(np.clip(np.diag(fixed), 1e-12, None))
    fixed = fixed / np.outer(scale, scale)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def _pseudo_observations(x: np.ndarray) -> np.ndarray:
    """列ごとの平均順位を (0,1) の擬似一様値へ変換する。"""
    n = x.shape[0]
    if n == 0:
        raise ValueError("連続値の学習データが空です")
    ranks = np.apply_along_axis(
        lambda col: pd.Series(col).rank(method="average").to_numpy(dtype=float),
        0,
        x,
    )
    return ranks / (n + 1.0)


def _fit_correlation(df: pd.DataFrame) -> np.ndarray:
    """B または detector 群から copula の相関行列を fit する。"""
    x = df[list(schema.CONTINUOUS)].to_numpy(dtype=float)
    if len(df) < 2:
        return np.eye(x.shape[1], dtype=float)
    z = norm.ppf(np.clip(_pseudo_observations(x), _UNIFORM_EPS, 1.0 - _UNIFORM_EPS))
    corr = np.corrcoef(z, rowvar=False)
    return _nearest_correlation(corr)


def _resample_continuous(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """経験分位逆変換付き Gaussian copula で任意の n 行を再合成する。"""
    if n < 1:
        return pd.DataFrame(columns=list(schema.CONTINUOUS), dtype=float)
    cols = list(schema.CONTINUOUS)
    x = df[cols].to_numpy(dtype=float)
    corr = _fit_correlation(df)
    z_samp = rng.multivariate_normal(np.zeros(len(cols)), corr, size=int(n), check_valid="raise")
    u_samp = np.clip(norm.cdf(z_samp), 0.0, 1.0)

    out = np.empty((int(n), len(cols)), dtype=float)
    for j in range(len(cols)):
        sorted_vals = np.sort(x[:, j])
        # NumPy 1.x/2.x のどちらでも動く呼び出しにする。
        try:
            out[:, j] = np.quantile(sorted_vals, u_samp[:, j], method="linear")
        except TypeError:
            out[:, j] = np.quantile(sorted_vals, u_samp[:, j], interpolation="linear")
    return pd.DataFrame(out, columns=cols)


def _format_continuous(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """提出の値域・型に合わせる（希少検出より前に必ず適用する）。"""
    out = df.copy()
    age_lo, age_hi = (float(v) for v in cfg["population"]["age_range"])
    out["age"] = np.floor(np.clip(out["age"].to_numpy(dtype=float), age_lo, age_hi)).astype(int)
    ranges = cfg["schema"]["ranges"]
    for col in _STATIC_CLINICAL:
        lo, hi = (float(v) for v in ranges[col])
        out[col] = np.clip(out[col].to_numpy(dtype=float), lo, hi)
    return out[list(schema.CONTINUOUS)].reset_index(drop=True)


def _load_detector(dist: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """utility_ref.json の公開 μ,Σ,band.lo を読む。"""
    path = dist / "utility_ref.json"
    if not path.exists():
        raise FileNotFoundError(f"公開希少検出器の参照ファイルがありません: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    # 配布物の現行形式はトップレベル。入れ子形式も読み込めるが、値は同じ公開値のみ。
    src = obj.get("rare_detector", obj)
    try:
        mu = np.asarray(src["mu"], dtype=float)
        sigma = np.asarray(src["sigma"], dtype=float)
        threshold = float(src["band"]["lo"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"utility_ref.json の希少検出器形式が不正です: {exc}") from exc
    if mu.shape != (len(schema.CONTINUOUS),) or sigma.shape != (
        len(schema.CONTINUOUS),
        len(schema.CONTINUOUS),
    ):
        raise ValueError("utility_ref.json の μ/Σ の次元が schema.CONTINUOUS と一致しません")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or not np.isfinite(threshold):
        raise ValueError("utility_ref.json の希少検出器パラメータが非有限です")
    return mu, np.linalg.pinv(sigma), threshold


def _mahalanobis(df: pd.DataFrame, mu: np.ndarray, inv_sigma: np.ndarray) -> np.ndarray:
    x = df[list(schema.CONTINUOUS)].to_numpy(dtype=float)
    diff = x - mu
    m2 = np.einsum("ij,jk,ik->i", diff, inv_sigma, diff)
    return np.sqrt(np.clip(m2, 0.0, None))


def _detector_mask(
    continuous: pd.DataFrame, mu: np.ndarray, inv_sigma: np.ndarray, threshold: float
) -> np.ndarray:
    return _mahalanobis(continuous, mu, inv_sigma) >= threshold


def _sample_rejected_group(
    source: pd.DataFrame,
    target_n: int,
    want_rare: bool,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    mu: np.ndarray,
    inv_sigma: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """群別 copula を整形後に detector で rejection sampling する。"""
    if target_n == 0:
        return pd.DataFrame(columns=list(schema.CONTINUOUS), dtype=float)
    accepted: list[pd.DataFrame] = []
    n_accepted = 0
    n_proposed = 0
    max_proposed = max(10000, int(target_n) * 2000)
    while n_accepted < int(target_n):
        remaining = int(target_n) - n_accepted
        batch = max(128, remaining * 4)
        candidate = _format_continuous(_resample_continuous(source, batch, rng), cfg)
        detected = _detector_mask(candidate, mu, inv_sigma, threshold)
        keep = detected if want_rare else ~detected
        if bool(np.any(keep)):
            chosen = candidate.loc[keep].reset_index(drop=True)
            accepted.append(chosen)
            n_accepted += len(chosen)
        n_proposed += len(candidate)
        if n_proposed > max_proposed:
            label = "希少" if want_rare else "非希少"
            rate = n_accepted / max(n_proposed, 1)
            raise RuntimeError(
                f"{label}群の detector rejection sampling が収束しません "
                f"(accepted={n_accepted}/{target_n}, rate={rate:.4f})"
            )
    return pd.concat(accepted, ignore_index=True).iloc[:target_n].reset_index(drop=True)


def _synthesize_continuous(
    b: pd.DataFrame,
    cfg: dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    detector: tuple[np.ndarray, np.ndarray, float],
) -> pd.DataFrame:
    """モードに応じて全連続値を再合成する。"""
    mu, inv_sigma, threshold = detector
    if mode == "aligned":
        return _format_continuous(_resample_continuous(b, len(b), rng), cfg)
    if mode != "rare_aligned":
        raise ValueError(f"未知の mode: {mode}")

    b_static = _format_continuous(b, cfg)
    b_rare = _detector_mask(b_static, mu, inv_sigma, threshold)
    rare_source = b.loc[b_rare].reset_index(drop=True)
    common_source = b.loc[~b_rare].reset_index(drop=True)
    if len(rare_source) == 0 or len(common_source) == 0:
        raise ValueError("B の公開 detector 群の一方が空です")

    rare_out = _sample_rejected_group(
        rare_source,
        int(b_rare.sum()),
        True,
        cfg,
        rng,
        mu,
        inv_sigma,
        threshold,
    )
    common_out = _sample_rejected_group(
        common_source,
        int((~b_rare).sum()),
        False,
        cfg,
        rng,
        mu,
        inv_sigma,
        threshold,
    )
    out = pd.concat([rare_out, common_out], ignore_index=True)
    # 群の順序自体は意味を持たせず、record_idも元Bから引き継がない。
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


class _StandardizedLogit:
    """連続予測子を標準化して fit/predict するロジスティック回帰。"""

    def __init__(self, features: list[str]) -> None:
        self.features = list(features)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.model: LogisticRegression | None = None

    def fit(self, df: pd.DataFrame, target: str) -> "_StandardizedLogit":
        x = df[self.features].to_numpy(dtype=float)
        y = df[target].to_numpy(dtype=int)
        if np.unique(y).size != 2:
            raise ValueError(f"{target} のロジスティック回帰に2クラスがありません")
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0, ddof=0)
        self.scale_[self.scale_ <= 0.0] = 1.0
        z = (x - self.mean_) / self.scale_
        self.model = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            max_iter=2000,
            random_state=0,
        )
        self.model.fit(z, y)
        return self

    def predict_probability(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ロジスティック回帰が fit されていません")
        x = df[self.features].to_numpy(dtype=float)
        z = (x - self.mean_) / self.scale_
        proba = self.model.predict_proba(z)[:, 1]
        return np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)


def _sample_categorical(
    b: pd.DataFrame, continuous: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """sex→smoking を条件付き、prefecture を B 周辺から独立に合成する。"""
    sex_features = list(schema.CONTINUOUS)
    smoking_features = list(schema.CONTINUOUS) + ["sex"]
    sex_model = _StandardizedLogit(sex_features).fit(b, "sex")
    sex_p = sex_model.predict_probability(continuous)
    sex = rng.binomial(1, sex_p, size=len(continuous)).astype(int)

    smoking_frame = continuous.copy()
    smoking_frame["sex"] = sex
    smoking_model = _StandardizedLogit(smoking_features).fit(b, "smoking")
    smoking_p = smoking_model.predict_probability(smoking_frame)
    smoking = rng.binomial(1, smoking_p, size=len(continuous)).astype(int)

    values, counts = np.unique(b["prefecture"].to_numpy(dtype=object), return_counts=True)
    if len(values) == 0:
        raise ValueError("B の prefecture が空です")
    prefecture = rng.choice(values, size=len(continuous), p=counts / counts.sum())
    out = continuous.copy()
    out["sex"] = sex
    out["smoking"] = smoking
    out["prefecture"] = prefecture
    return out


def _cox_frame(df: pd.DataFrame, event_col: str | None = None) -> pd.DataFrame:
    """採点側と同じ Cox 予測子名・log 変換のフレームを返す。"""
    out = df.copy()
    out["log_TG"] = np.log(np.clip(out["TG"].to_numpy(dtype=float), 1e-6, None))
    out["log_ALT"] = np.log(np.clip(out["ALT"].to_numpy(dtype=float), 1e-6, None))
    if event_col is not None:
        out["_event"] = df[event_col].to_numpy(dtype=int)
        return out[_COX_COLUMNS + ["time", "_event"]]
    return out[_COX_COLUMNS]


def _fit_cause_cox(df: pd.DataFrame, event_col: str) -> tuple[Any, np.ndarray, np.ndarray]:
    """無正則化 cause-specific Cox を fit し、baseline cumulative hazard を返す。"""
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceWarning

    fit_df = _cox_frame(df, event_col)
    model = CoxPHFitter(penalizer=0.0)
    # 収束失敗を別手法へのフォールバックで隠さない。
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(fit_df, duration_col="time", event_col="_event", robust=False)
    beta = model.params_.to_numpy(dtype=float)
    se = model.standard_errors_.to_numpy(dtype=float)
    if not (np.isfinite(beta).all() and np.isfinite(se).all() and (se > 0).all()):
        raise RuntimeError(f"{event_col} 用 Cox の係数/標準誤差が退化しています")
    h0 = model.baseline_cumulative_hazard_.iloc[:, 0]
    times = h0.index.to_numpy(dtype=float)
    values = h0.to_numpy(dtype=float)
    if len(times) == 0 or len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"{event_col} 用 Cox の baseline cumulative hazard が不正です")
    return model, times, values


def _sample_event_time(
    partial_hazard: np.ndarray,
    times: np.ndarray,
    h0_values: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """公式合成器と同じ H0(t)*partial_hazard の逆変換サンプリング。"""
    target = -np.log(np.clip(u, 1e-12, 1.0)) / np.clip(partial_hazard, 1e-12, None)
    t = np.interp(target, h0_values, times, left=times[0], right=np.inf)
    return np.where(target > h0_values[-1], np.inf, t)


def _sample_outcomes(
    b: pd.DataFrame,
    out: pd.DataFrame,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """B fit の cause-specific Cox から onset/death/time を生成する。"""
    horizon = float(cfg["onset"]["horizon_years"])
    cph_onset, t_onset, h0_onset = _fit_cause_cox(b, "onset")
    cph_death, t_death, h0_death = _fit_cause_cox(b, "death")
    x = _cox_frame(out)
    onset_time = _sample_event_time(
        cph_onset.predict_partial_hazard(x).to_numpy(),
        t_onset,
        h0_onset,
        rng.uniform(size=len(out)),
    )
    death_time = _sample_event_time(
        cph_death.predict_partial_hazard(x).to_numpy(),
        t_death,
        h0_death,
        rng.uniform(size=len(out)),
    )

    censored = np.minimum(onset_time, death_time) > horizon
    onset_event = (~censored) & (onset_time <= death_time)
    death_event = (~censored) & (death_time < onset_time)
    onset = np.zeros(len(out), dtype=int)
    death = np.zeros(len(out), dtype=int)
    time = np.full(len(out), horizon, dtype=float)
    onset[onset_event] = 1
    time[onset_event] = onset_time[onset_event]
    death[death_event] = 1
    time[death_event] = death_time[death_event]
    result = out.copy()
    result["time"] = np.clip(time, _TIME_FLOOR_YEARS, horizon)
    result["onset"] = onset
    result["death"] = death
    return result


def synthesize(
    b: pd.DataFrame,
    cfg: dict[str, Any],
    mode: str,
    seed: int,
    detector: tuple[np.ndarray, np.ndarray, float],
) -> pd.DataFrame:
    """1 seed 分の完全合成 distributed dataframe を返す。"""
    if mode not in _MODES:
        raise ValueError(f"mode は {_MODES} のいずれかです: {mode}")
    rng = np.random.default_rng(int(seed))
    continuous = _synthesize_continuous(b, cfg, mode, rng, detector)
    static = _sample_categorical(b, continuous, rng)
    # record_id は元 B から移さず、提出前の内部整形専用に新規採番する。
    static.insert(0, "record_id", np.arange(len(static), dtype=int))
    out = _sample_outcomes(b, static, cfg, rng)
    out = coerce_to_submission(out, cfg)
    expected = list(submission_schema.DISTRIBUTED_COLUMNS)
    if list(out.columns) != expected or len(out) != len(b):
        raise RuntimeError("合成結果の列または行数が配布スキーマと一致しません")
    return out


def _score_one(
    csv_path: Path, dist: Path, score_path: Path
) -> tuple[dict[str, Any] | None, str | None]:
    """公式 selfscore --json を実行し、JSONを保存する。"""
    command = [
        sys.executable,
        "-m",
        "pwscup2026.kit.selfscore",
        str(csv_path),
        "--dist",
        str(dist),
        "--json",
    ]
    proc = subprocess.run(
        command,
        cwd=str(_STARTER_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "selfscore failed"
        score_path.write_text(
            json.dumps({"error": message}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return None, message
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        message = f"selfscore JSON が解釈できません: {exc}"
        score_path.write_text(
            json.dumps({"error": message}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return None, message
    score_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, None


def _compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    facets = payload.get("facets", {})
    detail = payload.get("detail", {})
    spec = detail.get("U_spec", {})
    rare = detail.get("U_rare", {})
    return {
        "U": payload.get("utility"),
        "U_gen": facets.get("U_gen"),
        "U_spec": facets.get("U_spec"),
        "U_rare": facets.get("U_rare"),
        "U_valid": facets.get("U_valid"),
        "io_part": spec.get("io_part"),
        "tstr_cal": spec.get("tstr_cal"),
        "c2st_cal": rare.get("c2st_cal"),
        "rare_detected_C": rare.get("rare_detected_C"),
    }


def _print_metric_header() -> None:
    print(
        "mode\tseed\tU\tU_gen\tU_spec\tU_rare\tU_valid\t"
        "io_part\ttstr_cal\tc2st_cal\trare_detected_C"
    )


def _print_metrics(mode: str, seed: int, metrics: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    print(
        "\t".join(
            [
                mode,
                str(seed),
                fmt(metrics.get("U")),
                fmt(metrics.get("U_gen")),
                fmt(metrics.get("U_spec")),
                fmt(metrics.get("U_rare")),
                fmt(metrics.get("U_valid")),
                fmt(metrics.get("io_part")),
                fmt(metrics.get("tstr_cal")),
                fmt(metrics.get("c2st_cal")),
                fmt(metrics.get("rare_detected_C")),
            ]
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="aligned / rare_aligned fully synthetic 実験")
    ap.add_argument("--dist", required=True, help="participant_data を展開したディレクトリ")
    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", *_MODES],
        help="単一モード、または all（既定: 2モード）",
    )
    ap.add_argument("--seeds", default="0,1,2,3,4", help="カンマ区切り seed（既定: 0..4）")
    ap.add_argument("--out-dir", default=None, help="C.csv と scores の出力先（既定: --dist）")
    args = ap.parse_args(argv)

    try:
        seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
        if not seeds:
            raise ValueError("seed が空です")
        dist = dist_paths(args.dist)
        cfg = load_cfg()
        b = read_b(dist)
        detector = _load_detector(dist)
        out_dir = Path(args.out_dir) if args.out_dir is not None else dist
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_dir = out_dir / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 — CLIの入力エラーは理由を示して終了
        print(f"ERROR: 初期化失敗 {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    modes = list(_MODES) if args.mode == "all" else [args.mode]
    failures: list[str] = []
    _print_metric_header()
    for mode in modes:
        for seed in seeds:
            csv_path = out_dir / f"C_{mode}_s{seed}.csv"
            score_path = scores_dir / f"{mode}_s{seed}.json"
            try:
                distributed = synthesize(b, cfg, mode, seed, detector)
                write_c(distributed, csv_path)
                payload, error = _score_one(csv_path, dist, score_path)
                if error is not None or payload is None:
                    message = error or "selfscore returned no payload"
                    failures.append(f"{mode}/seed={seed}: {message}")
                    print(f"{mode}\t{seed}\tERROR\t{message}")
                    continue
                _print_metrics(mode, seed, _compact_metrics(payload))
            except Exception as exc:  # noqa: BLE001 — 他seedを継続しつつ失敗を明示
                message = f"{type(exc).__name__}: {exc}"
                failures.append(f"{mode}/seed={seed}: {message}")
                print(f"{mode}\t{seed}\tERROR\t{message}")

    if failures:
        print("Errors:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
