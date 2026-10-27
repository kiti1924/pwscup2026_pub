"""rare_aligned の静的共変量を固定した outcome-only seed sweep。

covariate seed=0 で ``rare_aligned`` の連続値・sex・smoking・prefecture を一度だけ
完全合成し、その静的表へ outcome seed=0..9 の cause-specific Cox サンプリングだけを
適用する。生成・整形・公式 selfscore は ``run_aligned_synth.py`` の実装を再利用する。

既定では ``--dist`` の直下へ ``C_rare_cov{covariate_seed}_o{seed}.csv``、``scores/``
へ公式 JSON を書き出す。covariate_seed=0 の既存命名はそのまま維持する。B の行・
record_id・token は出力へコピーしない。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ``reference`` と experiments は配布スターター内のスクリプト置き場であるため、
# 実行時に starter root だけを import path へ追加する。環境固有の絶対パスは埋め込まない。
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_STARTER_ROOT = _WORKSPACE_ROOT / "pwscup2026-kit" / "starter"
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
if str(_STARTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT))
if str(_STARTER_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT / "src"))

from experiments.run_aligned_synth import (  # noqa: E402
    _MODES,
    _compact_metrics,
    _load_detector,
    _sample_categorical,
    _sample_outcomes,
    _score_one,
    _synthesize_continuous,
)
from reference._common import (  # noqa: E402
    coerce_to_submission,
    dist_paths,
    load_cfg,
    read_b,
    write_c,
)
from pwscup2026.common import schema, submission_schema  # noqa: E402


def _build_static(
    b: Any,
    cfg: dict[str, Any],
    detector: tuple[np.ndarray, np.ndarray, float],
    covariate_seed: int = 0,
) -> Any:
    """rare_aligned の静的共変量・カテゴリを一度だけ構築する。"""
    rng = np.random.default_rng(int(covariate_seed))
    continuous = _synthesize_continuous(b, cfg, "rare_aligned", rng, detector)
    static = _sample_categorical(b, continuous, rng)
    # 元Bの record_id は参照せず、提出前の内部整形専用に新規採番する。
    static.insert(0, "record_id", np.arange(len(static), dtype=int))
    expected_static = ["record_id", *schema.CONTINUOUS, "sex", "smoking", "prefecture"]
    if list(static.columns) != expected_static or len(static) != len(b):
        raise RuntimeError("固定静的表の列または行数が想定と一致しません")
    return static


def _print_header() -> None:
    print(
        "outcome_seed\tU\tU_gen\tU_spec\tU_rare\tU_valid\t"
        "io_part\ttstr_cal\tmass_star\tprof_cal\tonset_cal\t"
        "c2st_cal\trare_detected_C"
    )


def _print_metrics(seed: int, payload: dict[str, Any]) -> None:
    metrics = _compact_metrics(payload)
    rare = payload.get("detail", {}).get("U_rare", {})
    fields = {
        **metrics,
        "mass_star": rare.get("mass_star"),
        "prof_cal": rare.get("prof_cal"),
        "onset_cal": rare.get("onset_cal"),
    }

    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    print(
        "\t".join(
            [
                str(seed),
                fmt(fields.get("U")),
                fmt(fields.get("U_gen")),
                fmt(fields.get("U_spec")),
                fmt(fields.get("U_rare")),
                fmt(fields.get("U_valid")),
                fmt(fields.get("io_part")),
                fmt(fields.get("tstr_cal")),
                fmt(fields.get("mass_star")),
                fmt(fields.get("prof_cal")),
                fmt(fields.get("onset_cal")),
                fmt(fields.get("c2st_cal")),
                fmt(fields.get("rare_detected_C")),
            ]
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="rare_aligned outcome-only seed sweep")
    ap.add_argument("--dist", required=True, help="participant_data を展開したディレクトリ")
    ap.add_argument("--covariate-seed", type=int, default=0, help="静的共変量のseed（既定: 0）")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="outcome seed の列")
    ap.add_argument("--out-dir", default=None, help="CSV/scores の出力先（既定: --dist）")
    args = ap.parse_args(argv)

    try:
        seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
        if not seeds:
            raise ValueError("outcome seed が空です")
        dist = dist_paths(args.dist)
        cfg = load_cfg()
        b = read_b(dist)
        detector = _load_detector(dist)
        out_dir = Path(args.out_dir) if args.out_dir is not None else dist
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_dir = out_dir / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        # ここだけで static を作り、各 outcome seed では決して再生成しない。
        static = _build_static(b, cfg, detector, covariate_seed=args.covariate_seed)
    except Exception as exc:  # noqa: BLE001 — 入力/fitエラーを理由付きで返す
        print(f"ERROR: 初期化失敗 {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    failures: list[str] = []
    _print_header()
    for seed in seeds:
        csv_path = out_dir / f"C_rare_cov{args.covariate_seed}_o{seed}.csv"
        score_path = scores_dir / f"rare_cov{args.covariate_seed}_o{seed}.json"
        try:
            # _sample_outcomes は入力静的表を変更しないが、seed間の独立性を明示するため
            # 毎回 fresh Generator と deep copy を渡す。
            outcome_rng = np.random.default_rng(int(seed))
            distributed = _sample_outcomes(b, static.copy(deep=True), cfg, outcome_rng)
            distributed = coerce_to_submission(distributed, cfg)
            expected = list(submission_schema.DISTRIBUTED_COLUMNS)
            if list(distributed.columns) != expected or len(distributed) != len(b):
                raise RuntimeError("outcome-only結果の列または行数が配布スキーマと一致しません")
            write_c(distributed, csv_path)

            payload, error = _score_one(csv_path, dist, score_path)
            if error is not None or payload is None:
                message = error or "selfscore returned no payload"
                failures.append(f"outcome_seed={seed}: {message}")
                print(f"{seed}\tERROR\t{message}")
                continue
            _print_metrics(seed, payload)
        except Exception as exc:  # noqa: BLE001 — seed間の結果を継続して表示
            message = f"{type(exc).__name__}: {exc}"
            failures.append(f"outcome_seed={seed}: {message}")
            print(f"{seed}\tERROR\t{message}")

    if failures:
        print("Errors:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
