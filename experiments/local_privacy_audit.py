"""Local privacy proxy audit for submitted C candidates.

This is deliberately a *proxy* audit.  It reuses the kit's disclosed attack
and scoring functions, but it is not the CodaBench/final attack evaluation.
Only aggregate values are emitted; per-row confidence values never leave
memory and are not written to disk.

Example::

    uv run python experiments/local_privacy_audit.py \
        --dist ../../_private_team10 \
        --c ../../_private_team10/C_seed4.csv \
           ../../_private_team10/C_perturb_fixed_k005_s1.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_STARTER_ROOT = _WORKSPACE_ROOT / "pwscup2026-kit" / "starter"
for _import_root in (_WORKSPACE_ROOT, _STARTER_ROOT, _STARTER_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from pwscup2026.attack import aia as aia_mod
from pwscup2026.attack import mia as mia_mod
from pwscup2026.common import schema
from pwscup2026.common import config as config_mod
from pwscup2026.rare import outlier as outlier_mod
from pwscup2026.scoring import attack_score
from pwscup2026.scoring.result import ReportLevel


_AIA_PARAMS = {
    "qi_continuous": ["age", "SBP", "TG", "HDL", "ALT"],
    "block_keys": ["sex", "prefecture"],
    "k_nn_onset": 5,
}
_QI_COLUMNS = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]


def _first_existing(dist: Path, names: Iterable[str]) -> Path:
    for name in names:
        path = dist / name
        if path.is_file():
            return path
    names_text = ", ".join(names)
    raise FileNotFoundError(f"{dist} に次のいずれもありません: {names_text}")


def _load_inputs(dist: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load only the aggregate-audit inputs from a participant distribution."""
    b = pd.read_csv(_first_existing(dist, ("B.csv", "B_practice.csv")))
    b_self = pd.read_csv(dist / "B_self.csv")
    a_bg = pd.read_csv(dist / "A_bg.csv")
    cfg = config_mod.load_config(dist / "scoring_config.yaml")
    cfg.setdefault("aia", {}).update(_AIA_PARAMS)
    if len(a_bg) < len(b):
        raise ValueError(f"A_bg ({len(a_bg)}) がB ({len(b)})より少なく、対照を無交換抽出できません")
    if "is_rare" not in b_self:
        raise ValueError("B_self.csv に is_rare 列がありません")
    if len(b_self) != len(b):
        raise ValueError(f"B ({len(b)}) とB_self ({len(b_self)})の行数が一致しません")
    return b, b_self, a_bg, cfg


def _member_rare_mask(b: pd.DataFrame, b_self: pd.DataFrame) -> np.ndarray:
    if "record_id" in b.columns and "record_id" in b_self.columns:
        rare_by_id = b_self.set_index("record_id")["is_rare"]
        values = rare_by_id.reindex(b["record_id"])
        if values.isna().any():
            raise ValueError("Bのrecord_idをB_selfで希少ラベルへ対応付けられません")
        return values.to_numpy(dtype=bool)
    return b_self["is_rare"].to_numpy(dtype=bool)


def _rare_mask_from_reference(df: pd.DataFrame, dist: Path) -> np.ndarray:
    """Apply the disclosed Mahalanobis detector to A_bg controls."""
    reference: dict[str, Any] | None = None
    utility_path = dist / "utility_ref.json"
    if utility_path.is_file():
        reference = json.loads(utility_path.read_text(encoding="utf-8"))
    if reference is None:
        schema_path = dist / "schema.json"
        schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
        reference = schema_obj.get("rare_detector")
    if not isinstance(reference, dict):
        raise ValueError("utility_ref.json/schema.json に rare detector がありません")
    try:
        mu = np.asarray(reference["mu"], dtype=float)
        sigma = np.asarray(reference["sigma"], dtype=float)
        band = float(reference["band"]["lo"])
        cols = list(reference.get("columns", schema.CONTINUOUS))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("rare detector の形式が不正です") from exc
    distance = outlier_mod.mahalanobis(df, mu, sigma, cols)
    return distance >= band


def _sample_controls(a_bg: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(a_bg), size=n, replace=False)
    return a_bg.iloc[indices].reset_index(drop=True)


def _audit_one(
    c_path: Path,
    c: pd.DataFrame,
    b: pd.DataFrame,
    controls: pd.DataFrame,
    b_rare: np.ndarray,
    controls_rare: np.ndarray,
    a_bg: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run official proxy attacks, retaining only aggregate values."""
    # The candidate order is not included in the report.  It is only used to
    # evaluate the disclosed attacks in this local proxy experiment.
    combined = pd.concat([b, controls], ignore_index=True)
    truth_membership = np.r_[np.ones(len(b), dtype=int), np.zeros(len(controls), dtype=int)]
    rare_mask = np.r_[b_rare, controls_rare]
    mia_confidence = mia_mod.overlap_mia(c, a_bg, combined, cfg)
    mia_result = attack_score.score_mia(
        mia_confidence,
        truth_membership,
        cfg,
        level=ReportLevel.AGGREGATE,
        rare_mask=rare_mask,
    )
    mia_breakdown = mia_result.breakdown

    member_onset_conf, member_time_hat = aia_mod.aia_predict(c, b, a_bg, cfg)
    control_onset_conf, control_time_hat = aia_mod.aia_predict(c, controls, a_bg, cfg)
    aia_result = attack_score.score_aia(
        member_onset_conf,
        member_time_hat,
        b["onset"].to_numpy(dtype=int),
        b["time"].to_numpy(dtype=float),
        control_onset_conf,
        control_time_hat,
        controls["onset"].to_numpy(dtype=int),
        controls["time"].to_numpy(dtype=float),
        cfg,
        level=ReportLevel.AGGREGATE,
        member_rare_mask=b_rare,
        control_rare_mask=controls_rare,
    )
    aia_breakdown = aia_result.breakdown

    def finite(value: Any) -> float | None:
        value = float(value)
        return value if np.isfinite(value) else None

    mia_exposure = finite(mia_result.score)
    aia_score = finite(aia_result.score)
    return {
        "candidate": c_path.name,
        "n_rows": int(len(c)),
        "mia": {
            "exposure": mia_exposure,
            "one_minus_exposure_proxy": (None if mia_exposure is None else 1.0 - mia_exposure),
            "overall_tpr_at_low_fpr": finite(mia_breakdown.get("tpr_at_low_fpr")),
            "rare_tpr_at_low_fpr": finite(mia_breakdown.get("tpr_rare")),
            "auc": finite(mia_breakdown.get("auc")),
            "low_fpr": finite(mia_breakdown.get("low_fpr")),
        },
        "aia": {
            "attack_score": aia_score,
            "overall_r_time": finite(aia_breakdown.get("r_time_all")),
            "rare_worst_r_time": finite(aia_breakdown.get("r_time_rare")),
            "rare_worst_r_time_raw": finite(aia_breakdown.get("r_time_rare_raw")),
            "member_time_success_rate": finite(aia_breakdown.get("succ_time_member")),
            "control_time_success_rate": finite(aia_breakdown.get("succ_time_control")),
            "tau_time": finite(aia_breakdown.get("tau_time")),
        },
    }


def _expand_candidates(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(p) for p in glob.glob(value)]
        paths.extend(matches if matches else [Path(value)])
    # Preserve command-line order, while avoiding duplicate paths.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ValueError("候補Cが指定されていません")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="公式攻撃器によるローカル匿名性プロキシ監査")
    parser.add_argument("--dist", required=True, help="B/B_self/A_bg/config を含む配布ディレクトリ")
    parser.add_argument("--c", nargs="+", required=True, help="監査対象C.csv（複数指定・glob可）")
    parser.add_argument(
        "--out",
        default=None,
        help="集約JSONの保存先（既定: <dist>/scores/privacy_proxy.json）",
    )
    args = parser.parse_args()

    dist = Path(args.dist).resolve()
    b, b_self, a_bg, cfg = _load_inputs(dist)
    candidates = _expand_candidates(args.c)
    controls = _sample_controls(a_bg, len(b), seed=0)
    b_rare = _member_rare_mask(b, b_self)
    controls_rare = _rare_mask_from_reference(controls, dist)

    results: list[dict[str, Any]] = []
    for c_path in candidates:
        c = pd.read_csv(c_path)
        required = set(schema.COVARIATES + ["prefecture", "time", "onset", "death"])
        missing = sorted(required - set(c.columns))
        if missing:
            raise ValueError(f"{c_path}: 必須列がありません: {missing}")
        if len(c) != len(b):
            raise ValueError(f"{c_path}: 行数 {len(c)} がBの行数 {len(b)} と異なります")
        result = _audit_one(c_path, c, b, controls, b_rare, controls_rare, a_bg, cfg)
        results.append(result)
        mia = result["mia"]
        aia = result["aia"]
        print(
            f"{c_path.name}: MIA exposure={mia['exposure']:.6f} "
            f"(1-exposure={mia['one_minus_exposure_proxy']:.6f}), "
            f"AIA overall={aia['overall_r_time']:.6f}, "
            f"rare-worst={aia['rare_worst_r_time'] if aia['rare_worst_r_time'] is not None else 'NA'}"
        )

    output = Path(args.out).resolve() if args.out else dist / "scores" / "privacy_proxy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label": "local privacy proxies; not exact CodaBench/final attacks",
        "method": {
            "mia": "official overlap_mia + official score_mia",
            "aia": "official aia_predict + official score_aia",
            "controls": "len(B) rows sampled without replacement from A_bg",
            "control_sample_seed": 0,
            "aggregate_only": True,
        },
        "dist": str(dist),
        "n_members": int(len(b)),
        "n_controls": int(len(controls)),
        "n_member_rare": int(b_rare.sum()),
        "n_control_rare": int(controls_rare.sum()),
        "candidates": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"集約JSONを書き出しました: {output}")


if __name__ == "__main__":
    main()
