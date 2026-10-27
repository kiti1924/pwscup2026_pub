"""Compose rare-tilt micro-sweep candidates on a fixed safe outcome base.

The rare-tilt files are generated separately by ``run_rare_truth_tilt.py``
from the original pre-tilt base.  This script overlays only the seven
continuous columns on the public-detector rare rows onto the current safe
candidate, then selfscores and runs fixed full-static/privacy proxy gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_STARTER_ROOT = _WORKSPACE_ROOT / "pwscup2026-kit" / "starter"
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
if str(_STARTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT))
if str(_STARTER_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT / "src"))

from experiments.local_privacy_audit import (  # noqa: E402
    _audit_one as _privacy_audit_one,
    _load_inputs as _load_privacy_inputs,
    _member_rare_mask,
    _rare_mask_from_reference,
    _sample_controls,
)
from experiments.run_aligned_synth import (  # noqa: E402
    _detector_mask,
    _format_continuous,
    _load_detector,
    _score_one,
)
from reference._common import dist_paths, load_cfg, read_b, write_c  # noqa: E402
from pwscup2026.common import schema, submission_schema  # noqa: E402
from pwscup2026.scoring import attack_score  # noqa: E402


NUMERIC = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
CATEGORICAL = ["sex", "smoking", "prefecture"]
EXACT_COLUMNS = [
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
    "smoking", "FPG", "time", "onset", "death",
]
STATIC_COLUMNS = [*schema.CONTINUOUS, "sex", "smoking", "prefecture"]
OUTCOME_COLUMNS = ["time", "onset", "death"]
ALL_TPR_LIMIT = 0.033365109628
TRUE_RARE_TPR_LIMIT = 1.0 / 24.0
MIA_LIMIT = 0.3333334
UREARE_BASELINE = 0.9721569112733572


def _truth_rare_mask(b: pd.DataFrame, b_self: pd.DataFrame) -> np.ndarray:
    values = b_self.set_index("record_id")["is_rare"].reindex(b["record_id"])
    if values.isna().any():
        raise ValueError("B/B_self record_id alignment failed")
    return values.to_numpy(dtype=bool)


def _nearest_distance(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    q_num = (query[NUMERIC].to_numpy(dtype=float) - mean) / std
    r_num = (reference[NUMERIC].to_numpy(dtype=float) - mean) / std
    d2 = ((q_num[:, None, :] - r_num[None, :, :]) ** 2).sum(axis=2)
    for col in CATEGORICAL:
        d2 += 2.0 * (
            query[col].to_numpy()[:, None] != reference[col].to_numpy()[None, :]
        )
    return np.sqrt(np.min(d2, axis=1))


def _exact_match_count(query: pd.DataFrame, reference: pd.DataFrame) -> int:
    reference_rows = set(reference[EXACT_COLUMNS].itertuples(index=False, name=None))
    return int(sum(row in reference_rows for row in query[EXACT_COLUMNS].itertuples(index=False, name=None)))


def _full_static_one(
    c_path: Path,
    c: pd.DataFrame,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    member_rare: np.ndarray,
    control_rare: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    """Fixed audit with member/control truth aligned independently of C."""
    member_distance = _nearest_distance(members, c, mean, std)
    control_distance = _nearest_distance(controls, c, mean, std)
    confidence = -np.r_[member_distance, control_distance]
    truth = np.r_[np.ones(len(members), dtype=int), np.zeros(len(controls), dtype=int)]
    all_tpr, auc = attack_score._tpr_at_fpr(confidence, truth, 0.01)
    rare_mask = np.r_[member_rare, control_rare]
    rare_tpr, _ = attack_score._tpr_at_fpr(confidence[rare_mask], truth[rare_mask], 0.01)
    return {
        "candidate": c_path.name,
        "all_tpr_at_fpr_0.01": float(all_tpr),
        "true_rare_tpr_at_fpr_0.01": float(rare_tpr),
        "roc_auc": float(auc),
        "member_d_le_0.5": int(np.count_nonzero(member_distance <= 0.5)),
        "control_d_le_0.5": int(np.count_nonzero(control_distance <= 0.5)),
        "member_exact_d0": int(np.count_nonzero(member_distance == 0.0)),
        "control_exact_d0": int(np.count_nonzero(control_distance == 0.0)),
        "member_exact_match_13col": _exact_match_count(members, c),
        "control_exact_match_13col": _exact_match_count(controls, c),
        "member_min_distance": float(np.min(member_distance)),
        "control_min_distance": float(np.min(control_distance)),
    }


def _full_static_gate(row: dict[str, Any]) -> bool:
    return bool(
        row["all_tpr_at_fpr_0.01"] <= ALL_TPR_LIMIT
        and row["true_rare_tpr_at_fpr_0.01"] <= TRUE_RARE_TPR_LIMIT
        and row["member_d_le_0.5"] <= 4
        and row["control_d_le_0.5"] == 0
        and row["member_exact_d0"] == 0
        and row["control_exact_d0"] == 0
        and row["member_exact_match_13col"] == 0
        and row["control_exact_match_13col"] == 0
    )


def _ks_similarity(candidate: pd.DataFrame, c_rare: np.ndarray, b_true_rare: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for col in schema.CONTINUOUS:
        statistic = float(
            ks_2samp(
                candidate.loc[c_rare, col].to_numpy(dtype=float),
                b_true_rare[col].to_numpy(dtype=float),
                alternative="two-sided",
                method="auto",
            ).statistic
        )
        result[col] = 1.0 - statistic
    return result


def _assert_composite(
    base: pd.DataFrame,
    source: pd.DataFrame,
    candidate: pd.DataFrame,
    base_rare: np.ndarray,
    source_rare: np.ndarray,
) -> None:
    if not np.array_equal(base_rare, source_rare):
        raise RuntimeError("source public rare mask differs from safe-base mask")
    if not np.array_equal(base_rare, _detector_mask(candidate, *_DETECTOR)):
        raise RuntimeError("composite public rare mask changed")
    common = ~base_rare
    for col in STATIC_COLUMNS:
        if not np.array_equal(
            base.loc[common, col].to_numpy(), candidate.loc[common, col].to_numpy()
        ):
            raise RuntimeError(f"common static column changed: {col}")
    for col in CATEGORICAL + OUTCOME_COLUMNS:
        if not np.array_equal(base[col].to_numpy(), candidate[col].to_numpy()):
            raise RuntimeError(f"non-target column changed: {col}")
    for col in schema.CONTINUOUS:
        if not np.array_equal(
            candidate.loc[base_rare, col].to_numpy(), source.loc[base_rare, col].to_numpy()
        ):
            raise RuntimeError(f"rare overlay failed: {col}")


_DETECTOR: tuple[np.ndarray, np.ndarray, float]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rare-tilt micro-sweep composite and privacy gates")
    parser.add_argument("--dist", required=True)
    parser.add_argument("--safe-base", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--strengths", default="0.301,0.305,0.310,0.315,0.320,0.325,0.330,0.335,0.340,0.350,0.360,0.370,0.380,0.390")
    args = parser.parse_args(argv)

    global _DETECTOR
    dist = dist_paths(args.dist)
    cfg = load_cfg()
    b = read_b(dist)
    b_self = pd.read_csv(dist / "B_self.csv")
    truth_rare = _truth_rare_mask(b, b_self)
    b_true_rare = b.loc[truth_rare].reset_index(drop=True)
    _DETECTOR = _load_detector(dist)
    base_path = Path(args.safe_base).resolve()
    base = pd.read_csv(base_path)
    expected = list(submission_schema.SUBMITTED_C_COLUMNS)
    if list(base.columns) != expected or len(base) != len(b):
        raise ValueError("safe base has an unexpected shape or columns")
    base_rare = _detector_mask(_format_continuous(base, cfg), *_DETECTOR)
    if int(base_rare.sum()) != 61:
        raise ValueError(f"safe base public rare count is {int(base_rare.sum())}, expected 61")

    strengths = [float(x.strip()) for x in str(args.strengths).split(",") if x.strip()]
    if not strengths:
        raise ValueError("strengths is empty")
    generated_dir = Path(args.generated_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for strength in strengths:
        label = f"{strength:.3f}".replace(".", "p")
        source_path = generated_dir / f"C_truth_tilt_{label}.csv"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = pd.read_csv(source_path)
        if list(source.columns) != expected or len(source) != len(base):
            raise ValueError(f"generated source has an unexpected shape: {source_path}")
        source_rare = _detector_mask(_format_continuous(source, cfg), *_DETECTOR)
        candidate = base.copy(deep=True)
        candidate.loc[base_rare, list(schema.CONTINUOUS)] = source.loc[
            base_rare, list(schema.CONTINUOUS)
        ].to_numpy()
        _assert_composite(base, source, candidate, base_rare, source_rare)
        csv_path = out_dir / f"C_composite_tilt_{label}.csv"
        score_path = scores_dir / f"composite_tilt_{label}.json"
        write_c(candidate, csv_path)
        payload, error = _score_one(csv_path, dist, score_path)
        if payload is None or error is not None:
            raise RuntimeError(error or f"selfscore failed: {csv_path}")
        facets = payload["facets"]
        rare_detail = payload.get("detail", {}).get("U_rare", {})
        row = {
            "strength": strength,
            "csv": csv_path.name,
            "score_json": score_path.name,
            "utility": float(payload["utility"]),
            **{key: float(value) for key, value in facets.items()},
            "prof_cal": rare_detail.get("prof_cal"),
            "c2st_cal": rare_detail.get("c2st_cal"),
            "public_rare_n": int(base_rare.sum()),
            "ks_1_minus": _ks_similarity(candidate, base_rare, b_true_rare),
        }
        candidates.append(row)
        print(
            f"{strength:.3f}\tU={row['utility']:.9f}\tUgen={row['U_gen']:.9f}\t"
            f"Uspec={row['U_spec']:.9f}\tUrare={row['U_rare']:.9f}\tUvalid={row['U_valid']:.9f}"
        )

    members = pd.read_csv(dist / "B_practice.csv")
    a_bg = pd.read_csv(dist / "A_bg.csv")
    controls = a_bg.iloc[np.random.default_rng(0).choice(len(a_bg), len(members), replace=False)].reset_index(drop=True)
    member_rare = _truth_rare_mask(members, b_self)
    control_rare = _detector_mask(controls, *_DETECTOR)
    mean = a_bg[NUMERIC].to_numpy(dtype=float).mean(axis=0)
    std = a_bg[NUMERIC].to_numpy(dtype=float).std(axis=0, ddof=0)
    std = np.where(std > 0.0, std, 1.0)
    for row in candidates:
        path = out_dir / row["csv"]
        c = pd.read_csv(path)
        static = _full_static_one(path, c, members, controls, member_rare, control_rare, mean, std)
        row["full_static"] = static
        row["full_static_safe"] = _full_static_gate(static)

    # Run the same disclosed local proxy implementation only after the fixed
    # full-static and Urare gates have passed.
    privacy_b, privacy_self, privacy_a, privacy_cfg = _load_privacy_inputs(dist)
    privacy_controls = _sample_controls(privacy_a, len(privacy_b), seed=0)
    privacy_member_rare = _member_rare_mask(privacy_b, privacy_self)
    privacy_control_rare = _rare_mask_from_reference(privacy_controls, dist)
    for row in candidates:
        if not row["full_static_safe"] or row["U_rare"] <= UREARE_BASELINE:
            row["privacy_gate"] = {"run": False, "reason": "full_static_or_urare_gate"}
            continue
        path = out_dir / row["csv"]
        c = pd.read_csv(path)
        privacy = _privacy_audit_one(
            path, c, privacy_b, privacy_controls, privacy_member_rare,
            privacy_control_rare, privacy_a, privacy_cfg,
        )
        row["privacy"] = privacy
        mia = privacy["mia"]["exposure"]
        aia = privacy["aia"]["attack_score"]
        row["privacy_safe"] = bool(
            mia is not None and mia <= MIA_LIMIT and aia is not None and aia == 0.0
        )
        row["privacy_gate"] = {"run": True, "mia_limit": MIA_LIMIT, "aia_limit": 0.0}

    candidates.sort(key=lambda item: item["strength"])
    safe = [
        row for row in candidates
        if row.get("full_static_safe", False)
        and row.get("privacy_safe", False)
        and row["U_rare"] > UREARE_BASELINE
    ]
    safe.sort(key=lambda item: item["strength"])
    minimum_strength = safe[0] if safe else None
    maximum_urare = max(safe, key=lambda item: item["U_rare"]) if safe else None
    summary = {
        "safe_base": str(base_path),
        "generated_dir": str(generated_dir),
        "strengths": strengths,
        "public_rare_n": int(base_rare.sum()),
        "true_rare_n": int(truth_rare.sum()),
        "gates": {
            "all_tpr_at_fpr_0.01_max": ALL_TPR_LIMIT,
            "true_rare_tpr_at_fpr_0.01_max": TRUE_RARE_TPR_LIMIT,
            "member_d_le_0.5_max": 4,
            "control_d_le_0.5": 0,
            "exact_matches": 0,
            "urare_strictly_gt": UREARE_BASELINE,
            "mia_exposure_max": MIA_LIMIT,
            "aia_attack_score": 0.0,
        },
        "candidates": candidates,
        "selected_min_strength_improvement": minimum_strength,
        "selected_max_urare": maximum_urare,
    }
    summary_path = out_dir / "rare_tilt_micro_composite_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
