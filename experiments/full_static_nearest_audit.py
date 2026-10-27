"""Full-static nearest-record privacy audit for candidate C files.

This reproduces the disclosed fixed audit: A_bg moments for numeric z-scores,
categorical mismatch weight 2, B_practice as members, and 1,049 controls
sampled from A_bg with ``default_rng(0)``.  Only aggregate diagnostics are
written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_STARTER_ROOT = _WORKSPACE_ROOT / "pwscup2026-kit" / "starter"
for _import_root in (_WORKSPACE_ROOT, _STARTER_ROOT, _STARTER_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

try:
    from experiments.run_aligned_synth import _detector_mask, _load_detector
except ModuleNotFoundError:  # direct ``python experiments/...py`` invocation
    from run_aligned_synth import _detector_mask, _load_detector
from pwscup2026.common import schema
from pwscup2026.scoring import attack_score


NUMERIC = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
CATEGORICAL = ["sex", "smoking", "prefecture"]
EXACT_COLUMNS = [
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
    "smoking", "FPG", "time", "onset", "death",
]


def _rare_by_id(b: pd.DataFrame, b_self: pd.DataFrame) -> np.ndarray:
    if "record_id" in b.columns and "record_id" in b_self.columns:
        values = b_self.set_index("record_id")["is_rare"].reindex(b["record_id"])
        if values.isna().any():
            raise ValueError("B/B_self record_id alignment failed")
        return values.to_numpy(dtype=bool)
    return b_self["is_rare"].to_numpy(dtype=bool)


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


def _audit_one(
    c_path: Path,
    c: pd.DataFrame,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    member_rare: np.ndarray,
    control_rare: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    member_distance = _nearest_distance(members, c, mean, std)
    control_distance = _nearest_distance(controls, c, mean, std)
    confidence = -np.r_[member_distance, control_distance]
    truth = np.r_[np.ones(len(c), dtype=int), np.zeros(len(c), dtype=int)]
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


def main() -> int:
    parser = argparse.ArgumentParser(description="fixed full-static nearest privacy audit")
    parser.add_argument("--dist", required=True)
    parser.add_argument("--c", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dist = Path(args.dist).resolve()
    members = pd.read_csv(dist / "B_practice.csv")
    b_self = pd.read_csv(dist / "B_self.csv")
    a_bg = pd.read_csv(dist / "A_bg.csv")
    if len(members) != len(b_self):
        raise ValueError("B_practice and B_self length mismatch")
    rng = np.random.default_rng(0)
    controls = a_bg.iloc[rng.choice(len(a_bg), size=len(members), replace=False)].reset_index(drop=True)
    member_rare = _rare_by_id(members, b_self)
    mu, inv_sigma, threshold = _load_detector(dist)
    control_rare = _detector_mask(controls, mu, inv_sigma, threshold)
    mean = a_bg[NUMERIC].to_numpy(dtype=float).mean(axis=0)
    std = a_bg[NUMERIC].to_numpy(dtype=float).std(axis=0, ddof=0)
    std = np.where(std > 0.0, std, 1.0)

    results: list[dict[str, Any]] = []
    for value in args.c:
        path = Path(value).resolve()
        c = pd.read_csv(path)
        if len(c) != len(members):
            raise ValueError(f"{path}: row count mismatch")
        results.append(_audit_one(path, c, members, controls, member_rare, control_rare, mean, std))
        row = results[-1]
        print(
            f"{row['candidate']}: allTPR={row['all_tpr_at_fpr_0.01']:.12f} "
            f"rare={row['true_rare_tpr_at_fpr_0.01']:.12f} AUC={row['roc_auc']:.12f} "
            f"member<=.5={row['member_d_le_0.5']} control<=.5={row['control_d_le_0.5']}"
        )

    report = {
        "label": "full-static nearest audit",
        "definition": {
            "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "numeric_z_reference": "A_bg mean/std ddof=0",
            "categorical_mismatch_weight": 2,
            "query_score": "-min_C distance",
            "members": "B_practice",
            "controls": "default_rng(0).choice(A_bg, 1049, replace=False)",
            "low_fpr": 0.01,
        },
        "n_members": int(len(members)),
        "n_controls": int(len(controls)),
        "n_member_rare": int(member_rare.sum()),
        "n_control_rare": int(control_rare.sum()),
        "baseline_reference": {
            "all_tpr_at_fpr_0.01": 0.033365109628,
            "true_rare_tpr_at_fpr_0.01": 1.0 / 24.0,
            "roc_auc": 0.578820811686,
            "member_d_le_0.5": 4,
            "control_d_le_0.5": 0,
        },
        "candidate_requirements": {
            "true_rare_tpr_at_fpr_0.01_le": 1.0 / 24.0,
            "member_d_le_0.5_le": 4,
            "control_d_le_0.5_eq": 0,
        },
        "candidates": results,
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
