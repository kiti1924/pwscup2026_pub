"""Build the predeclared S5 server probe without changing prior artifacts.

S5 is a fixed information probe.  It restores the frozen MIA source on every
target except the three predeclared rare-KDE columns, and evaluates exactly one
new AIA expert cell for each target.  The script performs all source and
destination checks before creating the final run directory and never submits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "attack_phase/src"))
sys.path.insert(0, str(ROOT / "attack_phase/scripts"))

from aia_stage1 import validate_f_aia_candidate  # noqa: E402
from build_attack_candidates import build_one  # noqa: E402
from build_s4_exploratory_probe import _aia_predict  # noqa: E402


TARGETS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 24, 26, 29)
RARE_KDE_TARGETS = (4, 19, 24)
RUN_NAME = "20260907_server_S5_adaptive_mux_v3"
ZIP_NAME = "PWSCup2026_attack_team10_20260907_server_S5_adaptive_mux_v3.zip"
ASSIGNMENT = {
    1: "weighted_modal_k25",
    2: "weighted_modal_window",
    3: "weighted_modal_window",
    4: "weighted_modal_window",
    5: "weighted_modal_window",
    6: "membership_gated",
    7: "weighted_modal_k25",
    8: "weighted_modal_k25",
    9: "weighted_modal_window",
    10: "weighted_modal_window",
    12: "weighted_modal_k25",
    13: "weighted_modal_k25",
    17: "stage3_huber_huber",
    18: "stage3_huber_huber",
    19: "membership_gated",
    20: "weighted_modal_window",
    22: "weighted_modal_k25",
    23: "weighted_modal_k25",
    24: "group_mean_corrected_donor",
    26: "any_event_cox_window_mode",
    29: "weighted_knn",
}

# Every global cell below is already server-tested.  The S4 entries are the
# exact target-specific cells recorded in the current-best matrix audit.
TESTED_CELLS = {
    (target, expert)
    for target in TARGETS
    for expert in ("official_1nn", "gated_abstain_donor_modal", "cell_modal_aggregate")
}
TESTED_CELLS.update(
    {
        (1, "stage3_l1_l2"), (2, "stage3_l1_l1"), (3, "weighted_modal_k25"),
        (4, "stage3_l1_l1"), (5, "weighted_modal_k25"), (6, "stage3_l1_l2"),
        (7, "weighted_modal_window"), (8, "membership_gated"),
        (9, "weighted_modal_k25"), (10, "stage3_l1_l2"),
        (12, "group_mean_corrected_donor"), (13, "stage3_l1_l1"),
        (17, "any_event_cox_window_mode"), (18, "weighted_knn"),
        (19, "weighted_knn"), (20, "weighted_modal_k25"),
        (22, "membership_gated"), (23, "weighted_modal_window"),
        (24, "stage3_huber_huber"), (26, "stage3_l1_l1"), (29, "membership_gated"),
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing")


def make_mia_candidate(base_path: Path, rare_path: Path) -> pd.DataFrame:
    """Compose 18 frozen columns and the three exact rare-KDE columns."""
    base = pd.read_csv(base_path)
    rare = pd.read_csv(rare_path)
    expected_columns = ["record_id", *[str(target) for target in TARGETS]]
    if list(base.columns) != expected_columns or list(rare.columns) != expected_columns:
        raise ValueError("MIA source columns do not match the official target schema")
    if not base["record_id"].equals(rare["record_id"]):
        raise ValueError("MIA source record_id order differs")
    result = base.copy()
    for target in RARE_KDE_TARGETS:
        result[str(target)] = rare[str(target)].to_numpy(copy=True)
    return result


def validate_mia_source_regression(candidate: pd.DataFrame, base_path: Path, rare_path: Path) -> None:
    """Assert every output target column is exactly from its intended source."""
    base = pd.read_csv(base_path)
    rare = pd.read_csv(rare_path)
    if not candidate["record_id"].equals(base["record_id"]):
        raise AssertionError("candidate record_id order changed")
    for target in TARGETS:
        source = rare if target in RARE_KDE_TARGETS else base
        if not candidate[str(target)].equals(source[str(target)]):
            raise AssertionError(f"MIA source regression failed for target {target}")


def validate_assignment_is_new() -> None:
    if tuple(ASSIGNMENT) != TARGETS:
        raise AssertionError("S5 assignment does not cover the official targets in order")
    duplicates = sorted((target, expert) for target, expert in ASSIGNMENT.items() if (target, expert) in TESTED_CELLS)
    if duplicates:
        raise AssertionError(f"S5 assignment repeats tested AIA cells: {duplicates}")


def build_aia_candidate(target_dir: Path, team_dir: Path) -> pd.DataFrame:
    background = pd.read_csv(team_dir / "A_bg.csv")
    rows: list[dict[str, object]] = []
    expected_ids: dict[int, list[int]] = {}
    for target in TARGETS:
        challenge = pd.read_csv(target_dir / f"aia_challenge_{target}.csv")
        challenge_ids = [int(value) for value in challenge["challenge_row_id"]]
        expected_ids[target] = challenge_ids
        target_c = pd.read_csv(target_dir / f"C_{target}.csv")
        values = _aia_predict(target_c, challenge, background, ASSIGNMENT[target])
        if len(values) != len(challenge) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"invalid AIA output for target {target}")
        rows.extend(
            {"target_k": target, "challenge_row_id": row_id, "time_hat": float(value)}
            for row_id, value in zip(challenge_ids, values)
        )
    result = pd.DataFrame(rows, columns=["target_k", "challenge_row_id", "time_hat"])
    validate_f_aia_candidate(result, expected_ids)
    for target in TARGETS:
        actual = result.loc[result["target_k"].eq(target), "challenge_row_id"].tolist()
        if actual != expected_ids[target]:
            raise AssertionError(f"AIA challenge order changed for target {target}")
    return result


def _safe_label(path: Path, roots: tuple[Path, ...]) -> str:
    for root in roots:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return "external-input"


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_assignment_is_new()
    for path, label in (
        (args.base_mia, "frozen MIA source"),
        (args.rare_mia, "rare-KDE MIA source"),
        (args.team_dir / "A_bg.csv", "background source"),
        (args.team_dir / "schema.json", "team schema"),
        (args.token, "Team 10 token"),
    ):
        _require_file(path, label)
    _require_dir(args.target_dir, "public target dist")
    output_root = ROOT / "attack_phase/outputs"
    submissions_root = ROOT / "attack_phase/submissions"
    run_dir = output_root / RUN_NAME
    zip_path = submissions_root / ZIP_NAME
    if run_dir.exists() or zip_path.exists():
        raise FileExistsError("S5 run or ZIP already exists; refusing overwrite")

    mia = make_mia_candidate(args.base_mia, args.rare_mia)
    validate_mia_source_regression(mia, args.base_mia, args.rare_mia)
    aia = build_aia_candidate(args.target_dir, args.team_dir)
    with tempfile.TemporaryDirectory(prefix="s5-build-input-", dir=ROOT) as temp_name:
        temp = Path(temp_name)
        mia_path = temp / "F_mia.csv"
        aia_path = temp / "F_aia.csv"
        mia.to_csv(mia_path, index=False)
        aia.to_csv(aia_path, index=False)
        metadata = build_one(
            run_name=RUN_NAME,
            method="S5 frozen MIA with Z3 rare-KDE + fixed AIA multiplex",
            mia_path=mia_path,
            aia_path=aia_path,
            other_aia_sha256="",
            output_root=output_root,
            submissions_root=submissions_root,
            token_path=args.token,
            dist_path=args.target_dir,
            seed=20260907,
            validate=True,
            zip_filename=ZIP_NAME,
        )

    run_metadata = {
        "run_name": RUN_NAME,
        "probe_name": "S5_adaptive_split_aia_multiplex",
        "purpose": "exploratory localization and target-wise AIA cell coverage",
        "comparison_probe": "S0_baseline",
        "hypothesis": "Restore frozen rare contribution from U6 while retaining rare-KDE only on the C-rare-zero prior group Z3; propagate S4 modal/window evidence through preregistered new AIA cells.",
        "base_commit": "4f8fd82",
        "seed": 20260907,
        "mia": {
            "frozen_source": _safe_label(args.base_mia, (ROOT,)),
            "rare_kde_source": "prior S3 rare-KDE candidate; only columns 4,19,24 copied",
            "frozen_targets": [target for target in TARGETS if target not in RARE_KDE_TARGETS],
            "rare_kde_targets": list(RARE_KDE_TARGETS),
            "source_regression": "all 21 target columns checked against intended source values and order",
        },
        "aia": {
            "assignment_seed": 20260907,
            "target_to_expert": {str(target): expert for target, expert in ASSIGNMENT.items()},
            "tested_cell_registry_asserted_new": True,
            "server_result_used_for_assignment": False,
        },
        "inputs": {
            "target_dist": _safe_label(args.target_dir, (ROOT,)),
            "team_inputs": "private Team 10 inputs (values not recorded)",
            "frozen_mia": "validated frozen_stage2_nn candidate",
            "rare_kde_mia": "validated S3 rare-KDE candidate",
        },
        "candidate_files": {"mia": "F_mia.csv", "aia": "F_aia.csv"},
        "candidate_sha256": {"mia": _sha256(run_dir / "F_mia.csv"), "aia": _sha256(run_dir / "F_aia.csv")},
        "zip_filename": ZIP_NAME,
        "zip_name_length": len(ZIP_NAME),
        "validator": metadata["validator"],
        "tests": "recorded after build in S5 stage report",
        "no_token_read_in_metadata": True,
        "no_submission": True,
    }
    (run_dir / "run.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run": str(run_dir), "zip": str(zip_path), "metadata": run_metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--team-dir", type=Path, required=True)
    parser.add_argument("--base-mia", type=Path, required=True)
    parser.add_argument("--rare-mia", type=Path, required=True)
    parser.add_argument("--token", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
