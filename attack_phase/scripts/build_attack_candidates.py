"""Freeze the selected MIA/AIA candidates into validator-ready attack packages.

The script intentionally reads ``token.txt`` only while constructing the ZIP.
Neither the token nor a token digest is written to ``run.json`` or any tracked
file.  Candidate CSVs and ZIPs live under ignored directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIA = ROOT / "attack_phase/outputs/20260828_mia_stage2_final/F_mia_candidate.csv"
DEFAULT_AIA_BASELINE = ROOT / "attack_phase/outputs/20260828_aia_stage2_final_v4/F_aia_baseline.csv"
DEFAULT_AIA_CHALLENGER = ROOT / "attack_phase/outputs/20260828_aia_stage2_final_v4/F_aia_challenger.csv"
DEFAULT_DIST = ROOT / "attack_phase/data/public/prelim_attack_20260825"
DEFAULT_TOKEN = ROOT / "_private_team10/token.txt"
DEFAULT_PYTHONPATH = ROOT / "pwscup2026-kit/starter/src"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} がありません: {path.as_posix()}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} がありません: {path.as_posix()}")


def _repo_label(path: Path) -> str:
    """Keep run metadata portable and free of machine-specific absolute paths."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _copy_candidates(run_dir: Path, mia: Path, aia: Path) -> None:
    shutil.copyfile(mia, run_dir / "F_mia.csv")
    shutil.copyfile(aia, run_dir / "F_aia.csv")


def _validate_zip(zip_path: Path, dist: Path) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pwscup2026.kit.validate",
        str(zip_path),
        "--kind",
        "attack",
        "--dist",
        str(dist),
    ]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(DEFAULT_PYTHONPATH)
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "command": ["python", "-m", "pwscup2026.kit.validate", "<zip>", "--kind", "attack", "--dist", str(dist)],
        "stdout_present": bool(result.stdout.strip()),
        "stderr_present": bool(result.stderr.strip()),
    }


def _make_zip(staging: Path, zip_path: Path, token_path: Path) -> None:
    # The token is deliberately read only in this short-lived staging scope.
    _require_file(token_path, "Team 10 token.txt")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token or "\n" in token or "\r" in token:
        raise ValueError("token.txt は空でない1行である必要があります")
    (staging / "token.txt").write_text(token + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("F_mia.csv", "F_aia.csv", "token.txt"):
            archive.write(staging / name, arcname=name)


def build_one(
    *,
    run_name: str,
    method: str,
    mia_path: Path,
    aia_path: Path,
    other_aia_sha256: str,
    output_root: Path,
    submissions_root: Path,
    token_path: Path,
    dist_path: Path,
    seed: int,
    validate: bool,
    zip_filename: str | None = None,
) -> dict[str, object]:
    run_dir = output_root / run_name
    zip_path = submissions_root / (zip_filename or f"PWSCup2026_attack_team10_{run_name}.zip")
    # Validate every fixed prerequisite and destination before creating the
    # run directory.  This prevents a missing token or occupied ZIP path from
    # leaving a partial ignored run behind.
    _require_file(token_path, "Team 10 token.txt")
    _require_dir(dist_path, "公開target dist")
    if run_dir.exists():
        raise FileExistsError(f"既存runを上書きしません: {run_dir.as_posix()}")
    if zip_path.exists():
        raise FileExistsError(f"既存ZIPを上書きしません: {zip_path.as_posix()}")
    run_dir.mkdir(parents=True)
    _copy_candidates(run_dir, mia_path, aia_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zip-staging-", dir=run_dir) as staging_name:
        staging = Path(staging_name)
        shutil.copyfile(run_dir / "F_mia.csv", staging / "F_mia.csv")
        shutil.copyfile(run_dir / "F_aia.csv", staging / "F_aia.csv")
        _make_zip(staging, zip_path, token_path)
    validation = _validate_zip(zip_path, dist_path) if validate else {"ok": None, "skipped": True}
    mia_sha256 = sha256_file(mia_path)
    aia_sha256 = sha256_file(aia_path)
    metadata = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "method": method,
        "inputs": {
            "mia": _repo_label(mia_path),
            "aia": _repo_label(aia_path),
            "target_dist": _repo_label(dist_path),
        },
        "input_sha256": {
            "mia": sha256_file(mia_path),
            "aia": aia_sha256,
        },
        "candidate_sha256": {
            "mia": mia_sha256,
            "aia": aia_sha256,
        },
        "paired_comparison": {
            "shared_mia_sha256": mia_sha256,
            "paired_aia_sha256": other_aia_sha256,
            "aia_differs_from_paired": aia_sha256 != other_aia_sha256,
        },
        "zip_members": ["F_mia.csv", "F_aia.csv", "token.txt"],
        "validator": validation,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_candidates(
    *,
    mia_path: Path = DEFAULT_MIA,
    aia_baseline_path: Path = DEFAULT_AIA_BASELINE,
    aia_challenger_path: Path = DEFAULT_AIA_CHALLENGER,
    token_path: Path = DEFAULT_TOKEN,
    dist_path: Path = DEFAULT_DIST,
    output_root: Path = ROOT / "attack_phase/outputs",
    submissions_root: Path = ROOT / "attack_phase/submissions",
    seed: int = 20260828,
    validate: bool = True,
) -> list[dict[str, object]]:
    for path, label in (
        (mia_path, "MIA候補"),
        (aia_baseline_path, "AIA baseline候補"),
        (aia_challenger_path, "AIA challenger候補"),
    ):
        _require_file(path, label)
    _require_dir(dist_path, "公開target dist")
    # Preflight credentials and all fixed destinations before creating any
    # run directory, so a missing token cannot leave a partial package.
    _require_file(token_path, "Team 10 token.txt")
    run_names = (
        "20260828_attack_candidate_baseline",
        "20260828_attack_candidate_challenger",
    )
    for run_name in run_names:
        run_dir = output_root / run_name
        zip_path = submissions_root / f"PWSCup2026_attack_team10_{run_name}.zip"
        if run_dir.exists():
            raise FileExistsError(f"既存runを上書きしません: {run_dir.as_posix()}")
        if zip_path.exists():
            raise FileExistsError(f"既存ZIPを上書きしません: {zip_path.as_posix()}")
    baseline_sha = sha256_file(aia_baseline_path)
    challenger_sha = sha256_file(aia_challenger_path)
    if baseline_sha == challenger_sha:
        raise ValueError("AIA baseline/challengerが同一バイト列です")
    common = {
        "mia_path": mia_path,
        "output_root": output_root,
        "submissions_root": submissions_root,
        "token_path": token_path,
        "dist_path": dist_path,
        "seed": seed,
        "validate": validate,
    }
    return [
        build_one(
            run_name=run_names[0],
            method="frozen_stage2_mia+official_1nn_aia",
            aia_path=aia_baseline_path,
            other_aia_sha256=challenger_sha,
            **common,
        ),
        build_one(
            run_name=run_names[1],
            method="frozen_stage2_mia+gated_abstain_donor_modal_aia",
            aia_path=aia_challenger_path,
            other_aia_sha256=baseline_sha,
            **common,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="MIA固定・AIA比較候補をZIP化してvalidator確認する")
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    try:
        results = build_candidates(token_path=args.token, seed=args.seed, validate=not args.no_validate)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"runs": results}, ensure_ascii=False, indent=2))
    return 0 if all(r["validator"].get("ok", True) for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
