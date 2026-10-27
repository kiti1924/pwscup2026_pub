"""Generate one B14 baseline; score/validate with the unmodified official kit.

The official reference CLI expects practice-only B_practice.csv and targets.
This adapter reads the final distribution's B.csv and calls the same generator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "pwscup2026-kit" / "starter"
sys.path[:0] = [str(ROOT), str(KIT), str(KIT / "src")]

import numpy as np
import pandas as pd
from reference._common import load_cfg, write_c
from pwscup2026.defense.synth import copula_survival
from experiments.run_aligned_synth import synthesize, _load_detector


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["reference", "rare_aligned"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dist = ROOT / "_private_team14"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}_s{args.seed}"
    csv = out / f"C_{tag}.csv"
    score = out / f"{tag}_score.json"
    archive = out / f"{tag}.zip"
    metadata = out / f"{tag}_manifest.json"
    if any(p.exists() for p in [csv, score, archive, metadata]):
        raise FileExistsError(f"Existing candidate {tag}; inspect saved checkpoint before rerunning")
    start = time.monotonic()
    b = pd.read_csv(dist / "B.csv")
    bs = pd.read_csv(dist / "B_self.csv")
    pd.testing.assert_frame_equal(b, bs.drop(columns="is_rare"))
    if len(b) != 1049 or int(bs.is_rare.sum()) != 24:
        raise ValueError("Unexpected B14 distribution")
    cfg = load_cfg()
    if args.method == "reference":
        c = copula_survival(b, cfg, np.random.default_rng(args.seed))
    else:
        c = synthesize(b, cfg, args.method, args.seed, _load_detector(dist))
    write_c(c, csv)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in [("C.csv", csv), ("token.txt", dist / "token.txt")]:
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    env = dict(os.environ, PYTHONPATH=str(KIT / "src"), PYTHONUTF8="1")
    commands = [
        ("validate", [sys.executable, "-m", "pwscup2026.kit.validate", str(archive), "--kind", "defense", "--dist", str(dist)]),
        ("selfscore", [sys.executable, "-m", "pwscup2026.kit.selfscore", str(csv), "--dist", str(dist), "--json"]),
    ]
    for name, command in commands:
        proc = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", timeout=1800)
        # Paths are normalized for cross-device reports; no token contents logged.
        stdout = proc.stdout.replace(str(ROOT), "<repo>")
        stderr = proc.stderr.replace(str(ROOT), "<repo>")
        (out / f"{tag}_{name}.log").write_text(stdout + stderr, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"{name} failed ({proc.returncode}); see saved log")
        if name == "selfscore":
            payload = json.loads(proc.stdout)
            score.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "method": args.method, "seed": args.seed, "round": "final", "cohort": 14,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate": csv.relative_to(ROOT).as_posix(), "candidate_sha256": sha(csv),
        "zip": archive.relative_to(ROOT).as_posix(), "zip_sha256": sha(archive),
        "input_sha256": {p.name: sha(p) for p in [dist / "B.csv", dist / "B_self.csv", dist / "scoring_config.yaml", dist / "utility_ref.json"]},
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "lifelines", "scikit-learn"]},
        "seconds": time.monotonic() - start, "validator": "OK", "utility_local": payload["utility"],
        "facets": payload["facets"], "official_server_score": None, "privacy_proxy": None,
    }
    metadata.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["method", "seed", "seconds", "utility_local", "facets"]}))


if __name__ == "__main__":
    main()
