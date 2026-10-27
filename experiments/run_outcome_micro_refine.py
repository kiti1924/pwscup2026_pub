"""Small second-stage seed-15 outcome search around a coordinate-search winner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_STARTER_ROOT = _WORKSPACE_ROOT / "pwscup2026-kit" / "starter"
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
if str(_STARTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT))
if str(_STARTER_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT / "src"))

from experiments.run_aligned_synth import (  # noqa: E402
    _COX_COLUMNS,
    _detector_mask,
    _fit_cause_cox,
    _load_detector,
    _score_one,
)
from experiments.run_calibrated_outcomes import _fixed_uniforms, _simulate_outcomes  # noqa: E402
from experiments.run_outcome_coordinate_search import STATIC_COLUMNS, STEP  # noqa: E402
from reference._common import dist_paths, load_cfg, read_b, write_c  # noqa: E402
from pwscup2026.common import submission_schema  # noqa: E402
from pwscup2026.scoring.utility import _score_u_spec  # noqa: E402


def _sig(frame: pd.DataFrame) -> str:
    a = np.column_stack([
        np.round(frame["time"].to_numpy(dtype=float), 12),
        frame["onset"].to_numpy(dtype=np.int8),
        frame["death"].to_numpy(dtype=np.int8),
    ])
    return hashlib.sha256(a.tobytes()).hexdigest()


def _static_only(path: Path, n: int) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    if set(header.columns) != set(submission_schema.SUBMITTED_C_COLUMNS):
        raise ValueError("static base schema mismatch")
    static = pd.read_csv(path, usecols=STATIC_COLUMNS)[STATIC_COLUMNS].copy(deep=True)
    if len(static) != n:
        raise ValueError("static base row count mismatch")
    static.insert(0, "record_id", np.arange(n, dtype=int))
    return static


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="micro refine an outcome-coordinate winner")
    ap.add_argument("--dist", required=True)
    ap.add_argument("--static-csv", required=True)
    ap.add_argument("--search-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--profile", choices=("fpg", "sex"), default="fpg")
    args = ap.parse_args(argv)

    dist = dist_paths(args.dist)
    out_dir = Path(args.out_dir).resolve()
    scores_dir = out_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.search_summary).read_text(encoding="utf-8"))
    best_row = summary["fast_results"][0]
    beta = np.asarray(best_row["beta_generator"], dtype=float)
    if beta.shape != (len(_COX_COLUMNS),):
        raise ValueError("best beta shape mismatch")

    cfg = load_cfg()
    b = read_b(dist)
    static = _static_only(Path(args.static_csv).resolve(), len(b))
    detector = _load_detector(dist)
    rare_mask = _detector_mask(static, *detector)
    if int(rare_mask.sum()) != 61:
        raise ValueError("static public rare count is not 61")
    onset_bundle = _fit_cause_cox(b, "onset")
    death_bundle = _fit_cause_cox(b, "death")
    target_beta = onset_bundle[0].params_.reindex(_COX_COLUMNS).to_numpy(dtype=float)
    onset_u, death_u = _fixed_uniforms(15, len(static))
    base_rare_onset = int(summary.get("base_fast", {}).get("base_rare_onset_count", 24))

    # 20 deterministic probes.  Coordinates are in _COX_COLUMNS order.
    # The base winner itself is already in the first-stage summary; only new
    # perturbations count toward this second-stage budget.
    probes: list[tuple[str, np.ndarray]] = []
    if args.profile == "fpg":
        for j in (7,):
            for mult in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3):
                d = np.zeros_like(beta)
                d[j] = mult * STEP[j]
                probes.append((f"FPG_{mult:+.1f}h", beta + d))
        for j in (4,):
            for mult in (-0.2, -0.1, 0.1, 0.2):
                d = np.zeros_like(beta)
                d[j] = mult * STEP[j]
                probes.append((f"log_TG_{mult:+.1f}h", beta + d))
        for j in (6,):
            for mult in (-0.1, 0.1):
                d = np.zeros_like(beta)
                d[j] = mult * STEP[j]
                probes.append((f"log_ALT_{mult:+.1f}h", beta + d))
        for left, right, label in ((7, 4, "FPG_log_TG"), (7, 6, "FPG_log_ALT")):
            for s_left in (-1.0, 1.0):
                for s_right in (-1.0, 1.0):
                    d = np.zeros_like(beta)
                    d[left] = s_left * 0.1 * STEP[left]
                    d[right] = s_right * 0.1 * STEP[right]
                    probes.append((f"{label}_{int(s_left):+d}{int(s_right):+d}", beta + d))
    else:
        # The first-stage winner already sits at sex=-2h.  Probe that axis
        # locally, then use the remaining budget on FPG/log_TG and sex/FPG.
        for j, name in ((1, "sex"),):
            for mult in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3):
                d = np.zeros_like(beta)
                d[j] = mult * STEP[j]
                probes.append((f"sex_{mult:+.1f}h", beta + d))
        for j, name in ((7, "FPG"),):
            for mult in (-0.2, -0.1, 0.1, 0.2):
                d = np.zeros_like(beta)
                d[j] = mult * STEP[j]
                probes.append((f"FPG_{mult:+.1f}h", beta + d))
        for j, name in ((4, "log_TG"),):
            for mult in (-0.2, -0.1, 0.1, 0.2):
                d = np.zeros_like(beta)
                d[j] = mult * STEP[j]
                probes.append((f"log_TG_{mult:+.1f}h", beta + d))
        for s_left in (-1.0, 1.0):
            for s_right in (-1.0, 1.0):
                d = np.zeros_like(beta)
                d[1] = s_left * 0.05 * STEP[1]
                d[7] = s_right * 0.05 * STEP[7]
                probes.append((f"sex_FPG_{int(s_left):+d}{int(s_right):+d}", beta + d))
    if len(probes) != 20:
        raise RuntimeError(f"unexpected probe count {len(probes)}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for key, candidate_beta in probes:
        frame = _simulate_outcomes(
            static, cfg,
            onset_bundle[0], onset_bundle[1], onset_bundle[2],
            death_bundle[0], death_bundle[1], death_bundle[2],
            candidate_beta, 1.0, onset_u, death_u,
        )
        signature = _sig(frame)
        if signature in seen:
            continue
        seen.add(signature)
        spec, detail = _score_u_spec(frame, b)
        rare_onset = int((rare_mask & frame["onset"].to_numpy(dtype=bool)).sum())
        row = {
            "key": key,
            "beta_generator": candidate_beta.tolist(),
            "fast_U_spec": float(spec),
            "fast_io_part": detail.get("io_part"),
            "fast_tstr_cal": detail.get("tstr_cal"),
            "cox_converged": bool(detail.get("cox_converged", False)),
            "event_count": int(frame["onset"].sum()),
            "rare_onset_count": rare_onset,
            "safe_guard": bool(detail.get("cox_converged", False) and rare_onset == base_rare_onset),
            "frame": frame,
        }
        rows.append(row)

    rows.sort(key=lambda x: (x["fast_U_spec"], x["fast_io_part"] or 0.0), reverse=True)
    pool = [r for r in rows if r["safe_guard"]]
    if not pool:
        pool = [r for r in rows if r["cox_converged"]]
    full_results: list[dict[str, Any]] = []
    print("rank\tkey\tfast_U_spec\tU\tU_gen\tU_spec\tU_rare\tU_valid\tsafe")
    expected = list(submission_schema.DISTRIBUTED_COLUMNS)
    for rank, row in enumerate(pool[: int(args.top_k)], start=1):
        frame = row["frame"]
        csv_path = out_dir / f"M_{rank:02d}_{row['key']}.csv"
        score_path = scores_dir / f"M_{rank:02d}_{row['key']}.json"
        if list(frame.columns) != expected:
            raise RuntimeError("generated frame schema mismatch")
        write_c(frame, csv_path)
        payload, error = _score_one(csv_path, dist, score_path)
        result = {"rank": rank, "key": row["key"], "csv": csv_path.name,
                  "score_json": score_path.name, "fast_U_spec": row["fast_U_spec"],
                  "safe_guard": row["safe_guard"]}
        if payload is None or error is not None:
            result["error"] = error or "selfscore failed"
            print(f"{rank}\t{row['key']}\t{row['fast_U_spec']:.9f}\tERROR\t{row['safe_guard']}")
        else:
            result.update({"utility": float(payload["utility"]), **{k: float(v) for k, v in payload["facets"].items()}})
            print(
                f"{rank}\t{row['key']}\t{row['fast_U_spec']:.9f}\t{result['utility']:.9f}\t"
                f"{result['U_gen']:.9f}\t{result['U_spec']:.9f}\t{result['U_rare']:.9f}\t"
                f"{result['U_valid']:.9f}\t{row['safe_guard']}"
            )
        full_results.append(result)

    out = {
        "base_key": best_row["key"],
        "base_fast_U_spec": best_row["fast_U_spec"],
        "n_fast_unique": len(rows),
        "fast_results": [{k: v for k, v in row.items() if k != "frame"} for row in rows],
        "full_results": full_results,
    }
    path = out_dir / "outcome_micro_refine_summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
