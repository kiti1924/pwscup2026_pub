"""Derivative-free outcome calibration on a fixed fully synthetic static table.

The static columns are read from a submitted C only.  ``time``, ``onset`` and
``death`` are deliberately excluded from the input read and are regenerated
from B-fitted cause-specific Cox baselines plus fixed seed-15 uniforms.  The
first stage evaluates at most 100 candidates with the official U_spec scorer;
only the best eight guarded candidates receive the full official selfscore.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from experiments.run_calibrated_outcomes import (  # noqa: E402
    _fixed_uniforms,
    _simulate_outcomes,
)
from reference._common import dist_paths, load_cfg, read_b, write_c  # noqa: E402
from pwscup2026.common import schema, submission_schema  # noqa: E402
from pwscup2026.scoring.utility import _score_u_spec  # noqa: E402


STATIC_COLUMNS = [*schema.CONTINUOUS, "sex", "smoking", "prefecture"]
OUTCOME_COLUMNS = ["time", "onset", "death"]
MAX_FAST = 100
TOP_K = 8

# Fixed seed-15 base used by the existing calibration bank.
BETA0_FALLBACK = np.asarray(
    [
        0.026908232645966167,
        -0.3775383161026733,
        -0.007815868217951225,
        0.006944401530932616,
        0.3369256232722189,
        -0.02531385929209147,
        0.42253914615630345,
        0.13191700004163648,
        0.24144530762426947,
    ],
    dtype=float,
)

# Coordinate steps in _COX_COLUMNS order.  They are deliberately on the scale
# of the observed beta-bank moves, not on the scale of raw clinical columns.
STEP = np.asarray(
    [0.00025, 0.0060, 0.00030, 0.00010, 0.0030, 0.00030, 0.00150, 0.00050, 0.0020],
    dtype=float,
)


@dataclass
class Candidate:
    key: str
    beta: np.ndarray
    source: str
    frame: pd.DataFrame | None = None
    frame_signature: str | None = None


def _frame_signature(frame: pd.DataFrame) -> str:
    """Stable outcome-only signature; ignore CSV float round-trip noise."""
    values = np.column_stack(
        [
            np.round(frame["time"].to_numpy(dtype=float), 12),
            frame["onset"].to_numpy(dtype=np.int8),
            frame["death"].to_numpy(dtype=np.int8),
        ]
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_static_only(path: Path, n_expected: int) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    if set(header.columns) != set(submission_schema.SUBMITTED_C_COLUMNS):
        raise ValueError(f"static base columns differ from submitted schema: {list(header.columns)}")
    # Do not even load the submitted outcomes.  Every outcome below is newly
    # generated from the fitted model and fixed uniforms.
    static = pd.read_csv(path, usecols=STATIC_COLUMNS)
    if len(static) != n_expected:
        raise ValueError(f"static base rows {len(static)} != B rows {n_expected}")
    static = static[STATIC_COLUMNS].copy(deep=True)
    static.insert(0, "record_id", np.arange(len(static), dtype=int))
    return static


def _summary_paths(dist: Path) -> list[Path]:
    return [
        dist / "calibrated_tilt030" / "calibrated_tilt030_summary.json",
        dist / "calibrated_tilt030_final_fine" / "calibrated_tilt030_finalfine_summary.json",
        dist / "calibrated_tilt030_pref0_fine" / "calibrated_tilt030_pref0_fine_summary.json",
        dist / "calibrated_tilt040_pref0" / "calibrated_tilt040_pref0_summary.json",
    ]


def _load_beta_bank(dist: Path) -> list[tuple[str, np.ndarray, float, str]]:
    bank: list[tuple[str, np.ndarray, float, str]] = []
    for path in _summary_paths(dist):
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in obj.get("finalists", []):
            if int(item.get("outcome_seed", 15)) != 15:
                continue
            raw = item.get("beta_generator")
            if raw is None:
                continue
            beta = np.asarray(raw, dtype=float)
            if beta.shape != (len(_COX_COLUMNS),) or not np.isfinite(beta).all():
                continue
            key = str(item.get("key") or path.stem)
            bank.append((key, beta, float(item.get("U_spec", 0.0)), str(path)))
    if not bank:
        bank.append(("fallback_u15_h1.000_lr0.350_i7", BETA0_FALLBACK.copy(), 0.0, "fallback"))
    return bank


def _unique_bank(bank: list[tuple[str, np.ndarray, float, str]], max_items: int = 24) -> list[Candidate]:
    # Keep the two anchors even if their previous score was not the highest on
    # their original static table.
    mandatory: list[tuple[str, np.ndarray, float, str]] = []
    for needle in ("u15_h1.000_lr0.350_i7", "u15_h1.000_lr0.500_i8"):
        mandatory.extend(item for item in bank if item[0] == needle)
    ordered = mandatory + sorted(bank, key=lambda x: x[2], reverse=True)
    out: list[Candidate] = []
    seen: set[tuple[float, ...]] = set()
    for key, beta, _prior, source in ordered:
        sig = tuple(np.round(beta, 14).tolist())
        if sig in seen:
            continue
        seen.add(sig)
        out.append(Candidate(key=f"bank_{key}", beta=beta.copy(), source=source))
        if len(out) >= max_items:
            break
    return out


def _clip_beta(beta: np.ndarray, target_beta: np.ndarray) -> np.ndarray:
    span = np.maximum(3.0 * np.abs(target_beta), 0.15)
    return np.clip(beta, target_beta - span, target_beta + span)


def _add_candidate(
    specs: list[Candidate], key: str, beta: np.ndarray, source: str, target_beta: np.ndarray
) -> None:
    specs.append(Candidate(key=key, beta=_clip_beta(np.asarray(beta, dtype=float), target_beta), source=source))


def _candidate_schedule(
    dist: Path, target_beta: np.ndarray, anchor_beta: np.ndarray | None = None,
    anchor_source: str = "micro_anchor",
) -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    bank = _load_beta_bank(dist)
    anchors = _unique_bank(bank, max_items=24)
    specs = list(anchors)
    bank_beta0 = next(
        (item.beta.copy() for item in anchors if "u15_h1.000_lr0.350_i7" in item.key),
        BETA0_FALLBACK.copy(),
    )
    beta40 = next(
        (item.beta.copy() for item in anchors if "u15_h1.000_lr0.500_i8" in item.key),
        bank_beta0.copy(),
    )
    beta0 = bank_beta0 if anchor_beta is None else np.asarray(anchor_beta, dtype=float).copy()
    if beta0.shape != (len(_COX_COLUMNS),) or not np.isfinite(beta0).all():
        raise ValueError("anchor beta shape/finite check failed")
    if not any(np.allclose(item.beta, beta0, rtol=0.0, atol=1e-14) for item in specs):
        specs.insert(
            0,
            Candidate(
                key="anchor_M01" if anchor_beta is not None else "bank_fallback_u15_h1.000_lr0.350_i7",
                beta=beta0.copy(),
                source=anchor_source if anchor_beta is not None else "fallback",
            ),
        )

    # Six beta0 -> beta40 interpolation points.
    for lam in (0.125, 0.25, 0.375, 0.5, 0.75, 1.0):
        _add_candidate(
            specs,
            f"ray40_l{lam:.3f}".replace(".", "p"),
            beta0 + lam * (beta40 - beta0),
            "beta0_to_beta40",
            target_beta,
        )

    for j, col in enumerate(_COX_COLUMNS):
        for mult in (-2.0, -1.0, 1.0, 2.0):
            delta = np.zeros_like(beta0)
            delta[j] = mult * STEP[j]
            _add_candidate(specs, f"coord_{col}_{mult:+g}h", beta0 + delta, "coordinate", target_beta)

    # Extra probes target the coefficients that limited the current U_spec.
    for j in (4, 6, 7, 8):  # log_TG, log_ALT, FPG, smoking
        for mult in (-3.0, 3.0):
            delta = np.zeros_like(beta0)
            delta[j] = mult * STEP[j]
            _add_candidate(specs, f"coord_{_COX_COLUMNS[j]}_{mult:+g}h", beta0 + delta, "binding_coordinate", target_beta)

    for left, right in ((6, 8), (6, 4), (8, 4), (7, 4)):
        for s_left in (-1.0, 1.0):
            for s_right in (-1.0, 1.0):
                delta = np.zeros_like(beta0)
                delta[left] = s_left * STEP[left]
                delta[right] = s_right * STEP[right]
                _add_candidate(
                    specs,
                    f"pair_{_COX_COLUMNS[left]}_{_COX_COLUMNS[right]}_{int(s_left):+d}{int(s_right):+d}",
                    beta0 + delta,
                    "pair_coordinate",
                    target_beta,
                )
    # The first 90 are evaluated before this adaptive refinement.  Ten more
    # probes are added around the first-stage winner in main().
    return specs, beta0, beta40


def _evaluate(
    candidate: Candidate,
    b: pd.DataFrame,
    static: pd.DataFrame,
    cfg: dict[str, Any],
    onset_bundle: tuple[Any, np.ndarray, np.ndarray],
    death_bundle: tuple[Any, np.ndarray, np.ndarray],
    onset_u: np.ndarray,
    death_u: np.ndarray,
    rare_mask: np.ndarray,
    base_onset_count: int,
    base_rare_onset_count: int,
    seen_frames: set[str],
) -> dict[str, Any] | None:
    frame = _simulate_outcomes(
        static,
        cfg,
        onset_bundle[0], onset_bundle[1], onset_bundle[2],
        death_bundle[0], death_bundle[1], death_bundle[2],
        candidate.beta,
        1.0,
        onset_u,
        death_u,
    )
    signature = _frame_signature(frame)
    if signature in seen_frames:
        return None
    seen_frames.add(signature)
    spec, detail = _score_u_spec(frame, b)
    onset_count = int(frame["onset"].sum())
    rare_onset_count = int((rare_mask & frame["onset"].to_numpy(dtype=bool)).sum())
    candidate.frame = frame
    candidate.frame_signature = signature
    converged = bool(detail.get("cox_converged", False))
    safe = bool(
        converged
        and int(rare_mask.sum()) == 61
        and rare_onset_count == int(base_rare_onset_count)
    )
    return {
        "key": candidate.key,
        "source": candidate.source,
        "beta_generator": candidate.beta.tolist(),
        "fast_U_spec": float(spec),
        "fast_io_part": detail.get("io_part"),
        "fast_tstr_cal": detail.get("tstr_cal"),
        "cox_converged": converged,
        "event_count": onset_count,
        "base_event_count": int(base_onset_count),
        "rare_onset_count": rare_onset_count,
        "base_rare_onset_count": int(base_rare_onset_count),
        "rare_detected": int(rare_mask.sum()),
        "safe_guard": safe,
        "candidate": candidate,
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    serializable = dict(payload)
    for row in serializable.get("fast_results", []):
        row.pop("candidate", None)
    for row in serializable.get("full_results", []):
        row.pop("candidate", None)
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed-15 derivative-free outcome coordinate search")
    parser.add_argument("--dist", required=True)
    parser.add_argument("--static-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--outcome-seed", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-fast", type=int, default=MAX_FAST)
    parser.add_argument("--anchor-summary", default=None)
    args = parser.parse_args(argv)
    if int(args.outcome_seed) != 15:
        raise ValueError("this search is intentionally fixed to outcome seed 15")
    max_fast = int(args.max_fast)
    if max_fast < 1 or max_fast > MAX_FAST:
        raise ValueError(f"--max-fast must be in [1, {MAX_FAST}]")

    dist = dist_paths(args.dist)
    static_path = Path(args.static_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    scores_dir = out_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg()
    b = read_b(dist)
    static = _load_static_only(static_path, len(b))
    detector = _load_detector(dist)
    rare_mask = _detector_mask(static, *detector)
    if int(rare_mask.sum()) != 61:
        raise ValueError(f"static base public rare count must be 61, got {int(rare_mask.sum())}")
    onset_bundle = _fit_cause_cox(b, "onset")
    death_bundle = _fit_cause_cox(b, "death")
    target_beta = onset_bundle[0].params_.reindex(_COX_COLUMNS).to_numpy(dtype=float)
    onset_u, death_u = _fixed_uniforms(args.outcome_seed, len(static))

    anchor_beta = None
    anchor_source = "micro_anchor"
    if args.anchor_summary:
        anchor_path = Path(args.anchor_summary).resolve()
        anchor_obj = json.loads(anchor_path.read_text(encoding="utf-8"))
        anchor_rows = anchor_obj.get("fast_results") or []
        if not anchor_rows or anchor_rows[0].get("beta_generator") is None:
            raise ValueError(f"anchor summary has no fast beta: {anchor_path}")
        anchor_beta = np.asarray(anchor_rows[0]["beta_generator"], dtype=float)
        anchor_source = f"anchor:{anchor_path}"
    specs, beta0, beta40 = _candidate_schedule(dist, target_beta, anchor_beta, anchor_source)
    # Ensure the known current anchor is evaluated first, then retain the
    # schedule's bank/interpolation/coordinate order for reproducibility.
    specs.sort(key=lambda item: (0 if np.allclose(item.beta, beta0, rtol=0.0, atol=1e-14) else 1))
    seen_frames: set[str] = set()
    fast_results: list[dict[str, Any]] = []
    base_record: dict[str, Any] | None = None
    base_event_count = 0
    base_rare_onset_count = 0

    for candidate in specs[:max_fast]:
        row = _evaluate(
            candidate, b, static, cfg, onset_bundle, death_bundle, onset_u, death_u,
            rare_mask, base_event_count, base_rare_onset_count, seen_frames,
        )
        if row is None:
            continue
        if base_record is None and np.allclose(candidate.beta, beta0, rtol=0.0, atol=1e-14):
            base_record = row
            base_event_count = int(row["event_count"])
            base_rare_onset_count = int(row["rare_onset_count"])
            row["base_event_count"] = base_event_count
            row["base_rare_onset_count"] = base_rare_onset_count
            row["safe_guard"] = bool(row["cox_converged"] and row["rare_detected"] == 61)
            # Re-evaluate guard values for all rows after establishing the base
            # count; their scores remain unchanged.
            for prior in fast_results:
                prior["base_event_count"] = base_event_count
                prior["base_rare_onset_count"] = base_rare_onset_count
                prior["safe_guard"] = bool(
                    prior["cox_converged"]
                    and prior["rare_detected"] == 61
                    and prior["rare_onset_count"] == base_rare_onset_count
                )
        fast_results.append(row)

    if base_record is None:
        raise RuntimeError("beta0 was not evaluated; cannot establish rare-onset guard")
    # Adaptive refinement around the best safe/converged first-stage result.
    eligible = [r for r in fast_results if r["cox_converged"] and r["safe_guard"]]
    if not eligible:
        eligible = [r for r in fast_results if r["cox_converged"]]
    eligible.sort(key=lambda row: (float(row["fast_U_spec"]), float(row["fast_io_part"] or 0.0)), reverse=True)
    best = eligible[0]["candidate"]
    refine_indices = (0, 4, 6, 7, 8)
    for j in refine_indices:
        for mult in (-0.5, 0.5):
            if len(fast_results) >= max_fast:
                break
            beta = best.beta.copy()
            beta[j] += mult * STEP[j]
            candidate = Candidate(
                key=f"refine_{_COX_COLUMNS[j]}_{mult:+g}h",
                beta=_clip_beta(beta, target_beta),
                source="adaptive_refine",
            )
            row = _evaluate(
                candidate, b, static, cfg, onset_bundle, death_bundle, onset_u, death_u,
                rare_mask, base_event_count, base_rare_onset_count, seen_frames,
            )
            if row is not None:
                fast_results.append(row)

    # Hard cap even if a future schedule edit accidentally adds probes.
    fast_results = fast_results[:max_fast]
    fast_results.sort(key=lambda row: (float(row["fast_U_spec"]), float(row["fast_io_part"] or 0.0)), reverse=True)
    full_pool = [row for row in fast_results if row["safe_guard"] and row["cox_converged"]]
    if not full_pool:
        full_pool = [row for row in fast_results if row["cox_converged"]]
    full_pool = full_pool[: max(1, int(args.top_k))]

    full_results: list[dict[str, Any]] = []
    expected = list(submission_schema.DISTRIBUTED_COLUMNS)
    print("rank\tkey\tfast_U_spec\tU\tU_gen\tU_spec\tU_rare\tU_valid\tsafe")
    for rank, row in enumerate(full_pool, start=1):
        candidate = row["candidate"]
        if candidate.frame is None or list(candidate.frame.columns) != expected:
            row["full_error"] = "frame schema mismatch"
            continue
        csv_path = out_dir / f"C_{rank:02d}_{candidate.key}.csv"
        score_path = scores_dir / f"C_{rank:02d}_{candidate.key}.json"
        write_c(candidate.frame, csv_path)
        payload, error = _score_one(csv_path, dist, score_path)
        result = {"rank": rank, "key": candidate.key, "csv": csv_path.name, "score_json": score_path.name,
                  "fast_U_spec": row["fast_U_spec"], "safe_guard": row["safe_guard"]}
        if payload is None or error is not None:
            result["error"] = error or "selfscore failed"
            print(f"{rank}\t{candidate.key}\t{row['fast_U_spec']:.9f}\tERROR\t{row['safe_guard']}")
        else:
            result.update({"utility": float(payload["utility"]), **{k: float(v) for k, v in payload["facets"].items()}})
            print(
                f"{rank}\t{candidate.key}\t{row['fast_U_spec']:.9f}\t{result['utility']:.9f}\t"
                f"{result['U_gen']:.9f}\t{result['U_spec']:.9f}\t{result['U_rare']:.9f}\t"
                f"{result['U_valid']:.9f}\t{row['safe_guard']}"
            )
        full_results.append(result)

    summary = {
        "static_csv": str(static_path),
        "outcome_seed": 15,
        "hazard_scale": 1.0,
        "max_fast": max_fast,
        "n_fast_unique": len(fast_results),
        "n_full": len(full_results),
        "beta0": beta0.tolist(),
        "beta40": beta40.tolist(),
        "static_public_rare_n": int(rare_mask.sum()),
        "base_fast": {k: v for k, v in (base_record or {}).items() if k != "candidate"},
        "fast_results": fast_results,
        "full_results": full_results,
        "go_thresholds": {
            "U_gen": 0.972926597,
            "U_spec": 0.972156912,
            "U_rare": 0.9721569112733572,
            "U_valid": 1.0,
            "utility_strict": 0.972036504155688,
        },
    }
    summary_path = out_dir / "outcome_coordinate_search_summary.json"
    _write_summary(summary_path, summary)
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
