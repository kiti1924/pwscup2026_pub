"""Indirectly calibrate fully synthetic outcomes to the official Cox utility.

The static synthetic covariates are built once with ``rare_aligned``.  For each
fixed pair of onset/death uniforms, the onset generator coefficient vector is
updated toward the Cox coefficients fitted on B.  No B row, record_id, or
observed individual time is copied to an output row.

This experiment deliberately changes only ``time/onset/death``.  Therefore the
reference continuous-covariate MIA input is invariant within a covariate seed;
all finalists still require a separate AIA audit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
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
    _TIME_FLOOR_YEARS,
    _cox_frame,
    _fit_cause_cox,
    _load_detector,
    _sample_event_time,
    _score_one,
)
from experiments.run_outcome_sweep import _build_static  # noqa: E402
from reference._common import (  # noqa: E402
    coerce_to_submission,
    dist_paths,
    load_cfg,
    read_b,
    write_c,
)
from pwscup2026.common import submission_schema  # noqa: E402
from pwscup2026.common import schema  # noqa: E402
from pwscup2026.scoring.utility import _score_u_spec  # noqa: E402


@dataclass
class Trial:
    key: str
    outcome_seed: int
    hazard_scale: float
    learning_rate: float
    iteration: int
    beta_generator: np.ndarray
    spec: float
    io_part: float
    tstr_cal: float
    event_count: int
    frame: pd.DataFrame


def _fixed_uniforms(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the draw order used by ``_sample_outcomes``."""
    rng = np.random.default_rng(int(seed))
    return rng.uniform(size=n), rng.uniform(size=n)


def _center_vector(model: Any) -> np.ndarray:
    center = getattr(model, "_norm_mean", None)
    if center is None:
        return np.zeros(len(_COX_COLUMNS), dtype=float)
    if isinstance(center, pd.Series):
        return center.reindex(_COX_COLUMNS).to_numpy(dtype=float)
    arr = np.asarray(center, dtype=float)
    if arr.shape != (len(_COX_COLUMNS),):
        raise RuntimeError("Cox centering vector has an unexpected shape")
    return arr


def _simulate_outcomes(
    static: pd.DataFrame,
    cfg: dict[str, Any],
    onset_model: Any,
    onset_times: np.ndarray,
    onset_h0: np.ndarray,
    death_model: Any,
    death_times: np.ndarray,
    death_h0: np.ndarray,
    beta_generator: np.ndarray,
    hazard_scale: float,
    onset_u: np.ndarray,
    death_u: np.ndarray,
) -> pd.DataFrame:
    """Generate competing-risk outcomes without reusing observed B times."""
    horizon = float(cfg["onset"]["horizon_years"])
    x = _cox_frame(static)
    x_matrix = x[_COX_COLUMNS].to_numpy(dtype=float)
    centered = x_matrix - _center_vector(onset_model)
    log_ph = np.clip(centered @ np.asarray(beta_generator, dtype=float), -40.0, 40.0)
    onset_ph = np.exp(log_ph) * float(hazard_scale)
    onset_time = _sample_event_time(onset_ph, onset_times, onset_h0, onset_u)

    death_ph = death_model.predict_partial_hazard(x).to_numpy(dtype=float)
    death_time = _sample_event_time(death_ph, death_times, death_h0, death_u)

    censored = np.minimum(onset_time, death_time) > horizon
    onset_event = (~censored) & (onset_time <= death_time)
    death_event = (~censored) & (death_time < onset_time)
    onset = np.zeros(len(static), dtype=int)
    death = np.zeros(len(static), dtype=int)
    time = np.full(len(static), horizon, dtype=float)
    onset[onset_event] = 1
    time[onset_event] = onset_time[onset_event]
    death[death_event] = 1
    time[death_event] = death_time[death_event]

    out = static.copy(deep=True)
    out["time"] = np.clip(time, _TIME_FLOOR_YEARS, horizon)
    out["onset"] = onset
    out["death"] = death
    return coerce_to_submission(out, cfg)


def _beta_from_detail(detail: dict[str, Any]) -> np.ndarray:
    per = detail.get("per_coefficient") or {}
    return np.asarray([float(per[col]["beta_C"]) for col in _COX_COLUMNS], dtype=float)


def _trial_key(seed: int, scale: float, lr: float, iteration: int) -> str:
    return f"u{seed}_h{scale:.3f}_lr{lr:.3f}_i{iteration}"


def _run_path(
    b: pd.DataFrame,
    static: pd.DataFrame,
    cfg: dict[str, Any],
    onset_bundle: tuple[Any, np.ndarray, np.ndarray],
    death_bundle: tuple[Any, np.ndarray, np.ndarray],
    outcome_seed: int,
    hazard_scale: float,
    learning_rate: float,
    iterations: int,
) -> list[Trial]:
    onset_model, onset_times, onset_h0 = onset_bundle
    death_model, death_times, death_h0 = death_bundle
    target_beta = onset_model.params_.reindex(_COX_COLUMNS).to_numpy(dtype=float)
    beta_generator = target_beta.copy()
    onset_u, death_u = _fixed_uniforms(outcome_seed, len(static))
    trials: list[Trial] = []

    for iteration in range(int(iterations) + 1):
        frame = _simulate_outcomes(
            static,
            cfg,
            onset_model,
            onset_times,
            onset_h0,
            death_model,
            death_times,
            death_h0,
            beta_generator,
            hazard_scale,
            onset_u,
            death_u,
        )
        spec, detail = _score_u_spec(frame, b)
        if not detail.get("cox_converged", False):
            break
        trials.append(
            Trial(
                key=_trial_key(outcome_seed, hazard_scale, learning_rate, iteration),
                outcome_seed=int(outcome_seed),
                hazard_scale=float(hazard_scale),
                learning_rate=float(learning_rate),
                iteration=int(iteration),
                beta_generator=beta_generator.copy(),
                spec=float(spec),
                io_part=float(detail["io_part"]),
                tstr_cal=float(detail["tstr_cal"]),
                event_count=int(frame["onset"].sum()),
                frame=frame,
            )
        )
        fitted_beta = _beta_from_detail(detail)
        delta = target_beta - fitted_beta
        beta_generator = beta_generator + float(learning_rate) * delta
        # Guard against a discrete-event update running away.  The bounds are
        # broad enough to allow sign correction while excluding pathological fits.
        span = np.maximum(3.0 * np.abs(target_beta), 0.15)
        beta_generator = np.clip(beta_generator, target_beta - span, target_beta + span)
    return trials


def _parse_float_list(value: str) -> list[float]:
    result = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not result:
        raise ValueError("float list is empty")
    return result


def _parse_int_list(value: str) -> list[int]:
    result = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not result:
        raise ValueError("integer list is empty")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Indirect Cox calibration for fully synthetic outcomes")
    ap.add_argument("--dist", required=True)
    ap.add_argument("--covariate-seed", type=int, default=4)
    ap.add_argument(
        "--static-csv",
        default=None,
        help="Optional fully synthetic C whose static columns replace rare_aligned generation",
    )
    ap.add_argument("--tag", default=None, help="Safe output label when --static-csv is used")
    ap.add_argument("--outcome-seeds", default="15,56")
    ap.add_argument("--hazard-scales", default="0.85,0.925,1.0,1.075,1.15")
    ap.add_argument("--learning-rates", default="0.25,0.5,0.75")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    try:
        dist = dist_paths(args.dist)
        cfg = load_cfg()
        b = read_b(dist)
        detector = _load_detector(dist)
        if args.static_csv:
            submitted = pd.read_csv(args.static_csv)
            if list(submitted.columns) != list(submission_schema.SUBMITTED_C_COLUMNS) or len(submitted) != len(b):
                raise ValueError("static CSV has an unexpected shape or columns")
            static_cols = [*schema.CONTINUOUS, "sex", "smoking", "prefecture"]
            static = submitted[static_cols].copy(deep=True)
            static.insert(0, "record_id", np.arange(len(static), dtype=int))
        else:
            static = _build_static(b, cfg, detector, covariate_seed=args.covariate_seed)
        outcome_seeds = _parse_int_list(args.outcome_seeds)
        hazard_scales = _parse_float_list(args.hazard_scales)
        learning_rates = _parse_float_list(args.learning_rates)
        if args.iterations < 0 or args.top_k < 1:
            raise ValueError("iterations/top-k must be positive")
        out_dir = Path(args.out_dir) if args.out_dir else dist / "calibrated"
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_dir = out_dir / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        onset_bundle = _fit_cause_cox(b, "onset")
        death_bundle = _fit_cause_cox(b, "death")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    trials: list[Trial] = []
    for outcome_seed in outcome_seeds:
        for hazard_scale in hazard_scales:
            for learning_rate in learning_rates:
                trials.extend(
                    _run_path(
                        b,
                        static,
                        cfg,
                        onset_bundle,
                        death_bundle,
                        outcome_seed,
                        hazard_scale,
                        learning_rate,
                        args.iterations,
                    )
                )

    if not trials:
        print("ERROR: no converged trial", file=sys.stderr)
        return 1
    trials.sort(key=lambda trial: (trial.spec, trial.io_part, trial.tstr_cal), reverse=True)

    # De-duplicate identical outcome frames before the expensive full scorer.
    finalists: list[Trial] = []
    seen: set[bytes] = set()
    for trial in trials:
        signature = pd.util.hash_pandas_object(
            trial.frame[["time", "onset", "death"]], index=False
        ).to_numpy(dtype=np.uint64).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        finalists.append(trial)
        if len(finalists) >= int(args.top_k):
            break

    output_tag = str(args.tag or f"cov{args.covariate_seed}")
    if not output_tag.replace("_", "").replace("-", "").isalnum():
        print("ERROR: --tag must contain only letters, digits, underscore, or hyphen", file=sys.stderr)
        return 2
    summary: dict[str, Any] = {
        "covariate_seed": int(args.covariate_seed),
        "static_csv": Path(args.static_csv).name if args.static_csv else None,
        "output_tag": output_tag,
        "n_trials": len(trials),
        "n_unique_finalists": len(finalists),
        "privacy_note": (
            "Static covariates are invariant within this run, so reference MIA inputs are invariant; "
            "AIA must be audited per finalist."
        ),
        "finalists": [],
    }
    expected = list(submission_schema.DISTRIBUTED_COLUMNS)
    print("rank\tkey\tU\tU_gen\tU_spec\tU_rare\tU_valid\tio\ttstr\tevents")
    failures = 0
    for rank, trial in enumerate(finalists, start=1):
        if list(trial.frame.columns) != expected or len(trial.frame) != len(b):
            failures += 1
            continue
        csv_path = out_dir / f"C_cal_{output_tag}_{trial.key}.csv"
        score_path = scores_dir / f"cal_{output_tag}_{trial.key}.json"
        write_c(trial.frame, csv_path)
        payload, error = _score_one(csv_path, dist, score_path)
        if payload is None or error is not None:
            failures += 1
            print(f"{rank}\t{trial.key}\tERROR\t{error}")
            continue
        facets = payload["facets"]
        detail = payload["detail"]["U_spec"]
        row = {
            "rank": rank,
            "key": trial.key,
            "csv": csv_path.name,
            "score_json": score_path.name,
            "outcome_seed": trial.outcome_seed,
            "hazard_scale": trial.hazard_scale,
            "learning_rate": trial.learning_rate,
            "iteration": trial.iteration,
            "event_count": trial.event_count,
            "utility": payload["utility"],
            **facets,
            "io_part": detail.get("io_part"),
            "tstr_cal": detail.get("tstr_cal"),
            "beta_generator": trial.beta_generator.tolist(),
        }
        summary["finalists"].append(row)
        print(
            f"{rank}\t{trial.key}\t{row['utility']:.6f}\t{row['U_gen']:.6f}\t"
            f"{row['U_spec']:.6f}\t{row['U_rare']:.6f}\t{row['U_valid']:.6f}\t"
            f"{row['io_part']:.6f}\t{row['tstr_cal']:.6f}\t{row['event_count']}"
        )

    summary_path = out_dir / f"calibrated_{output_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
