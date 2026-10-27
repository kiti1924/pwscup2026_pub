"""Outcome-triplet swap hill climb on a fixed static candidate.

This search deliberately keeps the static columns fixed.  The first phase
swaps complete outcome triplets only inside the public-common, onset-event
bucket.  Such swaps preserve every public rare/common outcome marginal and
all onset/death statuses while still changing Cox risk sets and coefficients.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError, ConvergenceWarning

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_STARTER_ROOT = _WORKSPACE_ROOT / "pwscup2026-kit" / "starter"
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
if str(_STARTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT))
if str(_STARTER_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_STARTER_ROOT / "src"))

from experiments.local_privacy_audit import (  # noqa: E402
    _audit_one,
    _load_inputs,
    _member_rare_mask,
    _rare_mask_from_reference,
    _sample_controls,
)
from experiments.full_static_nearest_audit import (  # noqa: E402
    CATEGORICAL as FULL_STATIC_CATEGORICAL,
    NUMERIC as FULL_STATIC_NUMERIC,
    _audit_one as _full_static_audit_one,
)
from experiments.run_aligned_synth import (  # noqa: E402
    _COX_COLUMNS,
    _fit_cause_cox,
    _load_detector,
    _score_one,
)
from pwscup2026.common import schema, submission_schema  # noqa: E402
from pwscup2026.scoring.utility import (  # noqa: E402
    _cox_frame,
    _fit_cox,
)
from reference._common import dist_paths, read_b, write_c  # noqa: E402


STATIC_COLUMNS = [*schema.CONTINUOUS, "sex", "smoking", "prefecture"]
OUTCOME_COLUMNS = ["time", "onset", "death"]
TRIPLET_COLUMNS = ["time", "onset", "death"]

BASE_UTILITY = 0.9725227593
BASE_UGEN = 0.9732125834
BASE_URARE = 0.9780967702
BASE_UVALID = 1.0


def _outcome_signature(frame: pd.DataFrame) -> str:
    values = np.column_stack(
        [
            np.round(frame["time"].to_numpy(dtype=float), 12),
            frame["onset"].to_numpy(dtype=np.int8),
            frame["death"].to_numpy(dtype=np.int8),
        ]
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def _triplet_counter(frame: pd.DataFrame, indices: Iterable[int] | None = None) -> Counter:
    if indices is None:
        rows = frame[TRIPLET_COLUMNS].itertuples(index=False, name=None)
    else:
        rows = frame.iloc[list(indices)][TRIPLET_COLUMNS].itertuples(index=False, name=None)
    return Counter((float(t), int(o), int(d)) for t, o, d in rows)


def _bucket_counters(frame: pd.DataFrame, rare_mask: np.ndarray) -> dict[tuple[bool, int, int], Counter]:
    out: dict[tuple[bool, int, int], Counter] = {}
    for is_rare, onset, death, time_value in zip(
        rare_mask,
        frame["onset"].to_numpy(dtype=int),
        frame["death"].to_numpy(dtype=int),
        frame["time"].to_numpy(dtype=float),
    ):
        key = (bool(is_rare), int(onset), int(death))
        out.setdefault(key, Counter())[(float(time_value), int(onset), int(death))] += 1
    return out


def _assert_invariants(
    base: pd.DataFrame,
    current: pd.DataFrame,
    base_rare: np.ndarray,
    current_rare: np.ndarray,
) -> None:
    if list(base[STATIC_COLUMNS].columns) != list(current[STATIC_COLUMNS].columns):
        raise AssertionError("static column order changed")
    if not base[STATIC_COLUMNS].equals(current[STATIC_COLUMNS]):
        raise AssertionError("static columns changed")
    if not np.array_equal(base_rare, current_rare):
        raise AssertionError("public rare detector mask changed")
    if _triplet_counter(base) != _triplet_counter(current):
        raise AssertionError("global outcome triplet multiset changed")
    if _bucket_counters(base, base_rare) != _bucket_counters(current, current_rare):
        raise AssertionError("public rare/status bucket outcome multiset changed")
    if not np.array_equal(base["onset"].to_numpy(), current["onset"].to_numpy()):
        raise AssertionError("same-status phase changed row-level onset")
    if not np.array_equal(base["death"].to_numpy(), current["death"].to_numpy()):
        raise AssertionError("same-status phase changed row-level death")
    if int(base["onset"].sum()) != int(current["onset"].sum()):
        raise AssertionError("global onset count changed")
    if int(base["death"].sum()) != int(current["death"].sum()):
        raise AssertionError("global death count changed")
    if int((base_rare & base["onset"].to_numpy(dtype=bool)).sum()) != int(
        (current_rare & current["onset"].to_numpy(dtype=bool)).sum()
    ):
        raise AssertionError("rare onset count changed")


class FastSpec:
    """Cached-B implementation of the official U_spec calculation."""

    def __init__(self, b: pd.DataFrame):
        self.b = b
        self.b_cox = _fit_cox(b)
        self.b_beta = self.b_cox.params_.reindex(_COX_COLUMNS).to_numpy(dtype=float)
        self.b_score = float(self.b_cox.score(_cox_frame(b), scoring_method="concordance_index"))
        self.cache: dict[str, tuple[float, dict[str, Any]]] = {}

    @staticmethod
    def _io_overlap(
        lo_b: float, hi_b: float, lo_c: float, hi_c: float, beta_b: float, beta_c: float, se_b: float
    ) -> float:
        width_b, width_c = hi_b - lo_b, hi_c - lo_c
        if width_b > 0 and width_c > 0:
            overlap = min(hi_b, hi_c) - max(lo_b, lo_c)
            value = 0.5 * (overlap / width_b + overlap / width_c)
        else:
            value = 1.0 - min(abs(beta_b - beta_c) / (4.0 * max(se_b, 1e-12)), 1.0)
        return float(np.clip(value, 0.0, 1.0))

    def score(self, frame: pd.DataFrame) -> tuple[float, dict[str, Any]]:
        key = _outcome_signature(frame)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        try:
            cox = _fit_cox(frame)
            per: dict[str, dict[str, Any]] = {}
            contribs: list[float] = []
            for col in _COX_COLUMNS:
                beta_b = float(self.b_cox.params_[col])
                beta_c = float(cox.params_[col])
                se_b = float(self.b_cox.standard_errors_[col])
                lo_b, hi_b = (float(v) for v in self.b_cox.confidence_intervals_.loc[col])
                lo_c, hi_c = (float(v) for v in cox.confidence_intervals_.loc[col])
                sig_b = (lo_b > 0.0) or (hi_b < 0.0)
                sig_c = (lo_c > 0.0) or (hi_c < 0.0)
                io = self._io_overlap(lo_b, hi_b, lo_c, hi_c, beta_b, beta_c, se_b)
                if sig_b != sig_c or (sig_b and sig_c and np.sign(beta_b) != np.sign(beta_c)):
                    contrib = 0.0
                else:
                    contrib = io
                contribs.append(contrib)
                per[col] = {
                    "beta_B": beta_b,
                    "beta_C": beta_c,
                    "sig_B": sig_b,
                    "sig_C": sig_c,
                    "io": io,
                    "contrib": contrib,
                }
            io_part = float(np.mean(contribs))
            c_index = float(cox.score(_cox_frame(self.b), scoring_method="concordance_index"))
            if not np.isfinite(c_index):
                raise ValueError("candidate c-index is non-finite")
            denom = self.b_score - 0.5
            tstr_cal = 1.0 if denom < 0.02 else float(np.clip((c_index - 0.5) / denom, 0.0, 1.0))
            score = float(min(io_part, tstr_cal))
            detail = {
                "cox_converged": True,
                "io_part": io_part,
                "tstr_c_index": c_index,
                "trtr_c_index": self.b_score,
                "tstr_cal": tstr_cal,
                "per_coefficient": per,
            }
        except (ConvergenceError, ConvergenceWarning, np.linalg.LinAlgError, ValueError, ZeroDivisionError) as exc:
            score = 0.0
            detail = {"cox_converged": False, "error": f"{type(exc).__name__}: {exc}"}
        self.cache[key] = (score, detail)
        return score, detail


def _swap(frame: pd.DataFrame, i: int, j: int) -> pd.DataFrame:
    candidate = frame.copy(deep=True)
    left = candidate.loc[i, TRIPLET_COLUMNS].copy()
    candidate.loc[i, TRIPLET_COLUMNS] = candidate.loc[j, TRIPLET_COLUMNS].to_numpy()
    candidate.loc[j, TRIPLET_COLUMNS] = left.to_numpy()
    return candidate


def _pair_pool(
    frame: pd.DataFrame,
    rare_mask: np.ndarray,
    bucket_key: tuple[bool, int, int],
    beta_error: np.ndarray,
    rng: np.random.Generator,
    max_proposals: int,
) -> list[tuple[int, int]]:
    indices = np.flatnonzero(
        (rare_mask == bucket_key[0])
        & (frame["onset"].to_numpy(dtype=int) == bucket_key[1])
        & (frame["death"].to_numpy(dtype=int) == bucket_key[2])
    )
    if len(indices) < 2:
        return []
    x = _cox_frame(frame)[_COX_COLUMNS].to_numpy(dtype=float)
    means = np.nanmean(x[indices], axis=0)
    scales = np.nanstd(x[indices], axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    xz = (x - means) / scales
    eta_error = xz @ beta_error
    times = frame["time"].to_numpy(dtype=float)
    n_pairs = len(indices) * (len(indices) - 1) // 2
    if n_pairs <= 120_000:
        pairs = [(int(indices[a]), int(indices[b])) for a in range(len(indices)) for b in range(a + 1, len(indices))]
    else:
        pairs = []
        seen: set[tuple[int, int]] = set()
        for _ in range(min(60_000, n_pairs)):
            a, b = sorted(rng.choice(indices, size=2, replace=False).tolist())
            pair = (int(a), int(b))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    if not pairs:
        return []
    score = np.asarray(
        [
            (eta_error[i] - eta_error[j]) * np.sign(times[j] - times[i])
            for i, j in pairs
        ],
        dtype=float,
    )
    order = np.argsort(score)[::-1]
    selected: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    guided_n = min(260, max_proposals)
    for pos in order[:guided_n]:
        pair = pairs[int(pos)]
        if score[int(pos)] > 0.0 and pair not in used:
            used.add(pair)
            selected.append(pair)
    # Four weak-coefficient-specific directional pools.
    weak = sorted(range(len(_COX_COLUMNS)), key=lambda j: abs(float(beta_error[j])), reverse=True)[:4]
    for col_idx in weak:
        local_scores = np.asarray(
            [
                (xz[i, col_idx] - xz[j, col_idx]) * float(beta_error[col_idx]) * np.sign(times[j] - times[i])
                for i, j in pairs
            ],
            dtype=float,
        )
        for pos in np.argsort(local_scores)[::-1][: max(1, min(45, max_proposals // 12))]:
            pair = pairs[int(pos)]
            if local_scores[int(pos)] > 0.0 and pair not in used:
                used.add(pair)
                selected.append(pair)
    # Random exploration protects against a misleading local Cox direction.
    remaining = max(0, max_proposals - len(selected))
    if remaining:
        random_order = rng.permutation(len(pairs))
        for pos in random_order:
            pair = pairs[int(pos)]
            if pair in used:
                continue
            used.add(pair)
            selected.append(pair)
            if len(selected) >= max_proposals:
                break
    return selected[:max_proposals]


def _score_payload_ok(
    payload: dict[str, Any],
    utility_floor: float = BASE_UTILITY,
    ugen_floor: float = BASE_UGEN,
    urare_floor: float = BASE_URARE,
    uvalid_floor: float = BASE_UVALID,
) -> bool:
    facets = payload.get("facets", {})
    return bool(
        float(payload.get("utility", 0.0)) > utility_floor
        and float(facets.get("U_gen", 0.0)) >= ugen_floor
        and float(facets.get("U_rare", 0.0)) >= urare_floor
        and float(facets.get("U_valid", 0.0)) >= uvalid_floor
    )


def _run_privacy(path: Path, dist: Path, b: pd.DataFrame, b_rare: np.ndarray, a_bg: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    c = pd.read_csv(path)
    controls = _sample_controls(a_bg, len(b), seed=0)
    controls_rare = _rare_mask_from_reference(controls, dist)
    return _audit_one(path, c, b, controls, b_rare, controls_rare, a_bg, cfg)


def _run_full_static(path: Path, dist: Path) -> dict[str, Any]:
    members = pd.read_csv(dist / "B_practice.csv")
    b_self = pd.read_csv(dist / "B_self.csv")
    a_bg = pd.read_csv(dist / "A_bg.csv")
    if len(members) != len(b_self):
        raise ValueError("B_practice and B_self length mismatch")
    controls = a_bg.iloc[
        np.random.default_rng(0).choice(len(a_bg), size=len(members), replace=False)
    ].reset_index(drop=True)
    member_rare = _member_rare_mask(members, b_self)
    detector = _load_detector(dist)
    from experiments.run_aligned_synth import _detector_mask  # noqa: E402

    control_rare = _detector_mask(controls, *detector)
    mean = a_bg[FULL_STATIC_NUMERIC].to_numpy(dtype=float).mean(axis=0)
    std = a_bg[FULL_STATIC_NUMERIC].to_numpy(dtype=float).std(axis=0, ddof=0)
    std = np.where(std > 0.0, std, 1.0)
    return _full_static_audit_one(
        path,
        pd.read_csv(path),
        members,
        controls,
        member_rare,
        control_rare,
        mean,
        std,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safe outcome-triplet swap hill climb")
    ap.add_argument("--dist", default="../../_private_team10")
    ap.add_argument("--base", default="../../_private_team10/prefecture_extended_best0350/C_prefext_prefix12.csv")
    ap.add_argument("--out-dir", default="../../_private_team10/outcome_swap_hillclimb")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--iter-proposals", type=int, default=500)
    ap.add_argument("--max-iterations", type=int, default=6)
    ap.add_argument("--max-evals", type=int, default=3000)
    ap.add_argument("--deadline-seconds", type=float, default=600.0)
    ap.add_argument("--phase2-max-evals", type=int, default=1000)
    ap.add_argument("--utility-floor", type=float, default=BASE_UTILITY)
    ap.add_argument("--ugen-floor", type=float, default=BASE_UGEN)
    ap.add_argument("--urare-floor", type=float, default=BASE_URARE)
    ap.add_argument("--uvalid-floor", type=float, default=BASE_UVALID)
    ap.add_argument("--static-auc-max", type=float, default=None)
    ap.add_argument("--static-tpr-max", type=float, default=None)
    ap.add_argument("--static-rare-tpr-max", type=float, default=1.0 / 24.0)
    ap.add_argument("--static-member-near-max", type=int, default=4)
    ap.add_argument("--static-control-near-exact", type=int, default=0)
    args = ap.parse_args(argv)

    dist = dist_paths(args.dist)
    base_path = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    scores_dir = out_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    b = read_b(dist)
    base = pd.read_csv(base_path)
    if list(base.columns) != list(submission_schema.SUBMITTED_C_COLUMNS):
        raise ValueError("base must have submitted C columns")
    if len(base) != len(b):
        raise ValueError(f"base rows {len(base)} != B rows {len(b)}")
    static = base[STATIC_COLUMNS].copy(deep=True)
    current = base.copy(deep=True)
    current.insert(0, "record_id", np.arange(len(current), dtype=int))
    base_distributed = current.copy(deep=True)

    detector = _load_detector(dist)
    # The detector consumes only static columns; this is fixed for the full run.
    from experiments.run_aligned_synth import _detector_mask  # noqa: E402

    rare_mask = _detector_mask(static, *detector)
    if int(rare_mask.sum()) != 61:
        raise ValueError(f"expected public rare count 61, got {int(rare_mask.sum())}")
    b_rare_for_privacy, b_self, a_bg, privacy_cfg = (None, None, None, None)
    try:
        _, b_self, a_bg, privacy_cfg = _load_inputs(dist)
        b_rare_for_privacy = _member_rare_mask(b, b_self)
    except Exception as exc:  # privacy is a final gate; preserve diagnostic context
        privacy_init_error = f"{type(exc).__name__}: {exc}"
    else:
        privacy_init_error = None

    _assert_invariants(base_distributed, current, rare_mask, rare_mask)
    scorer = FastSpec(b)
    base_fast, base_detail = scorer.score(current)
    rng = np.random.default_rng(int(args.seed))
    evaluation_count = 0
    accepted: list[dict[str, Any]] = []
    best = current.copy(deep=True)
    best_score = base_fast
    best_detail = base_detail
    phase1_start_score = best_score

    def evaluate_and_fullscore(frame: pd.DataFrame, label: str, fast_score: float, detail: dict[str, Any]) -> dict[str, Any]:
        csv_path = out_dir / f"{label}.csv"
        score_path = scores_dir / f"{label}.json"
        write_c(frame, csv_path)
        payload, error = _score_one(csv_path, dist, score_path)
        return {
            "label": label,
            "csv": csv_path.name,
            "fast_U_spec": float(fast_score),
            "fast_detail": detail,
            "payload": payload,
            "error": error,
        }

    base_full = evaluate_and_fullscore(current, "base", base_fast, base_detail)
    (out_dir / "base_full_summary.json").write_text(
        json.dumps(base_full, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    phase_specs = [
        ("phase1_common_onset", (False, 1, 0), int(args.max_evals)),
    ]
    no_improve_iterations = 0
    phase_results: list[dict[str, Any]] = []
    for phase_name, bucket_key, phase_budget in phase_specs:
        phase_best_before = best_score
        phase_evals = 0
        for iteration in range(int(args.max_iterations) if phase_name.startswith("phase1") else 2):
            if evaluation_count >= int(args.max_evals) or time.monotonic() - started >= float(args.deadline_seconds) - 25.0:
                break
            current_score, current_detail = scorer.score(current)
            beta_c = np.asarray(
                [float(current_detail.get("per_coefficient", {}).get(col, {}).get("beta_C", 0.0)) for col in _COX_COLUMNS],
                dtype=float,
            )
            beta_error = beta_c - scorer.b_beta
            pairs = _pair_pool(current, rare_mask, bucket_key, beta_error, rng, int(args.iter_proposals))
            best_local: tuple[float, dict[str, Any], pd.DataFrame, tuple[int, int]] | None = None
            for pair in pairs:
                if evaluation_count >= phase_budget or evaluation_count >= int(args.max_evals):
                    break
                if time.monotonic() - started >= float(args.deadline_seconds) - 25.0:
                    break
                proposal = _swap(current, *pair)
                evaluation_count += 1
                phase_evals += 1
                score, detail = scorer.score(proposal)
                if not detail.get("cox_converged", False):
                    continue
                candidate = (score, detail, proposal, pair)
                if best_local is None or (score, float(detail.get("io_part", 0.0))) > (
                    best_local[0], float(best_local[1].get("io_part", 0.0))
                ):
                    best_local = candidate
            improved = bool(best_local is not None and best_local[0] > current_score + 1e-10)
            if improved:
                _, current_detail, current, pair = best_local  # type: ignore[misc]
                current_score = float(best_local[0])
                _assert_invariants(base_distributed, current, rare_mask, rare_mask)
                accepted_label = f"accepted_{phase_name}_iter{iteration + 1:02d}"
                full = evaluate_and_fullscore(current, accepted_label, current_score, current_detail)
                accepted.append({"phase": phase_name, "iteration": iteration + 1, "pair": list(pair), **full})
                if current_score > best_score:
                    best = current.copy(deep=True)
                    best_score = current_score
                    best_detail = current_detail
                no_improve_iterations = 0
                print(f"accepted {phase_name} iter={iteration + 1} pair={pair} fast_U_spec={current_score:.12f} evals={evaluation_count}")
            else:
                no_improve_iterations += 1
                print(f"no improvement {phase_name} iter={iteration + 1} current={current_score:.12f} evals={evaluation_count}")
            if no_improve_iterations >= 2:
                break
        phase_results.append(
            {
                "phase": phase_name,
                "bucket": list(bucket_key),
                "evaluations": phase_evals,
                "score_before": phase_best_before,
                "score_after": best_score,
                "gain": best_score - phase_best_before,
            }
        )

    # Optional conservative Phase 2: common death-time bucket only, never rare/cross-status.
    phase2_gain = best_score - phase1_start_score
    if (
        # A gain below roughly 0.25 percentage points leaves useful room for
        # the lower-leverage common death-time bucket.
        phase2_gain < 0.0025
        and int(args.phase2_max_evals) > 0
        and evaluation_count < int(args.max_evals) + int(args.phase2_max_evals)
        and time.monotonic() - started < float(args.deadline_seconds) - 120.0
    ):
        phase_name = "phase2_common_death"
        bucket_key = (False, 0, 1)
        phase_evals = 0
        for iteration in range(2):
            if phase_evals >= int(args.phase2_max_evals) or time.monotonic() - started >= float(args.deadline_seconds) - 25.0:
                break
            current_score, current_detail = scorer.score(current)
            beta_c = np.asarray(
                [float(current_detail.get("per_coefficient", {}).get(col, {}).get("beta_C", 0.0)) for col in _COX_COLUMNS],
                dtype=float,
            )
            pairs = _pair_pool(current, rare_mask, bucket_key, beta_c - scorer.b_beta, rng, min(500, int(args.iter_proposals)))
            best_local = None
            for pair in pairs:
                if phase_evals >= int(args.phase2_max_evals) or time.monotonic() - started >= float(args.deadline_seconds) - 25.0:
                    break
                proposal = _swap(current, *pair)
                evaluation_count += 1
                phase_evals += 1
                score, detail = scorer.score(proposal)
                if detail.get("cox_converged", False) and (best_local is None or (score, float(detail.get("io_part", 0.0))) > (best_local[0], float(best_local[1].get("io_part", 0.0)))):
                    best_local = (score, detail, proposal, pair)
            if best_local is None or best_local[0] <= current_score + 1e-10:
                break
            current = best_local[2]
            _assert_invariants(base_distributed, current, rare_mask, rare_mask)
            current_score, current_detail = best_local[0], best_local[1]
            label = f"accepted_{phase_name}_iter{iteration + 1:02d}"
            full = evaluate_and_fullscore(current, label, current_score, current_detail)
            accepted.append({"phase": phase_name, "iteration": iteration + 1, "pair": list(best_local[3]), **full})
            if current_score > best_score:
                best = current.copy(deep=True)
                best_score = current_score
                best_detail = current_detail
        phase_results.append({"phase": phase_name, "bucket": list(bucket_key), "evaluations": phase_evals, "score_after": best_score})

    _assert_invariants(base_distributed, best, rare_mask, rare_mask)
    final_full = evaluate_and_fullscore(best, "final_best", best_score, best_detail)

    privacy: dict[str, Any] = {"init_error": privacy_init_error}
    if privacy_init_error is None and b_rare_for_privacy is not None and a_bg is not None and privacy_cfg is not None:
        base_privacy = _run_privacy(base_path, dist, b, b_rare_for_privacy, a_bg, privacy_cfg)
        final_privacy = _run_privacy(out_dir / "final_best.csv", dist, b, b_rare_for_privacy, a_bg, privacy_cfg)
        privacy = {
            "base": base_privacy,
            "final": final_privacy,
            "mia_same": abs(float(base_privacy["mia"]["exposure"]) - float(final_privacy["mia"]["exposure"])) <= 1e-12,
            "aia_zero": float(final_privacy["aia"]["attack_score"]) <= float(base_privacy["aia"]["attack_score"]) + 1e-12,
        }

    full_static: dict[str, Any] = {"init_error": None}
    try:
        base_full_static = _run_full_static(base_path, dist)
        final_full_static = _run_full_static(out_dir / "final_best.csv", dist)
        full_static = {
            "base": base_full_static,
            "final": final_full_static,
            "auc_ok": (
                args.static_auc_max is None
                # The requested threshold is rounded to 10 decimals; accept
                # the deterministic audit value's last few floating bits.
                or float(final_full_static["roc_auc"]) <= float(args.static_auc_max) + 5e-12
            ),
            "tpr_ok": (
                args.static_tpr_max is None
                or float(final_full_static["all_tpr_at_fpr_0.01"]) <= float(args.static_tpr_max) + 1e-12
            ),
            "rare_tpr_ok": float(final_full_static["true_rare_tpr_at_fpr_0.01"]) <= float(args.static_rare_tpr_max) + 1e-12,
            "member_near_ok": int(final_full_static["member_d_le_0.5"]) <= int(args.static_member_near_max),
            "control_near_ok": int(final_full_static["control_d_le_0.5"]) == int(args.static_control_near_exact),
            "exact_d0_ok": int(final_full_static["member_exact_d0"]) == 0 and int(final_full_static["control_exact_d0"]) == 0,
            "exact_13col_ok": int(final_full_static["member_exact_match_13col"]) == 0 and int(final_full_static["control_exact_match_13col"]) == 0,
        }
    except Exception as exc:  # full static is a hard final gate, retain error in summary
        full_static = {"init_error": f"{type(exc).__name__}: {exc}"}

    final_payload = final_full.get("payload") or {}
    gates = {
        "full_selfscore_ok": _score_payload_ok(
            final_payload,
            args.utility_floor,
            float(args.ugen_floor) - 1e-11,
            float(args.urare_floor) - 1e-11,
            float(args.uvalid_floor) - 1e-11,
        ),
        "utility_gt_base": float(final_payload.get("utility", 0.0)) > float(args.utility_floor),
        "U_gen_gate": float(final_payload.get("facets", {}).get("U_gen", 0.0)) >= float(args.ugen_floor) - 1e-11,
        "U_rare_gate": float(final_payload.get("facets", {}).get("U_rare", 0.0)) >= float(args.urare_floor) - 1e-11,
        "U_valid_gate": float(final_payload.get("facets", {}).get("U_valid", 0.0)) >= float(args.uvalid_floor) - 1e-11,
        "privacy_mia_same": bool(privacy.get("mia_same", False)),
        "privacy_aia_safe": bool(privacy.get("aia_zero", False)),
        "full_static_auc": bool(full_static.get("auc_ok", False)),
        "full_static_tpr": bool(full_static.get("tpr_ok", False)),
        "full_static_rare_tpr": bool(full_static.get("rare_tpr_ok", False)),
        "full_static_member_near": bool(full_static.get("member_near_ok", False)),
        "full_static_control_near": bool(full_static.get("control_near_ok", False)),
        "full_static_exact_d0": bool(full_static.get("exact_d0_ok", False)),
        "full_static_exact_13col": bool(full_static.get("exact_13col_ok", False)),
    }
    summary = {
        "base": str(base_path),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "elapsed_seconds": time.monotonic() - started,
        "evaluations": evaluation_count,
        "base_fast_U_spec": base_fast,
        "best_fast_U_spec": best_score,
        "best_fast_detail": best_detail,
        "phase_results": phase_results,
        "accepted": accepted,
        "final_full": final_full,
        "privacy": privacy,
        "full_static": full_static,
        "gates": gates,
        "go": bool(all(gates.values())),
    }
    (out_dir / "outcome_swap_hillclimb_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_fast_U_spec": best_score, "evaluations": evaluation_count, "gates": gates, "go": summary["go"]}, ensure_ascii=False))
    return 0 if summary["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
