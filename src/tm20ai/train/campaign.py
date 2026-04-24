from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data.parquet_writer import read_json
from .metrics import determinism_conversion_score
from .reporting import build_training_report


@dataclass(slots=True, frozen=True)
class RunValidation:
    run_dir: Path
    valid: bool
    reasons: tuple[str, ...]
    exact_final_eval_complete: bool
    incomplete_final_eval: bool
    final_eval_state: str | None
    deterministic_summary_path: str | None
    stochastic_summary_path: str | None


@dataclass(slots=True, frozen=True)
class RewardCandidate:
    run_dir: Path
    run_name: str
    exact_final_progress: float
    exact_final_progress_fraction: float | None
    exact_final_ghost_delta_ms: float | None
    exact_final_corridor_violation_truncation_rate: float | None
    corridor_nonrecovering_p95: float | None
    report: Mapping[str, Any]
    summary: Mapping[str, Any]


@dataclass(slots=True, frozen=True)
class ScheduledCheckpointCandidate:
    checkpoint_path: str
    progress: float
    env_step: int


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_p95(payload: Mapping[str, Any], key: str) -> float | None:
    metric = payload.get(key)
    if not isinstance(metric, Mapping):
        return None
    return _safe_float(metric.get("p95"))


def validate_campaign_run(run_dir: str | Path) -> RunValidation:
    resolved_run_dir = Path(run_dir).resolve()
    summary_path = resolved_run_dir / "summary.json"
    if not summary_path.exists():
        return RunValidation(
            run_dir=resolved_run_dir,
            valid=False,
            reasons=("summary_missing",),
            exact_final_eval_complete=False,
            incomplete_final_eval=True,
            final_eval_state=None,
            deterministic_summary_path=None,
            stochastic_summary_path=None,
        )
    summary = read_json(summary_path)
    exact_final_eval_complete = bool(summary.get("exact_final_eval_complete", False))
    incomplete_final_eval = bool(summary.get("incomplete_final_eval", True))
    final_eval_state = None if summary.get("final_eval_state") is None else str(summary.get("final_eval_state"))
    mode_paths = dict(summary.get("exact_final_eval_mode_summary_paths") or {})
    deterministic_summary_path = None if mode_paths.get("deterministic") is None else str(mode_paths.get("deterministic"))
    stochastic_summary_path = None if mode_paths.get("stochastic") is None else str(mode_paths.get("stochastic"))

    reasons: list[str] = []
    if not exact_final_eval_complete:
        reasons.append("exact_final_eval_complete_false")
    if incomplete_final_eval:
        reasons.append("incomplete_final_eval_true")
    if final_eval_state != "complete":
        reasons.append(f"final_eval_state={final_eval_state}")
    if deterministic_summary_path is None:
        reasons.append("deterministic_summary_path_missing")
    elif not Path(deterministic_summary_path).exists():
        reasons.append("deterministic_summary_missing_on_disk")
    if stochastic_summary_path is None:
        reasons.append("stochastic_summary_path_missing")
    elif not Path(stochastic_summary_path).exists():
        reasons.append("stochastic_summary_missing_on_disk")

    return RunValidation(
        run_dir=resolved_run_dir,
        valid=not reasons,
        reasons=tuple(reasons),
        exact_final_eval_complete=exact_final_eval_complete,
        incomplete_final_eval=incomplete_final_eval,
        final_eval_state=final_eval_state,
        deterministic_summary_path=deterministic_summary_path,
        stochastic_summary_path=stochastic_summary_path,
    )


def build_reward_candidates(run_dirs: Sequence[str | Path]) -> list[RewardCandidate]:
    candidates: list[RewardCandidate] = []
    for run_dir in run_dirs:
        resolved_run_dir = Path(run_dir).resolve()
        report = build_training_report(resolved_run_dir)
        summary = read_json(resolved_run_dir / "summary.json")
        exact_final_summary = dict(summary.get("exact_final_eval_summary") or {})
        episode_diagnostics = dict(report.get("episode_diagnostics") or {})
        candidates.append(
            RewardCandidate(
                run_dir=resolved_run_dir,
                run_name=str(report.get("run_name") or resolved_run_dir.name),
                exact_final_progress=float(exact_final_summary.get("mean_final_progress_index", 0.0) or 0.0),
                exact_final_progress_fraction=_safe_float(exact_final_summary.get("mean_progress_fraction_of_reference")),
                exact_final_ghost_delta_ms=_safe_float(exact_final_summary.get("mean_ghost_relative_time_delta_ms")),
                exact_final_corridor_violation_truncation_rate=_safe_float(
                    exact_final_summary.get("corridor_violation_truncation_rate")
                ),
                corridor_nonrecovering_p95=_metric_p95(episode_diagnostics, "corridor_nonrecovering_steps"),
                report=report,
                summary=summary,
            )
        )
    return candidates


def select_reward_winner(run_dirs: Sequence[str | Path]) -> tuple[RewardCandidate, list[RewardCandidate]]:
    candidates = build_reward_candidates(run_dirs)
    if not candidates:
        raise RuntimeError("At least one run is required to select a reward winner.")

    candidates.sort(key=lambda item: item.exact_final_progress, reverse=True)
    top_progress = max(candidate.exact_final_progress for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.exact_final_progress >= (top_progress * 0.95)
    ]
    if len(eligible) >= 2:
        eligible.sort(
            key=lambda item: (
                float("inf") if item.exact_final_ghost_delta_ms is None else item.exact_final_ghost_delta_ms,
                -1.0 if item.exact_final_progress_fraction is None else -item.exact_final_progress_fraction,
                float("inf")
                if item.exact_final_corridor_violation_truncation_rate is None
                else item.exact_final_corridor_violation_truncation_rate,
                float("inf") if item.corridor_nonrecovering_p95 is None else item.corridor_nonrecovering_p95,
                -item.exact_final_progress,
            )
        )
        winner = eligible[0]
    else:
        winner = candidates[0]
    return winner, candidates


def best_scheduled_deterministic_checkpoint(run_dir: str | Path) -> ScheduledCheckpointCandidate | None:
    summary = read_json(Path(run_dir).resolve() / "summary.json")
    best_candidate: ScheduledCheckpointCandidate | None = None
    for entry in summary.get("eval_history", []):
        if bool(entry.get("final_checkpoint_eval", False)):
            continue
        mode_summaries = dict(entry.get("mode_summaries") or {})
        deterministic_summary = dict(mode_summaries.get("deterministic") or entry.get("summary") or {})
        progress = float(deterministic_summary.get("mean_final_progress_index", 0.0) or 0.0)
        checkpoint_path = deterministic_summary.get("eval_checkpoint_path") or entry.get("eval_checkpoint_path")
        if checkpoint_path in (None, "") or not Path(str(checkpoint_path)).exists():
            continue
        candidate = ScheduledCheckpointCandidate(
            checkpoint_path=str(checkpoint_path),
            progress=progress,
            env_step=int(entry.get("env_step", deterministic_summary.get("env_step", 0)) or 0),
        )
        if best_candidate is None or candidate.progress > best_candidate.progress:
            best_candidate = candidate
    return best_candidate


def analyze_policy_mode_sweep_results(
    results: Mapping[str, Mapping[str, Any]],
    *,
    dcs_target: float = 0.85,
) -> dict[str, Any]:
    normalized_results = {str(name): dict(payload) for name, payload in results.items()}
    stochastic_reference = normalized_results.get("stochastic_temp_1")
    stochastic_reference_progress = None if stochastic_reference is None else _safe_float(
        stochastic_reference.get("mean_final_progress_index")
    )

    per_mode: dict[str, dict[str, Any]] = {}
    for mode_name, payload in normalized_results.items():
        progress = _safe_float(payload.get("mean_final_progress_index"))
        dcs = determinism_conversion_score(progress, stochastic_reference_progress)
        enriched = dict(payload)
        enriched["determinism_conversion_score"] = dcs
        per_mode[mode_name] = enriched

    deployment_choice = "deterministic_mean"
    deterministic_mean = per_mode.get("deterministic_mean")
    clipped_mean = per_mode.get("clipped_mean")
    if deterministic_mean is not None and clipped_mean is not None:
        baseline_progress = _safe_float(deterministic_mean.get("mean_final_progress_index")) or 0.0
        baseline_fraction = _safe_float(deterministic_mean.get("mean_progress_fraction_of_reference"))
        clipped_progress = _safe_float(clipped_mean.get("mean_final_progress_index")) or 0.0
        clipped_fraction = _safe_float(clipped_mean.get("mean_progress_fraction_of_reference"))
        if (
            clipped_progress >= baseline_progress * 1.05
            and (baseline_fraction is None or clipped_fraction is None or clipped_fraction >= baseline_fraction)
        ):
            deployment_choice = "clipped_mean"

    chosen_dcs = None if per_mode.get(deployment_choice) is None else per_mode[deployment_choice].get(
        "determinism_conversion_score"
    )
    return {
        "stochastic_reference_mode": "stochastic_temp_1" if stochastic_reference is not None else None,
        "deployment_choice": deployment_choice if deployment_choice in per_mode else None,
        "deployment_choice_meets_target": bool(
            chosen_dcs is not None and float(chosen_dcs) >= float(dcs_target)
        ),
        "dcs_target": float(dcs_target),
        "per_mode": per_mode,
    }
