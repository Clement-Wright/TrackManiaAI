from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from ..env.trajectory import RuntimeTrajectory


def _mean(values: Sequence[float]) -> float:
    return float(fmean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def determinism_conversion_score(
    deterministic_progress: float | int | None,
    stochastic_progress: float | int | None,
    *,
    epsilon: float = 1.0e-6,
) -> float | None:
    """Return how much stochastic progress survives deterministic extraction."""
    if deterministic_progress is None or stochastic_progress is None:
        return None
    denominator = max(float(stochastic_progress), float(epsilon))
    return float(deterministic_progress) / denominator


def mode_comparison_metrics(mode_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    deterministic = mode_summaries.get("deterministic")
    stochastic = mode_summaries.get("stochastic")
    if deterministic is None or stochastic is None:
        return {
            "determinism_conversion_score": None,
            "deterministic_stochastic_progress_gap": None,
            "deterministic_stochastic_completion_gap": None,
            "deterministic_stochastic_median_progress_gap": None,
        }
    deterministic_progress = float(deterministic.get("mean_final_progress_index", 0.0) or 0.0)
    stochastic_progress = float(stochastic.get("mean_final_progress_index", 0.0) or 0.0)
    deterministic_completion = float(deterministic.get("completion_rate", 0.0) or 0.0)
    stochastic_completion = float(stochastic.get("completion_rate", 0.0) or 0.0)
    deterministic_median = deterministic.get("median_final_progress_index")
    stochastic_median = stochastic.get("median_final_progress_index")
    median_gap = None
    if deterministic_median is not None and stochastic_median is not None:
        median_gap = float(deterministic_median) - float(stochastic_median)
    return {
        "determinism_conversion_score": determinism_conversion_score(
            deterministic_progress,
            stochastic_progress,
        ),
        "deterministic_stochastic_progress_gap": deterministic_progress - stochastic_progress,
        "deterministic_stochastic_completion_gap": deterministic_completion - stochastic_completion,
        "deterministic_stochastic_median_progress_gap": median_gap,
    }


def time_to_progress_thresholds(
    episode_summaries: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[int | float],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for threshold in thresholds:
        threshold_value = float(threshold)
        crossing_times = [
            float(episode.get("completion_time_ms"))
            for episode in episode_summaries
            if episode.get("completion_time_ms") is not None
            and float(episode.get("final_progress_index", 0.0) or 0.0) >= threshold_value
        ]
        result[str(threshold_value)] = min(crossing_times) if crossing_times else None
    return result


@dataclass(slots=True)
class ActiveStepBenchmark:
    step_dt_seconds: float
    active_steps: int = 0
    cumulative_step_duration_seconds: float = 0.0
    max_step_duration_seconds: float = 0.0
    cumulative_step_overrun_seconds: float = 0.0
    max_step_overrun_seconds: float = 0.0
    cumulative_active_drift_seconds: float = 0.0
    max_active_drift_seconds: float = 0.0
    reset_count: int = 0
    cumulative_reset_time_seconds: float = 0.0
    max_reset_duration_seconds: float = 0.0
    _expected_next: float | None = None

    def reanchor(self, now: float | None = None) -> None:
        self._expected_next = time.perf_counter() if now is None else now

    def record_step(self, *, end_time: float, duration_seconds: float) -> None:
        if self._expected_next is None:
            self.reanchor(end_time - duration_seconds)
        self.active_steps += 1
        self.cumulative_step_duration_seconds += duration_seconds
        self.max_step_duration_seconds = max(self.max_step_duration_seconds, duration_seconds)

        overrun = max(0.0, duration_seconds - self.step_dt_seconds)
        self.cumulative_step_overrun_seconds += overrun
        self.max_step_overrun_seconds = max(self.max_step_overrun_seconds, overrun)

        assert self._expected_next is not None
        self._expected_next += self.step_dt_seconds
        drift = max(0.0, end_time - self._expected_next)
        self.cumulative_active_drift_seconds += drift
        self.max_active_drift_seconds = max(self.max_active_drift_seconds, drift)

    def record_reset(self, duration_seconds: float, *, reanchor_time: float | None = None) -> None:
        self.reset_count += 1
        self.cumulative_reset_time_seconds += duration_seconds
        self.max_reset_duration_seconds = max(self.max_reset_duration_seconds, duration_seconds)
        self.reanchor(reanchor_time)

    def to_report(
        self,
        *,
        episodes: int,
        wall_clock_total_seconds: float,
        avg_obs_retrieval_seconds: float,
        avg_send_control_seconds: float,
        avg_reward_compute_seconds: float,
        avg_preprocess_seconds: float,
        raw_rtgym_benchmarks: Mapping[str, tuple[float | None, float | None]],
    ) -> dict[str, Any]:
        return {
            "active_steps": self.active_steps,
            "episodes": episodes,
            "step_dt_seconds": self.step_dt_seconds,
            "avg_step_duration_seconds": self.cumulative_step_duration_seconds / max(1, self.active_steps),
            "max_step_duration_seconds": self.max_step_duration_seconds,
            "avg_step_overrun_seconds": self.cumulative_step_overrun_seconds / max(1, self.active_steps),
            "max_step_overrun_seconds": self.max_step_overrun_seconds,
            "avg_obs_retrieval_seconds": avg_obs_retrieval_seconds,
            "avg_send_control_seconds": avg_send_control_seconds,
            "avg_reward_compute_seconds": avg_reward_compute_seconds,
            "avg_preprocess_seconds": avg_preprocess_seconds,
            "reset_count": self.reset_count,
            "avg_reset_duration_seconds": self.cumulative_reset_time_seconds / max(1, self.reset_count),
            "max_reset_duration_seconds": self.max_reset_duration_seconds,
            "cumulative_reset_time_seconds": self.cumulative_reset_time_seconds,
            "cumulative_active_drift_seconds": self.cumulative_active_drift_seconds,
            "max_active_drift_seconds": self.max_active_drift_seconds,
            "wall_clock_total_seconds": wall_clock_total_seconds,
            "rtgym_benchmarks": {
                key: {"average": average, "average_deviation": deviation}
                for key, (average, deviation) in raw_rtgym_benchmarks.items()
            },
        }


def summarize_episode_trace(
    *,
    episode_id: str,
    metadata: Mapping[str, Any],
    step_rows: Sequence[Mapping[str, Any]],
    trajectory: RuntimeTrajectory,
    sector_count: int,
) -> dict[str, Any]:
    progress_values = [int(row["progress_index"]) for row in step_rows]
    progress_meter_values = [
        float(row.get("progress_arc_length_m", row.get("trajectory_arc_length_m", 0.0)) or 0.0)
        for row in step_rows
    ]
    progress_fraction_values = [
        float(row["progress_fraction_of_reference"])
        for row in step_rows
        if row.get("progress_fraction_of_reference") is not None
    ]
    reference_total_arc_values = [
        float(row["reference_total_arc_length_m"])
        for row in step_rows
        if row.get("reference_total_arc_length_m") is not None
    ]
    ghost_relative_time_delta_values = [
        float(row["ghost_relative_time_delta_ms"])
        for row in step_rows
        if row.get("ghost_relative_time_delta_ms") is not None
    ]
    ghost_reference_time_values = [
        float(row["ghost_reference_time_ms"])
        for row in step_rows
        if row.get("ghost_reference_time_ms") is not None
    ]
    progress_spacing_values = [
        float(row["progress_spacing_meters"])
        for row in step_rows
        if row.get("progress_spacing_meters") is not None
    ]
    progress_semantics = next(
        (str(row["progress_index_semantics"]) for row in step_rows if row.get("progress_index_semantics")),
        None,
    )
    reward_values = [float(row["reward"]) for row in step_rows]
    completion_times = [int(row["race_time_ms"]) for row in step_rows if row.get("done_type") == "terminated" and row.get("terminal_reason") == "finished"]
    furthest_progress = max(progress_values) if progress_values else 0
    final_progress = progress_values[-1] if progress_values else 0
    furthest_progress_meters = max(progress_meter_values) if progress_meter_values else 0.0
    final_progress_meters = progress_meter_values[-1] if progress_meter_values else 0.0
    furthest_progress_fraction = max(progress_fraction_values) if progress_fraction_values else None
    final_progress_fraction = progress_fraction_values[-1] if progress_fraction_values else None
    final_ghost_relative_time_delta_ms = (
        ghost_relative_time_delta_values[-1] if ghost_relative_time_delta_values else None
    )
    furthest_sector = trajectory.sector_index_for_progress(furthest_progress, sector_count) if progress_values else 0

    sector_entry_speed = [0.0 for _ in range(sector_count)]
    sector_reward_totals = [0.0 for _ in range(sector_count)]
    seen_sector_entries: set[int] = set()
    for row in step_rows:
        sector_index = int(row["sector_index"])
        sector_reward_totals[sector_index] += float(row["reward"])
        if sector_index not in seen_sector_entries:
            seen_sector_entries.add(sector_index)
        sector_entry_speed[sector_index] = max(sector_entry_speed[sector_index], float(row["speed_kmh"]))

    termination_reason = metadata.get("termination_reason")
    done_type = metadata.get("done_type")
    completion_flag = bool(metadata.get("completion_flag"))
    return {
        "episode_id": episode_id,
        "map_uid": metadata.get("map_uid"),
        "run_id": metadata.get("run_id"),
        "episode_seed": metadata.get("episode_seed"),
        "start_timestamp": metadata.get("start_timestamp"),
        "end_timestamp": metadata.get("end_timestamp"),
        "step_count": len(step_rows),
        "best_progress_index": furthest_progress,
        "final_progress_index": final_progress,
        "best_progress_meters": furthest_progress_meters,
        "final_progress_meters": final_progress_meters,
        "best_arc_length_m": furthest_progress_meters,
        "final_arc_length_m": final_progress_meters,
        "best_progress_fraction_of_reference": furthest_progress_fraction,
        "progress_fraction_of_reference": final_progress_fraction,
        "reference_total_arc_length_m": reference_total_arc_values[-1] if reference_total_arc_values else None,
        "progress_spacing_meters": progress_spacing_values[-1] if progress_spacing_values else None,
        "progress_index_semantics": progress_semantics,
        "ghost_reference_time_ms": ghost_reference_time_values[-1] if ghost_reference_time_values else None,
        "ghost_relative_time_delta_ms": final_ghost_relative_time_delta_ms,
        "completion_flag": completion_flag,
        "completion_time_ms": completion_times[0] if completion_times else None,
        "termination_reason": termination_reason,
        "done_type": done_type,
        "episode_reward_total": sum(reward_values),
        "furthest_sector_reached": furthest_sector,
        "best_sector_entry_speed": sector_entry_speed,
        "average_sector_reward_gain": [
            sector_reward_totals[index] / max(1, sum(1 for row in step_rows if int(row["sector_index"]) == index))
            for index in range(sector_count)
        ],
        "video_path": metadata.get("video_path"),
        "sampled_frames_dir": metadata.get("sampled_frames_dir"),
    }


def aggregate_episode_summaries(episodes: Sequence[Mapping[str, Any]], *, sector_count: int) -> dict[str, Any]:
    completion_flags = [bool(episode.get("completion_flag")) for episode in episodes]
    final_progress = [float(episode.get("final_progress_index", 0)) for episode in episodes]
    final_progress_meters = [
        float(episode["final_progress_meters"])
        for episode in episodes
        if episode.get("final_progress_meters") is not None
    ]
    final_arc_length_values = [
        float(episode["final_arc_length_m"])
        for episode in episodes
        if episode.get("final_arc_length_m") is not None
    ]
    progress_fraction_values = [
        float(episode["progress_fraction_of_reference"])
        for episode in episodes
        if episode.get("progress_fraction_of_reference") is not None
    ]
    reference_total_arc_values = [
        float(episode["reference_total_arc_length_m"])
        for episode in episodes
        if episode.get("reference_total_arc_length_m") is not None
    ]
    ghost_relative_time_delta_values = [
        float(episode["ghost_relative_time_delta_ms"])
        for episode in episodes
        if episode.get("ghost_relative_time_delta_ms") is not None
    ]
    ghost_reference_time_values = [
        float(episode["ghost_reference_time_ms"])
        for episode in episodes
        if episode.get("ghost_reference_time_ms") is not None
    ]
    progress_spacings = [
        float(episode["progress_spacing_meters"])
        for episode in episodes
        if episode.get("progress_spacing_meters") is not None
    ]
    progress_semantics = next(
        (str(episode["progress_index_semantics"]) for episode in episodes if episode.get("progress_index_semantics")),
        None,
    )
    rewards = [float(episode.get("episode_reward_total", 0.0)) for episode in episodes]
    completion_times = [float(episode["completion_time_ms"]) for episode in episodes if episode.get("completion_time_ms") is not None]
    timeout_flags = [episode.get("termination_reason") in {"no_progress", "ep_max_length"} for episode in episodes]
    no_progress_flags = [episode.get("termination_reason") == "no_progress" for episode in episodes]
    stray_flags = [episode.get("termination_reason") == "stray" for episode in episodes]
    corridor_violation_flags = [episode.get("termination_reason") == "corridor_violation" for episode in episodes]

    per_sector_best_entry_speed = [0.0 for _ in range(sector_count)]
    per_sector_reward_gain: list[list[float]] = [[] for _ in range(sector_count)]
    for episode in episodes:
        speeds = episode.get("best_sector_entry_speed", [])
        gains = episode.get("average_sector_reward_gain", [])
        for index in range(sector_count):
            speed_value = float(speeds[index]) if index < len(speeds) else 0.0
            per_sector_best_entry_speed[index] = max(per_sector_best_entry_speed[index], speed_value)
            if index < len(gains):
                per_sector_reward_gain[index].append(float(gains[index]))

    return {
        "episode_count": len(episodes),
        "completion_rate": sum(completion_flags) / max(1, len(episodes)),
        "mean_final_progress_index": _mean(final_progress),
        "median_final_progress_index": _median(final_progress),
        "mean_final_progress_meters": _mean(final_progress_meters),
        "median_final_progress_meters": _median(final_progress_meters),
        "mean_final_arc_length_m": _mean(final_arc_length_values) if final_arc_length_values else None,
        "median_final_arc_length_m": _median(final_arc_length_values),
        "mean_progress_fraction_of_reference": _mean(progress_fraction_values) if progress_fraction_values else None,
        "median_progress_fraction_of_reference": _median(progress_fraction_values),
        "reference_total_arc_length_m": reference_total_arc_values[-1] if reference_total_arc_values else None,
        "progress_spacing_meters": progress_spacings[-1] if progress_spacings else None,
        "progress_index_semantics": progress_semantics,
        "mean_ghost_reference_time_ms": _mean(ghost_reference_time_values) if ghost_reference_time_values else None,
        "median_ghost_reference_time_ms": _median(ghost_reference_time_values),
        "mean_ghost_relative_time_delta_ms": (
            _mean(ghost_relative_time_delta_values) if ghost_relative_time_delta_values else None
        ),
        "median_ghost_relative_time_delta_ms": _median(ghost_relative_time_delta_values),
        "best_ghost_relative_time_delta_ms": min(ghost_relative_time_delta_values) if ghost_relative_time_delta_values else None,
        "timeout_rate": sum(timeout_flags) / max(1, len(episodes)),
        "no_progress_termination_rate": sum(no_progress_flags) / max(1, len(episodes)),
        "stray_termination_rate": sum(stray_flags) / max(1, len(episodes)),
        "corridor_violation_truncation_rate": sum(corridor_violation_flags) / max(1, len(episodes)),
        "mean_episode_reward": _mean(rewards),
        "best_reward": max(rewards) if rewards else None,
        "best_completion_time_ms": min(completion_times) if completion_times else None,
        "median_completion_time_ms": _median(completion_times),
        "sector_metrics": {
            "best_sector_entry_speed": per_sector_best_entry_speed,
            "average_sector_reward_gain": [_mean(values) for values in per_sector_reward_gain],
        },
    }


class TensorBoardScalarLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = EventFileWriter(str(log_dir))

    def add_scalar(self, tag: str, scalar_value: float, step: int) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(scalar_value))])
        event = Event(wall_time=time.time(), step=step, summary=summary)
        self._writer.add_event(event)
        self._writer.flush()

    def add_scalars_from_mapping(self, prefix: str, payload: Mapping[str, Any], step: int) -> None:
        for key, value in payload.items():
            if isinstance(value, Mapping):
                self.add_scalars_from_mapping(f"{prefix}/{key}" if prefix else str(key), value, step)
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                for index, item in enumerate(value):
                    if item is None:
                        continue
                    self.add_scalar(f"{prefix}/{key}/{index}" if prefix else f"{key}/{index}", float(item), step)
            elif value is not None and isinstance(value, (int, float, bool)):
                self.add_scalar(f"{prefix}/{key}" if prefix else str(key), float(value), step)

    def close(self) -> None:
        self._writer.close()
