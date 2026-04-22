from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq

from ..bridge import TelemetryFrame
from ..config import GhostConfig, RewardConfig
from .dataset import load_ghost_bundle_manifest


@dataclass(slots=True)
class RewardStepResult:
    reward: float
    done_type: str | None
    done_reason: str | None
    progress_index: int
    progress_delta: int
    stray_distance: float | None
    no_progress_steps: int
    info: dict[str, Any]


def _deduplicate_points(points: np.ndarray) -> np.ndarray:
    keep_indices = [0]
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[keep_indices[-1]])) > 1.0e-6:
            keep_indices.append(index)
    deduped = points[keep_indices]
    if len(deduped) < 2:
        raise ValueError("Ghost trajectory collapsed to fewer than two unique points.")
    return deduped


def _resample_fixed_spacing(
    points: np.ndarray,
    race_time_ms: np.ndarray,
    spacing_meters: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deduped = _deduplicate_points(np.asarray(points, dtype=np.float64))
    race_time = np.asarray(race_time_ms, dtype=np.float64)
    if race_time.shape[0] != points.shape[0]:
        raise ValueError("Ghost trajectory race_time_ms must align with positions.")
    dedup_keep_indices = [0]
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[dedup_keep_indices[-1]])) > 1.0e-6:
            dedup_keep_indices.append(index)
    dedup_time = race_time[dedup_keep_indices]
    segment_lengths = np.linalg.norm(np.diff(deduped, axis=0), axis=1)
    cumulative = np.concatenate((np.asarray([0.0], dtype=np.float64), np.cumsum(segment_lengths, dtype=np.float64)))
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError("Ghost trajectory length must be positive.")
    spacing = max(1.0e-6, float(spacing_meters))
    sample_arc = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if sample_arc.size == 0 or not np.isclose(sample_arc[-1], total_length):
        sample_arc = np.concatenate((sample_arc, np.asarray([total_length], dtype=np.float64)))
    x = np.interp(sample_arc, cumulative, deduped[:, 0])
    y = np.interp(sample_arc, cumulative, deduped[:, 1])
    z = np.interp(sample_arc, cumulative, deduped[:, 2])
    t = np.interp(sample_arc, cumulative, dedup_time)
    return (
        np.column_stack((x, y, z)).astype(np.float64, copy=False),
        sample_arc.astype(np.float64, copy=False),
        t.astype(np.float64, copy=False),
    )


@dataclass(frozen=True, slots=True)
class GhostLine:
    line_id: str
    rank: int | None
    trajectory_path: Path
    positions: np.ndarray
    arc_length: np.ndarray
    race_time_ms: np.ndarray
    source_row_index: np.ndarray
    spacing_meters: float

    @property
    def length(self) -> int:
        return int(self.positions.shape[0])

    def nearest_index(
        self,
        position: tuple[float, float, float],
        *,
        reference_index: int | None,
        check_backward: int,
        check_forward: int,
    ) -> tuple[int, float]:
        if self.length <= 0:
            return 0, float("inf")
        if reference_index is None:
            start = 0
            stop = self.length
        else:
            start = max(0, int(reference_index) - int(check_backward))
            stop = min(self.length, int(reference_index) + int(check_forward) + 1)
            if start >= stop:
                start = 0
                stop = self.length
        pos = np.asarray(position, dtype=np.float64)
        distances = np.linalg.norm(self.positions[start:stop] - pos[None, :], axis=1)
        local_index = int(np.argmin(distances))
        return start + local_index, float(distances[local_index])

    def arc_at(self, index: int) -> float:
        if self.length <= 0:
            return 0.0
        resolved = min(max(0, int(index)), len(self.arc_length) - 1)
        return float(self.arc_length[resolved])

    def progress_index_at(self, index: int) -> int:
        spacing = max(1.0e-6, float(self.spacing_meters))
        return int(np.floor((self.arc_at(index) / spacing) + 1.0e-6))

    def source_row_index_at(self, index: int) -> int:
        if self.length <= 0:
            return 0
        resolved = min(max(0, int(index)), len(self.source_row_index) - 1)
        return int(self.source_row_index[resolved])

    @property
    def total_length(self) -> float:
        return self.arc_at(self.length - 1)

    def race_time_at(self, index: int) -> float:
        if self.length <= 0:
            return 0.0
        resolved = min(max(0, int(index)), len(self.race_time_ms) - 1)
        return float(self.race_time_ms[resolved])

    def race_time_at_arc(self, arc_length_m: float) -> float:
        if self.length <= 0:
            return 0.0
        arc = float(arc_length_m)
        right = int(np.searchsorted(self.arc_length, arc, side="left"))
        if right <= 0:
            return float(self.race_time_ms[0])
        if right >= len(self.arc_length):
            return float(self.race_time_ms[-1])
        left = right - 1
        span = float(self.arc_length[right] - self.arc_length[left])
        if span <= 1.0e-9:
            return float(self.race_time_ms[right])
        ratio = (arc - float(self.arc_length[left])) / span
        return float((1.0 - ratio) * float(self.race_time_ms[left]) + ratio * float(self.race_time_ms[right]))

    def position_at_arc(self, arc_length_m: float) -> np.ndarray:
        if self.length <= 0:
            return np.zeros(3, dtype=np.float64)
        arc = float(arc_length_m)
        right = int(np.searchsorted(self.arc_length, arc, side="left"))
        if right <= 0:
            return self.positions[0]
        if right >= len(self.arc_length):
            return self.positions[-1]
        left = right - 1
        if abs(float(self.arc_length[left]) - arc) <= abs(float(self.arc_length[right]) - arc):
            return self.positions[left]
        return self.positions[right]


class GhostBundleReward:
    """Progress reward projected onto a selected bundle of leaderboard ghost lines."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        reward_config: RewardConfig,
        ghost_config: GhostConfig,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = load_ghost_bundle_manifest(self.manifest_path)
        self.map_uid = str(self.manifest.get("map_uid") or "")
        self.config = reward_config
        self.ghost_config = ghost_config
        self._canonical_reference_source = self.manifest.get("canonical_reference_source")
        self._canonical_reference_path = self.manifest.get("canonical_reference_path")
        self._strategy_classification_status = self.manifest.get("strategy_classification_status")
        self._selected_training_family = self.manifest.get("selected_training_family")
        self._mixed_fallback = bool(self.manifest.get("mixed_fallback", False))
        self._bundle_resolution_mode = self.manifest.get("bundle_resolution_mode")
        self._selected_ghost_selector = self.manifest.get("selected_ghost_selector")
        self._resolved_selected_ghost_rank = self.manifest.get("resolved_selected_ghost_rank")
        self._resolved_selected_ghost_name = self.manifest.get("resolved_selected_ghost_name")
        self._author_fallback_used = bool(self.manifest.get("author_fallback_used", False))
        self._intended_bundle_manifest_path = self.manifest.get("intended_bundle_manifest_path")
        self._exploit_bundle_manifest_path = self.manifest.get("exploit_bundle_manifest_path")
        self._selected_override_manifest_path = self.manifest.get("selected_override_manifest_path")
        self._author_fallback_manifest_path = self.manifest.get("author_fallback_manifest_path")
        self._strategy_family_counts = dict(self.manifest.get("strategy_family_counts") or {})
        self.lines = self._load_lines(self.manifest)
        if not self.lines:
            raise RuntimeError(f"Ghost bundle {self.manifest_path} does not contain selected trajectories.")
        self._line_index = 0
        self._current_index = 0
        self._step_count = 0
        self._no_progress_steps = 0
        self._line_switch_count = 0
        self._corridor_violation_steps = 0
        self._corridor_nonrecovering_steps = 0
        self._corridor_recovery_count = 0
        self._corridor_truncation_count = 0
        self._corridor_was_violating = False
        self._last_corridor_distance_m: float | None = None
        self._run_id: str | None = None
        total_lengths = [line.total_length for line in self.lines]
        self._reference_total_arc_length_m = float(np.median(np.asarray(total_lengths, dtype=np.float64))) if total_lengths else 0.0

    @staticmethod
    def _source_row_indices_for_arc(source_arc: np.ndarray, target_arc: np.ndarray) -> np.ndarray:
        if source_arc.size <= 0 or target_arc.size <= 0:
            return np.zeros((target_arc.size,), dtype=np.int32)
        result: list[int] = []
        for arc in target_arc:
            right = int(np.searchsorted(source_arc, float(arc), side="left"))
            if right <= 0:
                result.append(0)
                continue
            if right >= source_arc.size:
                result.append(int(source_arc.size - 1))
                continue
            left = right - 1
            if abs(float(source_arc[left]) - float(arc)) <= abs(float(source_arc[right]) - float(arc)):
                result.append(left)
            else:
                result.append(right)
        return np.asarray(result, dtype=np.int32)

    @staticmethod
    def _source_arc_lengths(positions: np.ndarray) -> np.ndarray:
        if positions.shape[0] <= 0:
            return np.zeros((0,), dtype=np.float64)
        if positions.shape[0] == 1:
            return np.zeros((1,), dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.concatenate((np.asarray([0.0], dtype=np.float64), np.cumsum(segment_lengths, dtype=np.float64)))

    def _load_lines(self, manifest: Mapping[str, Any]) -> list[GhostLine]:
        lines: list[GhostLine] = []
        for index, item in enumerate(manifest.get("selected_trajectories", [])):
            trajectory_path = Path(str(item["trajectory_parquet_path"])).resolve()
            rows = pq.read_table(trajectory_path).to_pylist()
            if not rows:
                continue
            positions = np.asarray(
                [
                    [
                        float(row.get("position_x") or 0.0),
                        float(row.get("position_y") or 0.0),
                        float(row.get("position_z") or 0.0),
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )
            race_time_ms = np.asarray(
                [float(row.get("race_time_ms") or row_idx * 50) for row_idx, row in enumerate(rows)],
                dtype=np.float64,
            )
            resampled_positions, resampled_arc_length, resampled_race_time_ms = _resample_fixed_spacing(
                positions,
                race_time_ms,
                self.config.spacing_meters,
            )
            source_row_index = self._source_row_indices_for_arc(
                self._source_arc_lengths(positions),
                resampled_arc_length,
            )
            lines.append(
                GhostLine(
                    line_id=str(item.get("trajectory_id") or f"ghost_line_{index:03d}"),
                    rank=None if item.get("rank") is None else int(item["rank"]),
                    trajectory_path=trajectory_path,
                    positions=resampled_positions,
                    arc_length=resampled_arc_length,
                    race_time_ms=resampled_race_time_ms,
                    source_row_index=source_row_index,
                    spacing_meters=float(self.config.spacing_meters),
                )
            )
        return lines

    @property
    def current_index(self) -> int:
        return self.current_line.progress_index_at(self._current_index)

    @property
    def current_line(self) -> GhostLine:
        return self.lines[self._line_index]

    def reset(self, *, run_id: str, initial_position: tuple[float, float, float] | None) -> None:
        self._run_id = run_id
        self._step_count = 0
        self._no_progress_steps = 0
        self._line_switch_count = 0
        self._corridor_violation_steps = 0
        self._corridor_nonrecovering_steps = 0
        self._corridor_recovery_count = 0
        self._corridor_truncation_count = 0
        self._corridor_was_violating = False
        self._last_corridor_distance_m = None
        self._line_index = 0
        self._current_index = 0
        if initial_position is None:
            return
        self._line_index, self._current_index, _ = self._best_projection(initial_position, reference_index=None)

    def _best_projection(
        self,
        position: tuple[float, float, float],
        *,
        reference_index: int | None,
    ) -> tuple[int, int, float]:
        best_line = self._line_index
        best_index = self._current_index
        best_distance = float("inf")
        current_arc_length = self.current_line.arc_at(self._current_index)
        for line_index, line in enumerate(self.lines):
            line_reference = reference_index
            candidate_index, distance = line.nearest_index(
                position,
                reference_index=line_reference,
                check_backward=self.config.check_backward,
                check_forward=self.config.check_forward,
            )
            if line_index != self._line_index and distance > self.config.line_switch_max_distance_m:
                continue
            candidate_arc_length = line.arc_at(candidate_index)
            candidate_advantage_m = candidate_arc_length - current_arc_length
            current_line_bonus_m = (
                0.0
                if line_index == self._line_index
                else float(self.ghost_config.line_switch_hysteresis) * max(1.0e-6, self.config.spacing_meters)
            )
            if distance < best_distance and candidate_advantage_m >= -current_line_bonus_m:
                best_line = line_index
                best_index = candidate_index
                best_distance = distance
        return best_line, best_index, best_distance

    def _local_spread_radius(self, arc_length_m: float) -> float:
        if self.config.corridor_mode != "map_calibrated" or len(self.lines) < 2:
            return 0.0
        positions = np.asarray([line.position_at_arc(arc_length_m) for line in self.lines], dtype=np.float64)
        center = np.median(positions, axis=0)
        distances = np.linalg.norm(positions - center[None, :], axis=1)
        spread = float(np.percentile(distances, 90))
        max_spread = max(0.0, self.config.corridor_catastrophic_distance_m - self.config.corridor_hard_margin_m)
        return min(max(0.0, spread), max_spread)

    def _corridor_radii(self, arc_length_m: float) -> tuple[float, float, float]:
        local_spread = self._local_spread_radius(arc_length_m)
        soft_radius = local_spread + self.config.corridor_soft_margin_m
        hard_radius = local_spread + self.config.corridor_hard_margin_m
        hard_radius = min(max(hard_radius, soft_radius), self.config.corridor_catastrophic_distance_m)
        soft_radius = min(max(0.0, soft_radius), hard_radius)
        return soft_radius, hard_radius, local_spread

    def _progress_fraction_of_reference(self, arc_length_m: float) -> float | None:
        if self._reference_total_arc_length_m <= 1.0e-9:
            return None
        return float(min(max(arc_length_m / self._reference_total_arc_length_m, 0.0), 1.0))

    def _ghost_reference_time_ms(self, arc_length_m: float) -> float | None:
        if not self.lines:
            return None
        values = [line.race_time_at_arc(arc_length_m) for line in self.lines if line.length > 0]
        if not values:
            return None
        return float(min(values))

    def _apply_corridor_reward(
        self,
        *,
        corridor_distance_m: float | None,
        progress_arc_length_m: float,
        progress_delta_m: float,
        speed_kmh: float,
    ) -> dict[str, Any]:
        soft_radius, hard_radius, local_spread = self._corridor_radii(progress_arc_length_m)
        penalty = 0.0
        recovery_bonus = 0.0
        soft_violation = False
        hard_violation = False
        catastrophic = False
        recovering = False
        distance_delta_m: float | None = None
        if corridor_distance_m is None:
            self._corridor_violation_steps = 0
            self._corridor_nonrecovering_steps = 0
            self._corridor_was_violating = False
            self._last_corridor_distance_m = None
        else:
            if self._last_corridor_distance_m is not None:
                distance_delta_m = float(corridor_distance_m - self._last_corridor_distance_m)
            soft_violation = corridor_distance_m > soft_radius
            hard_violation = corridor_distance_m > hard_radius
            catastrophic = corridor_distance_m > self.config.corridor_catastrophic_distance_m
            if soft_violation:
                excess = corridor_distance_m - soft_radius
                if hard_violation:
                    excess += corridor_distance_m - hard_radius
                    self._corridor_violation_steps += 1
                    recovering = (
                        progress_delta_m >= self.config.corridor_min_recovery_progress_m
                        or speed_kmh >= self.config.corridor_min_recovery_speed_kmh
                        or (
                            distance_delta_m is not None
                            and distance_delta_m <= -self.config.corridor_recovery_distance_delta_m
                        )
                    )
                    if recovering:
                        self._corridor_nonrecovering_steps = 0
                    else:
                        self._corridor_nonrecovering_steps += 1
                else:
                    self._corridor_violation_steps = 0
                    self._corridor_nonrecovering_steps = 0
                penalty = min(
                    self.config.corridor_penalty_max,
                    self.config.corridor_penalty_scale * max(0.0, excess),
                )
                self._corridor_was_violating = True
            else:
                if self._corridor_was_violating:
                    recovery_bonus = self.config.corridor_recovery_bonus
                    self._corridor_recovery_count += 1
                self._corridor_violation_steps = 0
                self._corridor_nonrecovering_steps = 0
                self._corridor_was_violating = False
            self._last_corridor_distance_m = float(corridor_distance_m)

        return {
            "corridor_distance_m": corridor_distance_m,
            "corridor_soft_radius_m": soft_radius,
            "corridor_hard_radius_m": hard_radius,
            "corridor_local_spread_m": local_spread,
            "corridor_penalty": penalty,
            "corridor_recovery_bonus": recovery_bonus,
            "corridor_violation_steps": self._corridor_violation_steps,
            "corridor_nonrecovering_steps": self._corridor_nonrecovering_steps,
            "corridor_recovery_count": self._corridor_recovery_count,
            "corridor_truncation_count": self._corridor_truncation_count,
            "corridor_soft_violation": soft_violation,
            "corridor_hard_violation": hard_violation,
            "corridor_catastrophic": catastrophic,
            "corridor_recovering": recovering,
            "corridor_distance_delta_m": distance_delta_m,
            "corridor_progress_delta_m": progress_delta_m,
            "corridor_speed_kmh": speed_kmh,
        }

    def evaluate(self, frame: TelemetryFrame) -> RewardStepResult:
        self._step_count += 1
        progress_delta = 0
        stray_distance: float | None = None
        done_type: str | None = None
        done_reason: str | None = None
        progress_delta_m = 0.0

        if frame.pos_xyz is not None:
            previous_line_index = self._line_index
            previous_index = self._current_index
            previous_arc_length = self.current_line.arc_at(previous_index)
            previous_progress_index = self.current_line.progress_index_at(previous_index)
            line_index, new_index, stray_distance = self._best_projection(
                frame.pos_xyz,
                reference_index=self._current_index,
            )
            if line_index != self._line_index:
                self._line_switch_count += 1
                self._line_index = line_index
            new_arc_length = self.current_line.arc_at(new_index)
            new_progress_index = self.current_line.progress_index_at(new_index)
            progress_delta = int(new_progress_index - previous_progress_index)
            progress_delta_m = float(new_arc_length - previous_arc_length)
            self._current_index = new_index
        else:
            self._no_progress_steps += 1

        reward = float(progress_delta) + self.config.constant_penalty
        if progress_delta > 0:
            self._no_progress_steps = 0
        elif frame.pos_xyz is not None:
            self._no_progress_steps += 1

        line = self.current_line
        arc_index = min(max(0, self._current_index), len(line.arc_length) - 1)
        progress_arc_length_m = float(line.arc_length[arc_index])
        progress_fraction_of_reference = self._progress_fraction_of_reference(progress_arc_length_m)
        ghost_reference_time_ms = self._ghost_reference_time_ms(progress_arc_length_m)
        ghost_relative_time_delta_ms = None
        if ghost_reference_time_ms is not None and frame.race_time_ms is not None:
            ghost_relative_time_delta_ms = float(frame.race_time_ms) - float(ghost_reference_time_ms)
        corridor = self._apply_corridor_reward(
            corridor_distance_m=stray_distance,
            progress_arc_length_m=progress_arc_length_m,
            progress_delta_m=progress_delta_m,
            speed_kmh=float(frame.speed_kmh or 0.0),
        )
        reward += float(corridor["corridor_recovery_bonus"]) - float(corridor["corridor_penalty"])

        if frame.finished:
            reward += self.config.end_of_track
            done_type = "terminated"
            done_reason = "finished"
        elif frame.terminal_reason in {"outside_active_race", "map_changed"}:
            done_type = "terminated"
            done_reason = frame.terminal_reason
        elif self._corridor_nonrecovering_steps >= self.config.corridor_patience_steps:
            self._corridor_truncation_count += 1
            corridor["corridor_truncation_count"] = self._corridor_truncation_count
            done_type = "truncated"
            done_reason = "corridor_violation"
        elif self._step_count >= self.config.min_steps and self._no_progress_steps >= self.config.failure_countdown:
            done_type = "truncated"
            done_reason = "no_progress"

        progress_index = line.progress_index_at(self._current_index)
        info = {
            "progress_index": progress_index,
            "progress_delta": progress_delta,
            "no_progress_steps": self._no_progress_steps,
            "reward_reason": done_reason,
            "stray_distance": stray_distance,
            "trajectory_arc_length_m": progress_arc_length_m,
            "final_arc_length_m": progress_arc_length_m,
            "progress_arc_length_m": progress_arc_length_m,
            "reference_total_arc_length_m": self._reference_total_arc_length_m,
            "progress_fraction_of_reference": progress_fraction_of_reference,
            "progress_spacing_meters": float(self.config.spacing_meters),
            "progress_index_semantics": "fixed_spacing_meters",
            "ghost_reference_time_ms": ghost_reference_time_ms,
            "ghost_relative_time_delta_ms": ghost_relative_time_delta_ms,
            "tm20ai_done_type": done_type,
            "ghost_bundle_manifest_path": str(self.manifest_path),
            "canonical_reference_source": self._canonical_reference_source,
            "canonical_reference_path": self._canonical_reference_path,
            "strategy_classification_status": self._strategy_classification_status,
            "selected_training_family": self._selected_training_family,
            "mixed_fallback": self._mixed_fallback,
            "bundle_resolution_mode": self._bundle_resolution_mode,
            "selected_ghost_selector": self._selected_ghost_selector,
            "resolved_selected_ghost_rank": self._resolved_selected_ghost_rank,
            "resolved_selected_ghost_name": self._resolved_selected_ghost_name,
            "author_fallback_used": self._author_fallback_used,
            "intended_bundle_manifest_path": self._intended_bundle_manifest_path,
            "exploit_bundle_manifest_path": self._exploit_bundle_manifest_path,
            "selected_override_manifest_path": self._selected_override_manifest_path,
            "author_fallback_manifest_path": self._author_fallback_manifest_path,
            "strategy_family_counts": dict(self._strategy_family_counts),
            "ghost_line_id": line.line_id,
            "ghost_line_rank": line.rank,
            "ghost_source_row_index": line.source_row_index_at(self._current_index),
            "ghost_line_switch_count": self._line_switch_count,
            **corridor,
        }
        return RewardStepResult(
            reward=reward,
            done_type=done_type,
            done_reason=done_reason,
            progress_index=progress_index,
            progress_delta=progress_delta,
            stray_distance=stray_distance,
            no_progress_steps=self._no_progress_steps,
            info=info,
        )
