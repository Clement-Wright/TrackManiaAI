from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq

from ..bridge import TelemetryFrame
from ..config import GhostConfig, RewardConfig
from ..env.reward import RewardStepResult
from .dataset import load_ghost_bundle_manifest


@dataclass(frozen=True, slots=True)
class GhostLine:
    line_id: str
    rank: int | None
    trajectory_path: Path
    positions: np.ndarray
    arc_length: np.ndarray

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
        self.lines = self._load_lines(self.manifest)
        if not self.lines:
            raise RuntimeError(f"Ghost bundle {self.manifest_path} does not contain selected trajectories.")
        self._line_index = 0
        self._current_index = 0
        self._step_count = 0
        self._no_progress_steps = 0
        self._line_switch_count = 0
        self._run_id: str | None = None

    @staticmethod
    def _load_lines(manifest: Mapping[str, Any]) -> list[GhostLine]:
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
            arc_length = np.asarray(
                [float(row.get("arc_length") or row_idx) for row_idx, row in enumerate(rows)],
                dtype=np.float64,
            )
            lines.append(
                GhostLine(
                    line_id=str(item.get("trajectory_id") or f"ghost_line_{index:03d}"),
                    rank=None if item.get("rank") is None else int(item["rank"]),
                    trajectory_path=trajectory_path,
                    positions=positions,
                    arc_length=arc_length,
                )
            )
        return lines

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def current_line(self) -> GhostLine:
        return self.lines[self._line_index]

    def reset(self, *, run_id: str, initial_position: tuple[float, float, float] | None) -> None:
        self._run_id = run_id
        self._step_count = 0
        self._no_progress_steps = 0
        self._line_switch_count = 0
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
        for line_index, line in enumerate(self.lines):
            line_reference = reference_index if line_index == self._line_index else None
            candidate_index, distance = line.nearest_index(
                position,
                reference_index=line_reference,
                check_backward=self.config.check_backward,
                check_forward=self.config.check_forward,
            )
            candidate_advantage = candidate_index - self._current_index
            current_line_bonus = 0 if line_index == self._line_index else self.ghost_config.line_switch_hysteresis
            if distance < best_distance and candidate_advantage >= -current_line_bonus:
                best_line = line_index
                best_index = candidate_index
                best_distance = distance
        return best_line, best_index, best_distance

    def evaluate(self, frame: TelemetryFrame) -> RewardStepResult:
        self._step_count += 1
        progress_delta = 0
        stray_distance: float | None = None
        done_type: str | None = None
        done_reason: str | None = None

        if frame.pos_xyz is not None:
            line_index, new_index, stray_distance = self._best_projection(
                frame.pos_xyz,
                reference_index=self._current_index,
            )
            if line_index != self._line_index:
                self._line_switch_count += 1
                self._line_index = line_index
            progress_delta = int(new_index - self._current_index)
            self._current_index = new_index
        else:
            self._no_progress_steps += 1

        reward = float(progress_delta) + self.config.constant_penalty
        if progress_delta > 0:
            self._no_progress_steps = 0
        elif frame.pos_xyz is not None:
            self._no_progress_steps += 1

        if frame.finished:
            reward += self.config.end_of_track
            done_type = "terminated"
            done_reason = "finished"
        elif frame.terminal_reason in {"outside_active_race", "map_changed"}:
            done_type = "terminated"
            done_reason = frame.terminal_reason
        elif stray_distance is not None and stray_distance > self.config.max_stray:
            done_type = "terminated"
            done_reason = "stray"
        elif self._step_count >= self.config.min_steps and self._no_progress_steps >= self.config.failure_countdown:
            done_type = "truncated"
            done_reason = "no_progress"

        line = self.current_line
        arc_index = min(max(0, self._current_index), len(line.arc_length) - 1)
        info = {
            "progress_index": self._current_index,
            "progress_delta": progress_delta,
            "no_progress_steps": self._no_progress_steps,
            "reward_reason": done_reason,
            "stray_distance": stray_distance,
            "trajectory_arc_length_m": float(line.arc_length[arc_index]),
            "tm20ai_done_type": done_type,
            "ghost_bundle_manifest_path": str(self.manifest_path),
            "ghost_line_id": line.line_id,
            "ghost_line_rank": line.rank,
            "ghost_line_switch_count": self._line_switch_count,
        }
        return RewardStepResult(
            reward=reward,
            done_type=done_type,
            done_reason=done_reason,
            progress_index=self._current_index,
            progress_delta=progress_delta,
            stray_distance=stray_distance,
            no_progress_steps=self._no_progress_steps,
            info=info,
        )
