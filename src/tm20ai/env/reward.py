from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..bridge import TelemetryFrame
from ..config import RewardConfig
from .trajectory import RuntimeTrajectory


@dataclass(slots=True)
class RewardStepResult:
    reward: float
    done_type: str | None
    done_reason: str | None
    progress_index: int
    progress_delta: int
    stray_distance: float | None
    no_progress_steps: int
    info: dict[str, Any] = field(default_factory=dict)


class TrajectoryProgressReward:
    """TMRL-style trajectory progress reward over a resampled reference path."""

    def __init__(self, trajectory: RuntimeTrajectory, config: RewardConfig):
        self.trajectory = trajectory
        self.config = config
        self._current_index = 0
        self._step_count = 0
        self._no_progress_steps = 0
        self._run_id: str | None = None

    @property
    def current_index(self) -> int:
        return self._current_index

    def reset(self, *, run_id: str, initial_position: tuple[float, float, float] | None) -> None:
        self._run_id = run_id
        self._step_count = 0
        self._no_progress_steps = 0
        if initial_position is None:
            self._current_index = 0
            return
        self._current_index, _ = self.trajectory.nearest_index(initial_position, reference_index=None)

    def evaluate(self, frame: TelemetryFrame) -> RewardStepResult:
        self._step_count += 1
        progress_delta = 0
        stray_distance: float | None = None
        done_type: str | None = None
        done_reason: str | None = None

        if frame.pos_xyz is not None:
            new_index, stray_distance = self.trajectory.nearest_index(
                frame.pos_xyz,
                reference_index=self._current_index,
                check_backward=self.config.check_backward,
                check_forward=self.config.check_forward,
            )
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

        info = {
            "progress_index": self._current_index,
            "progress_delta": progress_delta,
            "no_progress_steps": self._no_progress_steps,
            "reward_reason": done_reason,
            "stray_distance": stray_distance,
            "trajectory_arc_length_m": float(self.trajectory.arc_length[self._current_index]),
            "tm20ai_done_type": done_type,
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
