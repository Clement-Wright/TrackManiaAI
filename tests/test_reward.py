from __future__ import annotations

import numpy as np

from tm20ai.bridge import TelemetryFrame
from tm20ai.config import RewardConfig
from tm20ai.env.reward import TrajectoryProgressReward
from tm20ai.env.trajectory import RuntimeTrajectory


def make_trajectory() -> RuntimeTrajectory:
    points = np.asarray([[float(x), 0.0, 0.0] for x in range(11)], dtype=np.float32)
    tangents = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (len(points), 1))
    arc_length = np.asarray([float(x) for x in range(11)], dtype=np.float32)
    race_time_ms = np.asarray([float(x * 100) for x in range(11)], dtype=np.float32)
    return RuntimeTrajectory("test-map", points, tangents, arc_length, race_time_ms)


def make_frame(*, frame_id: int, position: tuple[float, float, float], finished: bool = False) -> TelemetryFrame:
    return TelemetryFrame(
        session_id="session",
        run_id="run",
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000_000,
        map_uid="test-map",
        race_time_ms=frame_id * 50,
        cp_count=0,
        cp_target=1,
        speed_kmh=100.0,
        gear=3,
        rpm=5000.0,
        pos_xyz=position,
        vel_xyz=(0.0, 0.0, 0.0),
        yaw_pitch_roll=(0.0, 0.0, 0.0),
        finished=finished,
        terminal_reason="finished" if finished else None,
    )


def test_reward_progress_and_backward_penalty() -> None:
    reward = TrajectoryProgressReward(make_trajectory(), RewardConfig())
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    step_forward = reward.evaluate(make_frame(frame_id=1, position=(1.1, 0.0, 0.0)))
    step_backward = reward.evaluate(make_frame(frame_id=2, position=(0.2, 0.0, 0.0)))

    assert step_forward.reward > 0.0
    assert step_forward.progress_delta > 0
    assert step_backward.reward < 0.0
    assert step_backward.progress_delta < 0


def test_reward_marks_stray_as_terminated() -> None:
    config = RewardConfig(max_stray=2.0)
    reward = TrajectoryProgressReward(make_trajectory(), config)
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    result = reward.evaluate(make_frame(frame_id=1, position=(100.0, 50.0, 0.0)))

    assert result.done_type == "terminated"
    assert result.done_reason == "stray"


def test_reward_marks_no_progress_as_truncated() -> None:
    config = RewardConfig(failure_countdown=3, min_steps=2)
    reward = TrajectoryProgressReward(make_trajectory(), config)
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    reward.evaluate(make_frame(frame_id=1, position=(0.0, 0.0, 0.0)))
    reward.evaluate(make_frame(frame_id=2, position=(0.0, 0.0, 0.0)))
    result = reward.evaluate(make_frame(frame_id=3, position=(0.0, 0.0, 0.0)))

    assert result.done_type == "truncated"
    assert result.done_reason == "no_progress"
