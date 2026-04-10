from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tm20ai.bridge import TelemetryFrame
from tm20ai.capture.preprocess import FrameStackPreprocessor
from tm20ai.config import BridgeConnectionConfig, CaptureConfig, FullObservationConfig, RuntimeLoopConfig
from tm20ai.env.reset_manager import ResetManager


def make_frame(frame_id: int, run_id: str, *, race_time_ms: int = 0) -> TelemetryFrame:
    return TelemetryFrame(
        session_id="session",
        run_id=run_id,
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000,
        map_uid="test-map",
        race_time_ms=race_time_ms,
        cp_count=0,
        cp_target=1,
        speed_kmh=50.0,
        gear=2,
        rpm=3000.0,
        pos_xyz=(1.0, 2.0, 3.0),
        vel_xyz=(0.0, 0.0, 0.0),
        yaw_pitch_roll=(0.0, 0.0, 0.0),
        finished=False,
        terminal_reason=None,
    )


@dataclass
class FakeResponse:
    success: bool
    payload: dict[str, object]
    message: str = ""


class FakeClient:
    def __init__(self) -> None:
        self.frames = [make_frame(1, "run-old", race_time_ms=50), make_frame(2, "run-new", race_time_ms=0)]
        self.pop_calls = 0

    def wait_for_frame(self, *, timeout=None, after_frame_id=None):  # noqa: ANN001
        del timeout, after_frame_id
        return self.frames.pop(0)

    def reset_to_start(self, *, timeout=None):  # noqa: ANN001
        del timeout
        return FakeResponse(
            success=True,
            payload={
                "run_id": "run-new",
                "map_uid": "test-map",
                "frame_id": 2,
            },
        )

    def pop_received_frames(self):
        self.pop_calls += 1
        return []

    def race_state(self, *, timeout=None):  # noqa: ANN001
        del timeout
        return FakeResponse(success=True, payload={"race_state": "start_line", "run_id": "run-new"})


class FakeGamepad:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    @staticmethod
    def neutral_action() -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def apply(self, action) -> np.ndarray:  # noqa: ANN001
        array = np.asarray(action, dtype=np.float32)
        self.actions.append(array)
        return array


class FakeCapture:
    def __init__(self) -> None:
        self.config = CaptureConfig(frame_timeout=0.1, post_reset_flush_seconds=0.25)
        self.ensure_started_calls = 0
        self.flushed_for: list[float] = []
        self.prime_calls = 0

    def ensure_started(self) -> None:
        self.ensure_started_calls += 1

    def flush_for_interval(self, duration_seconds: float) -> None:
        self.flushed_for.append(duration_seconds)

    def prime_frames(self, *, count: int, timeout: float | None = None):  # noqa: ANN001
        self.prime_calls += 1
        del timeout
        return [np.full((128, 256, 3), index, dtype=np.uint8) for index in range(count)]


def test_reset_manager_reuses_capture_and_flushes_frames(monkeypatch) -> None:
    monkeypatch.setattr("tm20ai.env.reset_manager.time.sleep", lambda *_args, **_kwargs: None)

    manager = ResetManager(
        client=FakeClient(),
        gamepad=FakeGamepad(),
        capture=FakeCapture(),
        preprocessor=FrameStackPreprocessor(FullObservationConfig()),
        runtime=RuntimeLoopConfig(sleep_time_at_reset=0.0),
        bridge_config=BridgeConnectionConfig(),
    )

    result = manager.reset_to_start()

    assert result.observation.shape == (4, 64, 64)
    assert result.info["run_id"] == "run-new"
    assert manager._capture.ensure_started_calls == 1
    assert manager._capture.flushed_for == [0.25]
    assert manager._capture.prime_calls == 1
