from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bridge import BridgeClient, TelemetryFrame
from ..capture import DXCamCapture, FrameStackPreprocessor
from ..config import BridgeConnectionConfig, RuntimeLoopConfig
from ..control import GamepadController


@dataclass(slots=True)
class ResetResult:
    observation: np.ndarray
    frame: TelemetryFrame
    info: dict[str, Any]


class ResetManager:
    """Reset flow separated from reward computation."""

    def __init__(
        self,
        *,
        client: BridgeClient,
        gamepad: GamepadController,
        capture: DXCamCapture,
        preprocessor: FrameStackPreprocessor,
        runtime: RuntimeLoopConfig,
        bridge_config: BridgeConnectionConfig,
    ) -> None:
        self._client = client
        self._gamepad = gamepad
        self._capture = capture
        self._preprocessor = preprocessor
        self._runtime = runtime
        self._bridge_config = bridge_config

    def reset_to_start(self) -> ResetResult:
        before = self._client.wait_for_frame(timeout=self._bridge_config.initial_frame_timeout)
        previous_run_id = before.run_id
        neutral_action = self._gamepad.neutral_action()
        self._gamepad.apply(neutral_action)

        response = self._client.reset_to_start(timeout=self._bridge_config.reset_timeout)
        if not response.success:
            raise RuntimeError(f"Bridge reset failed: {response.message}")

        expected_run_id = str(response.payload["run_id"])
        expected_map_uid = str(response.payload["map_uid"])
        expected_frame_id = int(response.payload["frame_id"])
        time.sleep(self._runtime.sleep_time_at_reset)

        self._capture.ensure_started()
        self._preprocessor.clear()
        self._client.pop_received_frames()
        self._capture.flush_for_interval(self._capture.config.post_reset_flush_seconds)

        deadline = time.monotonic() + max(self._bridge_config.reset_timeout, self._bridge_config.initial_frame_timeout)
        frame: TelemetryFrame | None = None
        after_frame_id = max(before.frame_id, expected_frame_id - 1)
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            candidate = self._client.wait_for_frame(after_frame_id=after_frame_id, timeout=min(0.5, remaining))
            after_frame_id = max(after_frame_id, candidate.frame_id)
            if candidate.run_id != expected_run_id or candidate.run_id == previous_run_id:
                continue
            if candidate.map_uid != expected_map_uid:
                continue
            if candidate.race_time_ms > 100:
                continue
            frame = candidate
            break

        if frame is None:
            raise TimeoutError("Timed out waiting for the first valid post-reset telemetry frame.")

        race_state_response = self._client.race_state(timeout=self._bridge_config.command_timeout)
        race_state_payload = race_state_response.payload
        if race_state_payload.get("race_state") != "start_line":
            raise RuntimeError(
                "Bridge reset acknowledged a new run, but race_state did not return to start_line: "
                f"{race_state_payload.get('race_state')!r}"
            )
        if race_state_payload.get("run_id") != frame.run_id:
            raise RuntimeError("Bridge race_state run_id did not match the new telemetry run_id after reset.")
        if frame.race_time_ms < 0 or frame.race_time_ms > 100:
            raise RuntimeError(f"Race timer did not return to a valid pre-run state: {frame.race_time_ms} ms")

        fresh_frames = self._capture.prime_frames(
            count=self._preprocessor.frame_stack,
            timeout=max(self._bridge_config.initial_frame_timeout, self._capture.config.frame_timeout),
        )
        observation = self._preprocessor.build_clean_stack(fresh_frames)
        info = {
            "map_uid": frame.map_uid,
            "run_id": frame.run_id,
            "session_id": frame.session_id,
            "race_state": race_state_payload.get("race_state"),
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
        }
        return ResetResult(observation=observation, frame=frame, info=info)
