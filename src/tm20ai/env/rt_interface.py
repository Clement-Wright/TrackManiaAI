from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from rtgym.envs.real_time_env import RealTimeGymInterface

from ..bridge import BridgeClient
from ..capture import DXCamCapture, FrameStackPreprocessor
from ..config import TM20AIConfig, load_tm20ai_config
from ..control import GamepadController
from .reset_manager import ResetManager
from .reward import TrajectoryProgressReward
from .trajectory import load_runtime_trajectory, runtime_trajectory_path_for_map


@dataclass(slots=True)
class InterfaceTimingMetrics:
    observation_calls: int = 0
    total_obs_retrieval_seconds: float = 0.0
    total_preprocess_seconds: float = 0.0
    total_reward_compute_seconds: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        count = max(1, self.observation_calls)
        return {
            "avg_obs_retrieval_seconds": self.total_obs_retrieval_seconds / count,
            "avg_preprocess_seconds": self.total_preprocess_seconds / count,
            "avg_reward_compute_seconds": self.total_reward_compute_seconds / count,
            "observation_calls": self.observation_calls,
        }


FROZEN_STEP_INFO_KEYS = (
    "session_id",
    "run_id",
    "map_uid",
    "frame_id",
    "timestamp_ns",
    "race_time_ms",
    "terminal_reason",
    "progress_index",
    "progress_delta",
    "no_progress_steps",
    "stray_distance",
    "trajectory_arc_length_m",
    "reward_reason",
    "tm20ai_done_type",
)


class TM20AIRtInterface(RealTimeGymInterface):
    """Real-time interface that binds the custom bridge, capture, control, and reward stack."""

    def __init__(self, *, config_path: str | Path):
        self.config: TM20AIConfig = load_tm20ai_config(config_path)
        if self.config.observation.mode != "full":
            raise NotImplementedError(
                f"Observation mode {self.config.observation.mode!r} is planned but not implemented in Phase 3-4."
            )

        self._bridge = BridgeClient(self.config.bridge)
        self._bridge.start()
        self._gamepad = GamepadController()
        self._capture = DXCamCapture(self.config.capture)
        self._preprocessor = FrameStackPreprocessor(self.config.full_observation)
        self._reset_manager = ResetManager(
            client=self._bridge,
            gamepad=self._gamepad,
            capture=self._capture,
            preprocessor=self._preprocessor,
            runtime=self.config.runtime,
            bridge_config=self.config.bridge,
        )
        self._reward_model: TrajectoryProgressReward | None = None
        self._last_frame = None
        self._timing_metrics = InterfaceTimingMetrics()

    def get_observation_space(self):
        obs_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.config.full_observation.frame_stack,
                self.config.full_observation.output_height,
                self.config.full_observation.output_width,
            ),
            dtype=np.uint8,
        )
        return spaces.Tuple((obs_space,))

    def get_action_space(self):
        return spaces.Box(
            low=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def get_default_action(self):
        return self._gamepad.neutral_action()

    def send_control(self, control):
        if control is None:
            return
        self._gamepad.apply(control)

    def _ensure_reward_model(self, map_uid: str) -> TrajectoryProgressReward:
        path = runtime_trajectory_path_for_map(map_uid, self.config.reward.spacing_meters)
        if self._reward_model is not None and self._reward_model.trajectory.map_uid == map_uid:
            return self._reward_model
        if not path.exists():
            raise FileNotFoundError(
                f"Reward trajectory artifact is missing for map_uid {map_uid!r}: {path}. "
                "Run scripts\\record_reward.py first."
            )
        self._reward_model = TrajectoryProgressReward(load_runtime_trajectory(path), self.config.reward)
        return self._reward_model

    def reset(self, seed=None, options=None):
        del seed, options
        self._capture.ensure_started()
        reset_result = self._reset_manager.reset_to_start()
        self._last_frame = reset_result.frame
        reward_model = self._ensure_reward_model(reset_result.frame.map_uid)
        reward_model.reset(run_id=reset_result.frame.run_id, initial_position=reset_result.frame.pos_xyz)
        return [reset_result.observation], dict(reset_result.info)

    def get_obs_rew_terminated_info(self):
        if self._last_frame is None:
            raise RuntimeError("reset() must be called before stepping the rtgym interface.")
        reward_model = self._ensure_reward_model(self._last_frame.map_uid)
        frame = self._bridge.wait_for_frame(
            after_frame_id=self._last_frame.frame_id,
            timeout=self.config.bridge.initial_frame_timeout,
        )
        retrieval_start = time.perf_counter()
        latest_frame = self._capture.get_latest_frame(timeout=self.config.capture.frame_timeout)
        retrieval_duration = time.perf_counter() - retrieval_start

        preprocess_start = time.perf_counter()
        observation = self._preprocessor.append_frame(latest_frame)
        preprocess_duration = time.perf_counter() - preprocess_start

        reward_start = time.perf_counter()
        reward_result = reward_model.evaluate(frame)
        reward_duration = time.perf_counter() - reward_start
        self._last_frame = frame
        self._timing_metrics.observation_calls += 1
        self._timing_metrics.total_obs_retrieval_seconds += retrieval_duration
        self._timing_metrics.total_preprocess_seconds += preprocess_duration
        self._timing_metrics.total_reward_compute_seconds += reward_duration
        info = {
            "session_id": frame.session_id,
            "run_id": frame.run_id,
            "map_uid": frame.map_uid,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "race_time_ms": frame.race_time_ms,
            "terminal_reason": frame.terminal_reason,
            "speed_kmh": frame.speed_kmh,
            "gear": frame.gear,
            "rpm": frame.rpm,
            "pos_xyz": frame.pos_xyz,
            "vel_xyz": frame.vel_xyz,
            "yaw_pitch_roll": frame.yaw_pitch_roll,
        }
        info.update(reward_result.info)
        for key in FROZEN_STEP_INFO_KEYS:
            info.setdefault(key, None)
        return [observation], float(reward_result.reward), bool(reward_result.done_type is not None), info

    def wait(self):
        self._gamepad.apply(self._gamepad.neutral_action())
        time.sleep(self.config.runtime.time_step_duration)

    def render(self):
        return None

    def get_runtime_metrics(self) -> dict[str, Any]:
        return self._timing_metrics.snapshot()

    def close(self) -> None:
        self._gamepad.close()
        self._capture.close()
        self._bridge.close()
