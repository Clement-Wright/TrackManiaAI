from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .bridge import BridgeConnectionConfig


class ConfigError(RuntimeError):
    """Raised when the repo runtime configuration is invalid."""


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping.")
    return value


def _bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be a bool.")
    return value


@dataclass(slots=True)
class RuntimeLoopConfig:
    time_step_duration: float = 0.05
    start_obs_capture: float = 0.04
    time_step_timeout_factor: float = 1.0
    act_buf_len: int = 2
    wait_on_done: bool = True
    ep_max_length: int = 1000
    sleep_time_at_reset: float = 1.5

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeLoopConfig":
        return cls(
            time_step_duration=float(payload.get("time_step_duration", 0.05)),
            start_obs_capture=float(payload.get("start_obs_capture", 0.04)),
            time_step_timeout_factor=float(payload.get("time_step_timeout_factor", 1.0)),
            act_buf_len=int(payload.get("act_buf_len", 2)),
            wait_on_done=_bool(payload.get("wait_on_done", True), context="runtime.wait_on_done"),
            ep_max_length=int(payload.get("ep_max_length", 1000)),
            sleep_time_at_reset=float(payload.get("sleep_time_at_reset", payload.get("reset_sleep", 1.5))),
        )


@dataclass(slots=True)
class ObservationModeConfig:
    mode: str = "full"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObservationModeConfig":
        mode = str(payload.get("mode", "full")).strip().lower()
        return cls(mode=mode)


@dataclass(slots=True)
class CaptureConfig:
    window_title: str = "Trackmania"
    target_fps: int = 60
    max_buffer_len: int = 64
    latest_frame_only: bool = True
    frame_timeout: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CaptureConfig":
        return cls(
            window_title=str(payload.get("window_title", "Trackmania")),
            target_fps=int(payload.get("target_fps", 60)),
            max_buffer_len=int(payload.get("max_buffer_len", 64)),
            latest_frame_only=_bool(payload.get("latest_frame_only", True), context="capture.latest_frame_only"),
            frame_timeout=float(payload.get("frame_timeout", 1.0)),
        )


@dataclass(slots=True)
class FullObservationConfig:
    window_width: int = 256
    window_height: int = 128
    output_width: int = 64
    output_height: int = 64
    grayscale: bool = True
    frame_stack: int = 4

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FullObservationConfig":
        return cls(
            window_width=int(payload.get("window_width", 256)),
            window_height=int(payload.get("window_height", 128)),
            output_width=int(payload.get("output_width", 64)),
            output_height=int(payload.get("output_height", 64)),
            grayscale=_bool(payload.get("grayscale", True), context="full_observation.grayscale"),
            frame_stack=int(payload.get("frame_stack", 4)),
        )


@dataclass(slots=True)
class RewardConfig:
    mode: str = "trajectory_progress"
    spacing_meters: float = 0.5
    end_of_track: float = 100.0
    constant_penalty: float = 0.0
    check_forward: int = 500
    check_backward: int = 10
    failure_countdown: int = 10
    min_steps: int = 70
    max_stray: float = 100.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RewardConfig":
        return cls(
            mode=str(payload.get("mode", "trajectory_progress")),
            spacing_meters=float(payload.get("spacing_meters", 0.5)),
            end_of_track=float(payload.get("end_of_track", payload.get("finish_bonus", 100.0))),
            constant_penalty=float(payload.get("constant_penalty", 0.0)),
            check_forward=int(payload.get("check_forward", 500)),
            check_backward=int(payload.get("check_backward", 10)),
            failure_countdown=int(payload.get("failure_countdown", 10)),
            min_steps=int(payload.get("min_steps", 70)),
            max_stray=float(payload.get("max_stray", 100.0)),
        )


@dataclass(slots=True)
class TM20AIConfig:
    runtime: RuntimeLoopConfig
    bridge: BridgeConnectionConfig
    observation: ObservationModeConfig
    capture: CaptureConfig
    full_observation: FullObservationConfig
    reward: RewardConfig

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TM20AIConfig":
        return cls(
            runtime=RuntimeLoopConfig.from_mapping(_mapping(payload.get("runtime", {}), context="runtime")),
            bridge=BridgeConnectionConfig.from_mapping(_mapping(payload.get("bridge", {}), context="bridge")),
            observation=ObservationModeConfig.from_mapping(
                _mapping(payload.get("observation", {}), context="observation")
            ),
            capture=CaptureConfig.from_mapping(_mapping(payload.get("capture", {}), context="capture")),
            full_observation=FullObservationConfig.from_mapping(
                _mapping(payload.get("full_observation", {}), context="full_observation")
            ),
            reward=RewardConfig.from_mapping(_mapping(payload.get("reward", {}), context="reward")),
        )


def load_tm20ai_config(path: str | Path) -> TM20AIConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(f"Config path does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Expected a mapping in {config_path}.")
    return TM20AIConfig.from_mapping(payload)
