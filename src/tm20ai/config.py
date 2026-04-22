from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _string_tuple(value: Any, *, context: str, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(item) for item in value)
    else:
        raise ConfigError(f"{context} must be a string or sequence of strings.")
    normalized = tuple(item.strip().lower() for item in values if item.strip())
    if not normalized:
        raise ConfigError(f"{context} must contain at least one value.")
    return normalized


def _int_tuple(value: Any, *, context: str, default: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(item) for item in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{context} must be a sequence of integers.")
    values = tuple(int(item) for item in value)
    if not values:
        raise ConfigError(f"{context} must contain at least one integer.")
    return values


def _float_sequence(value: Any, *, context: str, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, (int, float)):
        values = (float(value),)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(float(item) for item in value)
    else:
        raise ConfigError(f"{context} must be a number or sequence of numbers.")
    if not values:
        raise ConfigError(f"{context} must contain at least one value.")
    return values


def _float_tuple(
    value: Any,
    *,
    context: str,
    length: int,
    default: Sequence[float],
) -> tuple[float, ...]:
    if value is None:
        return tuple(float(item) for item in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{context} must be a sequence of {length} numeric values.")
    values = tuple(float(item) for item in value)
    if len(values) != length:
        raise ConfigError(f"{context} must contain exactly {length} values.")
    return values


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
        if mode not in {"full", "lidar"}:
            raise ConfigError(f"observation.mode must be one of ['full', 'lidar'], got {mode!r}.")
        return cls(mode=mode)


@dataclass(slots=True)
class CaptureConfig:
    window_title: str = "Trackmania"
    target_fps: int = 60
    max_buffer_len: int = 64
    backend: str = "dxgi"
    device_idx: int | None = None
    output_idx: int | None = None
    bootstrap_log: bool = True
    require_stable_window_polls: int = 5
    stable_window_poll_interval_seconds: float = 0.1
    latest_frame_only: bool = True
    frame_timeout: float = 1.0
    post_reset_flush_seconds: float = 0.25
    invalid_frame_limit: int = 3
    region_change_tolerance_pixels: int = 4

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CaptureConfig":
        backend = str(payload.get("backend", "dxgi")).strip().lower()
        if backend not in {"dxgi", "winrt", "auto"}:
            raise ConfigError(f"capture.backend must be one of ['dxgi', 'winrt', 'auto'], got {backend!r}.")
        device_idx = payload.get("device_idx")
        output_idx = payload.get("output_idx")
        return cls(
            window_title=str(payload.get("window_title", "Trackmania")),
            target_fps=int(payload.get("target_fps", 60)),
            max_buffer_len=int(payload.get("max_buffer_len", 64)),
            backend=backend,
            device_idx=None if device_idx in (None, "null") else int(device_idx),
            output_idx=None if output_idx in (None, "null") else int(output_idx),
            bootstrap_log=_bool(payload.get("bootstrap_log", True), context="capture.bootstrap_log"),
            require_stable_window_polls=int(payload.get("require_stable_window_polls", 5)),
            stable_window_poll_interval_seconds=float(payload.get("stable_window_poll_interval_seconds", 0.1)),
            latest_frame_only=_bool(payload.get("latest_frame_only", True), context="capture.latest_frame_only"),
            frame_timeout=float(payload.get("frame_timeout", 1.0)),
            post_reset_flush_seconds=float(payload.get("post_reset_flush_seconds", 0.25)),
            invalid_frame_limit=int(payload.get("invalid_frame_limit", 3)),
            region_change_tolerance_pixels=int(payload.get("region_change_tolerance_pixels", 4)),
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
class LidarObservationConfig:
    window_width: int = 958
    window_height: int = 488
    ray_count: int = 19
    lidar_hist_len: int = 4
    prev_action_hist_len: int = 2
    fixed_crop: tuple[float, float, float, float] = (0.18, 0.34, 0.82, 0.96)
    border_threshold: int = 48
    ray_min_angle_degrees: float = -80.0
    ray_max_angle_degrees: float = 80.0
    max_ray_length_fraction: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LidarObservationConfig":
        return cls(
            window_width=int(payload.get("window_width", 958)),
            window_height=int(payload.get("window_height", 488)),
            ray_count=int(payload.get("ray_count", 19)),
            lidar_hist_len=int(payload.get("lidar_hist_len", 4)),
            prev_action_hist_len=int(payload.get("prev_action_hist_len", 2)),
            fixed_crop=_float_tuple(
                payload.get("fixed_crop"),
                context="lidar_observation.fixed_crop",
                length=4,
                default=(0.18, 0.34, 0.82, 0.96),
            ),
            border_threshold=int(payload.get("border_threshold", 48)),
            ray_min_angle_degrees=float(payload.get("ray_min_angle_degrees", -80.0)),
            ray_max_angle_degrees=float(payload.get("ray_max_angle_degrees", 80.0)),
            max_ray_length_fraction=float(payload.get("max_ray_length_fraction", 1.0)),
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
    corridor_mode: str = "static"
    corridor_soft_margin_m: float = 25.0
    corridor_hard_margin_m: float = 100.0
    corridor_patience_steps: int = 60
    corridor_penalty_scale: float = 0.03
    corridor_penalty_max: float = 8.0
    corridor_recovery_bonus: float = 1.0
    corridor_catastrophic_distance_m: float = 350.0
    line_switch_max_distance_m: float = 140.0
    corridor_min_recovery_progress_m: float = 0.5
    corridor_min_recovery_speed_kmh: float = 8.0
    corridor_recovery_distance_delta_m: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RewardConfig":
        mode = str(payload.get("mode", "trajectory_progress")).strip().lower()
        if mode not in {"trajectory_progress", "ghost_bundle_progress"}:
            raise ConfigError(
                "reward.mode must be one of ['trajectory_progress', 'ghost_bundle_progress'], "
                f"got {mode!r}."
            )
        max_stray = float(payload.get("max_stray", 100.0))
        corridor_mode = str(payload.get("corridor_mode", "static")).strip().lower()
        if corridor_mode not in {"static", "map_calibrated"}:
            raise ConfigError(
                "reward.corridor_mode must be one of ['static', 'map_calibrated'], "
                f"got {corridor_mode!r}."
            )
        corridor_soft_margin_m = float(payload.get("corridor_soft_margin_m", 25.0))
        corridor_hard_margin_m = float(payload.get("corridor_hard_margin_m", max_stray))
        corridor_patience_steps = int(payload.get("corridor_patience_steps", 60))
        corridor_penalty_scale = float(payload.get("corridor_penalty_scale", 0.03))
        corridor_penalty_max = float(payload.get("corridor_penalty_max", 8.0))
        corridor_recovery_bonus = float(payload.get("corridor_recovery_bonus", 1.0))
        corridor_catastrophic_distance_m = float(payload.get("corridor_catastrophic_distance_m", 350.0))
        line_switch_max_distance_m = float(payload.get("line_switch_max_distance_m", 140.0))
        corridor_min_recovery_progress_m = float(payload.get("corridor_min_recovery_progress_m", 0.5))
        corridor_min_recovery_speed_kmh = float(payload.get("corridor_min_recovery_speed_kmh", 8.0))
        corridor_recovery_distance_delta_m = float(payload.get("corridor_recovery_distance_delta_m", 1.0))
        if corridor_soft_margin_m < 0.0:
            raise ConfigError("reward.corridor_soft_margin_m must be >= 0.")
        if corridor_hard_margin_m < corridor_soft_margin_m:
            raise ConfigError("reward.corridor_hard_margin_m must be >= reward.corridor_soft_margin_m.")
        if corridor_patience_steps < 1:
            raise ConfigError("reward.corridor_patience_steps must be >= 1.")
        if corridor_penalty_scale < 0.0:
            raise ConfigError("reward.corridor_penalty_scale must be >= 0.")
        if corridor_penalty_max < 0.0:
            raise ConfigError("reward.corridor_penalty_max must be >= 0.")
        if corridor_recovery_bonus < 0.0:
            raise ConfigError("reward.corridor_recovery_bonus must be >= 0.")
        if corridor_catastrophic_distance_m <= corridor_hard_margin_m:
            raise ConfigError(
                "reward.corridor_catastrophic_distance_m must be greater than reward.corridor_hard_margin_m."
            )
        if line_switch_max_distance_m <= 0.0:
            raise ConfigError("reward.line_switch_max_distance_m must be > 0.")
        if corridor_min_recovery_progress_m < 0.0:
            raise ConfigError("reward.corridor_min_recovery_progress_m must be >= 0.")
        if corridor_min_recovery_speed_kmh < 0.0:
            raise ConfigError("reward.corridor_min_recovery_speed_kmh must be >= 0.")
        if corridor_recovery_distance_delta_m < 0.0:
            raise ConfigError("reward.corridor_recovery_distance_delta_m must be >= 0.")
        return cls(
            mode=mode,
            spacing_meters=float(payload.get("spacing_meters", 0.5)),
            end_of_track=float(payload.get("end_of_track", payload.get("finish_bonus", 100.0))),
            constant_penalty=float(payload.get("constant_penalty", 0.0)),
            check_forward=int(payload.get("check_forward", 500)),
            check_backward=int(payload.get("check_backward", 10)),
            failure_countdown=int(payload.get("failure_countdown", 10)),
            min_steps=int(payload.get("min_steps", 70)),
            max_stray=max_stray,
            corridor_mode=corridor_mode,
            corridor_soft_margin_m=corridor_soft_margin_m,
            corridor_hard_margin_m=corridor_hard_margin_m,
            corridor_patience_steps=corridor_patience_steps,
            corridor_penalty_scale=corridor_penalty_scale,
            corridor_penalty_max=corridor_penalty_max,
            corridor_recovery_bonus=corridor_recovery_bonus,
            corridor_catastrophic_distance_m=corridor_catastrophic_distance_m,
            line_switch_max_distance_m=line_switch_max_distance_m,
            corridor_min_recovery_progress_m=corridor_min_recovery_progress_m,
            corridor_min_recovery_speed_kmh=corridor_min_recovery_speed_kmh,
            corridor_recovery_distance_delta_m=corridor_recovery_distance_delta_m,
        )


@dataclass(slots=True)
class EvalConfig:
    episodes: int = 20
    seed_base: int = 12345
    sector_count: int = 10
    record_video: bool = False
    video_fps: int = 20
    modes: tuple[str, ...] = ("deterministic",)
    trace_seconds: float = 3.0
    final_checkpoint_eval: bool = True
    extraction_modes: tuple[str, ...] = ("deterministic_mean", "stochastic")
    temperature_sweep: tuple[float, ...] = (1.0,)
    best_of_k: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvalConfig":
        modes = _string_tuple(payload.get("modes"), context="eval.modes", default=("deterministic",))
        invalid_modes = sorted({mode for mode in modes if mode not in {"deterministic", "stochastic"}})
        if invalid_modes:
            raise ConfigError(
                "eval.modes must only contain 'deterministic' and/or 'stochastic', "
                f"got {invalid_modes!r}."
            )
        trace_seconds = float(payload.get("trace_seconds", 3.0))
        if trace_seconds <= 0.0:
            raise ConfigError(f"eval.trace_seconds must be > 0, got {trace_seconds}.")
        extraction_modes = _string_tuple(
            payload.get("extraction_modes"),
            context="eval.extraction_modes",
            default=("deterministic_mean", "stochastic"),
        )
        allowed_extraction_modes = {
            "deterministic_mean",
            "stochastic",
            "clipped_mean",
            "sample_best_of_k",
        }
        invalid_extraction_modes = sorted(set(extraction_modes) - allowed_extraction_modes)
        if invalid_extraction_modes:
            raise ConfigError(
                "eval.extraction_modes must only contain deterministic_mean, stochastic, clipped_mean, "
                f"or sample_best_of_k; got {invalid_extraction_modes!r}."
            )
        temperature_sweep = _float_sequence(
            payload.get("temperature_sweep"),
            context="eval.temperature_sweep",
            default=(1.0,),
        )
        if any(value <= 0.0 for value in temperature_sweep):
            raise ConfigError(f"eval.temperature_sweep values must be > 0, got {temperature_sweep!r}.")
        best_of_k = int(payload.get("best_of_k", 1))
        if best_of_k < 1:
            raise ConfigError(f"eval.best_of_k must be >= 1, got {best_of_k}.")
        return cls(
            episodes=int(payload.get("episodes", 20)),
            seed_base=int(payload.get("seed_base", 12345)),
            sector_count=int(payload.get("sector_count", 10)),
            record_video=_bool(payload.get("record_video", False), context="eval.record_video"),
            video_fps=int(payload.get("video_fps", 20)),
            modes=modes,
            trace_seconds=trace_seconds,
            final_checkpoint_eval=_bool(
                payload.get("final_checkpoint_eval", True),
                context="eval.final_checkpoint_eval",
            ),
            extraction_modes=extraction_modes,
            temperature_sweep=temperature_sweep,
            best_of_k=best_of_k,
        )


@dataclass(slots=True)
class TrainConfig:
    algorithm: str = "sac"
    seed: int = 12345
    memory_size: int = 1_000_000
    batch_size: int = 256
    environment_steps_before_training: int = 1_000
    max_training_steps_per_environment_step: float = 4.0
    update_model_interval: int = 200
    update_buffer_interval: int = 200
    eval_interval_steps: int = 5_000
    checkpoint_interval_steps: int = 5_000
    queue_capacity: int = 8_192
    max_env_steps: int | None = None
    cuda_training: bool = True
    cuda_inference: bool = False
    single_live_env: bool = True
    broadcast_after_actor_update: bool = False
    actor_publish_every: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainConfig":
        algorithm = str(payload.get("algorithm", "sac")).strip().lower()
        if algorithm not in {"sac", "redq", "droq", "crossq"}:
            raise ConfigError(
                "train.algorithm must be one of ['sac', 'redq', 'droq', 'crossq'], "
                f"got {algorithm!r}."
            )
        max_env_steps = payload.get("max_env_steps")
        memory_size = payload.get("memory_size", payload.get("replay_capacity", 1_000_000))
        env_steps_before_training = payload.get(
            "environment_steps_before_training",
            payload.get("warmup_steps", 1_000),
        )
        max_train_ratio = payload.get(
            "max_training_steps_per_environment_step",
            payload.get("updates_per_step", 4.0),
        )
        actor_publish_every = int(payload.get("actor_publish_every", 1))
        if actor_publish_every < 1:
            raise ConfigError(f"train.actor_publish_every must be >= 1, got {actor_publish_every}.")
        return cls(
            algorithm=algorithm,
            seed=int(payload.get("seed", 12345)),
            memory_size=int(memory_size),
            batch_size=int(payload.get("batch_size", 256)),
            environment_steps_before_training=int(env_steps_before_training),
            max_training_steps_per_environment_step=float(max_train_ratio),
            update_model_interval=int(payload.get("update_model_interval", 200)),
            update_buffer_interval=int(payload.get("update_buffer_interval", 200)),
            eval_interval_steps=int(payload.get("eval_interval_steps", 5_000)),
            checkpoint_interval_steps=int(payload.get("checkpoint_interval_steps", 5_000)),
            queue_capacity=int(payload.get("queue_capacity", 8_192)),
            max_env_steps=None if max_env_steps in (None, "null") else int(max_env_steps),
            cuda_training=_bool(payload.get("cuda_training", True), context="train.cuda_training"),
            cuda_inference=_bool(payload.get("cuda_inference", False), context="train.cuda_inference"),
            single_live_env=_bool(payload.get("single_live_env", True), context="train.single_live_env"),
            broadcast_after_actor_update=_bool(
                payload.get("broadcast_after_actor_update", False),
                context="train.broadcast_after_actor_update",
            ),
            actor_publish_every=actor_publish_every,
        )


@dataclass(slots=True)
class SACConfig:
    gamma: float = 0.995
    polyak: float = 0.995
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    learn_entropy_coef: bool = False
    alpha: float = 0.01
    target_entropy: float = -2.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SACConfig":
        return cls(
            gamma=float(payload.get("gamma", 0.995)),
            polyak=float(payload.get("polyak", 0.995)),
            actor_lr=float(payload.get("actor_lr", 3e-4)),
            critic_lr=float(payload.get("critic_lr", 3e-4)),
            alpha_lr=float(payload.get("alpha_lr", 3e-4)),
            learn_entropy_coef=_bool(
                payload.get("learn_entropy_coef", payload.get("learn_alpha", False)),
                context="sac.learn_entropy_coef",
            ),
            alpha=float(payload.get("alpha", 0.01)),
            target_entropy=float(payload.get("target_entropy", -2.0)),
        )


@dataclass(slots=True)
class REDQConfig:
    n_critics: int = 10
    m_subset: int = 2
    q_updates_per_policy_update: int = 20
    share_encoders: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "REDQConfig":
        n_critics = int(payload.get("n_critics", 10))
        m_subset = int(payload.get("m_subset", 2))
        q_updates_per_policy_update = int(payload.get("q_updates_per_policy_update", 20))
        if n_critics < 2:
            raise ConfigError(f"redq.n_critics must be >= 2, got {n_critics}.")
        if m_subset < 1 or m_subset > n_critics:
            raise ConfigError(
                f"redq.m_subset must satisfy 1 <= m_subset <= n_critics, got m_subset={m_subset}, n_critics={n_critics}."
            )
        if q_updates_per_policy_update < 1:
            raise ConfigError(
                "redq.q_updates_per_policy_update must be >= 1, "
                f"got {q_updates_per_policy_update}."
            )
        return cls(
            n_critics=n_critics,
            m_subset=m_subset,
            q_updates_per_policy_update=q_updates_per_policy_update,
            share_encoders=_bool(payload.get("share_encoders", False), context="redq.share_encoders"),
        )


@dataclass(slots=True)
class DroQConfig:
    n_critics: int = 2
    m_subset: int = 2
    q_updates_per_policy_update: int = 5
    share_encoders: bool = True
    dropout_probability: float = 0.01

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DroQConfig":
        n_critics = int(payload.get("n_critics", 2))
        m_subset = int(payload.get("m_subset", 2))
        q_updates_per_policy_update = int(payload.get("q_updates_per_policy_update", 5))
        dropout_probability = float(payload.get("dropout_probability", 0.01))
        if n_critics < 2:
            raise ConfigError(f"droq.n_critics must be >= 2, got {n_critics}.")
        if m_subset < 1 or m_subset > n_critics:
            raise ConfigError(
                f"droq.m_subset must satisfy 1 <= m_subset <= n_critics, got m_subset={m_subset}, n_critics={n_critics}."
            )
        if q_updates_per_policy_update < 1:
            raise ConfigError(
                "droq.q_updates_per_policy_update must be >= 1, "
                f"got {q_updates_per_policy_update}."
            )
        if not 0.0 <= dropout_probability < 1.0:
            raise ConfigError(
                "droq.dropout_probability must satisfy 0.0 <= dropout_probability < 1.0, "
                f"got {dropout_probability}."
            )
        return cls(
            n_critics=n_critics,
            m_subset=m_subset,
            q_updates_per_policy_update=q_updates_per_policy_update,
            share_encoders=_bool(payload.get("share_encoders", True), context="droq.share_encoders"),
            dropout_probability=dropout_probability,
        )


@dataclass(slots=True)
class CrossQConfig:
    share_encoders: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CrossQConfig":
        return cls(
            share_encoders=_bool(payload.get("share_encoders", False), context="crossq.share_encoders"),
        )


@dataclass(slots=True)
class BCConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    validation_fraction: float = 0.2

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BCConfig":
        return cls(
            epochs=int(payload.get("epochs", 20)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            weight_decay=float(payload.get("weight_decay", 0.0)),
            validation_fraction=float(payload.get("validation_fraction", 0.2)),
        )


@dataclass(slots=True)
class GhostSelectionOverrideConfig:
    ghost_name_contains: str | None = None
    rank: int | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        context: str,
    ) -> "GhostSelectionOverrideConfig":
        ghost_name_contains = payload.get("ghost_name_contains")
        if ghost_name_contains is not None:
            ghost_name_contains = str(ghost_name_contains).strip()
            if not ghost_name_contains:
                raise ConfigError(f"{context}.ghost_name_contains must be a non-empty string when provided.")
        rank = payload.get("rank")
        if rank in (None, "null"):
            rank_value = None
        else:
            rank_value = int(rank)
            if rank_value < 1:
                raise ConfigError(f"{context}.rank must be >= 1, got {rank_value}.")
        if ghost_name_contains is None and rank_value is None:
            raise ConfigError(f"{context} must provide ghost_name_contains and/or rank.")
        return cls(
            ghost_name_contains=ghost_name_contains,
            rank=rank_value,
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.ghost_name_contains is not None:
            payload["ghost_name_contains"] = self.ghost_name_contains
        if self.rank is not None:
            payload["rank"] = self.rank
        return payload


@dataclass(slots=True)
class GhostConfig:
    enabled: bool = False
    root: str = "data/ghosts"
    bundle_manifest: str | None = None
    leaderboard_length: int = 100
    group_uid: str = "Personal_Best"
    only_world: bool = True
    canonical_reference_mode: str = "author_medal"
    author_reference_manifest: str | None = None
    unavailable_intended_policy: str = "selected_ghost_then_author_then_error"
    selected_ghost_overrides: dict[str, "GhostSelectionOverrideConfig"] = field(default_factory=dict)
    training_family: str = "intended_route"
    ambiguous_family_policy: str = "mixed_with_warning"
    anchor_count: int = 24
    anchor_radius_m: float = 12.0
    canonical_divergence_radius_m: float = 25.0
    early_reverse_window_ms: int = 10_000
    intended_anchor_fraction_min: float = 0.80
    exploit_anchor_fraction_max: float = 0.50
    exploit_reverse_progress_threshold_m: float = 15.0
    intended_candidate_pool: int = 30
    intended_bundle_size: int = 15
    exploit_bundle_size: int = 10
    default_bands: tuple[str, ...] = ("1-10", "11-30", "31-60", "61-100")
    max_representatives_per_band: int = 5
    line_switch_hysteresis: int = 5

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GhostConfig":
        leaderboard_length = int(payload.get("leaderboard_length", 100))
        if leaderboard_length < 1 or leaderboard_length > 100:
            raise ConfigError(f"ghosts.leaderboard_length must satisfy 1 <= value <= 100, got {leaderboard_length}.")
        max_representatives_per_band = int(payload.get("max_representatives_per_band", 5))
        if max_representatives_per_band < 1:
            raise ConfigError(
                "ghosts.max_representatives_per_band must be >= 1, "
                f"got {max_representatives_per_band}."
            )
        canonical_reference_mode = str(payload.get("canonical_reference_mode", "author_medal")).strip().lower()
        if canonical_reference_mode not in {"author_medal"}:
            raise ConfigError(
                "ghosts.canonical_reference_mode must be 'author_medal', "
                f"got {canonical_reference_mode!r}."
            )
        unavailable_intended_policy = str(
            payload.get("unavailable_intended_policy", "selected_ghost_then_author_then_error")
        ).strip().lower()
        if unavailable_intended_policy not in {"selected_ghost_then_author_then_error"}:
            raise ConfigError(
                "ghosts.unavailable_intended_policy must be 'selected_ghost_then_author_then_error', "
                f"got {unavailable_intended_policy!r}."
            )
        training_family = str(payload.get("training_family", "intended_route")).strip().lower()
        if training_family not in {"intended_route"}:
            raise ConfigError(
                "ghosts.training_family must be 'intended_route', "
                f"got {training_family!r}."
            )
        ambiguous_family_policy = str(payload.get("ambiguous_family_policy", "mixed_with_warning")).strip().lower()
        if ambiguous_family_policy not in {"mixed_with_warning"}:
            raise ConfigError(
                "ghosts.ambiguous_family_policy must be 'mixed_with_warning', "
                f"got {ambiguous_family_policy!r}."
            )
        anchor_count = int(payload.get("anchor_count", 24))
        if anchor_count < 2:
            raise ConfigError(f"ghosts.anchor_count must be >= 2, got {anchor_count}.")
        anchor_radius_m = float(payload.get("anchor_radius_m", 12.0))
        if anchor_radius_m <= 0.0:
            raise ConfigError(f"ghosts.anchor_radius_m must be > 0, got {anchor_radius_m}.")
        canonical_divergence_radius_m = float(payload.get("canonical_divergence_radius_m", 25.0))
        if canonical_divergence_radius_m <= 0.0:
            raise ConfigError(
                "ghosts.canonical_divergence_radius_m must be > 0, "
                f"got {canonical_divergence_radius_m}."
            )
        if canonical_divergence_radius_m < anchor_radius_m:
            raise ConfigError(
                "ghosts.canonical_divergence_radius_m must be >= ghosts.anchor_radius_m, "
                f"got {canonical_divergence_radius_m} < {anchor_radius_m}."
            )
        early_reverse_window_ms = int(payload.get("early_reverse_window_ms", 10_000))
        if early_reverse_window_ms < 0:
            raise ConfigError(f"ghosts.early_reverse_window_ms must be >= 0, got {early_reverse_window_ms}.")
        intended_anchor_fraction_min = float(payload.get("intended_anchor_fraction_min", 0.80))
        exploit_anchor_fraction_max = float(payload.get("exploit_anchor_fraction_max", 0.50))
        if not 0.0 <= exploit_anchor_fraction_max <= 1.0:
            raise ConfigError(
                "ghosts.exploit_anchor_fraction_max must be within [0, 1], "
                f"got {exploit_anchor_fraction_max}."
            )
        if not 0.0 <= intended_anchor_fraction_min <= 1.0:
            raise ConfigError(
                "ghosts.intended_anchor_fraction_min must be within [0, 1], "
                f"got {intended_anchor_fraction_min}."
            )
        if intended_anchor_fraction_min <= exploit_anchor_fraction_max:
            raise ConfigError(
                "ghosts.intended_anchor_fraction_min must be > ghosts.exploit_anchor_fraction_max, "
                f"got {intended_anchor_fraction_min} <= {exploit_anchor_fraction_max}."
            )
        exploit_reverse_progress_threshold_m = float(payload.get("exploit_reverse_progress_threshold_m", 15.0))
        if exploit_reverse_progress_threshold_m < 0.0:
            raise ConfigError(
                "ghosts.exploit_reverse_progress_threshold_m must be >= 0, "
                f"got {exploit_reverse_progress_threshold_m}."
            )
        intended_candidate_pool = int(payload.get("intended_candidate_pool", 30))
        if intended_candidate_pool < 1:
            raise ConfigError(f"ghosts.intended_candidate_pool must be >= 1, got {intended_candidate_pool}.")
        intended_bundle_size = int(payload.get("intended_bundle_size", 15))
        if intended_bundle_size < 1:
            raise ConfigError(f"ghosts.intended_bundle_size must be >= 1, got {intended_bundle_size}.")
        if intended_bundle_size > intended_candidate_pool:
            raise ConfigError(
                "ghosts.intended_bundle_size must be <= ghosts.intended_candidate_pool, "
                f"got {intended_bundle_size} > {intended_candidate_pool}."
            )
        exploit_bundle_size = int(payload.get("exploit_bundle_size", 10))
        if exploit_bundle_size < 1:
            raise ConfigError(f"ghosts.exploit_bundle_size must be >= 1, got {exploit_bundle_size}.")
        line_switch_hysteresis = int(payload.get("line_switch_hysteresis", 5))
        if line_switch_hysteresis < 0:
            raise ConfigError(f"ghosts.line_switch_hysteresis must be >= 0, got {line_switch_hysteresis}.")
        selected_ghost_overrides_payload = payload.get("selected_ghost_overrides", {})
        if selected_ghost_overrides_payload in (None, "null"):
            selected_ghost_overrides_payload = {}
        selected_ghost_overrides_mapping = _mapping(
            selected_ghost_overrides_payload,
            context="ghosts.selected_ghost_overrides",
        )
        selected_ghost_overrides: dict[str, GhostSelectionOverrideConfig] = {}
        for raw_map_uid, raw_selector in selected_ghost_overrides_mapping.items():
            map_uid = str(raw_map_uid).strip()
            if not map_uid:
                raise ConfigError("ghosts.selected_ghost_overrides keys must be non-empty map UIDs.")
            selected_ghost_overrides[map_uid] = GhostSelectionOverrideConfig.from_mapping(
                _mapping(raw_selector, context=f"ghosts.selected_ghost_overrides[{map_uid!r}]"),
                context=f"ghosts.selected_ghost_overrides[{map_uid!r}]",
            )
        bundle_manifest = payload.get("bundle_manifest")
        author_reference_manifest = payload.get("author_reference_manifest")
        return cls(
            enabled=_bool(payload.get("enabled", False), context="ghosts.enabled"),
            root=str(payload.get("root", "data/ghosts")),
            bundle_manifest=None if bundle_manifest in (None, "null") else str(bundle_manifest),
            leaderboard_length=leaderboard_length,
            group_uid=str(payload.get("group_uid", "Personal_Best")),
            only_world=_bool(payload.get("only_world", True), context="ghosts.only_world"),
            canonical_reference_mode=canonical_reference_mode,
            author_reference_manifest=(
                None if author_reference_manifest in (None, "null") else str(author_reference_manifest)
            ),
            unavailable_intended_policy=unavailable_intended_policy,
            selected_ghost_overrides=selected_ghost_overrides,
            training_family=training_family,
            ambiguous_family_policy=ambiguous_family_policy,
            anchor_count=anchor_count,
            anchor_radius_m=anchor_radius_m,
            canonical_divergence_radius_m=canonical_divergence_radius_m,
            early_reverse_window_ms=early_reverse_window_ms,
            intended_anchor_fraction_min=intended_anchor_fraction_min,
            exploit_anchor_fraction_max=exploit_anchor_fraction_max,
            exploit_reverse_progress_threshold_m=exploit_reverse_progress_threshold_m,
            intended_candidate_pool=intended_candidate_pool,
            intended_bundle_size=intended_bundle_size,
            exploit_bundle_size=exploit_bundle_size,
            default_bands=_string_tuple(
                payload.get("default_bands"),
                context="ghosts.default_bands",
                default=("1-10", "11-30", "31-60", "61-100"),
            ),
            max_representatives_per_band=max_representatives_per_band,
            line_switch_hysteresis=line_switch_hysteresis,
        )


@dataclass(slots=True)
class OfflinePretrainConfig:
    enabled: bool = False
    strategy: str = "bc_redq_awac"
    epochs: int = 5
    gradient_steps: int = 10_000
    batch_size: int = 256
    bc_weight: float = 1.0
    awac_lambda: float = 1.0
    cql_alpha: float = 0.0
    require_actions: bool = True
    seed_replay_buffer: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OfflinePretrainConfig":
        strategy = str(payload.get("strategy", "bc_redq_awac")).strip().lower()
        allowed = {"bc", "redq_critic", "awac", "iql", "cql", "bc_redq_awac"}
        if strategy not in allowed:
            raise ConfigError(f"offline_pretrain.strategy must be one of {sorted(allowed)}, got {strategy!r}.")
        epochs = int(payload.get("epochs", 5))
        gradient_steps = int(payload.get("gradient_steps", 10_000))
        batch_size = int(payload.get("batch_size", 256))
        if epochs < 1 or gradient_steps < 1 or batch_size < 1:
            raise ConfigError("offline_pretrain.epochs, gradient_steps, and batch_size must all be >= 1.")
        return cls(
            enabled=_bool(payload.get("enabled", False), context="offline_pretrain.enabled"),
            strategy=strategy,
            epochs=epochs,
            gradient_steps=gradient_steps,
            batch_size=batch_size,
            bc_weight=float(payload.get("bc_weight", 1.0)),
            awac_lambda=float(payload.get("awac_lambda", 1.0)),
            cql_alpha=float(payload.get("cql_alpha", 0.0)),
            require_actions=_bool(payload.get("require_actions", True), context="offline_pretrain.require_actions"),
            seed_replay_buffer=_bool(
                payload.get("seed_replay_buffer", True),
                context="offline_pretrain.seed_replay_buffer",
            ),
        )


@dataclass(slots=True)
class BalancedReplayConfig:
    enabled: bool = False
    offline_initial_fraction: float = 0.75
    offline_final_fraction: float = 0.10
    decay_env_steps: int = 50_000

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BalancedReplayConfig":
        initial = float(payload.get("offline_initial_fraction", 0.75))
        final = float(payload.get("offline_final_fraction", 0.10))
        decay_env_steps = int(payload.get("decay_env_steps", 50_000))
        if not 0.0 <= initial <= 1.0 or not 0.0 <= final <= 1.0:
            raise ConfigError("balanced_replay offline fractions must be within [0, 1].")
        if decay_env_steps < 1:
            raise ConfigError(f"balanced_replay.decay_env_steps must be >= 1, got {decay_env_steps}.")
        return cls(
            enabled=_bool(payload.get("enabled", False), context="balanced_replay.enabled"),
            offline_initial_fraction=initial,
            offline_final_fraction=final,
            decay_env_steps=decay_env_steps,
        )


@dataclass(slots=True)
class EliteArchiveConfig:
    enabled: bool = False
    root: str = "data/elite_archive"
    max_entries: int = 500
    novelty_bonus: float = 0.0
    min_progress_improvement: float = 25.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EliteArchiveConfig":
        max_entries = int(payload.get("max_entries", 500))
        if max_entries < 1:
            raise ConfigError(f"elite_archive.max_entries must be >= 1, got {max_entries}.")
        return cls(
            enabled=_bool(payload.get("enabled", False), context="elite_archive.enabled"),
            root=str(payload.get("root", "data/elite_archive")),
            max_entries=max_entries,
            novelty_bonus=float(payload.get("novelty_bonus", 0.0)),
            min_progress_improvement=float(payload.get("min_progress_improvement", 25.0)),
        )


@dataclass(slots=True)
class MetricsConfig:
    metric_version: str = "progress_v1"
    enable_plts: bool = False
    ghost_bundle_root: str | None = None
    progress_thresholds: tuple[int, ...] = (50, 100, 200, 400, 800, 1200, 1600, 2000)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MetricsConfig":
        return cls(
            metric_version=str(payload.get("metric_version", "progress_v1")),
            enable_plts=_bool(payload.get("enable_plts", False), context="metrics.enable_plts"),
            ghost_bundle_root=(
                None
                if payload.get("ghost_bundle_root") in (None, "null")
                else str(payload.get("ghost_bundle_root"))
            ),
            progress_thresholds=_int_tuple(
                payload.get("progress_thresholds"),
                context="metrics.progress_thresholds",
                default=(50, 100, 200, 400, 800, 1200, 1600, 2000),
            ),
        )


@dataclass(slots=True)
class ArtifactConfig:
    root: str = "artifacts"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactConfig":
        return cls(root=str(payload.get("root", "artifacts")))


@dataclass(slots=True)
class TM20AIConfig:
    runtime: RuntimeLoopConfig
    bridge: BridgeConnectionConfig
    observation: ObservationModeConfig
    capture: CaptureConfig
    full_observation: FullObservationConfig
    lidar_observation: LidarObservationConfig
    reward: RewardConfig
    eval: EvalConfig
    train: TrainConfig
    sac: SACConfig
    redq: REDQConfig
    droq: DroQConfig
    crossq: CrossQConfig
    bc: BCConfig
    ghosts: GhostConfig
    offline_pretrain: OfflinePretrainConfig
    balanced_replay: BalancedReplayConfig
    elite_archive: EliteArchiveConfig
    metrics: MetricsConfig
    artifacts: ArtifactConfig

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
            lidar_observation=LidarObservationConfig.from_mapping(
                _mapping(payload.get("lidar_observation", {}), context="lidar_observation")
            ),
            reward=RewardConfig.from_mapping(_mapping(payload.get("reward", {}), context="reward")),
            eval=EvalConfig.from_mapping(_mapping(payload.get("eval", {}), context="eval")),
            train=TrainConfig.from_mapping(_mapping(payload.get("train", {}), context="train")),
            sac=SACConfig.from_mapping(_mapping(payload.get("sac", {}), context="sac")),
            redq=REDQConfig.from_mapping(_mapping(payload.get("redq", {}), context="redq")),
            droq=DroQConfig.from_mapping(_mapping(payload.get("droq", {}), context="droq")),
            crossq=CrossQConfig.from_mapping(_mapping(payload.get("crossq", {}), context="crossq")),
            bc=BCConfig.from_mapping(_mapping(payload.get("bc", {}), context="bc")),
            ghosts=GhostConfig.from_mapping(_mapping(payload.get("ghosts", {}), context="ghosts")),
            offline_pretrain=OfflinePretrainConfig.from_mapping(
                _mapping(payload.get("offline_pretrain", {}), context="offline_pretrain")
            ),
            balanced_replay=BalancedReplayConfig.from_mapping(
                _mapping(payload.get("balanced_replay", {}), context="balanced_replay")
            ),
            elite_archive=EliteArchiveConfig.from_mapping(
                _mapping(payload.get("elite_archive", {}), context="elite_archive")
            ),
            metrics=MetricsConfig.from_mapping(_mapping(payload.get("metrics", {}), context="metrics")),
            artifacts=ArtifactConfig.from_mapping(_mapping(payload.get("artifacts", {}), context="artifacts")),
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
