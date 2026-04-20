from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ..capture import lidar_feature_dim
from ..train.features import ACTION_DIM, TELEMETRY_DIM

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class TimingAccumulator:
    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    def record(self, duration_seconds: float) -> None:
        duration = max(0.0, float(duration_seconds))
        self.count += 1
        self.total_seconds += duration
        self.max_seconds = max(self.max_seconds, duration)

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_seconds": self.total_seconds,
            "avg_seconds": self.total_seconds / max(1, self.count),
            "max_seconds": self.max_seconds,
        }


@dataclass(slots=True)
class QueueAccumulator:
    attempts: int = 0
    success_count: int = 0
    wait_count: int = 0
    full_retries: int = 0
    timeout_count: int = 0
    total_wait_seconds: float = 0.0
    max_wait_seconds: float = 0.0

    def record(
        self,
        wait_seconds: float,
        *,
        success: bool,
        full_retries: int = 0,
        timed_out: bool = False,
    ) -> None:
        wait_duration = max(0.0, float(wait_seconds))
        self.attempts += 1
        if success:
            self.success_count += 1
        if wait_duration > 0.0:
            self.wait_count += 1
            self.total_wait_seconds += wait_duration
            self.max_wait_seconds = max(self.max_wait_seconds, wait_duration)
        self.full_retries += int(full_retries)
        if timed_out:
            self.timeout_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "success_count": self.success_count,
            "wait_count": self.wait_count,
            "full_retries": self.full_retries,
            "timeout_count": self.timeout_count,
            "total_wait_seconds": self.total_wait_seconds,
            "avg_wait_seconds": self.total_wait_seconds / max(1, self.wait_count),
            "max_wait_seconds": self.max_wait_seconds,
        }


@dataclass(slots=True)
class RollingRatioTracker:
    window_env_steps: int = 1_000
    _points: list[tuple[int, int]] = field(default_factory=list)

    def record(self, *, env_step: int, learner_step: int) -> float | None:
        env = int(env_step)
        learner = int(learner_step)
        if self._points and self._points[-1] == (env, learner):
            return self.current_ratio()
        self._points.append((env, learner))
        cutoff = env - max(1, int(self.window_env_steps))
        while len(self._points) > 1 and self._points[1][0] <= cutoff:
            self._points.pop(0)
        return self.current_ratio()

    def current_ratio(self) -> float | None:
        if len(self._points) < 2:
            return None
        start_env, start_learner = self._points[0]
        end_env, end_learner = self._points[-1]
        delta_env = end_env - start_env
        if delta_env <= 0:
            return None
        return float((end_learner - start_learner) / delta_env)

    def snapshot(self) -> dict[str, Any]:
        return {
            "window_env_steps": int(self.window_env_steps),
            "current": self.current_ratio(),
        }


class JsonlEventLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def write(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "timestamp": time.time(),
            "event": event,
            "payload": {} if payload is None else dict(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True))
            handle.write("\n")


def summarize_values(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _count_parameters(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _count_unique_parameters(modules: Sequence[torch.nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            pointer = id(parameter)
            if pointer in seen:
                continue
            seen.add(pointer)
            total += int(parameter.numel())
    return total


def build_agent_resource_profile(agent: Any, device: torch.device | str) -> dict[str, Any]:
    import torch

    resolved_device = torch.device(device)
    actor = agent.actor
    algorithm = str(getattr(agent, "algorithm_name", getattr(agent, "algorithm", "sac")))
    if hasattr(agent, "critic1"):
        critics = [agent.critic1, agent.critic2]
        target_critics = [
            critic
            for critic in (getattr(agent, "target_critic1", None), getattr(agent, "target_critic2", None))
            if critic is not None
        ]
        share_encoders = False
    else:
        critics = list(agent.critics)
        target_critics = list(getattr(agent, "target_critics", []))
        share_encoders = bool(getattr(agent, "share_encoders", False))

    critic_encoder_modules: list[torch.nn.Module] = []
    for critic in critics:
        vision_encoder = getattr(critic, "vision_encoder", None)
        telemetry_encoder = getattr(critic, "telemetry_encoder", None)
        if vision_encoder is not None:
            critic_encoder_modules.append(vision_encoder)
        if telemetry_encoder is not None:
            critic_encoder_modules.append(telemetry_encoder)

    resource_profile = {
        "algorithm": algorithm,
        "device": str(resolved_device),
        "observation_mode": str(getattr(agent, "observation_mode", "unknown")),
        "n_critics": len(critics),
        "share_encoders": share_encoders,
        "actor_parameter_count": _count_parameters(actor),
        "critic_parameter_count": sum(_count_parameters(critic) for critic in critics),
        "target_critic_parameter_count": sum(_count_parameters(critic) for critic in target_critics),
        "critic_encoder_parameter_count": sum(_count_parameters(module) for module in critic_encoder_modules),
        "unique_critic_encoder_parameter_count": _count_unique_parameters(critic_encoder_modules),
        "total_parameter_count": _count_parameters(actor)
        + sum(_count_parameters(critic) for critic in critics)
        + sum(_count_parameters(critic) for critic in target_critics),
    }

    if resolved_device.type == "cuda" and torch.cuda.is_available():
        index = resolved_device.index if resolved_device.index is not None else torch.cuda.current_device()
        resource_profile.update(
            {
                "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    else:
        resource_profile.update(
            {
                "cuda_memory_allocated_bytes": None,
                "cuda_memory_reserved_bytes": None,
                "cuda_max_memory_allocated_bytes": None,
                "cuda_max_memory_reserved_bytes": None,
            }
        )
    return resource_profile


@dataclass(slots=True)
class EpisodeDiagnosticsTracker:
    episode_count: int = 0
    positive_progress_fractions: list[float] = field(default_factory=list)
    nonpositive_progress_fractions: list[float] = field(default_factory=list)
    max_no_progress_streaks: list[int] = field(default_factory=list)
    max_nonpositive_reward_streaks: list[int] = field(default_factory=list)
    final_no_progress_steps: list[int] = field(default_factory=list)
    termination_reason_counts: dict[str, int] = field(default_factory=dict)

    def record(self, payload: Mapping[str, Any]) -> None:
        self.episode_count += 1
        self.positive_progress_fractions.append(float(payload.get("positive_progress_fraction", 0.0) or 0.0))
        self.nonpositive_progress_fractions.append(float(payload.get("nonpositive_progress_fraction", 0.0) or 0.0))
        self.max_no_progress_streaks.append(int(payload.get("max_no_progress_streak", 0) or 0))
        self.max_nonpositive_reward_streaks.append(int(payload.get("max_nonpositive_reward_streak", 0) or 0))
        self.final_no_progress_steps.append(int(payload.get("final_no_progress_steps", 0) or 0))
        termination_reason = str(payload.get("termination_reason") or "unknown")
        self.termination_reason_counts[termination_reason] = self.termination_reason_counts.get(termination_reason, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "positive_progress_fraction": summarize_values(self.positive_progress_fractions),
            "nonpositive_progress_fraction": summarize_values(self.nonpositive_progress_fractions),
            "max_no_progress_streak": summarize_values(self.max_no_progress_streaks),
            "max_nonpositive_reward_streak": summarize_values(self.max_nonpositive_reward_streaks),
            "final_no_progress_steps": summarize_values(self.final_no_progress_steps),
            "termination_reason_counts": dict(sorted(self.termination_reason_counts.items())),
        }


@dataclass(slots=True)
class MovementDiagnosticsTracker:
    episode_count: int = 0
    movement_started_count: int = 0
    no_movement_episode_count: int = 0
    stall_episode_count: int = 0
    stall_counts: list[int] = field(default_factory=list)
    first_stall_race_time_ms: list[int] = field(default_factory=list)
    first_stall_delay_ms: list[int] = field(default_factory=list)
    termination_reason_counts: dict[str, int] = field(default_factory=dict)

    def record(self, payload: Mapping[str, Any]) -> None:
        self.episode_count += 1
        movement_started = bool(payload.get("movement_started", False))
        if movement_started:
            self.movement_started_count += 1
        else:
            self.no_movement_episode_count += 1
        stall_count = int(payload.get("stall_count", 0) or 0)
        self.stall_counts.append(stall_count)
        if stall_count > 0:
            self.stall_episode_count += 1
        first_stall = payload.get("first_stall")
        if isinstance(first_stall, Mapping):
            race_time_ms = int(first_stall.get("race_time_ms", 0) or 0)
            movement_start_race_time_ms = int(first_stall.get("movement_start_race_time_ms", 0) or 0)
            self.first_stall_race_time_ms.append(race_time_ms)
            self.first_stall_delay_ms.append(max(0, race_time_ms - movement_start_race_time_ms))
        termination_reason = str(payload.get("termination_reason") or "unknown")
        self.termination_reason_counts[termination_reason] = self.termination_reason_counts.get(termination_reason, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "movement_started_rate": self.movement_started_count / max(1, self.episode_count),
            "no_movement_episode_count": self.no_movement_episode_count,
            "stall_episode_rate": self.stall_episode_count / max(1, self.episode_count),
            "stall_count": summarize_values(self.stall_counts),
            "first_stall_race_time_ms": summarize_values(self.first_stall_race_time_ms),
            "first_stall_delay_ms": summarize_values(self.first_stall_delay_ms),
            "termination_reason_counts": dict(sorted(self.termination_reason_counts.items())),
        }


@dataclass(slots=True)
class ActorSyncTracker:
    run_start_monotonic: float
    broadcasts: dict[int, dict[str, Any]] = field(default_factory=dict)
    seen_lag_seconds: list[float] = field(default_factory=list)
    applied_lag_seconds: list[float] = field(default_factory=list)
    apply_duration_seconds: list[float] = field(default_factory=list)
    version_lag: list[int] = field(default_factory=list)
    learner_step_lag: list[int] = field(default_factory=list)
    env_step_lag: list[int] = field(default_factory=list)
    first_ready_actor_seconds: float | None = None
    first_applied_ready_actor_seconds: float | None = None
    first_policy_control_window_seconds: float | None = None
    policy_control_fraction: float | None = None
    current_actor_staleness: int | None = None
    current_versions_behind: int | None = None

    def record_broadcast(
        self,
        version: int,
        *,
        env_step: int,
        learner_step: int,
        actor_step: int | None,
        ready_for_control: bool,
        broadcast_monotonic: float,
    ) -> None:
        self.broadcasts[int(version)] = {
            "broadcast_monotonic": float(broadcast_monotonic),
            "env_step": int(env_step),
            "learner_step": int(learner_step),
            "actor_step": None if actor_step is None else int(actor_step),
            "ready_for_control": bool(ready_for_control),
        }
        if ready_for_control and self.first_ready_actor_seconds is None:
            self.first_ready_actor_seconds = float(broadcast_monotonic - self.run_start_monotonic)

    def record_desired_seen(self, payload: Mapping[str, Any], *, received_monotonic: float) -> None:
        version = payload.get("desired_actor_version")
        if version is None:
            return
        broadcast = self.broadcasts.get(int(version))
        if broadcast is None:
            return
        self.seen_lag_seconds.append(max(0.0, float(received_monotonic - broadcast["broadcast_monotonic"])))

    def record_applied(
        self,
        payload: Mapping[str, Any],
        *,
        received_monotonic: float,
        current_learner_step: int,
        current_env_step: int,
    ) -> None:
        version = payload.get("applied_actor_version")
        if version is not None:
            broadcast = self.broadcasts.get(int(version))
            if broadcast is not None:
                self.applied_lag_seconds.append(max(0.0, float(received_monotonic - broadcast["broadcast_monotonic"])))
        if payload.get("apply_duration_seconds") is not None:
            self.apply_duration_seconds.append(float(payload["apply_duration_seconds"]))
        applied_source_learner_step = payload.get("applied_source_learner_step")
        if applied_source_learner_step is not None:
            staleness = max(0, int(current_learner_step - int(applied_source_learner_step)))
            self.learner_step_lag.append(staleness)
            self.current_actor_staleness = staleness
        requested_env_step = payload.get("requested_env_step")
        if requested_env_step is not None:
            self.env_step_lag.append(max(0, int(current_env_step - int(requested_env_step))))
        if payload.get("actor_ready_for_control") and self.first_applied_ready_actor_seconds is None:
            self.first_applied_ready_actor_seconds = float(received_monotonic - self.run_start_monotonic)

    def record_control_window(
        self,
        payload: Mapping[str, Any],
        *,
        received_monotonic: float,
        current_learner_step: int,
        current_env_step: int,
    ) -> None:
        desired_version = payload.get("desired_actor_version")
        applied_version = payload.get("applied_actor_version")
        if desired_version is not None and applied_version is not None:
            versions_behind = max(0, int(desired_version) - int(applied_version))
            self.version_lag.append(versions_behind)
            self.current_versions_behind = versions_behind
        applied_source_learner_step = payload.get("applied_source_learner_step")
        if applied_source_learner_step is not None:
            staleness = max(0, int(current_learner_step - int(applied_source_learner_step)))
            self.learner_step_lag.append(staleness)
            self.current_actor_staleness = staleness
        last_actor_apply_env_step = payload.get("last_actor_apply_env_step")
        if last_actor_apply_env_step is not None:
            self.env_step_lag.append(max(0, int(current_env_step - int(last_actor_apply_env_step))))
        if payload.get("policy_control_fraction") is not None:
            self.policy_control_fraction = float(payload["policy_control_fraction"])
        if payload.get("control_source") == "policy" and self.first_policy_control_window_seconds is None:
            self.first_policy_control_window_seconds = float(received_monotonic - self.run_start_monotonic)

    def snapshot(self) -> dict[str, Any]:
        return {
            "broadcast_count": len(self.broadcasts),
            "time_to_seen_seconds": summarize_values(self.seen_lag_seconds),
            "time_to_applied_seconds": summarize_values(self.applied_lag_seconds),
            "worker_apply_duration_seconds": summarize_values(self.apply_duration_seconds),
            "versions_behind": summarize_values(self.version_lag),
            "learner_step_lag": summarize_values(self.learner_step_lag),
            "env_step_lag": summarize_values(self.env_step_lag),
            "time_to_first_ready_actor_seconds": self.first_ready_actor_seconds,
            "time_to_first_applied_ready_actor_seconds": self.first_applied_ready_actor_seconds,
            "time_to_first_policy_control_window_seconds": self.first_policy_control_window_seconds,
            "policy_control_fraction": self.policy_control_fraction,
            "current_actor_staleness": self.current_actor_staleness,
            "current_versions_behind": self.current_versions_behind,
        }


def build_bottleneck_verdict(
    *,
    learner_backprop_seconds: float,
    worker_env_seconds: float,
    ipc_backpressure_seconds: float,
    actor_sync_seconds: float,
) -> dict[str, Any]:
    breakdown = {
        "learner_backprop": max(0.0, float(learner_backprop_seconds)),
        "worker_env": max(0.0, float(worker_env_seconds)),
        "ipc_backpressure": max(0.0, float(ipc_backpressure_seconds)),
        "actor_sync": max(0.0, float(actor_sync_seconds)),
    }
    label = max(breakdown, key=breakdown.get) if breakdown else "unknown"
    return {
        "label": label,
        "breakdown_seconds": breakdown,
    }


def _build_full_replay(*, rng_seed: int, capacity: int, observation_shape: tuple[int, int, int]) -> ReplayBuffer:
    from .replay import ReplayBuffer

    replay = ReplayBuffer(
        mode="full",
        capacity=capacity,
        observation_shape=observation_shape,
        telemetry_dim=TELEMETRY_DIM,
        rng_seed=rng_seed,
    )
    channels, height, width = observation_shape
    for step in range(max(capacity // 2, 8)):
        replay.add(
            {
                "obs_uint8": np.full((channels, height, width), step % 255, dtype=np.uint8),
                "telemetry_float": np.full((TELEMETRY_DIM,), 0.01 * step, dtype=np.float32),
                "action": np.asarray([0.1, -0.1], dtype=np.float32),
                "reward": 1.0,
                "next_obs_uint8": np.full((channels, height, width), (step + 1) % 255, dtype=np.uint8),
                "next_telemetry_float": np.full((TELEMETRY_DIM,), 0.01 * (step + 1), dtype=np.float32),
                "terminated": False,
                "truncated": False,
            }
        )
    return replay


def _sync_cuda(device: torch.device) -> None:
    import torch

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def benchmark_redq_sweep(
    *,
    device: torch.device | str,
    observation_shape: tuple[int, int, int] = (4, 64, 64),
    batch_size: int = 32,
    warmup_updates: int = 2,
    measured_updates: int = 6,
    rng_seed: int = 12345,
    sweep: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    import torch

    from ..algos.redq import REDQSACAgent
    from ..config import REDQConfig, SACConfig

    resolved_device = torch.device(device)
    replay = _build_full_replay(rng_seed=rng_seed, capacity=max(batch_size * 4, 64), observation_shape=observation_shape)
    batch = replay.sample(batch_size, device=resolved_device)
    sweep_rows = list(sweep) if sweep is not None else [
        {"n_critics": 4, "m_subset": 2, "share_encoders": True},
        {"n_critics": 10, "m_subset": 2, "share_encoders": True},
        {"n_critics": 4, "m_subset": 2, "share_encoders": False},
        {"n_critics": 10, "m_subset": 2, "share_encoders": False},
    ]

    results: list[dict[str, Any]] = []
    for row in sweep_rows:
        agent = REDQSACAgent(
            sac_config=SACConfig(actor_lr=3.0e-4, critic_lr=5.0e-5, learn_entropy_coef=False, alpha=0.37),
            redq_config=REDQConfig(
                n_critics=int(row.get("n_critics", 10)),
                m_subset=int(row.get("m_subset", 2)),
                q_updates_per_policy_update=int(row.get("q_updates_per_policy_update", 20)),
                share_encoders=bool(row.get("share_encoders", False)),
            ),
            observation_mode="full",
            device=resolved_device,
            observation_shape=observation_shape,
            telemetry_dim=TELEMETRY_DIM,
            action_dim=ACTION_DIM,
        )
        if resolved_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(resolved_device)
        for _ in range(max(0, int(warmup_updates))):
            agent.update_critics(batch)
            agent.maybe_update_actor_and_alpha(batch)

        critic_update_durations_ms: list[float] = []
        actor_update_durations_ms: list[float] = []
        actor_update_count = 0
        total_start = time.perf_counter()
        for _ in range(max(1, int(measured_updates))):
            _sync_cuda(resolved_device)
            critic_start = time.perf_counter()
            agent.update_critics(batch)
            _sync_cuda(resolved_device)
            critic_update_durations_ms.append((time.perf_counter() - critic_start) * 1000.0)

            _sync_cuda(resolved_device)
            actor_start = time.perf_counter()
            actor_result = agent.maybe_update_actor_and_alpha(batch)
            _sync_cuda(resolved_device)
            if actor_result is not None:
                actor_update_count += 1
                actor_update_durations_ms.append((time.perf_counter() - actor_start) * 1000.0)
        total_elapsed_seconds = max(1e-9, time.perf_counter() - total_start)
        resource_profile = build_agent_resource_profile(agent, resolved_device)
        results.append(
            {
                "n_critics": int(row.get("n_critics", 10)),
                "m_subset": int(row.get("m_subset", 2)),
                "q_updates_per_policy_update": int(row.get("q_updates_per_policy_update", 20)),
                "share_encoders": bool(row.get("share_encoders", False)),
                "critic_update_ms": summarize_values(critic_update_durations_ms),
                "actor_update_ms": summarize_values(actor_update_durations_ms),
                "critic_updates_per_second": float(measured_updates / total_elapsed_seconds),
                "actor_update_count": actor_update_count,
                "resource_profile": resource_profile,
            }
        )

    return {
        "device": str(resolved_device),
        "batch_size": int(batch_size),
        "warmup_updates": int(warmup_updates),
        "measured_updates": int(measured_updates),
        "observation_shape": tuple(int(value) for value in observation_shape),
        "results": results,
    }


def default_benchmark_observation_shape() -> tuple[int, int, int]:
    return (4, 64, 64)


def default_lidar_benchmark_dim() -> int:
    from ..config import LidarObservationConfig

    return int(lidar_feature_dim(LidarObservationConfig()))
