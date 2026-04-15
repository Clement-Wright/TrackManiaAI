from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
import math
from time import monotonic
from typing import Any, Callable, Mapping

import numpy as np

from ..action_space import clamp_action
from ..capture import lidar_feature_dim
from ..config import TM20AIConfig, load_tm20ai_config
from ..env import TM20AIGymEnv, make_env
from .diagnostics import QueueAccumulator, TimingAccumulator
from .features import TELEMETRY_DIM, TelemetryFeatureBuilder
from .protocol import EvalResult


@dataclass(slots=True)
class TrainingEpisodeState:
    episode_index: int = 0
    step_count: int = 0
    episode_reward: float = 0.0

    @property
    def episode_id(self) -> str:
        return f"train_episode_{self.episode_index:06d}"


@dataclass(slots=True)
class RewardEpisodeState:
    step_count: int = 0
    positive_progress_steps: int = 0
    nonpositive_progress_steps: int = 0
    current_no_progress_streak: int = 0
    max_no_progress_streak: int = 0
    current_nonpositive_reward_streak: int = 0
    max_nonpositive_reward_streak: int = 0
    final_no_progress_steps: int = 0


@dataclass(slots=True)
class MovementEpisodeState:
    run_id: str | None = None
    episode_index: int | None = None
    start_pos: tuple[float, float, float] | None = None
    movement_started: bool = False
    movement_start_env_step: int | None = None
    movement_start_frame_id: int | None = None
    movement_start_race_time_ms: int | None = None
    candidate_stall_env_step: int | None = None
    candidate_stall_frame_id: int | None = None
    candidate_stall_race_time_ms: int | None = None
    candidate_stall_timestamp_ns: int | None = None
    candidate_stall_pos: tuple[float, float, float] | None = None
    first_stall: dict[str, Any] | None = None
    stall_count: int = 0
    max_speed_kmh: float = 0.0
    max_distance_from_start_m: float = 0.0


class SACWorker:
    def __init__(
        self,
        *,
        config_path: str,
        command_queue,
        output_queue,
        eval_result_queue,
        shutdown_event,
        worker_done_event,
        bootstrap_log_path: str | None = None,
        env_factory: Callable[[str, bool], TM20AIGymEnv] | None = None,
    ) -> None:
        self.config_path = config_path
        self.config: TM20AIConfig = load_tm20ai_config(config_path)
        self.command_queue = command_queue
        self.output_queue = output_queue
        self.eval_result_queue = eval_result_queue
        self.shutdown_event = shutdown_event
        self.worker_done_event = worker_done_event
        self.bootstrap_log_path = None if bootstrap_log_path is None else Path(bootstrap_log_path)
        self.worker_events_log_path = None if self.bootstrap_log_path is None else self.bootstrap_log_path.with_name(
            "worker_events.log"
        )
        self.desired_actor_path = None if self.bootstrap_log_path is None else self.bootstrap_log_path.parent / "worker_sync" / "desired_actor.json"
        self.worker_actor_status_path = (
            None if self.bootstrap_log_path is None else self.bootstrap_log_path.parent / "worker_sync" / "worker_actor_status.json"
        )
        self.env_factory = env_factory or (lambda path, benchmark=False: make_env(path, benchmark=benchmark))
        self._rng = np.random.default_rng(self.config.train.seed)

        self.observation_mode = self.config.observation.mode
        if self.observation_mode == "full":
            self._features: TelemetryFeatureBuilder | None = TelemetryFeatureBuilder()
        elif self.observation_mode == "lidar":
            self._features = None
        else:
            raise ValueError(f"Unsupported observation mode: {self.observation_mode!r}")

        self.actor = None
        self._torch = None
        self._has_policy_weights = False
        self._env_step = 0
        self._episode_state = TrainingEpisodeState()
        self._reward_episode_state = RewardEpisodeState()
        self._last_heartbeat = monotonic()
        self._pending_transitions: list[dict[str, Any]] = []
        self._shutdown_requested = False
        self._action_stats_interval_steps = 100
        self._desired_actor_version: int | None = None
        self._desired_actor_ready_for_control = False
        self._control_ready_reason: str | None = None
        self._seen_actor_version: int | None = None
        self._ready_for_control_seen = False
        self._applied_actor_version: int | None = None
        self._actor_ready_for_control = False
        self._applied_source_learner_step: int | None = None
        self._last_actor_apply_env_step: int | None = None
        self._last_actor_apply_episode_index: int | None = None
        self._eval_actor_version: int | None = None
        self._eval_actor_source_learner_step: int | None = None
        self._last_action_stats: dict[str, Any] | None = None
        self._recent_executed_actions: deque[np.ndarray] = deque(maxlen=self._action_stats_interval_steps)
        self._recent_raw_actions: deque[np.ndarray] = deque(maxlen=self._action_stats_interval_steps)
        self._action_guard_active = False
        self._status_mode = "train"
        self._movement_state = MovementEpisodeState()
        self._movement_start_speed_threshold_kmh = 5.0
        self._movement_start_distance_threshold_m = 0.5
        self._movement_stall_speed_threshold_kmh = 2.0
        self._movement_stall_position_epsilon_m = 0.15
        self._movement_stall_duration_seconds = 1.0
        self._control_step_count = 0
        self._policy_control_step_count = 0
        self._worker_runtime = {
            "env_step": TimingAccumulator(),
            "env_reset": TimingAccumulator(),
            "command_drain": TimingAccumulator(),
            "actor_apply": TimingAccumulator(),
        }
        self._worker_queue = {
            "output_put": QueueAccumulator(),
            "eval_result_put": QueueAccumulator(),
        }
        self._latest_env_benchmarks: dict[str, Any] = {}

    def _write_worker_event(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        if self.worker_events_log_path is None:
            return
        self.worker_events_log_path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "event": event,
            "env_step": self._env_step,
            "observation_mode": self.observation_mode,
        }
        if payload is not None:
            body["payload"] = dict(payload)
        with self.worker_events_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True))
            handle.write("\n")

    def _write_bootstrap_log(self, phase: str, payload: Mapping[str, Any]) -> None:
        if not self.config.capture.bootstrap_log or self.bootstrap_log_path is None:
            return
        self.bootstrap_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "phase": phase,
            "pid": os.getpid(),
            "payload": dict(payload),
        }
        with self.bootstrap_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")

    def _write_actor_status(self) -> None:
        if self.worker_actor_status_path is None:
            return
        self.worker_actor_status_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "desired_actor_version": self._desired_actor_version,
            "desired_actor_ready_for_control": self._desired_actor_ready_for_control,
            "seen_actor_version": self._seen_actor_version,
            "ready_for_control_seen": self._ready_for_control_seen,
            "applied_actor_version": self._applied_actor_version,
            "actor_ready_for_control": self._actor_ready_for_control,
            "applied_source_learner_step": self._applied_source_learner_step,
            "applied_at_env_step": self._last_actor_apply_env_step,
            "applied_at_episode_index": self._last_actor_apply_episode_index,
            "control_ready_reason": self._control_ready_reason,
            "mode": self._status_mode,
            "updated_at": time.time(),
        }
        temp_path = self.worker_actor_status_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.worker_actor_status_path)

    def _actor_sync_payload(
        self,
        *,
        desired_version: int | None = None,
        seen_version: int | None = None,
        applied_version: int | None = None,
        actor_state_path: str | None = None,
        written_at: str | None = None,
        learner_step: int | None = None,
        requested_env_step: int | None = None,
    ) -> dict[str, Any]:
        return {
            "desired_actor_version": self._desired_actor_version if desired_version is None else desired_version,
            "desired_actor_ready_for_control": self._desired_actor_ready_for_control,
            "seen_actor_version": self._seen_actor_version if seen_version is None else seen_version,
            "ready_for_control_seen": self._ready_for_control_seen,
            "applied_actor_version": self._applied_actor_version if applied_version is None else applied_version,
            "actor_ready_for_control": self._actor_ready_for_control,
            "applied_source_learner_step": self._applied_source_learner_step,
            "env_step": self._env_step,
            "episode_index": self._episode_state.episode_index,
            "last_actor_apply_env_step": self._last_actor_apply_env_step,
            "last_actor_apply_episode_index": self._last_actor_apply_episode_index,
            "actor_state_path": actor_state_path,
            "written_at": written_at,
            "learner_step": learner_step,
            "requested_env_step": requested_env_step,
            "control_ready_reason": self._control_ready_reason,
        }

    def _begin_reward_episode(self) -> None:
        self._reward_episode_state = RewardEpisodeState()

    def _observe_reward_step(self, *, reward: float, info: Mapping[str, Any]) -> None:
        state = self._reward_episode_state
        state.step_count += 1
        progress_delta = int(info.get("progress_delta", 0) or 0)
        if progress_delta > 0:
            state.positive_progress_steps += 1
            state.current_no_progress_streak = 0
        else:
            state.nonpositive_progress_steps += 1
            state.current_no_progress_streak += 1
            state.max_no_progress_streak = max(state.max_no_progress_streak, state.current_no_progress_streak)
        if float(reward) <= 0.0:
            state.current_nonpositive_reward_streak += 1
            state.max_nonpositive_reward_streak = max(
                state.max_nonpositive_reward_streak,
                state.current_nonpositive_reward_streak,
            )
        else:
            state.current_nonpositive_reward_streak = 0
        state.final_no_progress_steps = int(info.get("no_progress_steps", 0) or 0)

    def _reward_episode_summary(self, info: Mapping[str, Any]) -> dict[str, Any]:
        state = self._reward_episode_state
        steps = max(1, state.step_count)
        return {
            "positive_progress_fraction": state.positive_progress_steps / steps,
            "nonpositive_progress_fraction": state.nonpositive_progress_steps / steps,
            "max_no_progress_streak": state.max_no_progress_streak,
            "max_nonpositive_reward_streak": state.max_nonpositive_reward_streak,
            "final_no_progress_steps": state.final_no_progress_steps,
            "reset_reason": info.get("reward_reason") or info.get("terminal_reason"),
        }

    def _control_metrics_snapshot(self) -> dict[str, Any]:
        return {
            "control_step_count": self._control_step_count,
            "policy_control_step_count": self._policy_control_step_count,
            "policy_control_fraction": self._policy_control_step_count / max(1, self._control_step_count),
        }

    def _update_env_benchmarks(self, env: TM20AIGymEnv | None) -> None:
        if env is None:
            return
        try:
            benchmarks = env.benchmarks()
        except Exception:  # noqa: BLE001
            return
        tm20ai_metrics = dict(benchmarks.get("tm20ai", {}))
        observation_calls = int(tm20ai_metrics.get("observation_calls", 0) or 0)
        if observation_calls > 0:
            for key in ("avg_obs_retrieval_seconds", "avg_preprocess_seconds", "avg_reward_compute_seconds"):
                if tm20ai_metrics.get(key) is not None:
                    total_key = key.replace("avg_", "total_")
                    tm20ai_metrics[total_key] = float(tm20ai_metrics[key]) * observation_calls
        self._latest_env_benchmarks = {
            "tm20ai": tm20ai_metrics,
            "rtgym": dict(benchmarks.get("rtgym", {})),
        }

    def _runtime_profile_snapshot(self) -> dict[str, Any]:
        env_step_snapshot = self._worker_runtime["env_step"].snapshot()
        env_reset_snapshot = self._worker_runtime["env_reset"].snapshot()
        return {
            "env_step": env_step_snapshot,
            "env_reset": env_reset_snapshot,
            "command_drain": self._worker_runtime["command_drain"].snapshot(),
            "actor_apply": self._worker_runtime["actor_apply"].snapshot(),
            "env_loop_total_seconds": env_step_snapshot["total_seconds"] + env_reset_snapshot["total_seconds"],
            "env_benchmarks": dict(self._latest_env_benchmarks),
        }

    def _queue_profile_snapshot(self) -> dict[str, Any]:
        return {key: value.snapshot() for key, value in self._worker_queue.items()}

    def _initialize_policy_components(self) -> None:
        if self.actor is not None and self._torch is not None:
            return

        import torch

        torch.manual_seed(self.config.train.seed)
        self._torch = torch
        if self.observation_mode == "full":
            from ..models.full_actor_critic import FullObservationActor

            self.actor = FullObservationActor(telemetry_dim=TELEMETRY_DIM).cpu().eval()
        else:
            from ..models.lidar_actor_critic import LidarActor

            self.actor = LidarActor(observation_dim=lidar_feature_dim(self.config.lidar_observation)).cpu().eval()

    def _bootstrap_capture(self, env: TM20AIGymEnv) -> None:
        interface = getattr(env, "interface", None)
        if interface is None or not hasattr(interface, "bootstrap_capture"):
            return
        context = interface.describe_capture_bootstrap()
        self._write_bootstrap_log("pre_capture", context)
        started = interface.bootstrap_capture()
        self._write_bootstrap_log("post_capture", started)

    def run(self, *, max_env_steps: int | None = None) -> None:
        env: TM20AIGymEnv | None = None
        try:
            env = self.env_factory(self.config_path, False)
            self._bootstrap_capture(env)
            self._initialize_policy_components()
            if self._should_shutdown():
                return
            observation, info, aux = self._start_training_episode(env)
            if self._should_shutdown():
                return
            while not self._should_shutdown():
                command_drain_start = time.perf_counter()
                pending_eval = self._drain_commands()
                self._worker_runtime["command_drain"].record(time.perf_counter() - command_drain_start)
                if self._should_shutdown() and pending_eval is None:
                    break
                if pending_eval is not None:
                    self._flush_pending_transitions(force=True)
                    self._apply_latest_desired_actor(mode="eval")
                    if not self._actor_ready_for_control:
                        raise RuntimeError("run_eval received before a control-ready actor was applied.")
                    self._status_mode = "eval"
                    self._eval_actor_version = self._applied_actor_version
                    self._eval_actor_source_learner_step = self._applied_source_learner_step
                    self._write_actor_status()
                    eval_metadata = {
                        "run_name": pending_eval.get("run_name"),
                        "env_step": int(pending_eval.get("env_step", self._env_step)),
                        "learner_step": int(pending_eval.get("learner_step", 0)),
                        "episodes": int(pending_eval.get("episodes", self.config.eval.episodes)),
                        "eval_actor_version": self._eval_actor_version,
                        "eval_actor_source_learner_step": self._eval_actor_source_learner_step,
                    }
                    self._put_message(
                        {
                            "type": "eval_started",
                            **eval_metadata,
                            "timestamp": time.time(),
                        }
                    )
                    self._write_worker_event("eval_begin", eval_metadata)
                    try:
                        result = self._run_eval(env, pending_eval)
                    except Exception:
                        self._write_worker_event(
                            "eval_exception",
                            {
                                **eval_metadata,
                                "error": traceback.format_exc(),
                            },
                        )
                        raise
                    self._write_worker_event(
                        "eval_end",
                        {
                            **eval_metadata,
                            "summary_path": str(result.get("summary_path")),
                        },
                    )
                    eval_result = EvalResult.from_run_result(
                        checkpoint_step=int(pending_eval.get("checkpoint_step", pending_eval.get("env_step", self._env_step))),
                        env_step=int(pending_eval.get("env_step", self._env_step)),
                        learner_step=int(pending_eval.get("learner_step", 0)),
                        result=result,
                    )
                    self._put_eval_result(eval_result)
                    self._write_worker_event(
                        "eval_result_published",
                        {
                            **eval_metadata,
                            "summary_path": eval_result.summary_path,
                        },
                    )
                    if self._should_shutdown():
                        break
                    self._status_mode = "train"
                    self._eval_actor_version = None
                    self._eval_actor_source_learner_step = None
                    self._write_actor_status()
                    observation, info, aux = self._start_training_episode(env)
                    if self._should_shutdown():
                        break
                    continue

                if self._should_shutdown():
                    break
                raw_action = self._select_action(observation, aux)
                action = self._execute_training_action(raw_action)
                step_start = time.perf_counter()
                next_observation, reward, terminated, truncated, next_info = env.step(action)
                self._worker_runtime["env_step"].record(time.perf_counter() - step_start)
                next_aux = self._build_aux_features(next_info, action)

                self._episode_state.step_count += 1
                self._episode_state.episode_reward += float(reward)
                self._env_step += 1
                self._observe_reward_step(reward=reward, info=next_info)
                self._observe_movement(next_info)
                self._record_action_stats(raw_action=raw_action, executed_action=action)

                transition = self._build_transition(
                    observation=observation,
                    aux=aux,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    next_aux=next_aux,
                    terminated=terminated,
                    truncated=truncated,
                    info=next_info,
                )
                self._pending_transitions.append(transition)
                should_flush = (
                    len(self._pending_transitions) >= self.config.train.update_buffer_interval
                    or terminated
                    or truncated
                )
                if should_flush:
                    self._flush_pending_transitions(force=True)

                if monotonic() - self._last_heartbeat >= 1.0:
                    self._last_heartbeat = monotonic()
                    self._update_env_benchmarks(env)
                    self._put_message(
                        {
                            "type": "heartbeat",
                            "env_step": self._env_step,
                            "episode_index": self._episode_state.episode_index,
                            "desired_actor_version": self._desired_actor_version,
                            "desired_actor_ready_for_control": self._desired_actor_ready_for_control,
                            "seen_actor_version": self._seen_actor_version,
                            "ready_for_control_seen": self._ready_for_control_seen,
                            "applied_actor_version": self._applied_actor_version,
                            "actor_ready_for_control": self._actor_ready_for_control,
                            "applied_source_learner_step": self._applied_source_learner_step,
                            "last_actor_apply_env_step": self._last_actor_apply_env_step,
                            "last_actor_apply_episode_index": self._last_actor_apply_episode_index,
                            "latest_action_stats": self._last_action_stats,
                            "control_ready_reason": self._control_ready_reason,
                            "runtime_profile": self._runtime_profile_snapshot(),
                            "queue_profile": self._queue_profile_snapshot(),
                        }
                    )

                if self._should_shutdown():
                    observation, info, aux = next_observation, next_info, next_aux
                    break
                if terminated or truncated:
                    self._put_message(
                        {
                            "type": "episode_summary",
                            "summary": {
                                "episode_id": self._episode_state.episode_id,
                                "episode_index": self._episode_state.episode_index,
                                "episode_reward": self._episode_state.episode_reward,
                                "step_count": self._episode_state.step_count,
                                "final_progress_index": float(next_info.get("progress_index", 0.0) or 0.0),
                                "termination_reason": next_info.get("reward_reason") or next_info.get("terminal_reason"),
                                "map_uid": next_info.get("map_uid"),
                                "run_id": next_info.get("run_id"),
                                "observation_mode": self.observation_mode,
                                **self._reward_episode_summary(next_info),
                            },
                        }
                    )
                    self._finalize_movement_episode(next_info)
                    if self._should_shutdown():
                        observation, info, aux = next_observation, next_info, next_aux
                        break
                    observation, info, aux = self._start_training_episode(env)
                    if self._should_shutdown():
                        break
                else:
                    observation, info, aux = next_observation, next_info, next_aux

                if max_env_steps is not None and self._env_step >= max_env_steps:
                    self._shutdown_requested = True
                    break
        except Exception:  # noqa: BLE001
            self._put_message({"type": "fatal_error", "error": traceback.format_exc()})
            raise
        finally:
            self._shutdown_cleanup(env)

    def run_bootstrap_probe(self) -> None:
        env: TM20AIGymEnv | None = None
        try:
            env = self.env_factory(self.config_path, False)
            self._bootstrap_capture(env)
        except Exception:  # noqa: BLE001
            self._put_message({"type": "fatal_error", "error": traceback.format_exc()})
            raise
        finally:
            self._shutdown_cleanup(env)

    def _start_training_episode(self, env: TM20AIGymEnv) -> tuple[np.ndarray, Mapping[str, Any], np.ndarray | None]:
        self._clear_action_guard(reason="episode_reset")
        seed = self.config.train.seed + self._episode_state.episode_index
        reset_start = time.perf_counter()
        observation, info = env.reset(seed=seed)
        self._worker_runtime["env_reset"].record(time.perf_counter() - reset_start)
        self._episode_state = TrainingEpisodeState(episode_index=self._episode_state.episode_index + 1)
        self._begin_reward_episode()
        self._status_mode = "train"
        self._begin_movement_episode(info)
        aux = self._reset_aux_features(info)
        self._apply_latest_desired_actor(mode="train")
        self._write_actor_status()
        return observation, info, aux

    def _clear_action_guard(self, *, reason: str) -> None:
        if self._action_guard_active:
            payload = self._actor_sync_payload()
            payload["reason"] = reason
            self._write_worker_event("action_guard_cleared", payload)
        self._action_guard_active = False
        self._recent_executed_actions.clear()
        self._recent_raw_actions.clear()

    def _reset_aux_features(self, info: Mapping[str, Any]) -> np.ndarray | None:
        if self.observation_mode == "full":
            assert self._features is not None
            self._features.reset(None if info.get("run_id") is None else str(info["run_id"]))
            return self._features.encode(info)
        return None

    @staticmethod
    def _normalize_position(value: Any) -> tuple[float, float, float] | None:
        if value is None or not isinstance(value, (tuple, list)) or len(value) != 3:
            return None
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _distance(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float | None:
        if a is None or b is None:
            return None
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        dz = float(a[2] - b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _begin_movement_episode(self, info: Mapping[str, Any]) -> None:
        self._movement_state = MovementEpisodeState(
            run_id=None if info.get("run_id") is None else str(info.get("run_id")),
            episode_index=self._episode_state.episode_index,
            start_pos=self._normalize_position(info.get("pos_xyz")),
        )

    def _clear_stall_candidate(self) -> None:
        self._movement_state.candidate_stall_env_step = None
        self._movement_state.candidate_stall_frame_id = None
        self._movement_state.candidate_stall_race_time_ms = None
        self._movement_state.candidate_stall_timestamp_ns = None
        self._movement_state.candidate_stall_pos = None

    def _observe_movement(self, info: Mapping[str, Any]) -> None:
        state = self._movement_state
        run_id = None if info.get("run_id") is None else str(info.get("run_id"))
        if run_id is None:
            return
        if state.run_id is None:
            self._begin_movement_episode(info)
            state = self._movement_state
        elif run_id != state.run_id:
            self._finalize_movement_episode(info={"run_id": state.run_id, "reward_reason": "run_id_changed"})
            self._begin_movement_episode(info)
            state = self._movement_state

        speed_kmh = float(info.get("speed_kmh", 0.0) or 0.0)
        pos_xyz = self._normalize_position(info.get("pos_xyz"))
        distance_from_start = self._distance(state.start_pos, pos_xyz)
        state.max_speed_kmh = max(state.max_speed_kmh, speed_kmh)
        if distance_from_start is not None:
            state.max_distance_from_start_m = max(state.max_distance_from_start_m, distance_from_start)

        frame_id = int(info.get("frame_id", 0) or 0)
        race_time_ms = int(info.get("race_time_ms", 0) or 0)
        timestamp_ns = int(info.get("timestamp_ns", 0) or 0)

        if not state.movement_started:
            started = speed_kmh >= self._movement_start_speed_threshold_kmh
            if distance_from_start is not None and distance_from_start >= self._movement_start_distance_threshold_m:
                started = True
            if started:
                state.movement_started = True
                state.movement_start_env_step = self._env_step
                state.movement_start_frame_id = frame_id
                state.movement_start_race_time_ms = race_time_ms
                self._write_worker_event(
                    "movement_started",
                    {
                        "run_id": state.run_id,
                        "episode_index": state.episode_index,
                        "env_step": self._env_step,
                        "frame_id": frame_id,
                        "race_time_ms": race_time_ms,
                        "speed_kmh": speed_kmh,
                        "distance_from_start_m": distance_from_start,
                    },
                )

        if not state.movement_started or state.first_stall is not None:
            return
        if speed_kmh > self._movement_stall_speed_threshold_kmh:
            self._clear_stall_candidate()
            return
        if state.candidate_stall_frame_id is None:
            state.candidate_stall_env_step = self._env_step
            state.candidate_stall_frame_id = frame_id
            state.candidate_stall_race_time_ms = race_time_ms
            state.candidate_stall_timestamp_ns = timestamp_ns
            state.candidate_stall_pos = pos_xyz
            return

        distance_in_candidate = self._distance(state.candidate_stall_pos, pos_xyz)
        if distance_in_candidate is not None and distance_in_candidate > self._movement_stall_position_epsilon_m:
            state.candidate_stall_env_step = self._env_step
            state.candidate_stall_frame_id = frame_id
            state.candidate_stall_race_time_ms = race_time_ms
            state.candidate_stall_timestamp_ns = timestamp_ns
            state.candidate_stall_pos = pos_xyz
            return

        if state.candidate_stall_timestamp_ns is None:
            return
        stall_duration = max(0.0, (timestamp_ns - state.candidate_stall_timestamp_ns) / 1_000_000_000.0)
        if stall_duration < self._movement_stall_duration_seconds:
            return
        state.stall_count += 1
        state.first_stall = {
            "run_id": state.run_id,
            "episode_index": state.episode_index,
            "env_step": state.candidate_stall_env_step,
            "frame_id": state.candidate_stall_frame_id,
            "race_time_ms": state.candidate_stall_race_time_ms,
            "timestamp_ns": state.candidate_stall_timestamp_ns,
            "speed_kmh": speed_kmh,
            "pos_xyz": state.candidate_stall_pos,
            "stall_duration_seconds": stall_duration,
            "movement_start_env_step": state.movement_start_env_step,
            "movement_start_frame_id": state.movement_start_frame_id,
            "movement_start_race_time_ms": state.movement_start_race_time_ms,
        }
        self._write_worker_event("movement_stall_detected", dict(state.first_stall))

    def _finalize_movement_episode(self, info: Mapping[str, Any] | None = None) -> None:
        state = self._movement_state
        if state.run_id is None:
            return
        payload = {
            "run_id": state.run_id,
            "episode_index": state.episode_index,
            "movement_started": state.movement_started,
            "movement_start_env_step": state.movement_start_env_step,
            "movement_start_frame_id": state.movement_start_frame_id,
            "movement_start_race_time_ms": state.movement_start_race_time_ms,
            "max_speed_kmh": state.max_speed_kmh,
            "max_distance_from_start_m": state.max_distance_from_start_m,
            "stall_count": state.stall_count,
            "first_stall": state.first_stall,
        }
        if info is not None:
            payload["final_frame_id"] = int(info.get("frame_id", 0) or 0)
            payload["final_race_time_ms"] = int(info.get("race_time_ms", 0) or 0)
            payload["termination_reason"] = info.get("reward_reason") or info.get("terminal_reason")
        self._write_worker_event("movement_episode_summary", payload)
        self._put_message({"type": "movement_episode_summary", "summary": payload})
        self._movement_state = MovementEpisodeState()

    def _build_aux_features(self, info: Mapping[str, Any], action: np.ndarray) -> np.ndarray | None:
        if self.observation_mode == "full":
            assert self._features is not None
            self._features.observe_action(action, run_id=None if info.get("run_id") is None else str(info["run_id"]))
            return self._features.encode(info)
        return None

    def _select_action(self, observation: np.ndarray, aux: np.ndarray | None) -> np.ndarray:
        if self.actor is None or self._torch is None:
            raise RuntimeError("Worker policy components are not initialized.")
        if self._env_step < self.config.train.environment_steps_before_training or not self._actor_ready_for_control:
            return np.asarray(
                [
                    self._rng.uniform(-1.0, 1.0),
                    self._rng.uniform(-1.0, 1.0),
                ],
                dtype=np.float32,
            )
        torch = self._torch
        if self.observation_mode == "full":
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0) / 255.0
            assert aux is not None
            aux_tensor = torch.as_tensor(aux, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.actor.act(observation_tensor, aux_tensor, deterministic=False)
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.actor.act(observation_tensor, deterministic=False)
        return clamp_action(action.squeeze(0).cpu().numpy().astype(np.float32))

    def _build_transition(
        self,
        *,
        observation: np.ndarray,
        aux: np.ndarray | None,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        next_aux: np.ndarray | None,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_id": self._episode_state.episode_id,
            "map_uid": None if info.get("map_uid") is None else str(info.get("map_uid")),
            "step_idx": self._episode_state.step_count - 1,
        }
        if self.observation_mode == "full":
            payload.update(
                {
                    "obs_uint8": np.asarray(observation, dtype=np.uint8),
                    "telemetry_float": np.asarray(aux, dtype=np.float32),
                    "next_obs_uint8": np.asarray(next_observation, dtype=np.uint8),
                    "next_telemetry_float": np.asarray(next_aux, dtype=np.float32),
                }
            )
        else:
            payload.update(
                {
                    "obs_float": np.asarray(observation, dtype=np.float32),
                    "next_obs_float": np.asarray(next_observation, dtype=np.float32),
                }
            )
        return payload

    def _drain_commands(self) -> Mapping[str, Any] | None:
        pending_eval: Mapping[str, Any] | None = None
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                break
            command_type = str(command.get("type", ""))
            if command_type == "run_eval":
                self._write_worker_event(
                    "command_received",
                    {
                        "command_type": "run_eval",
                        "run_name": command.get("run_name"),
                        "learner_step": int(command.get("learner_step", 0)),
                        "requested_env_step": int(command.get("env_step", self._env_step)),
                        "episodes": int(command.get("episodes", self.config.eval.episodes)),
                    },
                )
                pending_eval = dict(command)
            elif command_type == "set_actor":
                self._write_worker_event(
                    "command_received",
                    {
                        "command_type": "set_actor",
                        "ignored": True,
                    },
                )
            elif command_type == "shutdown":
                self._write_worker_event(
                    "command_received",
                    {
                        "command_type": "shutdown",
                        "requested_env_step": self._env_step,
                    },
                )
                self._shutdown_requested = True
        if self._shutdown_requested:
            return None
        return pending_eval

    def _apply_latest_desired_actor(self, *, mode: str) -> None:
        if self.desired_actor_path is None:
            return
        if self._torch is None or self.actor is None:
            return
        try:
            desired_actor = json.loads(self.desired_actor_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            self._write_worker_event(
                "actor_sync_error",
                {
                    **self._actor_sync_payload(),
                    "error": traceback.format_exc(),
                    "desired_actor_path": str(self.desired_actor_path),
                },
            )
            self._write_actor_status()
            return
        try:
            desired_version = int(desired_actor["desired_actor_version"])
            actor_state_path = Path(str(desired_actor["actor_state_path"]))
            learner_step = int(desired_actor.get("learner_step", 0))
            requested_env_step = int(desired_actor.get("env_step", self._env_step))
            written_at = desired_actor.get("written_at")
            ready_for_control = bool(desired_actor.get("ready_for_control", False))
            control_ready_reason = str(desired_actor.get("control_ready_reason", "unspecified"))
            self._desired_actor_version = desired_version
            self._desired_actor_ready_for_control = ready_for_control
            self._control_ready_reason = control_ready_reason
            self._status_mode = mode
            if self._seen_actor_version is None or desired_version > self._seen_actor_version:
                self._seen_actor_version = desired_version
                self._ready_for_control_seen = ready_for_control
                seen_payload = self._actor_sync_payload(
                    desired_version=desired_version,
                    seen_version=desired_version,
                    actor_state_path=str(actor_state_path),
                    written_at=written_at,
                    learner_step=learner_step,
                    requested_env_step=requested_env_step,
                )
                self._write_worker_event("actor_sync_desired_seen", seen_payload)
                self._put_message({"type": "actor_sync_desired_seen", **seen_payload})
            self._write_actor_status()
            if self._applied_actor_version is not None and desired_version <= self._applied_actor_version:
                return
            apply_start = time.perf_counter()
            actor_state_dict = self._torch.load(actor_state_path, map_location="cpu")
            self.actor.load_state_dict(actor_state_dict)
            self.actor.eval()
            apply_duration_seconds = time.perf_counter() - apply_start
            self._worker_runtime["actor_apply"].record(apply_duration_seconds)
            self._has_policy_weights = True
            self._applied_actor_version = desired_version
            self._actor_ready_for_control = ready_for_control
            self._applied_source_learner_step = learner_step
            self._last_actor_apply_env_step = self._env_step
            self._last_actor_apply_episode_index = self._episode_state.episode_index
            payload = self._actor_sync_payload(
                desired_version=desired_version,
                seen_version=self._seen_actor_version,
                applied_version=desired_version,
                actor_state_path=str(actor_state_path),
                written_at=written_at,
                learner_step=learner_step,
                requested_env_step=requested_env_step,
            )
            payload["apply_duration_seconds"] = apply_duration_seconds
            self._write_worker_event("actor_sync_applied", payload)
            self._put_message({"type": "actor_sync_applied", **payload})
            self._write_actor_status()
        except Exception:
            self._write_worker_event(
                "actor_sync_error",
                {
                    **self._actor_sync_payload(),
                    "error": traceback.format_exc(),
                    "desired_actor_path": str(self.desired_actor_path),
                },
            )
            self._write_actor_status()

    def _execute_training_action(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = clamp_action(raw_action)
        if self._status_mode != "train":
            return raw_action
        if self._env_step < self.config.train.environment_steps_before_training:
            return raw_action
        self._control_step_count += 1
        if self._actor_ready_for_control:
            self._policy_control_step_count += 1
        executed_action = raw_action.copy()
        self._recent_raw_actions.append(raw_action.copy())
        self._recent_executed_actions.append(executed_action.copy())
        return executed_action

    def _record_action_stats(self, *, raw_action: np.ndarray, executed_action: np.ndarray) -> None:
        del raw_action, executed_action
        if self._status_mode != "train" or self._env_step < self.config.train.environment_steps_before_training:
            return
        if not self._recent_executed_actions:
            return
        if self._env_step % self._action_stats_interval_steps != 0:
            return
        executed_window = np.stack(list(self._recent_executed_actions), axis=0)
        raw_window = np.stack(list(self._recent_raw_actions), axis=0) if self._recent_raw_actions else executed_window
        throttle = executed_window[:, 0]
        steer = np.abs(executed_window[:, 1])
        raw_throttle = raw_window[:, 0]
        steer_raw = executed_window[:, 1]
        gas = np.maximum(throttle, 0.0)
        brake = np.maximum(-throttle, 0.0)
        raw_gas = np.maximum(raw_throttle, 0.0)
        raw_brake = np.maximum(-raw_throttle, 0.0)
        throttle_delta = np.diff(throttle) if executed_window.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
        steer_delta = np.diff(steer_raw) if executed_window.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
        throttle_sign_flip_rate = (
            float(np.mean(np.sign(throttle[:-1]) != np.sign(throttle[1:])))
            if executed_window.shape[0] > 1
            else 0.0
        )
        steer_sign_flip_rate = (
            float(np.mean(np.sign(steer_raw[:-1]) != np.sign(steer_raw[1:])))
            if executed_window.shape[0] > 1
            else 0.0
        )
        stats = {
            "env_step": self._env_step,
            "window_size": int(executed_window.shape[0]),
            "desired_actor_version": self._desired_actor_version,
            "desired_actor_ready_for_control": self._desired_actor_ready_for_control,
            "seen_actor_version": self._seen_actor_version,
            "ready_for_control_seen": self._ready_for_control_seen,
            "applied_actor_version": self._applied_actor_version,
            "actor_ready_for_control": self._actor_ready_for_control,
            "applied_source_learner_step": self._applied_source_learner_step,
            "last_actor_apply_env_step": self._last_actor_apply_env_step,
            "last_actor_apply_episode_index": self._last_actor_apply_episode_index,
            "steps_since_actor_apply": (
                None
                if self._last_actor_apply_env_step is None
                else int(self._env_step - self._last_actor_apply_env_step)
            ),
            "versions_behind": (
                None
                if self._desired_actor_version is None or self._applied_actor_version is None
                else int(max(0, self._desired_actor_version - self._applied_actor_version))
            ),
            "action_guard_active": self._action_guard_active,
            "control_source": "policy" if self._actor_ready_for_control else "exploration",
            "mean_throttle": float(np.mean(throttle)),
            "mean_gas": float(np.mean(gas)),
            "mean_brake": float(np.mean(brake)),
            "mean_abs_steer": float(np.mean(steer)),
            "mean_abs_throttle_delta": float(np.mean(np.abs(throttle_delta))) if throttle_delta.size else 0.0,
            "mean_abs_steer_delta": float(np.mean(np.abs(steer_delta))) if steer_delta.size else 0.0,
            "throttle_sign_flip_rate": throttle_sign_flip_rate,
            "steer_sign_flip_rate": steer_sign_flip_rate,
            "fraction_gas_gt_0_1": float(np.mean(gas > 0.1)),
            "fraction_brake_gt_0_1": float(np.mean(brake > 0.1)),
            "fraction_gas_and_brake_gt_0_1": float(np.mean((gas > 0.1) & (brake > 0.1))),
            "raw_mean_throttle": float(np.mean(raw_throttle)),
            "raw_mean_gas": float(np.mean(raw_gas)),
            "raw_mean_brake": float(np.mean(raw_brake)),
            "raw_fraction_gas_and_brake_gt_0_1": float(np.mean((raw_gas > 0.1) & (raw_brake > 0.1))),
            **self._control_metrics_snapshot(),
            "control_ready_reason": self._control_ready_reason,
        }
        self._last_action_stats = stats
        self._write_worker_event("action_stats", stats)
        self._put_message({"type": "action_stats", **stats})

    def _run_eval(self, env: TM20AIGymEnv, command: Mapping[str, Any]) -> dict[str, Any]:
        if self.actor is None:
            raise RuntimeError("Worker actor is not initialized for evaluation.")
        from .evaluator import ActorPolicyAdapter, run_policy_episodes_on_env

        policy = ActorPolicyAdapter(self.actor, observation_mode=self.observation_mode, name="checkpoint")
        return run_policy_episodes_on_env(
            env=env,
            config_path=self.config_path,
            mode="eval",
            policy=policy,
            episodes=int(command.get("episodes", self.config.eval.episodes)),
            seed_base=int(command.get("seed_base", self.config.eval.seed_base)),
            record_video=bool(command.get("record_video", self.config.eval.record_video)),
            checkpoint_path=command.get("checkpoint_path"),
            run_name=None if command.get("run_name") is None else str(command.get("run_name")),
            summary_extra={
                "learner_step": int(command.get("learner_step", 0)),
                "env_step": int(command.get("env_step", self._env_step)),
                "observation_mode": self.observation_mode,
                "eval_actor_version": self._eval_actor_version,
                "eval_actor_source_learner_step": self._eval_actor_source_learner_step,
            },
            close_env=False,
        )

    def _flush_pending_transitions(self, *, force: bool = False) -> None:
        if not self._pending_transitions:
            return
        if not force and len(self._pending_transitions) < self.config.train.update_buffer_interval:
            return
        self._put_message(
            {
                "type": "transition_batch",
                "env_step": self._env_step,
                "observation_mode": self.observation_mode,
                "transitions": list(self._pending_transitions),
            }
        )
        self._pending_transitions.clear()

    def _put_eval_result(self, result: EvalResult) -> None:
        deadline = monotonic() + 5.0
        wait_start = time.perf_counter()
        full_retries = 0
        while True:
            try:
                self.eval_result_queue.put(result, timeout=0.25)
                self._worker_queue["eval_result_put"].record(
                    time.perf_counter() - wait_start,
                    success=True,
                    full_retries=full_retries,
                )
                return
            except queue.Full:
                full_retries += 1
                if monotonic() >= deadline:
                    self._worker_queue["eval_result_put"].record(
                        time.perf_counter() - wait_start,
                        success=False,
                        full_retries=full_retries,
                        timed_out=True,
                    )
                    raise RuntimeError("Timed out while publishing eval result during worker shutdown.")

    def _put_message(self, payload: Mapping[str, Any]) -> None:
        deadline = monotonic() + 5.0
        wait_start = time.perf_counter()
        full_retries = 0
        while True:
            try:
                self.output_queue.put(dict(payload), timeout=0.25)
                self._worker_queue["output_put"].record(
                    time.perf_counter() - wait_start,
                    success=True,
                    full_retries=full_retries,
                )
                return
            except queue.Full:
                full_retries += 1
                if monotonic() >= deadline:
                    self._worker_queue["output_put"].record(
                        time.perf_counter() - wait_start,
                        success=False,
                        full_retries=full_retries,
                        timed_out=True,
                    )
                    return

    def _send_neutral_action(self, env: TM20AIGymEnv | None) -> None:
        if env is None:
            return
        try:
            interface = getattr(env, "interface", None)
            default_action = getattr(env, "default_action", None)
            if interface is not None and default_action is not None and hasattr(interface, "send_control"):
                interface.send_control(default_action)
        except Exception:
            return

    def _should_shutdown(self) -> bool:
        return self._shutdown_requested or self.shutdown_event.is_set()

    def _shutdown_cleanup(self, env: TM20AIGymEnv | None) -> None:
        try:
            self._send_neutral_action(env)
            self._flush_pending_transitions(force=True)
            self._finalize_movement_episode()
            self._update_env_benchmarks(env)
            self._put_message(
                {
                    "type": "heartbeat",
                    "env_step": self._env_step,
                    "episode_index": self._episode_state.episode_index,
                    "desired_actor_version": self._desired_actor_version,
                    "desired_actor_ready_for_control": self._desired_actor_ready_for_control,
                    "seen_actor_version": self._seen_actor_version,
                    "ready_for_control_seen": self._ready_for_control_seen,
                    "applied_actor_version": self._applied_actor_version,
                    "actor_ready_for_control": self._actor_ready_for_control,
                    "applied_source_learner_step": self._applied_source_learner_step,
                    "last_actor_apply_env_step": self._last_actor_apply_env_step,
                    "last_actor_apply_episode_index": self._last_actor_apply_episode_index,
                    "latest_action_stats": self._last_action_stats,
                    "control_ready_reason": self._control_ready_reason,
                    "runtime_profile": self._runtime_profile_snapshot(),
                    "queue_profile": self._queue_profile_snapshot(),
                }
            )
            self._write_actor_status()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                self.worker_done_event.set()


def worker_entry(
    config_path: str,
    command_queue,
    output_queue,
    eval_result_queue,
    shutdown_event,
    worker_done_event,
    bootstrap_log_path: str | None = None,
    max_env_steps: int | None = None,
) -> None:  # noqa: ANN001
    worker = SACWorker(
        config_path=config_path,
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
        bootstrap_log_path=bootstrap_log_path,
    )
    worker.run(max_env_steps=max_env_steps)


def worker_bootstrap_probe_entry(config_path: str, bootstrap_log_path: str | None = None) -> None:
    worker = SACWorker(
        config_path=config_path,
        command_queue=queue.Queue(),
        output_queue=queue.Queue(),
        eval_result_queue=queue.Queue(),
        shutdown_event=threading.Event(),
        worker_done_event=threading.Event(),
        bootstrap_log_path=bootstrap_log_path,
    )
    worker.run_bootstrap_probe()
