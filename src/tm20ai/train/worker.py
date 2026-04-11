from __future__ import annotations

import queue
import traceback
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ..capture import lidar_feature_dim
from ..config import TM20AIConfig, load_tm20ai_config
from ..env import TM20AIGymEnv, make_env
from ..models.full_actor_critic import FullObservationActor
from ..models.lidar_actor_critic import LidarActor
from .evaluator import ActorPolicyAdapter, run_policy_episodes_on_env
from .features import TELEMETRY_DIM, TelemetryFeatureBuilder


@dataclass(slots=True)
class TrainingEpisodeState:
    episode_index: int = 0
    step_count: int = 0
    episode_reward: float = 0.0

    @property
    def episode_id(self) -> str:
        return f"train_episode_{self.episode_index:06d}"


class SACWorker:
    def __init__(
        self,
        *,
        config_path: str,
        command_queue,
        output_queue,
        shutdown_event,
        env_factory: Callable[[str, bool], TM20AIGymEnv] | None = None,
    ) -> None:
        self.config_path = config_path
        self.config: TM20AIConfig = load_tm20ai_config(config_path)
        self.command_queue = command_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event
        self.env_factory = env_factory or (lambda path, benchmark=False: make_env(path, benchmark=benchmark))
        self._rng = np.random.default_rng(self.config.train.seed)
        torch.manual_seed(self.config.train.seed)

        self.observation_mode = self.config.observation.mode
        if self.observation_mode == "full":
            self.actor = FullObservationActor(telemetry_dim=TELEMETRY_DIM).cpu().eval()
            self._features: TelemetryFeatureBuilder | None = TelemetryFeatureBuilder()
        elif self.observation_mode == "lidar":
            self.actor = LidarActor(observation_dim=lidar_feature_dim(self.config.lidar_observation)).cpu().eval()
            self._features = None
        else:
            raise ValueError(f"Unsupported observation mode: {self.observation_mode!r}")

        self._has_policy_weights = False
        self._env_step = 0
        self._episode_state = TrainingEpisodeState()
        self._last_heartbeat = monotonic()
        self._pending_transitions: list[dict[str, Any]] = []

    def run(self, *, max_env_steps: int | None = None) -> None:
        env: TM20AIGymEnv | None = None
        try:
            env = self.env_factory(self.config_path, False)
            observation, info, aux = self._reset_training_env(env)
            while not self.shutdown_event.is_set():
                pending_eval = self._drain_commands()
                if pending_eval is not None:
                    self._flush_pending_transitions(force=True)
                    result = self._run_eval(env, pending_eval)
                    self._put_message({"type": "eval_result", "result": result})
                    observation, info, aux = self._reset_training_env(env)
                    continue

                action = self._select_action(observation, aux)
                next_observation, reward, terminated, truncated, next_info = env.step(action)
                next_aux = self._build_aux_features(next_info, action)

                self._episode_state.step_count += 1
                self._episode_state.episode_reward += float(reward)
                self._env_step += 1

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
                    self._put_message(
                        {
                            "type": "heartbeat",
                            "env_step": self._env_step,
                            "episode_index": self._episode_state.episode_index,
                        }
                    )

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
                            },
                        }
                    )
                    observation, info, aux = self._reset_training_env(env)
                else:
                    observation, info, aux = next_observation, next_info, next_aux

                if max_env_steps is not None and self._env_step >= max_env_steps:
                    self.shutdown_event.set()
                    break
        except Exception:  # noqa: BLE001
            self._put_message({"type": "fatal_error", "error": traceback.format_exc()})
            raise
        finally:
            self._flush_pending_transitions(force=True)
            if env is not None:
                env.close()

    def _reset_training_env(self, env: TM20AIGymEnv) -> tuple[np.ndarray, Mapping[str, Any], np.ndarray | None]:
        seed = self.config.train.seed + self._episode_state.episode_index
        observation, info = env.reset(seed=seed)
        aux = self._reset_aux_features(info)
        self._episode_state = TrainingEpisodeState(episode_index=self._episode_state.episode_index + 1)
        return observation, info, aux

    def _reset_aux_features(self, info: Mapping[str, Any]) -> np.ndarray | None:
        if self.observation_mode == "full":
            assert self._features is not None
            self._features.reset(None if info.get("run_id") is None else str(info["run_id"]))
            return self._features.encode(info)
        return None

    def _build_aux_features(self, info: Mapping[str, Any], action: np.ndarray) -> np.ndarray | None:
        if self.observation_mode == "full":
            assert self._features is not None
            self._features.observe_action(action, run_id=None if info.get("run_id") is None else str(info["run_id"]))
            return self._features.encode(info)
        return None

    def _select_action(self, observation: np.ndarray, aux: np.ndarray | None) -> np.ndarray:
        if self._env_step < self.config.train.environment_steps_before_training or not self._has_policy_weights:
            return np.asarray(
                [
                    self._rng.random(),
                    self._rng.random(),
                    self._rng.uniform(-1.0, 1.0),
                ],
                dtype=np.float32,
            )
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
        return action.squeeze(0).cpu().numpy().astype(np.float32)

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
            if command_type == "set_actor":
                self.actor.load_state_dict(command["actor_state_dict"])
                self.actor.eval()
                self._has_policy_weights = True
            elif command_type == "run_eval":
                pending_eval = dict(command)
            elif command_type == "shutdown":
                self.shutdown_event.set()
        return pending_eval

    def _run_eval(self, env: TM20AIGymEnv, command: Mapping[str, Any]) -> dict[str, Any]:
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

    def _put_message(self, payload: Mapping[str, Any]) -> None:
        while True:
            try:
                self.output_queue.put(dict(payload), timeout=0.25)
                return
            except queue.Full:
                if self.shutdown_event.is_set():
                    return


def worker_entry(config_path: str, command_queue, output_queue, shutdown_event, max_env_steps: int | None = None) -> None:  # noqa: ANN001
    worker = SACWorker(
        config_path=config_path,
        command_queue=command_queue,
        output_queue=output_queue,
        shutdown_event=shutdown_event,
    )
    worker.run(max_env_steps=max_env_steps)
