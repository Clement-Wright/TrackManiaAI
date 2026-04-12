from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

import numpy as np

from ..capture import lidar_feature_dim
from ..config import TM20AIConfig, load_tm20ai_config
from ..data.demo_recorder import DemoRecorder
from ..data.parquet_writer import build_run_artifact_paths, sha256_file, timestamp_tag, write_json
from ..env import TM20AIGymEnv, make_env
from ..env.trajectory import load_runtime_trajectory, runtime_trajectory_path_for_map
from .features import TELEMETRY_DIM, TelemetryFeatureBuilder
from .metrics import TensorBoardScalarLogger, aggregate_episode_summaries

if TYPE_CHECKING:
    import torch


class PolicyAdapter(Protocol):
    name: str

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray: ...


@dataclass(slots=True)
class ZeroPolicy:
    name: str = "zero"

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del observation, info
        return np.zeros(3, dtype=np.float32)


@dataclass(slots=True)
class FixedActionPolicy:
    action: np.ndarray
    name: str = "fixed"

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del observation, info
        return np.asarray(self.action, dtype=np.float32)


VK_UP = 0x26
VK_DOWN = 0x28
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44


def _load_win32api():
    try:
        import win32api
    except ImportError as exc:  # pragma: no cover - Windows dependency guard
        raise RuntimeError("pywin32 is required for the human teleop policy.") from exc
    return win32api


@dataclass(slots=True)
class KeyboardTeleopPolicy:
    name: str = "human"
    key_state_reader: Callable[[int], int] | None = None

    def __post_init__(self) -> None:
        if self.key_state_reader is None:
            win32api = _load_win32api()
            self.key_state_reader = win32api.GetAsyncKeyState

    def _pressed(self, virtual_key: int) -> bool:
        assert self.key_state_reader is not None
        return bool(self.key_state_reader(virtual_key) & 0x8000)

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del observation, info
        gas = 1.0 if (self._pressed(VK_UP) or self._pressed(VK_W)) else 0.0
        brake = 1.0 if (self._pressed(VK_DOWN) or self._pressed(VK_S)) else 0.0
        steer_left = self._pressed(VK_LEFT) or self._pressed(VK_A)
        steer_right = self._pressed(VK_RIGHT) or self._pressed(VK_D)
        if steer_left and steer_right:
            steer = 0.0
        elif steer_left:
            steer = -1.0
        elif steer_right:
            steer = 1.0
        else:
            steer = 0.0
        return np.asarray([gas, brake, steer], dtype=np.float32)


@dataclass(slots=True)
class ScriptedPolicyAdapter:
    callback: Callable[[np.ndarray, Mapping[str, Any]], np.ndarray]
    name: str = "scripted"

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        action = self.callback(observation, info)
        return np.asarray(action, dtype=np.float32)


class ActorPolicyAdapter:
    def __init__(self, actor: "torch.nn.Module", *, observation_mode: str, name: str = "checkpoint") -> None:
        self.name = name
        self._actor = actor.eval()
        self._observation_mode = observation_mode
        self._features = TelemetryFeatureBuilder() if self._observation_mode == "full" else None

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        import torch

        if self._observation_mode == "full":
            assert self._features is not None
            telemetry = self._features.encode(info)
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0) / 255.0
            telemetry_tensor = torch.as_tensor(telemetry, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self._actor.act(observation_tensor, telemetry_tensor, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
            self._features.observe_action(action_np, run_id=None if info.get("run_id") is None else str(info.get("run_id")))
            return action_np
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self._actor.act(observation_tensor, deterministic=True)
        return action.squeeze(0).cpu().numpy().astype(np.float32)


class TorchCheckpointPolicyAdapter:
    def __init__(self, checkpoint_path: str | Path):
        import torch

        self.name = "checkpoint"
        self.checkpoint_path = Path(checkpoint_path).resolve()
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        self._policy = self._resolve_policy(payload)

    def _resolve_actor_checkpoint(self, payload: Mapping[str, Any]) -> ActorPolicyAdapter | None:
        actor_state = payload.get("actor_state_dict")
        if actor_state is None:
            return None
        observation_mode = str(payload.get("observation_mode", "full"))
        action_dim = int(payload.get("action_dim", 3))
        if observation_mode == "full":
            from ..models.full_actor_critic import FullObservationActor

            telemetry_dim = int(payload.get("telemetry_dim", TELEMETRY_DIM))
            observation_shape = tuple(payload.get("observation_shape", (4, 64, 64)))
            actor = FullObservationActor(
                observation_shape=tuple(int(value) for value in observation_shape),
                telemetry_dim=telemetry_dim,
                action_dim=action_dim,
            )
        else:
            from ..models.lidar_actor_critic import LidarActor

            observation_shape = tuple(payload.get("observation_shape", (83,)))
            actor = LidarActor(observation_dim=int(observation_shape[0]), action_dim=action_dim)
        actor.load_state_dict(actor_state)
        return ActorPolicyAdapter(actor, observation_mode=observation_mode, name=self.name)

    def _resolve_policy(self, payload: Any) -> PolicyAdapter | Callable[[np.ndarray, Mapping[str, Any]], Any]:
        if isinstance(payload, Mapping):
            actor_policy = self._resolve_actor_checkpoint(payload)
            if actor_policy is not None:
                return actor_policy
        if hasattr(payload, "act") and callable(payload.act):
            return lambda obs, info: payload.act(obs, info)
        if callable(payload):
            return lambda obs, info: payload(obs)
        if isinstance(payload, Mapping):
            for key in ("policy", "model", "policy_module"):
                candidate = payload.get(key)
                if hasattr(candidate, "act") and callable(candidate.act):
                    return lambda obs, info, candidate=candidate: candidate.act(obs, info)
                if callable(candidate):
                    return lambda obs, info, candidate=candidate: candidate(obs)
        raise RuntimeError(
            f"Checkpoint {self.checkpoint_path} does not expose a supported actor checkpoint or callable policy."
        )

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        result = self._policy.act(observation, info) if hasattr(self._policy, "act") else self._policy(observation, info)
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        array = np.asarray(result, dtype=np.float32)
        if array.ndim > 1:
            array = array.reshape(-1)
        return array


def load_scripted_policy(spec: str) -> ScriptedPolicyAdapter:
    module_ref, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("Scripted policy must use 'module:attr' or 'path.py:attr'.")

    module: Any
    module_path = Path(module_ref)
    if module_path.suffix == ".py" or module_path.exists():
        resolved = module_path.resolve()
        module_name = f"tm20ai_scripted_{resolved.stem}"
        spec_obj = importlib.util.spec_from_file_location(module_name, resolved)
        if spec_obj is None or spec_obj.loader is None:
            raise RuntimeError(f"Could not load scripted policy module from {resolved}.")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    callback = getattr(module, attr)
    if not callable(callback):
        raise RuntimeError(f"Scripted policy target {spec!r} is not callable.")
    return ScriptedPolicyAdapter(callback=callback)


def resolve_policy_adapter(
    *,
    policy: str,
    fixed_action: np.ndarray | None = None,
    script: str | None = None,
    checkpoint: str | Path | None = None,
) -> PolicyAdapter:
    if policy == "human":
        return KeyboardTeleopPolicy()
    if policy == "zero":
        return ZeroPolicy()
    if policy == "fixed":
        if fixed_action is None:
            raise ValueError("fixed policy requires a fixed_action.")
        return FixedActionPolicy(action=np.asarray(fixed_action, dtype=np.float32))
    if policy == "scripted":
        if script is None:
            raise ValueError("scripted policy requires --script.")
        return load_scripted_policy(script)
    if policy == "checkpoint":
        if checkpoint is None:
            raise ValueError("checkpoint policy requires --checkpoint.")
        return TorchCheckpointPolicyAdapter(checkpoint)
    raise ValueError(f"Unsupported policy: {policy}")


def run_policy_episodes_on_env(
    *,
    env: TM20AIGymEnv,
    config_path: str | Path,
    mode: str,
    policy: PolicyAdapter,
    episodes: int,
    seed_base: int,
    record_video: bool,
    checkpoint_path: str | Path | None = None,
    run_name: str | None = None,
    summary_extra: Mapping[str, Any] | None = None,
    close_env: bool = False,
) -> dict[str, Any]:
    config_path = str(Path(config_path).resolve())
    config: TM20AIConfig = load_tm20ai_config(config_path)
    seed_schedule = [seed_base + index for index in range(episodes)]
    effective_run_name = run_name or f"{mode}_{policy.name}_{timestamp_tag()}"
    run_paths = build_run_artifact_paths(config, mode=mode, run_name=effective_run_name)
    writer = TensorBoardScalarLogger(run_paths.tensorboard_dir)

    try:
        observation, info = env.reset(seed=seed_schedule[0])
        trajectory_path = runtime_trajectory_path_for_map(str(info["map_uid"]), config.reward.spacing_meters)
        trajectory = load_runtime_trajectory(trajectory_path)
        demo_feature_builder = TelemetryFeatureBuilder() if mode == "demos" and config.observation.mode == "full" else None
        recorder = DemoRecorder(
            run_paths=run_paths,
            trajectory=trajectory,
            sector_count=config.eval.sector_count,
            record_video=record_video,
            observation_mode=config.observation.mode,
            record_observation_sidecar=mode == "demos" and config.observation.mode == "full",
        )

        for episode_index, seed in enumerate(seed_schedule):
            if episode_index > 0:
                observation, info = env.reset(seed=seed)
            if demo_feature_builder is not None:
                demo_feature_builder.reset(None if info.get("run_id") is None else str(info["run_id"]))
            episode_id = f"episode_{episode_index:05d}"
            recorder.start_episode(
                episode_id=episode_id,
                map_uid=str(info["map_uid"]),
                run_id=str(info["run_id"]),
                episode_seed=seed,
            )

            terminated = False
            truncated = False
            while not (terminated or truncated):
                demo_telemetry = demo_feature_builder.encode(info) if demo_feature_builder is not None else None
                action = np.asarray(policy.act(observation, info), dtype=np.float32)
                next_observation, reward, terminated, truncated, next_info = env.step(action)
                recorder.record_step(
                    observation=next_observation,
                    action=action,
                    reward=reward,
                    info=next_info,
                    policy_observation=observation if demo_feature_builder is not None else None,
                    policy_telemetry=demo_telemetry,
                )
                if demo_feature_builder is not None:
                    demo_feature_builder.observe_action(
                        action,
                        run_id=None if next_info.get("run_id") is None else str(next_info.get("run_id")),
                    )
                observation, info = next_observation, next_info
            recorder.finish_episode(terminated=terminated, truncated=truncated, final_info=info)

        recorder.write_episode_index()
        aggregate = aggregate_episode_summaries(recorder.episode_index_rows, sector_count=config.eval.sector_count)
        action_metrics = recorder.run_action_metrics()
        summary = {
            "mode": mode,
            "run_name": effective_run_name,
            "config_path": config_path,
            "config_sha256": sha256_file(Path(config_path)),
            "reward_trajectory_path": str(trajectory_path),
            "policy_descriptor": {
                "name": policy.name,
                "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None,
            },
            "seed_schedule": seed_schedule,
            "map_uid": str(info["map_uid"]),
            "observation_mode": config.observation.mode,
            **aggregate,
            **action_metrics,
        }
        invalid_reason = None
        valid_for_bc = True
        if mode == "demos" and policy.name == "human" and int(action_metrics["total_nonzero_action_steps"]) == 0:
            valid_for_bc = False
            invalid_reason = "all_zero_actions"
        summary["valid_for_bc"] = valid_for_bc
        summary["invalid_reason"] = invalid_reason
        if summary_extra is not None:
            summary.update(dict(summary_extra))
        write_json(run_paths.summary_json, summary)
        writer.add_scalars_from_mapping(mode, aggregate, step=episodes)
        if mode == "demos" and policy.name == "human" and not valid_for_bc:
            raise RuntimeError(
                "Human demo run did not record any non-zero actions. Focus Trackmania and retry with --episodes 1 first."
            )
        return {
            "summary": summary,
            "summary_path": run_paths.summary_json,
            "episode_index_path": run_paths.episode_index_parquet,
            "tensorboard_dir": run_paths.tensorboard_dir,
            "run_dir": run_paths.run_dir,
            "episode_summaries": recorder.episode_index_rows,
        }
    finally:
        writer.close()
        if close_env:
            env.close()


def run_policy_episodes(
    *,
    config_path: str | Path,
    mode: str,
    policy: PolicyAdapter,
    episodes: int,
    seed_base: int,
    record_video: bool,
    env_factory: Callable[[str | Path, bool], TM20AIGymEnv] | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    env_builder = env_factory or (lambda path, benchmark=False: make_env(path, benchmark=benchmark))
    env = env_builder(config_path, False)
    return run_policy_episodes_on_env(
        env=env,
        config_path=config_path,
        mode=mode,
        policy=policy,
        episodes=episodes,
        seed_base=seed_base,
        record_video=record_video,
        checkpoint_path=checkpoint_path,
        close_env=True,
    )
