from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from ..config import TM20AIConfig, load_tm20ai_config
from ..data.demo_recorder import DemoRecorder
from ..data.parquet_writer import build_run_artifact_paths, sha256_file, timestamp_tag, write_json
from ..env import TM20AIGymEnv, make_env
from ..env.trajectory import load_runtime_trajectory, runtime_trajectory_path_for_map
from .metrics import TensorBoardScalarLogger, aggregate_episode_summaries


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


@dataclass(slots=True)
class ScriptedPolicyAdapter:
    callback: Callable[[np.ndarray, Mapping[str, Any]], np.ndarray]
    name: str = "scripted"

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        action = self.callback(observation, info)
        return np.asarray(action, dtype=np.float32)


class TorchCheckpointPolicyAdapter:
    def __init__(self, checkpoint_path: str | Path):
        import torch

        self.name = "checkpoint"
        self.checkpoint_path = Path(checkpoint_path).resolve()
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        self._policy = self._resolve_policy(payload)

    def _resolve_policy(self, payload: Any) -> Callable[[np.ndarray, Mapping[str, Any]], Any]:
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
            f"Checkpoint {self.checkpoint_path} does not expose a callable policy or act(obs, info) adapter."
        )

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        result = self._policy(observation, info)
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
    config_path = str(Path(config_path).resolve())
    config: TM20AIConfig = load_tm20ai_config(config_path)
    env_builder = env_factory or (lambda path, benchmark=False: make_env(path, benchmark=benchmark))
    env = env_builder(config_path, False)

    seed_schedule = [seed_base + index for index in range(episodes)]
    run_name = f"{mode}_{policy.name}_{timestamp_tag()}"
    run_paths = build_run_artifact_paths(config, mode=mode, run_name=run_name)
    writer = TensorBoardScalarLogger(run_paths.tensorboard_dir)

    try:
        observation, info = env.reset(seed=seed_schedule[0])
        trajectory_path = runtime_trajectory_path_for_map(str(info["map_uid"]), config.reward.spacing_meters)
        trajectory = load_runtime_trajectory(trajectory_path)
        recorder = DemoRecorder(
            run_paths=run_paths,
            trajectory=trajectory,
            sector_count=config.eval.sector_count,
            record_video=record_video,
        )

        for episode_index, seed in enumerate(seed_schedule):
            if episode_index > 0:
                observation, info = env.reset(seed=seed)
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
                action = np.asarray(policy.act(observation, info), dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                recorder.record_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    info=info,
                )
            recorder.finish_episode(terminated=terminated, truncated=truncated, final_info=info)

        recorder.write_episode_index()
        aggregate = aggregate_episode_summaries(recorder.episode_index_rows, sector_count=config.eval.sector_count)
        summary = {
            "mode": mode,
            "run_name": run_name,
            "config_path": config_path,
            "config_sha256": sha256_file(Path(config_path)),
            "reward_trajectory_path": str(trajectory_path),
            "policy_descriptor": {
                "name": policy.name,
                "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None,
            },
            "seed_schedule": seed_schedule,
            **aggregate,
        }
        write_json(run_paths.summary_json, summary)
        writer.add_scalars_from_mapping(mode, aggregate, step=episodes)
        return {
            "summary": summary,
            "summary_path": run_paths.summary_json,
            "episode_index_path": run_paths.episode_index_parquet,
            "tensorboard_dir": run_paths.tensorboard_dir,
            "run_dir": run_paths.run_dir,
        }
    finally:
        writer.close()
        env.close()
