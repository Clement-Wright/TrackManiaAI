from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ..action_space import clamp_action
from ..capture import lidar_feature_dim
from ..config import TM20AIConfig
from .features import ACTION_DIM, TELEMETRY_DIM


FULL_OBSERVATION_SHAPE = (4, 64, 64)


@dataclass(frozen=True, slots=True)
class ReplaySample:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    indices: np.ndarray
    telemetry: torch.Tensor | None = None
    next_telemetry: torch.Tensor | None = None
    source: torch.Tensor | None = None


class ReplayBuffer:
    def __init__(
        self,
        *,
        mode: str,
        capacity: int,
        observation_shape: tuple[int, ...],
        telemetry_dim: int = 0,
        action_dim: int = ACTION_DIM,
        rng_seed: int = 12345,
    ) -> None:
        self.mode = mode
        self.capacity = int(capacity)
        self.observation_shape = observation_shape
        self.telemetry_dim = int(telemetry_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(rng_seed)

        obs_dtype = np.uint8 if self.mode == "full" else np.float32
        self._obs = np.zeros((self.capacity, *observation_shape), dtype=obs_dtype)
        self._next_obs = np.zeros((self.capacity, *observation_shape), dtype=obs_dtype)
        self._telemetry = (
            np.zeros((self.capacity, telemetry_dim), dtype=np.float32) if telemetry_dim > 0 else None
        )
        self._next_telemetry = (
            np.zeros((self.capacity, telemetry_dim), dtype=np.float32) if telemetry_dim > 0 else None
        )
        self._action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((self.capacity,), dtype=np.float32)
        self._terminated = np.zeros((self.capacity,), dtype=np.bool_)
        self._truncated = np.zeros((self.capacity,), dtype=np.bool_)
        self._episode_ids: list[str | None] = [None for _ in range(self.capacity)]
        self._map_uids: list[str | None] = [None for _ in range(self.capacity)]
        self._step_idx = np.zeros((self.capacity,), dtype=np.int32)

        self._size = 0
        self._cursor = 0

    @classmethod
    def from_config(cls, config: TM20AIConfig) -> "ReplayBuffer":
        if config.observation.mode == "full":
            observation_shape = (
                config.full_observation.frame_stack,
                config.full_observation.output_height,
                config.full_observation.output_width,
            )
            telemetry_dim = TELEMETRY_DIM
        else:
            observation_shape = (lidar_feature_dim(config.lidar_observation),)
            telemetry_dim = 0
        return cls(
            mode=config.observation.mode,
            capacity=config.train.memory_size,
            observation_shape=observation_shape,
            telemetry_dim=telemetry_dim,
            rng_seed=config.train.seed,
        )

    @property
    def size(self) -> int:
        return self._size

    @property
    def empty(self) -> bool:
        return self._size <= 0

    def add(self, transition: Mapping[str, Any]) -> None:
        index = self._cursor
        if self.mode == "full":
            self._obs[index] = np.asarray(transition["obs_uint8"], dtype=np.uint8)
            self._next_obs[index] = np.asarray(transition["next_obs_uint8"], dtype=np.uint8)
            assert self._telemetry is not None
            assert self._next_telemetry is not None
            self._telemetry[index] = np.asarray(transition["telemetry_float"], dtype=np.float32)
            self._next_telemetry[index] = np.asarray(transition["next_telemetry_float"], dtype=np.float32)
        else:
            self._obs[index] = np.asarray(transition["obs_float"], dtype=np.float32)
            self._next_obs[index] = np.asarray(transition["next_obs_float"], dtype=np.float32)

        self._action[index] = clamp_action(transition["action"])
        self._reward[index] = float(transition["reward"])
        self._terminated[index] = bool(transition["terminated"])
        self._truncated[index] = bool(transition["truncated"])
        self._episode_ids[index] = None if transition.get("episode_id") is None else str(transition.get("episode_id"))
        self._map_uids[index] = None if transition.get("map_uid") is None else str(transition.get("map_uid"))
        self._step_idx[index] = int(transition.get("step_idx", 0))

        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if self._size <= 0:
            raise RuntimeError("Replay buffer is empty.")
        return self._rng.integers(0, self._size, size=int(batch_size))

    def _gather(self, indices: np.ndarray, *, device: torch.device | str, source_id: int | None = None) -> ReplaySample:
        if self.mode == "full":
            obs = torch.from_numpy(self._obs[indices]).to(device=device, dtype=torch.float32) / 255.0
            next_obs = torch.from_numpy(self._next_obs[indices]).to(device=device, dtype=torch.float32) / 255.0
            telemetry = torch.from_numpy(self._telemetry[indices]).to(device=device, dtype=torch.float32)
            next_telemetry = torch.from_numpy(self._next_telemetry[indices]).to(device=device, dtype=torch.float32)
        else:
            obs = torch.from_numpy(self._obs[indices]).to(device=device, dtype=torch.float32)
            next_obs = torch.from_numpy(self._next_obs[indices]).to(device=device, dtype=torch.float32)
            telemetry = None
            next_telemetry = None
        action = torch.from_numpy(self._action[indices]).to(device=device, dtype=torch.float32)
        reward = torch.from_numpy(self._reward[indices]).to(device=device, dtype=torch.float32).unsqueeze(-1)
        terminated = torch.from_numpy(self._terminated[indices].astype(np.float32)).to(device=device).unsqueeze(-1)
        truncated = torch.from_numpy(self._truncated[indices].astype(np.float32)).to(device=device).unsqueeze(-1)
        done = torch.clamp(terminated + truncated, min=0.0, max=1.0)
        source = (
            None
            if source_id is None
            else torch.full((len(indices), 1), float(source_id), dtype=torch.float32, device=device)
        )
        return ReplaySample(
            obs=obs,
            telemetry=telemetry,
            action=action,
            reward=reward,
            next_obs=next_obs,
            next_telemetry=next_telemetry,
            done=done,
            terminated=terminated,
            truncated=truncated,
            indices=indices,
            source=source,
        )

    def sample(self, batch_size: int, *, device: torch.device | str) -> ReplaySample:
        return self._gather(self._sample_indices(batch_size), device=device)


def concat_replay_samples(samples: list[ReplaySample]) -> ReplaySample:
    if not samples:
        raise ValueError("concat_replay_samples requires at least one sample.")
    telemetry = None
    next_telemetry = None
    if samples[0].telemetry is not None:
        telemetry = torch.cat([sample.telemetry for sample in samples if sample.telemetry is not None], dim=0)
    if samples[0].next_telemetry is not None:
        next_telemetry = torch.cat(
            [sample.next_telemetry for sample in samples if sample.next_telemetry is not None],
            dim=0,
        )
    source = None
    if samples[0].source is not None:
        source = torch.cat([sample.source for sample in samples if sample.source is not None], dim=0)
    return ReplaySample(
        obs=torch.cat([sample.obs for sample in samples], dim=0),
        telemetry=telemetry,
        action=torch.cat([sample.action for sample in samples], dim=0),
        reward=torch.cat([sample.reward for sample in samples], dim=0),
        next_obs=torch.cat([sample.next_obs for sample in samples], dim=0),
        next_telemetry=next_telemetry,
        done=torch.cat([sample.done for sample in samples], dim=0),
        terminated=torch.cat([sample.terminated for sample in samples], dim=0),
        truncated=torch.cat([sample.truncated for sample in samples], dim=0),
        indices=np.concatenate([sample.indices for sample in samples], axis=0),
        source=source,
    )


class BalancedReplayBuffer:
    """Online replay plus optional offline ghost/demo replay with a decaying sample mix."""

    def __init__(
        self,
        *,
        online: ReplayBuffer,
        offline: ReplayBuffer,
        offline_initial_fraction: float,
        offline_final_fraction: float,
        decay_env_steps: int,
        rng_seed: int = 12345,
    ) -> None:
        if online.mode != offline.mode:
            raise ValueError("BalancedReplayBuffer requires matching online/offline observation modes.")
        self.online = online
        self.offline = offline
        self.mode = online.mode
        self.offline_initial_fraction = float(offline_initial_fraction)
        self.offline_final_fraction = float(offline_final_fraction)
        self.decay_env_steps = max(1, int(decay_env_steps))
        self._env_step = 0
        self._rng = np.random.default_rng(rng_seed)
        self.last_sample_profile: dict[str, Any] = {
            "offline_fraction": self.offline_initial_fraction,
            "offline_batch_size": 0,
            "online_batch_size": 0,
        }

    @classmethod
    def from_config(cls, config: TM20AIConfig) -> "BalancedReplayBuffer":
        online = ReplayBuffer.from_config(config)
        offline = ReplayBuffer.from_config(config)
        return cls(
            online=online,
            offline=offline,
            offline_initial_fraction=config.balanced_replay.offline_initial_fraction,
            offline_final_fraction=config.balanced_replay.offline_final_fraction,
            decay_env_steps=config.balanced_replay.decay_env_steps,
            rng_seed=config.train.seed,
        )

    @property
    def size(self) -> int:
        return self.online.size + self.offline.size

    @property
    def online_size(self) -> int:
        return self.online.size

    @property
    def offline_size(self) -> int:
        return self.offline.size

    def set_progress(self, *, env_step: int) -> None:
        self._env_step = max(0, int(env_step))

    def offline_fraction(self) -> float:
        progress = min(1.0, float(self._env_step) / float(self.decay_env_steps))
        return self.offline_initial_fraction + progress * (self.offline_final_fraction - self.offline_initial_fraction)

    def add(self, transition: Mapping[str, Any]) -> None:
        self.online.add(transition)

    def add_offline(self, transition: Mapping[str, Any]) -> None:
        self.offline.add(transition)

    def sample(self, batch_size: int, *, device: torch.device | str) -> ReplaySample:
        if self.size <= 0:
            raise RuntimeError("Replay buffer is empty.")
        batch_size = int(batch_size)
        if self.offline.empty:
            offline_count = 0
        elif self.online.empty:
            offline_count = batch_size
        else:
            offline_count = int(round(batch_size * self.offline_fraction()))
            offline_count = max(0, min(batch_size, offline_count))
        online_count = batch_size - offline_count
        samples: list[ReplaySample] = []
        if offline_count > 0:
            samples.append(
                self.offline._gather(
                    self.offline._sample_indices(offline_count),
                    device=device,
                    source_id=1,
                )
            )
        if online_count > 0:
            samples.append(
                self.online._gather(
                    self.online._sample_indices(online_count),
                    device=device,
                    source_id=0,
                )
            )
        if len(samples) == 1:
            sample = samples[0]
        else:
            order = self._rng.permutation(batch_size)
            merged = concat_replay_samples(samples)
            sample = ReplaySample(
                obs=merged.obs[order],
                telemetry=None if merged.telemetry is None else merged.telemetry[order],
                action=merged.action[order],
                reward=merged.reward[order],
                next_obs=merged.next_obs[order],
                next_telemetry=None if merged.next_telemetry is None else merged.next_telemetry[order],
                done=merged.done[order],
                terminated=merged.terminated[order],
                truncated=merged.truncated[order],
                indices=merged.indices[order],
                source=None if merged.source is None else merged.source[order],
            )
        self.last_sample_profile = {
            "offline_fraction": self.offline_fraction(),
            "offline_batch_size": offline_count,
            "online_batch_size": online_count,
            "offline_size": self.offline.size,
            "online_size": self.online.size,
        }
        return sample
