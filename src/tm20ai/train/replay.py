from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

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

        self._action[index] = np.asarray(transition["action"], dtype=np.float32)
        self._reward[index] = float(transition["reward"])
        self._terminated[index] = bool(transition["terminated"])
        self._truncated[index] = bool(transition["truncated"])
        self._episode_ids[index] = None if transition.get("episode_id") is None else str(transition.get("episode_id"))
        self._map_uids[index] = None if transition.get("map_uid") is None else str(transition.get("map_uid"))
        self._step_idx[index] = int(transition.get("step_idx", 0))

        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int, *, device: torch.device | str) -> ReplaySample:
        if self._size <= 0:
            raise RuntimeError("Replay buffer is empty.")
        indices = self._rng.integers(0, self._size, size=int(batch_size))
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
        )
