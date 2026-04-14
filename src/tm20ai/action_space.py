from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


ACTION_DIM = 2
LEGACY_ACTION_DIM = 3
ACTION_LOW = np.asarray([-1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.asarray([1.0, 1.0], dtype=np.float32)


def _as_array(action: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(action, dtype=np.float32).reshape(-1)


def clamp_action(action: Sequence[float] | np.ndarray) -> np.ndarray:
    array = _as_array(action)
    if array.shape == (ACTION_DIM,):
        return np.asarray(
            [
                np.clip(array[0], -1.0, 1.0),
                np.clip(array[1], -1.0, 1.0),
            ],
            dtype=np.float32,
        )
    if array.shape == (LEGACY_ACTION_DIM,):
        gas = float(np.clip(array[0], 0.0, 1.0))
        brake = float(np.clip(array[1], 0.0, 1.0))
        steer = float(np.clip(array[2], -1.0, 1.0))
        throttle = float(np.clip(gas - brake, -1.0, 1.0))
        return np.asarray([throttle, steer], dtype=np.float32)
    raise ValueError(
        f"Expected an action with shape ({ACTION_DIM},) or legacy shape ({LEGACY_ACTION_DIM},), got {array.shape}."
    )


def neutral_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


@dataclass(slots=True, frozen=True)
class ThrottleAction:
    throttle: float
    steer: float

    @classmethod
    def from_iterable(cls, action: Iterable[float]) -> "ThrottleAction":
        throttle, steer = clamp_action(list(action))
        return cls(throttle=float(throttle), steer=float(steer))

    @property
    def gas(self) -> float:
        return max(0.0, self.throttle)

    @property
    def brake(self) -> float:
        return max(0.0, -self.throttle)

    def as_array(self) -> np.ndarray:
        return np.asarray([self.throttle, self.steer], dtype=np.float32)

    def as_legacy_array(self) -> np.ndarray:
        return np.asarray([self.gas, self.brake, self.steer], dtype=np.float32)
