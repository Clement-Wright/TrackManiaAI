from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from ..action_space import neutral_action


def _load_vgamepad_module():
    try:
        import vgamepad as vg
    except ImportError as exc:  # pragma: no cover - exercised on machines without vgamepad installed
        raise RuntimeError(
            "vgamepad is not installed. Install the runtime dependencies before using the env control path."
        ) from exc
    return vg


class XboxPadLike(Protocol):
    def left_trigger(self, value: int) -> None: ...
    def right_trigger(self, value: int) -> None: ...
    def left_joystick(self, x_value: int, y_value: int) -> None: ...
    def update(self) -> None: ...


@dataclass(slots=True, frozen=True)
class AnalogAction:
    throttle: float
    steer: float

    @classmethod
    def from_iterable(cls, action: Iterable[float]) -> "AnalogAction":
        from ..action_space import ThrottleAction

        throttle_action = ThrottleAction.from_iterable(action)
        return cls(throttle=throttle_action.throttle, steer=throttle_action.steer)

    @property
    def gas(self) -> float:
        return max(0.0, self.throttle)

    @property
    def brake(self) -> float:
        return max(0.0, -self.throttle)

    def as_array(self) -> np.ndarray:
        return np.asarray([self.throttle, self.steer], dtype=np.float32)


def _steer_to_thumb_x(steer: float) -> int:
    if steer >= 0.0:
        return int(round(steer * 32767.0))
    return int(round(steer * 32768.0))


class GamepadController:
    """Thin vgamepad shim for the canonical [throttle, steer] action space."""

    def __init__(self, backend: XboxPadLike | None = None):
        if backend is None:
            backend = _load_vgamepad_module().VX360Gamepad()
        self._backend = backend

    @staticmethod
    def neutral_action() -> np.ndarray:
        return neutral_action()

    def apply(self, action: Iterable[float]) -> np.ndarray:
        analog = AnalogAction.from_iterable(action)
        self._backend.left_trigger(int(round(analog.brake * 255.0)))
        self._backend.right_trigger(int(round(analog.gas * 255.0)))
        self._backend.left_joystick(_steer_to_thumb_x(analog.steer), 0)
        self._backend.update()
        return analog.as_array()

    def close(self) -> None:
        try:
            self.apply(self.neutral_action())
        except Exception:  # noqa: BLE001 - shutdown should not mask the main task
            pass
