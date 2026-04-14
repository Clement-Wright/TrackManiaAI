from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..action_space import ACTION_DIM, clamp_action, neutral_action


ACTION_HISTORY_LENGTH = 2
TELEMETRY_DIM = 2 + 6 + (ACTION_HISTORY_LENGTH * ACTION_DIM)
MAX_SPEED_KMH = 1_000.0
MAX_RPM = 11_000.0


@dataclass(slots=True)
class TelemetryFeatureBuilder:
    action_history_len: int = ACTION_HISTORY_LENGTH
    _run_id: str | None = None
    _history: deque[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.action_history_len)
        self.reset()

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def reset(self, run_id: str | None = None) -> None:
        self._run_id = run_id
        self._history.clear()
        neutral = neutral_action()
        for _ in range(self.action_history_len):
            self._history.append(neutral.copy())

    def observe_action(self, action: Sequence[float] | np.ndarray, *, run_id: str | None = None) -> None:
        if run_id is not None and run_id != self._run_id:
            self.reset(run_id)
        self._history.append(clamp_action(action))

    def encode(self, info: Mapping[str, Any]) -> np.ndarray:
        run_id = info.get("run_id")
        run_id_str = None if run_id is None else str(run_id)
        if run_id_str != self._run_id:
            self.reset(run_id_str)

        speed = float(info.get("speed_kmh", 0.0) or 0.0)
        rpm = float(info.get("rpm", 0.0) or 0.0)
        gear = int(info.get("gear", 0) or 0)
        gear = max(0, min(5, gear))

        gear_one_hot = np.zeros(6, dtype=np.float32)
        gear_one_hot[gear] = 1.0
        history = np.concatenate(list(self._history), dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(
                    [
                        np.clip(speed / MAX_SPEED_KMH, 0.0, 1.0),
                        np.clip(rpm / MAX_RPM, 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
                gear_one_hot,
                history,
            ],
            dtype=np.float32,
        )
