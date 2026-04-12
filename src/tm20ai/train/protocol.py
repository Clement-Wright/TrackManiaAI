from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class EvalResult:
    checkpoint_step: int
    env_step: int
    learner_step: int
    summary_path: str
    summary: dict[str, Any]
    timestamp: float

    @classmethod
    def from_run_result(
        cls,
        *,
        checkpoint_step: int,
        env_step: int,
        learner_step: int,
        result: Mapping[str, Any],
    ) -> "EvalResult":
        return cls(
            checkpoint_step=int(checkpoint_step),
            env_step=int(env_step),
            learner_step=int(learner_step),
            summary_path=str(result["summary_path"]),
            summary=dict(result["summary"]),
            timestamp=time.time(),
        )
