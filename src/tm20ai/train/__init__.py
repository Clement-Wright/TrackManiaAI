"""Training-adjacent evaluation and metrics helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "FixedActionPolicy",
    "PolicyAdapter",
    "ScriptedPolicyAdapter",
    "TorchCheckpointPolicyAdapter",
    "ZeroPolicy",
    "resolve_policy_adapter",
    "run_policy_episodes",
]

if TYPE_CHECKING:
    from .evaluator import (
        FixedActionPolicy,
        PolicyAdapter,
        ScriptedPolicyAdapter,
        TorchCheckpointPolicyAdapter,
        ZeroPolicy,
        resolve_policy_adapter,
        run_policy_episodes,
    )


def __getattr__(name: str) -> Any:
    if name in set(__all__):
        from .evaluator import (
            FixedActionPolicy,
            PolicyAdapter,
            ScriptedPolicyAdapter,
            TorchCheckpointPolicyAdapter,
            ZeroPolicy,
            resolve_policy_adapter,
            run_policy_episodes,
        )

        return {
            "FixedActionPolicy": FixedActionPolicy,
            "PolicyAdapter": PolicyAdapter,
            "ScriptedPolicyAdapter": ScriptedPolicyAdapter,
            "TorchCheckpointPolicyAdapter": TorchCheckpointPolicyAdapter,
            "ZeroPolicy": ZeroPolicy,
            "resolve_policy_adapter": resolve_policy_adapter,
            "run_policy_episodes": run_policy_episodes,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
