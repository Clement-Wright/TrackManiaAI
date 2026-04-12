"""Training-adjacent evaluation and metrics helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "ActorPolicyAdapter",
    "FullBehaviorCloningDataset",
    "FixedActionPolicy",
    "PolicyAdapter",
    "ReplayBuffer",
    "SACLearner",
    "ScriptedPolicyAdapter",
    "TelemetryFeatureBuilder",
    "TorchCheckpointPolicyAdapter",
    "EvalResult",
    "run_policy_episodes_on_env",
    "seed_replay_from_demo_sidecars",
    "split_demo_dataset",
    "worker_entry",
    "ZeroPolicy",
    "resolve_policy_adapter",
    "run_policy_episodes",
]

if TYPE_CHECKING:
    from .evaluator import (
        ActorPolicyAdapter,
        FixedActionPolicy,
        PolicyAdapter,
        ScriptedPolicyAdapter,
        TorchCheckpointPolicyAdapter,
        ZeroPolicy,
        resolve_policy_adapter,
        run_policy_episodes,
        run_policy_episodes_on_env,
    )
    from ..data.dataset import FullBehaviorCloningDataset, seed_replay_from_demo_sidecars, split_demo_dataset
    from .features import TelemetryFeatureBuilder
    from .learner import SACLearner
    from .protocol import EvalResult
    from .replay import ReplayBuffer
    from .worker import worker_entry


def __getattr__(name: str) -> Any:
    if name in set(__all__):
        from .evaluator import (
            ActorPolicyAdapter,
            FixedActionPolicy,
            PolicyAdapter,
            ScriptedPolicyAdapter,
            TorchCheckpointPolicyAdapter,
            ZeroPolicy,
            resolve_policy_adapter,
            run_policy_episodes,
            run_policy_episodes_on_env,
        )
        from ..data.dataset import FullBehaviorCloningDataset, seed_replay_from_demo_sidecars, split_demo_dataset
        from .features import TelemetryFeatureBuilder
        from .learner import SACLearner
        from .protocol import EvalResult
        from .replay import ReplayBuffer
        from .worker import worker_entry

        return {
            "ActorPolicyAdapter": ActorPolicyAdapter,
            "EvalResult": EvalResult,
            "FullBehaviorCloningDataset": FullBehaviorCloningDataset,
            "FixedActionPolicy": FixedActionPolicy,
            "PolicyAdapter": PolicyAdapter,
            "ReplayBuffer": ReplayBuffer,
            "SACLearner": SACLearner,
            "ScriptedPolicyAdapter": ScriptedPolicyAdapter,
            "seed_replay_from_demo_sidecars": seed_replay_from_demo_sidecars,
            "split_demo_dataset": split_demo_dataset,
            "TelemetryFeatureBuilder": TelemetryFeatureBuilder,
            "TorchCheckpointPolicyAdapter": TorchCheckpointPolicyAdapter,
            "run_policy_episodes_on_env": run_policy_episodes_on_env,
            "worker_entry": worker_entry,
            "ZeroPolicy": ZeroPolicy,
            "resolve_policy_adapter": resolve_policy_adapter,
            "run_policy_episodes": run_policy_episodes,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
