from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tm20ai.action_space import ACTION_DIM
from tm20ai.algos.crossq import CrossQAgent
from tm20ai.algos.droq import DroQSACAgent
from tm20ai.config import ConfigError, CrossQConfig, DroQConfig, SACConfig, TM20AIConfig
from tm20ai.train.diagnostics import build_agent_resource_profile
from tm20ai.train.evaluator import resolve_policy_adapter
from tm20ai.train.features import TELEMETRY_DIM
from tm20ai.train.replay import ReplayBuffer


def _build_full_replay(*, rng_seed: int) -> ReplayBuffer:
    replay = ReplayBuffer(
        mode="full",
        capacity=16,
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
        rng_seed=rng_seed,
    )
    for step in range(8):
        replay.add(
            {
                "obs_uint8": np.full((4, 64, 64), step, dtype=np.uint8),
                "telemetry_float": np.full((TELEMETRY_DIM,), 0.1 * step, dtype=np.float32),
                "action": np.asarray([0.1, -0.1], dtype=np.float32),
                "reward": 1.0,
                "next_obs_uint8": np.full((4, 64, 64), step + 1, dtype=np.uint8),
                "next_telemetry_float": np.full((TELEMETRY_DIM,), 0.1 * (step + 1), dtype=np.float32),
                "terminated": False,
                "truncated": False,
            }
        )
    return replay


def test_tm20ai_config_parses_droq_and_crossq_blocks() -> None:
    config = TM20AIConfig.from_mapping(
        {
            "train": {"algorithm": "droq"},
            "droq": {
                "n_critics": 2,
                "m_subset": 2,
                "q_updates_per_policy_update": 5,
                "share_encoders": True,
                "dropout_probability": 0.02,
            },
            "crossq": {"share_encoders": False},
        }
    )

    assert config.train.algorithm == "droq"
    assert config.droq.dropout_probability == 0.02
    assert config.crossq.share_encoders is False


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"train": {"algorithm": "crosq"}}, "train.algorithm"),
        ({"droq": {"n_critics": 1}}, "droq.n_critics"),
        ({"droq": {"m_subset": 3}}, "droq.m_subset"),
        ({"droq": {"dropout_probability": 1.0}}, "droq.dropout_probability"),
    ],
)
def test_tm20ai_config_rejects_invalid_droq_crossq_settings(payload: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        TM20AIConfig.from_mapping(payload)


def test_droq_agent_smoke_checkpoint_and_worker_compatibility(tmp_path: Path) -> None:
    agent = DroQSACAgent(
        sac_config=SACConfig(),
        droq_config=DroQConfig(),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )
    replay = _build_full_replay(rng_seed=41)
    batch = replay.sample(4, device=torch.device("cpu"))

    critic_update = agent.update_critics(batch)
    actor_update = agent.maybe_update_actor_and_alpha(batch)
    assert np.isfinite(critic_update.critic_loss)
    assert actor_update is None
    for _ in range(4):
        agent.update_critics(batch)
    actor_update = agent.maybe_update_actor_and_alpha(batch)
    assert actor_update is not None
    assert np.isfinite(actor_update.actor_loss)

    checkpoint_path = tmp_path / "droq_checkpoint.pt"
    torch.save(
        {
            **agent.state_dict(),
            "learner_step": 5,
            "actor_step": 1,
            "env_step": 5,
        },
        checkpoint_path,
    )
    checkpoint_policy = resolve_policy_adapter(policy="checkpoint", checkpoint=checkpoint_path)
    action = checkpoint_policy.act(
        np.zeros((4, 64, 64), dtype=np.uint8),
        {"run_id": "droq-run", "speed_kmh": 0.0, "rpm": 0.0, "gear": 0},
    )
    assert action.shape == (ACTION_DIM,)
    resource_profile = build_agent_resource_profile(agent, torch.device("cpu"))
    assert resource_profile["algorithm"] == "droq"


def test_crossq_agent_smoke_checkpoint_and_worker_compatibility(tmp_path: Path) -> None:
    agent = CrossQAgent(
        sac_config=SACConfig(),
        crossq_config=CrossQConfig(),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )
    replay = _build_full_replay(rng_seed=43)
    batch = replay.sample(4, device=torch.device("cpu"))

    update = agent.update(batch)
    assert np.isfinite(update.critic_loss)
    assert np.isfinite(update.actor_loss)

    checkpoint_path = tmp_path / "crossq_checkpoint.pt"
    torch.save(
        {
            **agent.state_dict(),
            "learner_step": 1,
            "env_step": 1,
        },
        checkpoint_path,
    )
    checkpoint_policy = resolve_policy_adapter(policy="checkpoint", checkpoint=checkpoint_path)
    action = checkpoint_policy.act(
        np.zeros((4, 64, 64), dtype=np.uint8),
        {"run_id": "crossq-run", "speed_kmh": 0.0, "rpm": 0.0, "gear": 0},
    )
    assert action.shape == (ACTION_DIM,)
    resource_profile = build_agent_resource_profile(agent, torch.device("cpu"))
    assert resource_profile["algorithm"] == "crossq"
