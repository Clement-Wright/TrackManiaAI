from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tm20ai.action_space import ACTION_DIM
from tm20ai.algos.redq import REDQSACAgent
from tm20ai.capture import lidar_feature_dim
from tm20ai.config import ConfigError, LidarObservationConfig, REDQConfig, SACConfig, TM20AIConfig
from tm20ai.models.full_actor_critic import FullObservationActor
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


def _build_lidar_replay(*, rng_seed: int) -> tuple[ReplayBuffer, int]:
    lidar_dim = lidar_feature_dim(LidarObservationConfig())
    replay = ReplayBuffer(mode="lidar", capacity=16, observation_shape=(lidar_dim,), telemetry_dim=0, rng_seed=rng_seed)
    for step in range(8):
        replay.add(
            {
                "obs_float": np.full((lidar_dim,), 0.01 * step, dtype=np.float32),
                "action": np.asarray([0.1, -0.1], dtype=np.float32),
                "reward": 1.0,
                "next_obs_float": np.full((lidar_dim,), 0.01 * (step + 1), dtype=np.float32),
                "terminated": False,
                "truncated": False,
            }
        )
    return replay, lidar_dim


def _write_bc_actor_checkpoint(
    path: Path,
    *,
    observation_shape: tuple[int, int, int] = (4, 64, 64),
    telemetry_dim: int = TELEMETRY_DIM,
    action_dim: int = ACTION_DIM,
) -> dict:
    actor = FullObservationActor(
        observation_shape=observation_shape,
        telemetry_dim=telemetry_dim,
        action_dim=action_dim,
    )
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.fill_(0.125)
    payload = {
        "checkpoint_kind": "bc_actor",
        "observation_mode": "full",
        "actor_state_dict": actor.state_dict(),
        "observation_shape": observation_shape,
        "telemetry_dim": telemetry_dim,
        "action_dim": action_dim,
        "map_uid": "test-map",
        "demo_root": "C:/demo-root",
        "epoch": 3,
        "train_loss": 0.1,
        "validation_loss": 0.2,
    }
    torch.save(payload, path)
    return payload


def test_tm20ai_config_parses_redq_block_and_normalizes_algorithm() -> None:
    config = TM20AIConfig.from_mapping(
        {
            "train": {"algorithm": "REDQ"},
            "redq": {
                "n_critics": 12,
                "m_subset": 3,
                "q_updates_per_policy_update": 5,
                "share_encoders": True,
            },
        }
    )

    assert config.train.algorithm == "redq"
    assert config.redq.n_critics == 12
    assert config.redq.m_subset == 3
    assert config.redq.q_updates_per_policy_update == 5
    assert config.redq.share_encoders is True


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"train": {"algorithm": "redqsac"}}, "train.algorithm"),
        ({"redq": {"n_critics": 1}}, "redq.n_critics"),
        ({"redq": {"n_critics": 4, "m_subset": 0}}, "redq.m_subset"),
        ({"redq": {"n_critics": 4, "m_subset": 5}}, "redq.m_subset"),
        ({"redq": {"q_updates_per_policy_update": 0}}, "redq.q_updates_per_policy_update"),
    ],
)
def test_tm20ai_config_rejects_invalid_redq_settings(payload, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        TM20AIConfig.from_mapping(payload)


def test_redq_agent_update_smoke_for_full() -> None:
    agent = REDQSACAgent(
        sac_config=SACConfig(),
        redq_config=REDQConfig(n_critics=4, m_subset=2, q_updates_per_policy_update=2),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )
    replay = _build_full_replay(rng_seed=17)
    batch = replay.sample(4, device=torch.device("cpu"))

    critic_update_one = agent.update_critics(batch)
    actor_update_one = agent.maybe_update_actor_and_alpha(batch)
    critic_update_two = agent.update_critics(batch)
    actor_update_two = agent.maybe_update_actor_and_alpha(batch)

    assert np.isfinite(critic_update_one.critic_loss)
    assert np.isfinite(critic_update_two.critic_loss)
    assert actor_update_one is None
    assert actor_update_two is not None
    assert np.isfinite(actor_update_two.actor_loss)
    assert np.isfinite(actor_update_two.alpha)
    assert agent.critic_updates_since_actor == 0


def test_redq_agent_update_smoke_for_lidar() -> None:
    replay, lidar_dim = _build_lidar_replay(rng_seed=19)
    agent = REDQSACAgent(
        sac_config=SACConfig(),
        redq_config=REDQConfig(n_critics=4, m_subset=2, q_updates_per_policy_update=2, share_encoders=True),
        observation_mode="lidar",
        device=torch.device("cpu"),
        observation_shape=(lidar_dim,),
        telemetry_dim=0,
    )
    batch = replay.sample(4, device=torch.device("cpu"))

    critic_update_one = agent.update_critics(batch)
    assert agent.maybe_update_actor_and_alpha(batch) is None
    critic_update_two = agent.update_critics(batch)
    actor_update = agent.maybe_update_actor_and_alpha(batch)

    assert np.isfinite(critic_update_one.critic_loss)
    assert np.isfinite(critic_update_two.critic_loss)
    assert actor_update is not None
    assert np.isfinite(actor_update.actor_loss)
    assert np.isfinite(actor_update.alpha)


def test_redq_checkpoint_roundtrip_and_evaluator_compatibility(tmp_path) -> None:
    agent = REDQSACAgent(
        sac_config=SACConfig(),
        redq_config=REDQConfig(n_critics=4, m_subset=2, q_updates_per_policy_update=2),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )
    replay = _build_full_replay(rng_seed=23)
    batch = replay.sample(4, device=torch.device("cpu"))
    agent.update_critics(batch)
    agent.update_critics(batch)
    assert agent.maybe_update_actor_and_alpha(batch) is not None

    checkpoint_path = tmp_path / "redq_checkpoint.pt"
    payload = {
        **agent.state_dict(),
        "learner_step": 2,
        "actor_step": 1,
        "env_step": 2,
    }
    torch.save(payload, checkpoint_path)

    restored = REDQSACAgent(
        sac_config=SACConfig(),
        redq_config=REDQConfig(n_critics=4, m_subset=2, q_updates_per_policy_update=2),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    observation = np.zeros((4, 64, 64), dtype=np.uint8)
    telemetry = np.zeros((TELEMETRY_DIM,), dtype=np.float32)
    original_action = agent.select_action(observation, telemetry, deterministic=True, device=torch.device("cpu"))
    restored_action = restored.select_action(observation, telemetry, deterministic=True, device=torch.device("cpu"))
    np.testing.assert_allclose(restored_action, original_action, rtol=1e-6, atol=1e-6)

    checkpoint_policy = resolve_policy_adapter(policy="checkpoint", checkpoint=checkpoint_path)
    action = checkpoint_policy.act(
        observation,
        {
            "run_id": "redq-checkpoint-run",
            "speed_kmh": 0.0,
            "rpm": 0.0,
            "gear": 0,
        },
    )
    assert action.shape == (ACTION_DIM,)


def test_redq_agent_load_bc_warm_start_actor_only(tmp_path) -> None:
    checkpoint_path = tmp_path / "bc_actor.pt"
    payload = _write_bc_actor_checkpoint(checkpoint_path)
    agent = REDQSACAgent(
        sac_config=SACConfig(),
        redq_config=REDQConfig(n_critics=4, m_subset=2, q_updates_per_policy_update=2),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )
    original_critic_state = {
        key: value.clone()
        for key, value in agent.critics[0].vision_encoder.state_dict().items()
    }

    metadata = agent.load_bc_warm_start(checkpoint_path, init_mode="actor_only")

    for key, value in payload["actor_state_dict"].items():
        assert torch.equal(agent.actor.state_dict()[key], value)
    for key, value in original_critic_state.items():
        assert torch.equal(agent.critics[0].vision_encoder.state_dict()[key], value)
    assert metadata["checkpoint_kind"] == "bc_actor"
    assert metadata["map_uid"] == "test-map"


def test_redq_agent_load_bc_warm_start_actor_plus_critic_encoders(tmp_path) -> None:
    checkpoint_path = tmp_path / "bc_actor.pt"
    payload = _write_bc_actor_checkpoint(checkpoint_path)
    agent = REDQSACAgent(
        sac_config=SACConfig(),
        redq_config=REDQConfig(
            n_critics=4,
            m_subset=2,
            q_updates_per_policy_update=2,
            share_encoders=True,
        ),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )

    agent.load_bc_warm_start(checkpoint_path, init_mode="actor_plus_critic_encoders")

    actor_state = payload["actor_state_dict"]
    expected_vision = {
        key.removeprefix("vision_encoder."): value
        for key, value in actor_state.items()
        if key.startswith("vision_encoder.")
    }
    expected_telemetry = {
        key.removeprefix("telemetry_encoder."): value
        for key, value in actor_state.items()
        if key.startswith("telemetry_encoder.")
    }
    for critic in (*agent.critics, *agent.target_critics):
        for key, value in expected_vision.items():
            assert torch.equal(critic.vision_encoder.state_dict()[key], value)
        for key, value in expected_telemetry.items():
            assert torch.equal(critic.telemetry_encoder.state_dict()[key], value)
