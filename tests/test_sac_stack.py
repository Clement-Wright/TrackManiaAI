from __future__ import annotations

import numpy as np
import torch
import pytest

from tm20ai.action_space import ACTION_DIM
from tm20ai.capture import lidar_feature_dim
from tm20ai.algos.sac import SACAgent
from tm20ai.config import LidarObservationConfig, SACConfig
from tm20ai.models.full_actor_critic import FullObservationActor, FullObservationCritic, random_shift_augmentation
from tm20ai.models.lidar_actor_critic import LidarActor, LidarCritic
from tm20ai.train.features import TELEMETRY_DIM
from tm20ai.train.replay import ReplayBuffer


def test_full_actor_outputs_valid_action_ranges_and_log_probs() -> None:
    actor = FullObservationActor()
    observation = torch.rand(2, 4, 64, 64)
    telemetry = torch.rand(2, TELEMETRY_DIM)

    action, log_prob = actor.sample(observation, telemetry, deterministic=False)

    assert action.shape == (2, ACTION_DIM)
    assert log_prob is not None
    assert log_prob.shape == (2, 1)
    assert torch.isfinite(log_prob).all()
    assert torch.all(action[:, 0] >= -1.0)
    assert torch.all(action[:, 0] <= 1.0)
    assert torch.all(action[:, 1] >= -1.0)
    assert torch.all(action[:, 1] <= 1.0)


def test_lidar_actor_outputs_valid_action_ranges_and_log_probs() -> None:
    actor = LidarActor()
    observation = torch.rand(2, lidar_feature_dim(LidarObservationConfig()))

    action, log_prob = actor.sample(observation, deterministic=False)

    assert action.shape == (2, ACTION_DIM)
    assert log_prob is not None
    assert log_prob.shape == (2, 1)
    assert torch.isfinite(log_prob).all()
    assert torch.all(action[:, 0] >= -1.0)
    assert torch.all(action[:, 0] <= 1.0)
    assert torch.all(action[:, 1] >= -1.0)
    assert torch.all(action[:, 1] <= 1.0)


def test_full_critic_outputs_scalar_q_values() -> None:
    critic = FullObservationCritic()
    q_value = critic(torch.rand(4, 4, 64, 64), torch.rand(4, TELEMETRY_DIM), torch.rand(4, ACTION_DIM))
    assert q_value.shape == (4, 1)


def test_lidar_critic_outputs_scalar_q_values() -> None:
    critic = LidarCritic()
    q_value = critic(torch.rand(4, lidar_feature_dim(LidarObservationConfig())), torch.rand(4, ACTION_DIM))
    assert q_value.shape == (4, 1)


def test_random_shift_augmentation_preserves_shape() -> None:
    observation = torch.randint(0, 256, (3, 4, 64, 64), dtype=torch.uint8)
    augmented = random_shift_augmentation(observation, padding=4)
    assert augmented.shape == (3, 4, 64, 64)
    assert augmented.dtype == torch.float32


def test_sac_agent_update_smoke_for_full() -> None:
    agent = SACAgent(
        config=SACConfig(),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )
    replay = ReplayBuffer(
        mode="full",
        capacity=16,
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
        rng_seed=11,
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

    update = agent.update(replay.sample(4, device=torch.device("cpu")))
    assert np.isfinite(update.actor_loss)
    assert np.isfinite(update.critic_loss)
    assert np.isfinite(update.alpha)


def test_sac_agent_update_smoke_for_lidar() -> None:
    lidar_dim = lidar_feature_dim(LidarObservationConfig())
    agent = SACAgent(
        config=SACConfig(),
        observation_mode="lidar",
        device=torch.device("cpu"),
        observation_shape=(lidar_dim,),
        telemetry_dim=0,
    )
    replay = ReplayBuffer(mode="lidar", capacity=16, observation_shape=(lidar_dim,), telemetry_dim=0, rng_seed=13)
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

    update = agent.update(replay.sample(4, device=torch.device("cpu")))
    assert np.isfinite(update.actor_loss)
    assert np.isfinite(update.critic_loss)
    assert np.isfinite(update.alpha)


def _write_bc_actor_checkpoint(path, *, observation_shape=(4, 64, 64), telemetry_dim=TELEMETRY_DIM, action_dim=ACTION_DIM) -> dict:
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


def test_sac_agent_load_bc_warm_start_actor_only(tmp_path) -> None:
    checkpoint_path = tmp_path / "bc_actor.pt"
    payload = _write_bc_actor_checkpoint(checkpoint_path)
    agent = SACAgent(
        config=SACConfig(),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )
    original_critic_state = {
        key: value.clone()
        for key, value in agent.critic1.vision_encoder.state_dict().items()
    }

    metadata = agent.load_bc_warm_start(checkpoint_path, init_mode="actor_only")

    for key, value in payload["actor_state_dict"].items():
        assert torch.equal(agent.actor.state_dict()[key], value)
    for key, value in original_critic_state.items():
        assert torch.equal(agent.critic1.vision_encoder.state_dict()[key], value)
    assert metadata["checkpoint_kind"] == "bc_actor"
    assert metadata["map_uid"] == "test-map"


def test_sac_agent_load_bc_warm_start_actor_plus_critic_encoders(tmp_path) -> None:
    checkpoint_path = tmp_path / "bc_actor.pt"
    payload = _write_bc_actor_checkpoint(checkpoint_path)
    agent = SACAgent(
        config=SACConfig(),
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
    for critic in (agent.critic1, agent.critic2, agent.target_critic1, agent.target_critic2):
        for key, value in expected_vision.items():
            assert torch.equal(critic.vision_encoder.state_dict()[key], value)
        for key, value in expected_telemetry.items():
            assert torch.equal(critic.telemetry_encoder.state_dict()[key], value)


def test_sac_agent_rejects_incompatible_bc_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "bc_actor_bad.pt"
    _write_bc_actor_checkpoint(checkpoint_path, observation_shape=(3, 64, 64))
    agent = SACAgent(
        config=SACConfig(),
        observation_mode="full",
        device=torch.device("cpu"),
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
    )

    with pytest.raises(RuntimeError, match="observation_shape"):
        agent.load_bc_warm_start(checkpoint_path, init_mode="actor_only")
