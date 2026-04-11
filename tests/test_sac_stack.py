from __future__ import annotations

import numpy as np
import torch

from tm20ai.algos.sac import SACAgent
from tm20ai.config import SACConfig
from tm20ai.models.full_actor_critic import FullObservationActor, FullObservationCritic, random_shift_augmentation
from tm20ai.models.lidar_actor_critic import LidarActor, LidarCritic
from tm20ai.train.replay import ReplayBuffer


def test_full_actor_outputs_valid_action_ranges_and_log_probs() -> None:
    actor = FullObservationActor()
    observation = torch.rand(2, 4, 64, 64)
    telemetry = torch.rand(2, 14)

    action, log_prob = actor.sample(observation, telemetry, deterministic=False)

    assert action.shape == (2, 3)
    assert log_prob is not None
    assert log_prob.shape == (2, 1)
    assert torch.isfinite(log_prob).all()
    assert torch.all(action[:, 0] >= 0.0)
    assert torch.all(action[:, 0] <= 1.0)
    assert torch.all(action[:, 1] >= 0.0)
    assert torch.all(action[:, 1] <= 1.0)
    assert torch.all(action[:, 2] >= -1.0)
    assert torch.all(action[:, 2] <= 1.0)


def test_lidar_actor_outputs_valid_action_ranges_and_log_probs() -> None:
    actor = LidarActor()
    observation = torch.rand(2, 83)

    action, log_prob = actor.sample(observation, deterministic=False)

    assert action.shape == (2, 3)
    assert log_prob is not None
    assert log_prob.shape == (2, 1)
    assert torch.isfinite(log_prob).all()
    assert torch.all(action[:, 0] >= 0.0)
    assert torch.all(action[:, 0] <= 1.0)
    assert torch.all(action[:, 1] >= 0.0)
    assert torch.all(action[:, 1] <= 1.0)
    assert torch.all(action[:, 2] >= -1.0)
    assert torch.all(action[:, 2] <= 1.0)


def test_full_critic_outputs_scalar_q_values() -> None:
    critic = FullObservationCritic()
    q_value = critic(torch.rand(4, 4, 64, 64), torch.rand(4, 14), torch.rand(4, 3))
    assert q_value.shape == (4, 1)


def test_lidar_critic_outputs_scalar_q_values() -> None:
    critic = LidarCritic()
    q_value = critic(torch.rand(4, 83), torch.rand(4, 3))
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
        telemetry_dim=14,
    )
    replay = ReplayBuffer(mode="full", capacity=16, observation_shape=(4, 64, 64), telemetry_dim=14, rng_seed=11)
    for step in range(8):
        replay.add(
            {
                "obs_uint8": np.full((4, 64, 64), step, dtype=np.uint8),
                "telemetry_float": np.full((14,), 0.1 * step, dtype=np.float32),
                "action": np.asarray([0.1, 0.2, -0.1], dtype=np.float32),
                "reward": 1.0,
                "next_obs_uint8": np.full((4, 64, 64), step + 1, dtype=np.uint8),
                "next_telemetry_float": np.full((14,), 0.1 * (step + 1), dtype=np.float32),
                "terminated": False,
                "truncated": False,
            }
        )

    update = agent.update(replay.sample(4, device=torch.device("cpu")))
    assert np.isfinite(update.actor_loss)
    assert np.isfinite(update.critic_loss)
    assert np.isfinite(update.alpha)


def test_sac_agent_update_smoke_for_lidar() -> None:
    agent = SACAgent(
        config=SACConfig(),
        observation_mode="lidar",
        device=torch.device("cpu"),
        observation_shape=(83,),
        telemetry_dim=0,
    )
    replay = ReplayBuffer(mode="lidar", capacity=16, observation_shape=(83,), telemetry_dim=0, rng_seed=13)
    for step in range(8):
        replay.add(
            {
                "obs_float": np.full((83,), 0.01 * step, dtype=np.float32),
                "action": np.asarray([0.1, 0.2, -0.1], dtype=np.float32),
                "reward": 1.0,
                "next_obs_float": np.full((83,), 0.01 * (step + 1), dtype=np.float32),
                "terminated": False,
                "truncated": False,
            }
        )

    update = agent.update(replay.sample(4, device=torch.device("cpu")))
    assert np.isfinite(update.actor_loss)
    assert np.isfinite(update.critic_loss)
    assert np.isfinite(update.alpha)
