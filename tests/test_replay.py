from __future__ import annotations

import numpy as np
import torch

from tm20ai.capture import lidar_feature_dim
from tm20ai.config import LidarObservationConfig
from tm20ai.train.features import ACTION_DIM, TELEMETRY_DIM
from tm20ai.train.replay import BalancedReplayBuffer, ReplayBuffer


def make_full_transition(step: int) -> dict[str, object]:
    obs = np.full((4, 64, 64), step, dtype=np.uint8)
    next_obs = np.full((4, 64, 64), step + 1, dtype=np.uint8)
    telemetry = np.full((TELEMETRY_DIM,), float(step), dtype=np.float32)
    next_telemetry = np.full((TELEMETRY_DIM,), float(step + 1), dtype=np.float32)
    return {
        "obs_uint8": obs,
        "telemetry_float": telemetry,
        "action": np.asarray([0.1, -0.3], dtype=np.float32),
        "reward": float(step),
        "next_obs_uint8": next_obs,
        "next_telemetry_float": next_telemetry,
        "terminated": False,
        "truncated": step % 2 == 0,
        "episode_id": f"episode-{step}",
        "map_uid": "map",
        "step_idx": step,
    }


def make_lidar_transition(step: int) -> dict[str, object]:
    lidar_dim = lidar_feature_dim(LidarObservationConfig())
    obs = np.full((lidar_dim,), 0.1 * step, dtype=np.float32)
    next_obs = np.full((lidar_dim,), 0.1 * (step + 1), dtype=np.float32)
    return {
        "obs_float": obs,
        "action": np.asarray([0.4, -0.2], dtype=np.float32),
        "reward": float(step),
        "next_obs_float": next_obs,
        "terminated": step % 2 == 0,
        "truncated": False,
        "episode_id": f"episode-{step}",
        "map_uid": "map",
        "step_idx": step,
    }


def test_full_replay_buffer_wraparound_and_sample_conversion() -> None:
    replay = ReplayBuffer(
        mode="full",
        capacity=2,
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
        rng_seed=7,
    )
    replay.add(make_full_transition(1))
    replay.add(make_full_transition(2))
    replay.add(make_full_transition(3))

    assert replay.size == 2

    sample = replay.sample(2, device=torch.device("cpu"))
    assert sample.obs.shape == (2, 4, 64, 64)
    assert sample.next_obs.shape == (2, 4, 64, 64)
    assert sample.telemetry is not None
    assert sample.next_telemetry is not None
    assert sample.telemetry.shape == (2, TELEMETRY_DIM)
    assert sample.action.shape == (2, ACTION_DIM)
    assert sample.reward.shape == (2, 1)
    assert sample.done.shape == (2, 1)
    assert sample.obs.dtype == torch.float32
    assert sample.next_obs.dtype == torch.float32
    assert float(sample.obs.max().item()) <= 1.0
    assert float(sample.obs.min().item()) >= 0.0


def test_lidar_replay_buffer_preserves_vector_observations() -> None:
    lidar_dim = lidar_feature_dim(LidarObservationConfig())
    replay = ReplayBuffer(mode="lidar", capacity=3, observation_shape=(lidar_dim,), telemetry_dim=0, rng_seed=11)
    replay.add(make_lidar_transition(1))
    replay.add(make_lidar_transition(2))
    replay.add(make_lidar_transition(3))

    sample = replay.sample(2, device=torch.device("cpu"))
    assert sample.obs.shape == (2, lidar_dim)
    assert sample.next_obs.shape == (2, lidar_dim)
    assert sample.telemetry is None
    assert sample.next_telemetry is None
    assert sample.action.shape == (2, ACTION_DIM)
    assert sample.reward.shape == (2, 1)
    assert sample.done.shape == (2, 1)
    assert sample.obs.dtype == torch.float32
    assert float(sample.obs.max().item()) <= 0.3


def test_balanced_replay_buffer_samples_offline_and_online_sources() -> None:
    online = ReplayBuffer(
        mode="full",
        capacity=8,
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
        rng_seed=1,
    )
    offline = ReplayBuffer(
        mode="full",
        capacity=8,
        observation_shape=(4, 64, 64),
        telemetry_dim=TELEMETRY_DIM,
        rng_seed=2,
    )
    balanced = BalancedReplayBuffer(
        online=online,
        offline=offline,
        offline_initial_fraction=0.75,
        offline_final_fraction=0.25,
        decay_env_steps=100,
        rng_seed=3,
    )
    for step in range(4):
        balanced.add(make_full_transition(step))
        balanced.add_offline(make_full_transition(step + 10))

    balanced.set_progress(env_step=0)
    sample = balanced.sample(4, device=torch.device("cpu"))

    assert sample.source is not None
    assert int((sample.source == 1).sum().item()) == 3
    assert int((sample.source == 0).sum().item()) == 1
    assert balanced.last_sample_profile["offline_batch_size"] == 3
    assert balanced.last_sample_profile["online_batch_size"] == 1
