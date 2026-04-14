from __future__ import annotations

import numpy as np

from tm20ai.capture.lidar import LidarExtractor, LidarObservationBuilder, lidar_feature_dim
from tm20ai.config import LidarObservationConfig


def make_test_frame() -> np.ndarray:
    frame = np.full((488, 958, 3), 255, dtype=np.uint8)
    frame[:, :40] = 0
    frame[:, -40:] = 0
    frame[-120:, 420:540] = 0
    return frame


def test_lidar_extractor_returns_normalized_rays() -> None:
    config = LidarObservationConfig()
    extractor = LidarExtractor(config)
    distances = extractor.extract(make_test_frame())

    assert distances.shape == (config.ray_count,)
    assert distances.dtype == np.float32
    assert np.all(distances >= 0.0)
    assert np.all(distances <= 1.0)


def test_lidar_observation_builder_resets_and_tracks_action_history() -> None:
    config = LidarObservationConfig()
    builder = LidarObservationBuilder(config)
    fresh_frames = [make_test_frame() for _ in range(config.lidar_hist_len)]

    observation = builder.reset(fresh_frames, speed_norm=0.25)
    assert observation.shape == (lidar_feature_dim(config),)
    assert observation.dtype == np.float32
    assert observation[0] == np.float32(0.25)

    builder.observe_action(np.asarray([1.0, -0.5], dtype=np.float32))
    next_observation = builder.append_frame(make_test_frame(), speed_norm=0.5)
    assert next_observation.shape == (lidar_feature_dim(config),)
    assert np.isclose(next_observation[0], 0.5)
    assert np.isclose(next_observation[-2], 1.0)
    assert np.isclose(next_observation[-1], -0.5)
