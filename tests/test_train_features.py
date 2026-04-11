from __future__ import annotations

import numpy as np

from tm20ai.train.features import TELEMETRY_DIM, TelemetryFeatureBuilder, clamp_action


def test_clamp_action_respects_env_ranges() -> None:
    action = clamp_action([2.0, -1.0, 2.5])
    assert np.allclose(action, np.asarray([1.0, 0.0, 1.0], dtype=np.float32))


def test_telemetry_feature_builder_freezes_14d_layout_and_history() -> None:
    builder = TelemetryFeatureBuilder()
    info = {
        "run_id": "run-1",
        "speed_kmh": 500.0,
        "rpm": 5500.0,
        "gear": 3,
    }

    features = builder.encode(info)

    assert features.shape == (TELEMETRY_DIM,)
    assert np.isclose(features[0], 0.5)
    assert np.isclose(features[1], 0.5)
    assert np.allclose(features[2:8], np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(features[8:], np.zeros(6, dtype=np.float32))

    builder.observe_action([1.0, 0.5, -0.25], run_id="run-1")
    next_features = builder.encode(info)
    assert np.allclose(next_features[-3:], np.asarray([1.0, 0.5, -0.25], dtype=np.float32))

    reset_features = builder.encode({"run_id": "run-2", "speed_kmh": 0.0, "rpm": 0.0, "gear": 0})
    assert np.allclose(reset_features[8:], np.zeros(6, dtype=np.float32))
