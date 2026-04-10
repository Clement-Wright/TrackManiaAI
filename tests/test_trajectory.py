from __future__ import annotations

import numpy as np

from tm20ai.env.trajectory import build_runtime_trajectory


def test_trajectory_resampling_is_monotonic() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    race_time_ms = np.asarray([0.0, 100.0, 200.0], dtype=np.float32)

    trajectory = build_runtime_trajectory("test-map", positions, race_time_ms, 0.5)

    assert trajectory.points.shape[1] == 3
    assert np.all(np.diff(trajectory.arc_length) > 0.0)
    assert np.isclose(float(trajectory.arc_length[-1]), 2.0)
    assert np.allclose(trajectory.tangents[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
