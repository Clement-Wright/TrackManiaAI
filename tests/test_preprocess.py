from __future__ import annotations

import numpy as np

from tm20ai.capture.preprocess import FrameStackPreprocessor
from tm20ai.config import FullObservationConfig


def make_frame(value: int) -> np.ndarray:
    return np.full((128, 256, 3), value, dtype=np.uint8)


def test_preprocess_outputs_expected_shape_and_dtype() -> None:
    config = FullObservationConfig()
    preprocessor = FrameStackPreprocessor(config)

    stacked = preprocessor.reset_stack(make_frame(32))

    assert stacked.shape == (4, 64, 64)
    assert stacked.dtype == np.uint8
    assert int(stacked[0, 0, 0]) == int(stacked[-1, 0, 0])


def test_preprocess_reset_clears_stale_history() -> None:
    config = FullObservationConfig()
    preprocessor = FrameStackPreprocessor(config)

    preprocessor.reset_stack(make_frame(10))
    updated = preprocessor.append_frame(make_frame(200))
    reset = preprocessor.reset_stack(make_frame(80))

    assert int(updated[-1, 0, 0]) != int(updated[0, 0, 0])
    assert np.all(reset[0] == reset[-1])
    assert int(reset[0, 0, 0]) != int(updated[-1, 0, 0])
