from __future__ import annotations

from collections import deque
from typing import Iterable

import cv2
import numpy as np

from ..config import FullObservationConfig


class FrameStackPreprocessor:
    """Resize, grayscale, and stack FULL observations."""

    def __init__(self, config: FullObservationConfig):
        if not config.grayscale:
            raise ValueError("Phase 3-4 only supports grayscale FULL observations.")
        self.config = config
        self._frames: deque[np.ndarray] = deque(maxlen=config.frame_stack)

    @property
    def frame_stack(self) -> int:
        return self.config.frame_stack

    def clear(self) -> None:
        self._frames.clear()

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
            raise ValueError("Expected an HWC frame with 3 or 4 channels.")
        resized = cv2.resize(
            frame,
            (self.config.output_width, self.config.output_height),
            interpolation=cv2.INTER_AREA,
        )
        code = cv2.COLOR_BGRA2GRAY if resized.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(resized, code)
        return gray.astype(np.uint8, copy=False)

    def _stack(self) -> np.ndarray:
        if len(self._frames) != self.config.frame_stack:
            raise RuntimeError("Frame stack is not full yet.")
        return np.stack(list(self._frames), axis=0).astype(np.uint8, copy=False)

    def reset_stack(self, frame: np.ndarray) -> np.ndarray:
        processed = self.transform_frame(frame)
        self._frames.clear()
        for _ in range(self.config.frame_stack):
            self._frames.append(processed.copy())
        return self._stack()

    def build_clean_stack(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        processed_frames = [self.transform_frame(frame) for frame in frames]
        if not processed_frames:
            raise ValueError("At least one frame is required to build the observation stack.")
        self._frames.clear()
        for processed in processed_frames[-self.config.frame_stack :]:
            self._frames.append(processed)
        while len(self._frames) < self.config.frame_stack:
            self._frames.appendleft(self._frames[0].copy())
        return self._stack()

    def append_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._frames:
            return self.reset_stack(frame)
        self._frames.append(self.transform_frame(frame))
        return self._stack()
