from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np

from ..config import LidarObservationConfig


LIDAR_SPEED_DIM = 1
LIDAR_ACTION_DIM = 3


def _clamp_action(action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape != (LIDAR_ACTION_DIM,):
        raise ValueError(f"Expected an action with shape ({LIDAR_ACTION_DIM},), got {array.shape}.")
    return np.asarray(
        [
            np.clip(array[0], 0.0, 1.0),
            np.clip(array[1], 0.0, 1.0),
            np.clip(array[2], -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def lidar_feature_dim(config: LidarObservationConfig) -> int:
    return LIDAR_SPEED_DIM + (config.lidar_hist_len * config.ray_count) + (
        config.prev_action_hist_len * LIDAR_ACTION_DIM
    )


@dataclass(slots=True, frozen=True)
class LidarDebugResult:
    crop_bounds: tuple[int, int, int, int]
    rays: np.ndarray
    overlay: np.ndarray


class LidarExtractor:
    def __init__(self, config: LidarObservationConfig) -> None:
        self.config = config
        self._angles_radians = np.deg2rad(
            np.linspace(
                self.config.ray_min_angle_degrees,
                self.config.ray_max_angle_degrees,
                self.config.ray_count,
                dtype=np.float32,
            )
        )

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
            raise ValueError("Expected an HWC frame with 3 or 4 channels.")
        code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(frame, code)

    def crop_frame(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        gray = self._to_grayscale(frame)
        height, width = gray.shape
        left = max(0, min(width - 1, int(round(width * self.config.fixed_crop[0]))))
        top = max(0, min(height - 1, int(round(height * self.config.fixed_crop[1]))))
        right = max(left + 1, min(width, int(round(width * self.config.fixed_crop[2]))))
        bottom = max(top + 1, min(height, int(round(height * self.config.fixed_crop[3]))))
        return gray[top:bottom, left:right], (left, top, right, bottom)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        crop, _ = self.crop_frame(frame)
        mask = crop <= self.config.border_threshold
        height, width = mask.shape
        origin_x = (width - 1) * 0.5
        origin_y = float(height - 1)
        max_ray_length = max(
            1.0,
            np.sqrt(float(width * width + height * height)) * float(self.config.max_ray_length_fraction),
        )
        rays = np.ones((self.config.ray_count,), dtype=np.float32)
        for ray_index, angle in enumerate(self._angles_radians):
            dx = float(np.sin(angle))
            dy = float(-np.cos(angle))
            distance = max_ray_length
            for step in range(1, ceil(max_ray_length) + 1):
                sample_x = int(round(origin_x + dx * step))
                sample_y = int(round(origin_y + dy * step))
                if sample_x < 0 or sample_x >= width or sample_y < 0 or sample_y >= height:
                    distance = float(step)
                    break
                if mask[sample_y, sample_x]:
                    distance = float(step)
                    break
            rays[ray_index] = np.clip(distance / max_ray_length, 0.0, 1.0)
        return rays

    def build_debug_result(self, frame: np.ndarray) -> LidarDebugResult:
        crop, crop_bounds = self.crop_frame(frame)
        overlay = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        rays = self.extract(frame)
        height, width = crop.shape
        origin = (int(round((width - 1) * 0.5)), int(round(height - 1)))
        max_ray_length = max(
            1.0,
            np.sqrt(float(width * width + height * height)) * float(self.config.max_ray_length_fraction),
        )
        for normalized_distance, angle in zip(rays, self._angles_radians, strict=True):
            distance = float(normalized_distance) * max_ray_length
            endpoint = (
                int(round(origin[0] + np.sin(angle) * distance)),
                int(round(origin[1] - np.cos(angle) * distance)),
            )
            cv2.line(overlay, origin, endpoint, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(overlay, origin, 3, (0, 0, 255), -1, cv2.LINE_AA)
        return LidarDebugResult(crop_bounds=crop_bounds, rays=rays, overlay=overlay)


class LidarObservationBuilder:
    def __init__(self, config: LidarObservationConfig) -> None:
        self.config = config
        self.extractor = LidarExtractor(config)
        self._lidar_history: deque[np.ndarray] = deque(maxlen=config.lidar_hist_len)
        self._action_history: deque[np.ndarray] = deque(maxlen=config.prev_action_hist_len)
        self.reset_action_history()

    @property
    def feature_dim(self) -> int:
        return lidar_feature_dim(self.config)

    def reset_action_history(self) -> None:
        self._action_history.clear()
        neutral = np.zeros((LIDAR_ACTION_DIM,), dtype=np.float32)
        for _ in range(self.config.prev_action_hist_len):
            self._action_history.append(neutral.copy())

    def observe_action(self, action: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self._action_history.append(_clamp_action(action))

    def reset(self, fresh_frames: list[np.ndarray], *, speed_norm: float) -> np.ndarray:
        if not fresh_frames:
            raise ValueError("At least one frame is required to reset the LIDAR observation history.")
        self._lidar_history.clear()
        self.reset_action_history()
        recent_frames = fresh_frames[-self.config.lidar_hist_len :]
        for frame in recent_frames:
            self._lidar_history.append(self.extractor.extract(frame))
        while len(self._lidar_history) < self.config.lidar_hist_len:
            self._lidar_history.appendleft(self._lidar_history[0].copy())
        return self._build_observation(speed_norm)

    def append_frame(self, frame: np.ndarray, *, speed_norm: float) -> np.ndarray:
        if not self._lidar_history:
            return self.reset([frame], speed_norm=speed_norm)
        self._lidar_history.append(self.extractor.extract(frame))
        return self._build_observation(speed_norm)

    def _build_observation(self, speed_norm: float) -> np.ndarray:
        speed = np.asarray([np.clip(speed_norm, 0.0, 1.0)], dtype=np.float32)
        lidar_hist = np.stack(list(self._lidar_history), axis=0).reshape(-1).astype(np.float32, copy=False)
        prev_actions = np.concatenate(list(self._action_history), dtype=np.float32)
        return np.concatenate((speed, lidar_hist, prev_actions), dtype=np.float32)
