from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ..config import CaptureConfig
from .window import WindowGeometry, find_window, get_client_geometry


def _default_camera_factory(max_buffer_len: int):
    try:
        import dxcam
    except ImportError as exc:  # pragma: no cover - exercised on machines without dxcam installed
        raise RuntimeError("dxcam is not installed. Install the runtime dependencies before using screen capture.") from exc

    try:
        return dxcam.create(max_buffer_len=max_buffer_len, output_color="RGB")
    except TypeError:
        return dxcam.create(max_buffer_len=max_buffer_len)


class DXCamCapture:
    """DXcam-backed latest-frame capture for the Trackmania client region."""

    def __init__(
        self,
        config: CaptureConfig,
        *,
        camera_factory: Callable[[int], Any] | None = None,
        window_title: str | None = None,
    ) -> None:
        self.config = config
        self._camera_factory = camera_factory or _default_camera_factory
        self._window_title = window_title or config.window_title
        self._camera: Any | None = None
        self._geometry: WindowGeometry | None = None

    @property
    def geometry(self) -> WindowGeometry | None:
        return self._geometry

    def open(self) -> None:
        if self._camera is not None:
            return
        hwnd = find_window(self._window_title)
        self._geometry = get_client_geometry(hwnd)
        self._camera = self._camera_factory(self.config.max_buffer_len)
        self._start_camera()

    def _start_camera(self) -> None:
        assert self._camera is not None
        assert self._geometry is not None
        region = self._geometry.as_region()
        try:
            self._camera.start(region=region, target_fps=self.config.target_fps, video_mode=True)
        except TypeError:
            self._camera.start(region=region, target_fps=self.config.target_fps)

    def stop(self) -> None:
        if self._camera is None:
            return
        try:
            self._camera.stop()
        finally:
            self._camera = None

    def restart(self) -> None:
        self.stop()
        self.open()

    def get_latest_frame(self, *, timeout: float | None = None, copy: bool = True) -> np.ndarray:
        self.open()
        assert self._camera is not None
        deadline = time.monotonic() + (self.config.frame_timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            frame = self._camera.get_latest_frame()
            if frame is not None:
                return np.array(frame, copy=copy)
            time.sleep(0.005)
        raise TimeoutError("Timed out waiting for a DXcam frame.")

    def prime_frames(
        self,
        *,
        count: int,
        timeout: float | None = None,
        interval_seconds: float | None = None,
    ) -> list[np.ndarray]:
        if count <= 0:
            return []
        interval = interval_seconds if interval_seconds is not None else max(0.005, 1.0 / float(self.config.target_fps))
        deadline = time.monotonic() + (self.config.frame_timeout if timeout is None else timeout)
        frames: list[np.ndarray] = []
        while len(frames) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"Timed out while priming {count} DXcam frames.")
            frames.append(self.get_latest_frame(timeout=min(remaining, self.config.frame_timeout)))
            if len(frames) < count:
                time.sleep(interval)
        return frames

    def close(self) -> None:
        self.stop()
