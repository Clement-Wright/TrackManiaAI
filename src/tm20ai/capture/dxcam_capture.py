from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable

import numpy as np

from ..config import CaptureConfig
from .window import TrackmaniaWindowLocator, WindowGeometry


class CaptureState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    RUNNING = "RUNNING"
    BROKEN = "BROKEN"


def _default_camera_factory(config: CaptureConfig):
    try:
        import dxcam
    except ImportError as exc:  # pragma: no cover - exercised on machines without dxcam installed
        raise RuntimeError("dxcam is not installed. Install the runtime dependencies before using screen capture.") from exc

    try:
        return dxcam.create(output_idx=0, max_buffer_len=config.max_buffer_len, output_color="RGB")
    except TypeError:
        return dxcam.create(output_idx=0, max_buffer_len=config.max_buffer_len)


def _region_delta(old: WindowGeometry, new: WindowGeometry) -> int:
    return max(
        abs(old.left - new.left),
        abs(old.top - new.top),
        abs(old.right - new.right),
        abs(old.bottom - new.bottom),
    )


def _is_valid_frame(frame: Any) -> bool:
    return isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[0] > 0 and frame.shape[1] > 0


class DXCamCapture:
    """DXcam-backed latest-frame capture for the Trackmania client region."""

    def __init__(
        self,
        config: CaptureConfig,
        *,
        camera_factory: Callable[[CaptureConfig], Any] | None = None,
        window_locator: TrackmaniaWindowLocator | None = None,
        window_title: str | None = None,
        region_refresh_interval_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self._camera_factory = camera_factory or _default_camera_factory
        self._window_locator = window_locator or TrackmaniaWindowLocator(window_title or config.window_title)
        self._region_refresh_interval_seconds = region_refresh_interval_seconds

        self._camera: Any | None = None
        self._geometry: WindowGeometry | None = None
        self._window_handle: int | None = None
        self._invalid_frame_count = 0
        self._last_refresh_monotonic: float | None = None
        self._state = CaptureState.UNINITIALIZED

    @property
    def geometry(self) -> WindowGeometry | None:
        return self._geometry

    @property
    def window_handle(self) -> int | None:
        return self._window_handle

    @property
    def state(self) -> CaptureState:
        return self._state

    def _locate_window(self) -> tuple[int, WindowGeometry]:
        hwnd, geometry = self._window_locator.locate_tm_window()
        return hwnd, geometry

    def _start_camera(self) -> None:
        assert self._camera is not None
        assert self._geometry is not None
        region = self._geometry.as_region()
        try:
            self._camera.start(region=region, target_fps=self.config.target_fps, video_mode=True)
        except TypeError:
            self._camera.start(region=region, target_fps=self.config.target_fps)
        self._state = CaptureState.RUNNING
        self._invalid_frame_count = 0

    def ensure_started(self) -> None:
        hwnd, geometry = self._locate_window()
        if self._camera is None:
            self._camera = self._camera_factory(self.config)
            self._window_handle = hwnd
            self._geometry = geometry
            self._state = CaptureState.READY
            self._start_camera()
            self._last_refresh_monotonic = time.monotonic()
            return

        self.refresh_region_if_needed(hwnd=hwnd, geometry=geometry, force=self._state in {CaptureState.BROKEN, CaptureState.READY})
        if self._state != CaptureState.RUNNING:
            self._start_camera()

    def refresh_region_if_needed(
        self,
        *,
        hwnd: int | None = None,
        geometry: WindowGeometry | None = None,
        force: bool = False,
    ) -> bool:
        now = time.monotonic()
        if not force and self._last_refresh_monotonic is not None:
            if (now - self._last_refresh_monotonic) < self._region_refresh_interval_seconds:
                return False

        if hwnd is None or geometry is None:
            hwnd, geometry = self._locate_window()
        assert geometry is not None
        assert hwnd is not None
        self._last_refresh_monotonic = now

        if self._window_handle is None or self._geometry is None:
            self._window_handle = hwnd
            self._geometry = geometry
            return True

        handle_changed = hwnd != self._window_handle
        region_changed = _region_delta(self._geometry, geometry) > self.config.region_change_tolerance_pixels
        if self._state == CaptureState.BROKEN or handle_changed or region_changed:
            self.restart(hwnd=hwnd, geometry=geometry, recreate=self._state == CaptureState.BROKEN)
            return True
        return False

    def stop(self) -> None:
        if self._camera is None:
            self._state = CaptureState.UNINITIALIZED
            return
        if self._state == CaptureState.RUNNING:
            try:
                self._camera.stop()
            finally:
                self._state = CaptureState.READY
        elif self._state == CaptureState.BROKEN:
            self._state = CaptureState.READY

    def restart(
        self,
        *,
        hwnd: int | None = None,
        geometry: WindowGeometry | None = None,
        recreate: bool = False,
    ) -> None:
        if hwnd is None or geometry is None:
            hwnd, geometry = self._locate_window()

        self.stop()
        self._window_handle = hwnd
        self._geometry = geometry
        self._invalid_frame_count = 0

        if recreate or self._camera is None:
            self._camera = None
            self._camera = self._camera_factory(self.config)
        self._state = CaptureState.READY
        self._start_camera()

    def get_latest_frame(self, *, timeout: float | None = None, copy: bool = True) -> np.ndarray:
        self.ensure_started()
        assert self._camera is not None

        deadline = time.monotonic() + (self.config.frame_timeout if timeout is None else timeout)
        last_exception: Exception | None = None
        while time.monotonic() < deadline:
            self.refresh_region_if_needed()
            try:
                frame = self._camera.get_latest_frame()
            except Exception as exc:  # noqa: BLE001 - restart on DXcam failure
                self._state = CaptureState.BROKEN
                last_exception = exc
                self.restart(recreate=True)
                continue

            if _is_valid_frame(frame):
                self._invalid_frame_count = 0
                return np.array(frame, copy=copy)

            self._invalid_frame_count += 1
            if self._invalid_frame_count >= self.config.invalid_frame_limit:
                self._state = CaptureState.BROKEN
                self.restart(recreate=True)
            time.sleep(0.005)

        if last_exception is not None:
            raise TimeoutError(f"Timed out waiting for a DXcam frame after camera restart: {last_exception}") from last_exception
        raise TimeoutError("Timed out waiting for a DXcam frame.")

    def flush_for_interval(self, duration_seconds: float) -> None:
        if duration_seconds <= 0.0:
            return
        deadline = time.monotonic() + duration_seconds
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            try:
                self.get_latest_frame(timeout=min(0.05, remaining), copy=False)
            except TimeoutError:
                pass
            time.sleep(0.005)

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
        self._camera = None
        self._window_handle = None
        self._geometry = None
        self._invalid_frame_count = 0
        self._state = CaptureState.UNINITIALIZED
