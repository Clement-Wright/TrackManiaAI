from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from ..config import CaptureConfig
from .window import (
    MonitorGeometry,
    TrackmaniaWindowLocator,
    WindowGeometry,
    get_window_monitor_geometry,
)


class CaptureState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    RUNNING = "RUNNING"
    BROKEN = "BROKEN"


@dataclass(slots=True, frozen=True)
class CaptureBinding:
    device_idx: int
    output_idx: int
    backend: str
    output_left: int
    output_top: int
    output_right: int
    output_bottom: int
    monitor_handle: int | None
    monitor_device_name: str | None
    output_device_name: str | None
    is_primary: bool | None

    def region_for_geometry(self, geometry: WindowGeometry) -> tuple[int, int, int, int]:
        return (
            geometry.left - self.output_left,
            geometry.top - self.output_top,
            geometry.right - self.output_left,
            geometry.bottom - self.output_top,
        )

    def same_target(self, other: "CaptureBinding | None") -> bool:
        if other is None:
            return False
        return (
            self.device_idx == other.device_idx
            and self.output_idx == other.output_idx
            and self.output_left == other.output_left
            and self.output_top == other.output_top
            and self.output_right == other.output_right
            and self.output_bottom == other.output_bottom
            and self.monitor_handle == other.monitor_handle
            and self.monitor_device_name == other.monitor_device_name
            and self.output_device_name == other.output_device_name
        )

    def with_backend(self, backend: str) -> "CaptureBinding":
        return CaptureBinding(
            device_idx=self.device_idx,
            output_idx=self.output_idx,
            backend=backend,
            output_left=self.output_left,
            output_top=self.output_top,
            output_right=self.output_right,
            output_bottom=self.output_bottom,
            monitor_handle=self.monitor_handle,
            monitor_device_name=self.monitor_device_name,
            output_device_name=self.output_device_name,
            is_primary=self.is_primary,
        )


def _load_dxcam_module():
    try:
        import dxcam
    except ImportError as exc:  # pragma: no cover - exercised on machines without dxcam installed
        raise RuntimeError("dxcam is not installed. Install the runtime dependencies before using screen capture.") from exc
    return dxcam


def _default_camera_factory(
    config: CaptureConfig,
    *,
    device_idx: int = 0,
    output_idx: int | None = None,
    backend: str = "dxgi",
):
    dxcam = _load_dxcam_module()
    create_attempts: list[dict[str, Any]] = [
        {
            "device_idx": device_idx,
            "output_idx": output_idx,
            "max_buffer_len": config.max_buffer_len,
            "output_color": "RGB",
            "backend": backend,
        },
        {
            "output_idx": output_idx,
            "max_buffer_len": config.max_buffer_len,
            "output_color": "RGB",
            "backend": backend,
        },
        {
            "device_idx": device_idx,
            "output_idx": output_idx,
            "max_buffer_len": config.max_buffer_len,
            "output_color": "RGB",
        },
        {
            "output_idx": output_idx,
            "max_buffer_len": config.max_buffer_len,
            "output_color": "RGB",
        },
    ]
    if output_idx is None:
        create_attempts = [
            attempt
            for attempt in create_attempts
            if "output_idx" not in attempt or attempt["output_idx"] is not None
        ]

    last_exception: Exception | None = None
    for kwargs in create_attempts:
        try:
            return dxcam.create(**kwargs)
        except TypeError as exc:
            last_exception = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            continue
    assert last_exception is not None
    raise last_exception


def _region_delta(old: WindowGeometry, new: WindowGeometry) -> int:
    return max(
        abs(old.left - new.left),
        abs(old.top - new.top),
        abs(old.right - new.right),
        abs(old.bottom - new.bottom),
    )


def _is_valid_frame(frame: Any) -> bool:
    return isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[0] > 0 and frame.shape[1] > 0


def _foreground_window() -> int | None:
    try:
        import win32gui
    except ImportError:  # pragma: no cover - Windows dependency guard
        return None
    try:
        return int(win32gui.GetForegroundWindow())
    except Exception:  # noqa: BLE001
        return None


def _backend_candidates(config: CaptureConfig) -> list[str]:
    if config.backend == "auto":
        return ["dxgi", "winrt"]
    if config.backend == "dxgi":
        return ["dxgi", "winrt"]
    if config.backend == "winrt":
        return ["winrt", "dxgi"]
    return [config.backend]


def _enumerate_dxcam_outputs(dxcam_module) -> list[dict[str, Any]]:  # noqa: ANN001
    factory = dxcam_module.__dict__.get("__factory")
    if factory is None:
        return []
    output_metadata = getattr(factory, "output_metadata", {})
    outputs: list[dict[str, Any]] = []
    for device_idx, device_outputs in enumerate(getattr(factory, "outputs", [])):
        for output_idx, output in enumerate(device_outputs):
            output.update_desc()
            desc = getattr(output, "desc", None)
            coords = getattr(desc, "DesktopCoordinates", None)
            output_name = str(getattr(output, "devicename", ""))
            metadata = output_metadata.get(output_name)
            outputs.append(
                {
                    "device_idx": int(device_idx),
                    "output_idx": int(output_idx),
                    "output_device_name": output_name,
                    "monitor_handle": int(getattr(getattr(output, "hmonitor", None), "value", getattr(output, "hmonitor", 0)) or 0),
                    "left": 0 if coords is None else int(coords.left),
                    "top": 0 if coords is None else int(coords.top),
                    "right": int(getattr(output, "resolution", (0, 0))[0]) if coords is None else int(coords.right),
                    "bottom": int(getattr(output, "resolution", (0, 0))[1]) if coords is None else int(coords.bottom),
                    "is_primary": None if metadata is None else bool(metadata[1]),
                }
            )
    return outputs


class DXCamCapture:
    """DXcam-backed latest-frame capture for the Trackmania client region."""

    def __init__(
        self,
        config: CaptureConfig,
        *,
        camera_factory: Callable[..., Any] | None = None,
        window_locator: TrackmaniaWindowLocator | None = None,
        window_title: str | None = None,
        region_refresh_interval_seconds: float = 1.0,
        binding_resolver: Callable[[Any, int, WindowGeometry], CaptureBinding] | None = None,
        expected_client_size: tuple[int, int] | None = None,
    ) -> None:
        self.config = config
        self._camera_factory = camera_factory or _default_camera_factory
        self._window_locator = window_locator or TrackmaniaWindowLocator(window_title or config.window_title)
        self._region_refresh_interval_seconds = region_refresh_interval_seconds
        self._binding_resolver = binding_resolver
        self._expected_client_size = expected_client_size

        self._camera: Any | None = None
        self._geometry: WindowGeometry | None = None
        self._window_handle: int | None = None
        self._binding: CaptureBinding | None = None
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

    @property
    def binding(self) -> CaptureBinding | None:
        return self._binding

    def _locate_window(self) -> tuple[int, WindowGeometry]:
        hwnd, geometry = self._window_locator.locate_tm_window()
        self._validate_geometry(geometry)
        return hwnd, geometry

    def _validate_geometry(self, geometry: WindowGeometry) -> None:
        if self._expected_client_size is None:
            return
        expected_width, expected_height = self._expected_client_size
        if geometry.width == expected_width and geometry.height == expected_height:
            return
        raise RuntimeError(
            "Trackmania window client rect drifted from the configured observation size: "
            f"observed {geometry.width}x{geometry.height}, expected {expected_width}x{expected_height}."
        )

    def _wait_for_stable_window(self) -> tuple[int, WindowGeometry]:
        required = max(1, self.config.require_stable_window_polls)
        interval = max(0.01, self.config.stable_window_poll_interval_seconds)
        deadline = time.monotonic() + max(5.0, required * interval * 10.0)
        last_signature: tuple[int, int, int, int, int] | None = None
        stable_count = 0

        while time.monotonic() < deadline:
            hwnd, geometry = self._locate_window()
            if geometry.width > 0 and geometry.height > 0:
                signature = (hwnd, geometry.left, geometry.top, geometry.right, geometry.bottom)
                if signature == last_signature:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_signature = signature
                if stable_count >= required:
                    return hwnd, geometry
            else:
                stable_count = 0
                last_signature = None
            time.sleep(interval)

        raise RuntimeError(
            "Trackmania window did not stabilize before capture startup. "
            f"Last seen geometry={last_signature!r}."
        )

    def _resolve_capture_binding(self, dxcam_module, hwnd: int, geometry: WindowGeometry) -> CaptureBinding:  # noqa: ANN001
        if self._binding_resolver is not None:
            return self._binding_resolver(dxcam_module, hwnd, geometry)

        monitor = get_window_monitor_geometry(hwnd)
        outputs = _enumerate_dxcam_outputs(dxcam_module)
        if not outputs:
            raise RuntimeError("DXcam reported no available outputs.")

        chosen: dict[str, Any] | None = None
        if self.config.device_idx is not None and self.config.output_idx is not None:
            chosen = next(
                (
                    item
                    for item in outputs
                    if item["device_idx"] == self.config.device_idx and item["output_idx"] == self.config.output_idx
                ),
                None,
            )
        elif self.config.device_idx is not None:
            candidates = [item for item in outputs if item["device_idx"] == self.config.device_idx]
            chosen = next(
                (
                    item
                    for item in candidates
                    if item["monitor_handle"] == monitor.handle or item["output_device_name"] == monitor.device_name
                ),
                None,
            )
            if chosen is None:
                chosen = next((item for item in candidates if item["is_primary"]), candidates[0] if candidates else None)
        elif self.config.output_idx is not None:
            candidates = [item for item in outputs if item["output_idx"] == self.config.output_idx]
            chosen = next(
                (
                    item
                    for item in candidates
                    if item["monitor_handle"] == monitor.handle or item["output_device_name"] == monitor.device_name
                ),
                None,
            )
            if chosen is None and candidates:
                chosen = candidates[0]
        else:
            chosen = next(
                (
                    item
                    for item in outputs
                    if item["monitor_handle"] == monitor.handle or item["output_device_name"] == monitor.device_name
                ),
                None,
            )
            if chosen is None:
                chosen = next((item for item in outputs if item["is_primary"]), outputs[0])

        if chosen is None:
            raise RuntimeError(
                "Could not resolve a DXcam output for the Trackmania window. "
                f"monitor={monitor.device_name!r} outputs={outputs!r}"
            )

        return CaptureBinding(
            device_idx=int(chosen["device_idx"]),
            output_idx=int(chosen["output_idx"]),
            backend=_backend_candidates(self.config)[0],
            output_left=int(chosen["left"]),
            output_top=int(chosen["top"]),
            output_right=int(chosen["right"]),
            output_bottom=int(chosen["bottom"]),
            monitor_handle=monitor.handle,
            monitor_device_name=monitor.device_name,
            output_device_name=str(chosen["output_device_name"]),
            is_primary=None if chosen["is_primary"] is None else bool(chosen["is_primary"]),
        )

    def describe_bootstrap_context(self) -> dict[str, Any]:
        if self._window_handle is not None and self._geometry is not None:
            hwnd = self._window_handle
            geometry = self._geometry
        else:
            hwnd, geometry = self._wait_for_stable_window()
        monitor = get_window_monitor_geometry(hwnd)
        context = {
            "pid": os.getpid(),
            "thread_id": threading.get_native_id(),
            "start_method": mp.get_start_method(allow_none=True),
            "cwd": os.getcwd(),
            "window_handle": hwnd,
            "window_title": geometry.title,
            "window_client_rect": {
                "left": geometry.left,
                "top": geometry.top,
                "right": geometry.right,
                "bottom": geometry.bottom,
                "width": geometry.width,
                "height": geometry.height,
            },
            "foreground_window": _foreground_window(),
            "is_foreground": _foreground_window() == hwnd,
            "monitor": asdict(monitor),
        }
        if self._binding is not None:
            context["binding"] = asdict(self._binding)
        return context

    def _start_camera(self) -> None:
        assert self._camera is not None
        assert self._geometry is not None
        assert self._binding is not None
        region = self._binding.region_for_geometry(self._geometry)
        try:
            self._camera.start(region=region, target_fps=self.config.target_fps, video_mode=True)
        except TypeError:
            self._camera.start(region=region, target_fps=self.config.target_fps)
        self._state = CaptureState.RUNNING
        self._invalid_frame_count = 0

    def _create_camera(self, binding: CaptureBinding) -> tuple[Any, CaptureBinding]:
        last_exception: Exception | None = None
        for backend in _backend_candidates(self.config):
            candidate = binding.with_backend(backend)
            try:
                try:
                    camera = self._camera_factory(
                        self.config,
                        device_idx=candidate.device_idx,
                        output_idx=candidate.output_idx,
                        backend=candidate.backend,
                    )
                except TypeError:
                    try:
                        camera = self._camera_factory(self.config, candidate)
                    except TypeError:
                        camera = self._camera_factory(self.config)
                return camera, candidate
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                continue
        assert last_exception is not None
        raise last_exception

    def ensure_started(self) -> None:
        if self._camera is not None and self._state == CaptureState.RUNNING:
            return

        dxcam_module = _load_dxcam_module()
        if self._camera is None:
            hwnd, geometry = self._wait_for_stable_window()
            binding = self._resolve_capture_binding(dxcam_module, hwnd, geometry)
            self._camera, self._binding = self._create_camera(binding)
            self._window_handle = hwnd
            self._geometry = geometry
            self._state = CaptureState.READY
            self._start_camera()
            self._last_refresh_monotonic = time.monotonic()
            return

        hwnd, geometry = self._locate_window()
        binding = self._resolve_capture_binding(dxcam_module, hwnd, geometry)
        self.refresh_region_if_needed(
            hwnd=hwnd,
            geometry=geometry,
            binding=binding,
            force=self._state in {CaptureState.BROKEN, CaptureState.READY},
        )
        if self._state != CaptureState.RUNNING:
            self._start_camera()

    def refresh_region_if_needed(
        self,
        *,
        hwnd: int | None = None,
        geometry: WindowGeometry | None = None,
        binding: CaptureBinding | None = None,
        force: bool = False,
    ) -> bool:
        now = time.monotonic()
        if not force and self._last_refresh_monotonic is not None:
            if (now - self._last_refresh_monotonic) < self._region_refresh_interval_seconds:
                return False

        if hwnd is None or geometry is None:
            hwnd, geometry = self._locate_window()
        if binding is None:
            dxcam_module = _load_dxcam_module()
            binding = self._resolve_capture_binding(dxcam_module, hwnd, geometry)
        assert geometry is not None
        assert hwnd is not None
        self._last_refresh_monotonic = now

        if self._window_handle is None or self._geometry is None or self._binding is None:
            self._window_handle = hwnd
            self._geometry = geometry
            self._binding = binding
            return True

        handle_changed = hwnd != self._window_handle
        region_changed = _region_delta(self._geometry, geometry) > self.config.region_change_tolerance_pixels
        binding_changed = not binding.same_target(self._binding)
        if self._state == CaptureState.BROKEN or handle_changed or region_changed or binding_changed:
            self.restart(
                hwnd=hwnd,
                geometry=geometry,
                binding=binding if binding_changed else self._binding,
                recreate=self._state == CaptureState.BROKEN or binding_changed,
            )
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
        binding: CaptureBinding | None = None,
        recreate: bool = False,
    ) -> None:
        if hwnd is None or geometry is None:
            hwnd, geometry = self._locate_window()
        if binding is None:
            dxcam_module = _load_dxcam_module()
            binding = self._resolve_capture_binding(dxcam_module, hwnd, geometry)

        self.stop()
        self._window_handle = hwnd
        self._geometry = geometry
        self._invalid_frame_count = 0

        if recreate or self._camera is None or not binding.same_target(self._binding):
            self._camera = None
            self._camera, self._binding = self._create_camera(binding)
        else:
            self._binding = binding
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
        self._binding = None
        self._invalid_frame_count = 0
        self._state = CaptureState.UNINITIALIZED
