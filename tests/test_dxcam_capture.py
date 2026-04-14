from __future__ import annotations

import numpy as np

from tm20ai.capture.dxcam_capture import CaptureBinding, CaptureState, DXCamCapture
from tm20ai.capture.window import WindowGeometry
from tm20ai.config import CaptureConfig


class FakeCamera:
    def __init__(self, frames: list[np.ndarray | None], *, raises: bool = False) -> None:
        self.frames = list(frames)
        self.raises = raises
        self.start_calls: list[tuple[tuple[int, int, int, int], int]] = []
        self.stop_calls = 0

    def start(self, *, region, target_fps, video_mode=True):  # noqa: ANN001
        del video_mode
        self.start_calls.append((region, target_fps))

    def stop(self) -> None:
        self.stop_calls += 1

    def get_latest_frame(self):
        if self.raises:
            raise RuntimeError("camera failed")
        if not self.frames:
            return None
        return self.frames.pop(0)


class FakeWindowLocator:
    def __init__(self, windows: list[tuple[int, WindowGeometry]]) -> None:
        self.windows = windows
        self._index = 0

    def locate_tm_window(self) -> tuple[int, WindowGeometry]:
        if self._index >= len(self.windows):
            return self.windows[-1]
        result = self.windows[self._index]
        self._index += 1
        return result


def make_geometry(left: int, top: int, right: int, bottom: int) -> WindowGeometry:
    return WindowGeometry(hwnd=100, title="Trackmania", left=left, top=top, right=right, bottom=bottom)


def make_frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def fake_binding(_dxcam_module, _hwnd: int, _geometry: WindowGeometry) -> CaptureBinding:  # noqa: ANN001
    return CaptureBinding(
        device_idx=0,
        output_idx=0,
        backend="dxgi",
        output_left=0,
        output_top=0,
        output_right=256,
        output_bottom=128,
        monitor_handle=1,
        monitor_device_name="\\\\.\\DISPLAY1",
        output_device_name="\\\\.\\DISPLAY1",
        is_primary=True,
    )


def test_dxcam_capture_starts_once_and_keeps_running() -> None:
    config = CaptureConfig(frame_timeout=0.01, require_stable_window_polls=1)
    camera = FakeCamera([make_frame(10)])
    created: list[FakeCamera] = []

    def factory(cfg: CaptureConfig) -> FakeCamera:
        created.append(camera)
        return camera

    locator = FakeWindowLocator(windows=[(100, make_geometry(0, 0, 256, 128))])
    capture = DXCamCapture(
        config,
        camera_factory=factory,
        window_locator=locator,
        region_refresh_interval_seconds=0.0,
        binding_resolver=fake_binding,
    )

    frame = capture.get_latest_frame()

    assert frame.shape == (8, 8, 3)
    assert len(created) == 1
    assert camera.stop_calls == 0
    assert len(camera.start_calls) == 1
    assert capture.state == CaptureState.RUNNING


def test_dxcam_capture_ensure_started_is_noop_while_running() -> None:
    config = CaptureConfig(frame_timeout=0.01, require_stable_window_polls=1)
    camera = FakeCamera([make_frame(10)])
    create_count = 0

    def factory(cfg: CaptureConfig) -> FakeCamera:
        nonlocal create_count
        create_count += 1
        return camera

    locator = FakeWindowLocator(
        windows=[
            (100, make_geometry(0, 0, 256, 128)),
            (100, make_geometry(20, 0, 276, 128)),
        ]
    )
    capture = DXCamCapture(
        config,
        camera_factory=factory,
        window_locator=locator,
        region_refresh_interval_seconds=0.0,
        binding_resolver=fake_binding,
    )

    capture.ensure_started()
    capture.ensure_started()

    assert create_count == 1
    assert camera.stop_calls == 0
    assert len(camera.start_calls) == 1
    assert capture.state == CaptureState.RUNNING


def test_dxcam_capture_restarts_on_region_change_without_recreate() -> None:
    config = CaptureConfig(frame_timeout=0.01, require_stable_window_polls=1)
    camera = FakeCamera([make_frame(10), make_frame(11)])
    create_count = 0

    def factory(cfg: CaptureConfig) -> FakeCamera:
        nonlocal create_count
        create_count += 1
        return camera

    locator = FakeWindowLocator(
        windows=[
            (100, make_geometry(0, 0, 256, 128)),
            (100, make_geometry(20, 0, 276, 128)),
        ]
    )
    capture = DXCamCapture(
        config,
        camera_factory=factory,
        window_locator=locator,
        region_refresh_interval_seconds=0.0,
        binding_resolver=fake_binding,
    )

    capture.ensure_started()
    changed = capture.refresh_region_if_needed(force=True)

    assert changed is True
    assert create_count == 1
    assert camera.stop_calls == 1
    assert len(camera.start_calls) == 2


def test_dxcam_capture_recreates_camera_after_repeated_invalid_frames() -> None:
    config = CaptureConfig(frame_timeout=0.05, invalid_frame_limit=2, require_stable_window_polls=1)
    cameras = [
        FakeCamera([None, None]),
        FakeCamera([make_frame(42)]),
    ]

    def factory(cfg: CaptureConfig) -> FakeCamera:
        return cameras.pop(0)

    locator = FakeWindowLocator(windows=[(100, make_geometry(0, 0, 256, 128))])
    capture = DXCamCapture(
        config,
        camera_factory=factory,
        window_locator=locator,
        region_refresh_interval_seconds=0.0,
        binding_resolver=fake_binding,
    )

    frame = capture.get_latest_frame()

    assert int(frame[0, 0, 0]) == 42
    assert capture.state == CaptureState.RUNNING


def test_dxcam_capture_rejects_unexpected_client_size() -> None:
    config = CaptureConfig(frame_timeout=0.01, require_stable_window_polls=1)
    camera = FakeCamera([make_frame(10)])

    locator = FakeWindowLocator(windows=[(100, make_geometry(0, 0, 320, 180))])
    capture = DXCamCapture(
        config,
        camera_factory=lambda _cfg: camera,
        window_locator=locator,
        region_refresh_interval_seconds=0.0,
        binding_resolver=fake_binding,
        expected_client_size=(256, 128),
    )

    with np.testing.assert_raises_regex(RuntimeError, "expected 256x128"):
        capture.ensure_started()
