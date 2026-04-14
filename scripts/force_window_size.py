from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.capture.window import TrackmaniaWindowLocator, get_client_geometry
from tm20ai.config import load_tm20ai_config


def log(message: str) -> None:
    print(f"[force-window-size] {message}", flush=True)


def expected_window_shape(config_path: Path) -> tuple[int, int]:
    config = load_tm20ai_config(config_path)
    if config.observation.mode == "lidar":
        return config.lidar_observation.window_width, config.lidar_observation.window_height
    return config.full_observation.window_width, config.full_observation.window_height


def force_client_rect(hwnd: int, *, width: int, height: int, retries: int = 10) -> tuple[int, int]:
    import win32con
    import win32gui

    user32 = ctypes.windll.user32
    user32.AllowSetForegroundWindow(-1)

    last_geometry = None
    for attempt in range(1, retries + 1):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        try:
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
        except Exception:  # noqa: BLE001
            pass

        geometry = get_client_geometry(hwnd)
        outer_left, outer_top, outer_right, outer_bottom = win32gui.GetWindowRect(hwnd)
        outer_width = int(outer_right - outer_left)
        outer_height = int(outer_bottom - outer_top)
        delta_width = outer_width - geometry.width
        delta_height = outer_height - geometry.height
        target_outer_width = max(1, width + delta_width)
        target_outer_height = max(1, height + delta_height)
        win32gui.MoveWindow(hwnd, 0, 0, target_outer_width, target_outer_height, True)
        time.sleep(0.5)

        geometry = get_client_geometry(hwnd)
        last_geometry = geometry
        log(
            f"attempt={attempt} client_rect={geometry.width}x{geometry.height} "
            f"target={width}x{height}"
        )
        if geometry.width == width and geometry.height == height:
            return geometry.width, geometry.height

    assert last_geometry is not None
    raise RuntimeError(
        "Failed to force the Trackmania client rect to the requested size. "
        f"Last observed client rect was {last_geometry.width}x{last_geometry.height}; expected {width}x{height}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Force the Trackmania window client rect to the configured size.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_sac.yaml"))
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    target_width, target_height = expected_window_shape(config_path)
    if args.width is not None:
        target_width = int(args.width)
    if args.height is not None:
        target_height = int(args.height)

    hwnd, geometry = TrackmaniaWindowLocator("Trackmania").locate_tm_window()
    log(f"before client_rect={geometry.width}x{geometry.height}")
    width, height = force_client_rect(hwnd, width=target_width, height=target_height, retries=max(1, args.retries))
    log(f"after client_rect={width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
