import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import win32con
import win32gui
from tmrl import get_environment


APP_ID = "2225070"
TRACKMANIA_EXE = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Trackmania\Trackmania.exe")
USER_PROFILE = Path(os.environ["USERPROFILE"])
TMRL_CONFIG_PATH = USER_PROFILE / "TmrlData" / "config" / "config.json"
OPENPLANET_PLUGIN = USER_PROFILE / "OpenplanetNext" / "Plugins" / "TMRL_GrabData.op"
INTERFACE_BY_ENV = {
    "lidar": "TM20LIDAR",
    "full": "TM20FULL",
}


def log(message: str) -> None:
    print(f"[phase1] {message}", flush=True)


def load_tmrl_config() -> dict[str, Any]:
    if not TMRL_CONFIG_PATH.exists():
        raise RuntimeError(
            f"TMRL config is missing at {TMRL_CONFIG_PATH}. "
            "Run scripts\\bootstrap_phase1.ps1 and scripts\\apply_tmrl_config.py first."
        )
    with TMRL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_action(action_text: str) -> np.ndarray:
    parts = [part.strip() for part in action_text.split(",")]
    if len(parts) != 3:
        raise ValueError("Action must have exactly three comma-separated values: gas, brake, steer")
    values = np.array([float(part) for part in parts], dtype=np.float32)
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("Action values must be between -1.0 and 1.0")
    return values


def describe_observation(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [describe_observation(item) for item in value]
    if isinstance(value, dict):
        return {key: describe_observation(item) for key, item in value.items()}
    if np.isscalar(value):
        return {"type": type(value).__name__, "value": value.item() if hasattr(value, "item") else value}
    return {"type": type(value).__name__}


def enumerate_trackmania_windows() -> list[int]:
    hwnds: list[int] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if "Trackmania" in title:
            hwnds.append(hwnd)
        return True

    win32gui.EnumWindows(callback, 0)
    return hwnds


def get_trackmania_window() -> int | None:
    hwnds = enumerate_trackmania_windows()
    return hwnds[0] if hwnds else None


def launch_trackmania() -> None:
    try:
        os.startfile(f"steam://rungameid/{APP_ID}")  # type: ignore[attr-defined]
        log("Launched Trackmania via Steam URI.")
    except OSError as steam_error:
        log(f"Steam launch failed ({steam_error}); falling back to Trackmania.exe.")
        if not TRACKMANIA_EXE.exists():
            raise RuntimeError(f"Trackmania executable not found at {TRACKMANIA_EXE}") from steam_error
        subprocess.Popen([str(TRACKMANIA_EXE)], close_fds=True)


def wait_for_trackmania_window(timeout_seconds: float) -> int:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        hwnd = get_trackmania_window()
        if hwnd is not None:
            return hwnd
        time.sleep(1.0)
    raise RuntimeError("Timed out waiting for a Trackmania window.")


def focus_and_resize_window(hwnd: int, width: int, height: int, retries: int = 5) -> None:
    user32 = ctypes.windll.user32
    user32.AllowSetForegroundWindow(-1)

    last_rect: tuple[int, int, int, int] | None = None
    foreground_matched = False
    for attempt in range(1, retries + 1):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.MoveWindow(hwnd, 0, 0, width, height, True)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.4)

        foreground = win32gui.GetForegroundWindow()
        rect = win32gui.GetWindowRect(hwnd)
        last_rect = rect
        current_width = rect[2] - rect[0]
        current_height = rect[3] - rect[1]
        foreground_matched = foreground == hwnd

        width_ok = abs(current_width - width) <= 1
        height_ok = abs(current_height - height) <= 1
        if rect[0] == 0 and rect[1] == 0 and width_ok and height_ok:
            if foreground_matched:
                log(f"Trackmania window focused and resized on attempt {attempt}.")
            else:
                log(
                    "Trackmania window resized correctly, but Windows did not keep it in the foreground. "
                    "Continuing with a warning."
                )
            return

    rect_text = str(last_rect) if last_rect is not None else "unknown"
    if not foreground_matched:
        raise RuntimeError(
            f"Repeated resize failure for Trackmania window; latest rect was {rect_text}. "
            f"Expected top-left {width}x{height}."
        )
    raise RuntimeError(
        f"Repeated focus or resize failure for Trackmania window; latest rect was {rect_text}. "
        f"Expected top-left {width}x{height}."
    )


def run_smoke_loop(env_name: str, seconds: float, action: np.ndarray) -> None:
    tmrl_config = load_tmrl_config()
    env_config = tmrl_config["ENV"]
    expected_interface = INTERFACE_BY_ENV[env_name]
    actual_interface = env_config["RTGYM_INTERFACE"]
    if actual_interface != expected_interface:
        raise RuntimeError(
            f"TMRL config currently targets {actual_interface}, expected {expected_interface}. "
            f"Run scripts\\apply_tmrl_config.py --env {env_name} first."
        )

    if not OPENPLANET_PLUGIN.exists():
        raise RuntimeError(
            f"Openplanet plugin missing at {OPENPLANET_PLUGIN}. "
            "Run scripts\\bootstrap_phase1.ps1 and confirm Openplanet is installed."
        )

    hwnd = get_trackmania_window()
    if hwnd is None:
        launch_trackmania()
        hwnd = wait_for_trackmania_window(timeout_seconds=60.0)
    else:
        log("Reusing the existing Trackmania window.")

    width = int(env_config["WINDOW_WIDTH"])
    height = int(env_config["WINDOW_HEIGHT"])
    focus_and_resize_window(hwnd, width=width, height=height)
    time.sleep(1.0)

    max_steps = max(1, int(round(seconds / 0.05)))
    env = None
    try:
        log(f"Creating TMRL environment for {actual_interface}.")
        env = get_environment()

        log("Resetting environment.")
        obs, info = env.reset()
        if obs is None:
            raise RuntimeError("Environment returned no observation on the first reset.")
        log(f"First reset observation: {json.dumps(describe_observation(obs))}")
        log(f"First reset info keys: {sorted(info.keys()) if isinstance(info, dict) else type(info).__name__}")

        for step_index in range(max_steps):
            obs, reward, terminated, truncated, info = env.step(action)
            if obs is None:
                raise RuntimeError(f"Environment returned no observation at step {step_index + 1}.")
            log(
                "step="
                f"{step_index + 1}/{max_steps} "
                f"reward={reward:.6f} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )
            if terminated or truncated:
                log("Environment signaled termination; stopping the action loop early.")
                break

        log("Resetting environment again to confirm loop stability.")
        obs, info = env.reset()
        if obs is None:
            raise RuntimeError("Environment returned no observation on the second reset.")
        log(f"Second reset observation: {json.dumps(describe_observation(obs))}")
        log("Smoke loop completed successfully.")
    except ConnectionRefusedError as error:
        raise RuntimeError(
            "TMRL connection was refused. Reload the TMRL Grab Data plugin in Openplanet and retry."
        ) from error
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a plain TMRL smoke loop for Trackmania 2020.")
    parser.add_argument("--env", choices=sorted(INTERFACE_BY_ENV), required=True)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--action", default="0.8,0.0,0.0")
    args = parser.parse_args()

    action = parse_action(args.action)
    log(f"Using fixed analog action: {action.tolist()}")
    run_smoke_loop(env_name=args.env, seconds=args.seconds, action=action)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"ERROR: {error}")
        raise SystemExit(1)
