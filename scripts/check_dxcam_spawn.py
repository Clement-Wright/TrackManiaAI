from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _worker(*, mode: str, config_path: str | None) -> None:
    import dxcam

    if mode == "dxcam-create":
        camera = dxcam.create(output_idx=0, max_buffer_len=8, output_color="RGB")
        print(type(camera).__name__, flush=True)
        camera.stop()
        return

    if config_path is None:
        raise RuntimeError(f"--config is required for mode={mode}")

    if mode == "dxcam-capture":
        from tm20ai.capture.dxcam_capture import DXCamCapture
        from tm20ai.config import load_tm20ai_config

        config = load_tm20ai_config(config_path)
        if config.observation.mode == "lidar":
            expected_client_size = (
                config.lidar_observation.window_width,
                config.lidar_observation.window_height,
            )
        else:
            expected_client_size = (
                config.full_observation.window_width,
                config.full_observation.window_height,
            )
        capture = DXCamCapture(config.capture, expected_client_size=expected_client_size)
        try:
            capture.ensure_started()
            print(capture.state.value, flush=True)
        finally:
            capture.close()
        return

    if mode == "interface-bootstrap":
        from tm20ai.env.rt_interface import TM20AIRtInterface

        interface = TM20AIRtInterface(config_path=config_path)
        try:
            result = interface.bootstrap_capture()
            print(result.get("capture_state"), flush=True)
        finally:
            interface.close()
        return

    if mode == "gym-env-bootstrap":
        from tm20ai.env import make_env

        env = make_env(config_path, benchmark=False)
        try:
            result = env.interface.bootstrap_capture()
            print(result.get("capture_state"), flush=True)
        finally:
            env.close()
        return

    raise RuntimeError(f"Unsupported mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dxcam-create", "dxcam-capture", "interface-bootstrap", "gym-env-bootstrap"),
        default="dxcam-create",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_worker,
        kwargs={
            "mode": str(args.mode),
            "config_path": None if args.config is None else str(Path(args.config).resolve()),
        },
        name="dxcam-spawn-check",
    )
    process.start()
    process.join()
    print(f"exitcode={process.exitcode}", flush=True)
    return int(process.exitcode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
