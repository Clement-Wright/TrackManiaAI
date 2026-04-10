from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.bridge import BridgeClient
from tm20ai.config import load_tm20ai_config
from tm20ai.env.trajectory import (
    build_runtime_trajectory_from_raw_lap,
    raw_lap_path_for_map,
    runtime_trajectory_path_for_map,
    save_raw_lap_records,
    save_runtime_trajectory,
)


def log(message: str) -> None:
    print(f"[record-reward] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one manual reward lap and build the runtime trajectory artifact.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    sample_period = config.runtime.time_step_duration

    with BridgeClient(config.bridge) as client:
        initial_frame = client.wait_for_frame(timeout=config.bridge.initial_frame_timeout)
        state = client.race_state(timeout=config.bridge.command_timeout)
        if state.payload.get("race_state") != "start_line":
            log("ERROR: reward recording must start with the car at the map start line.")
            return 1

        map_uid = str(state.payload["map_uid"])
        run_id = str(state.payload["run_id"])
        raw_path = raw_lap_path_for_map(map_uid)
        trajectory_path = runtime_trajectory_path_for_map(map_uid, config.reward.spacing_meters)
        log(f"Recording one manual lap for map_uid={map_uid!r}. Drive now.")

        response = client.set_recording_mode(True, timeout=config.bridge.command_timeout)
        if not response.success:
            log(f"ERROR: failed to enable bridge recording mode: {response.message}")
            return 1

        records: list[dict[str, object]] = []
        next_sample_at = time.monotonic()
        try:
            while True:
                sleep_for = next_sample_at - time.monotonic()
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
                next_sample_at += sample_period

                frame = client.get_latest_frame() or client.wait_for_frame(timeout=config.bridge.initial_frame_timeout)
                if frame.map_uid != map_uid:
                    log("ERROR: map changed while recording the reward lap.")
                    return 1
                if frame.run_id != run_id and frame.frame_id > initial_frame.frame_id:
                    log("ERROR: run_id changed during reward recording; aborting the lap capture.")
                    return 1
                if frame.pos_xyz is None:
                    log("ERROR: telemetry frame is missing pos_xyz; reward recording cannot continue.")
                    return 1

                records.append(
                    {
                        "session_id": frame.session_id,
                        "run_id": frame.run_id,
                        "frame_id": frame.frame_id,
                        "timestamp_ns": frame.timestamp_ns,
                        "map_uid": frame.map_uid,
                        "race_time_ms": frame.race_time_ms,
                        "x": frame.pos_xyz[0],
                        "y": frame.pos_xyz[1],
                        "z": frame.pos_xyz[2],
                        "finished": frame.finished,
                    }
                )

                if frame.finished:
                    break
                if frame.terminal_reason in {"outside_active_race", "map_changed"}:
                    log(f"ERROR: lap recording terminated early with terminal_reason={frame.terminal_reason!r}.")
                    return 1
        finally:
            try:
                client.set_recording_mode(False, timeout=config.bridge.command_timeout)
            except Exception as exc:  # noqa: BLE001 - cleanup failure should not hide the real recording result
                log(f"WARNING: failed to disable bridge recording mode cleanly: {exc}")

    save_raw_lap_records(records, raw_path)
    trajectory = build_runtime_trajectory_from_raw_lap(raw_path, config.reward.spacing_meters)
    save_runtime_trajectory(trajectory, trajectory_path)
    log(f"Saved raw lap: {raw_path}")
    log(f"Saved runtime trajectory: {trajectory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
