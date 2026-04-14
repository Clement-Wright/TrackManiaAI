from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.bridge import BridgeClient
from tm20ai.bridge.messages import TelemetryFrame
from tm20ai.config import load_tm20ai_config


def log(message: str) -> None:
    print(f"[movement-monitor] {message}", flush=True)


def _distance(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    dz = float(a[2] - b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _parse_env_step(line: str) -> int | None:
    marker = "env_step="
    start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = start
    while end < len(line) and line[end].isdigit():
        end += 1
    if end == start:
        return None
    return int(line[start:end])


@dataclass(slots=True)
class StallEvent:
    run_id: str
    frame_id: int
    race_time_ms: int
    timestamp_ns: int
    wall_time: float
    speed_kmh: float
    pos_xyz: tuple[float, float, float] | None
    approx_env_step: int | None
    duration_seconds: float


@dataclass(slots=True)
class RunSummary:
    run_id: str
    first_frame_id: int
    last_frame_id: int
    first_race_time_ms: int
    last_race_time_ms: int
    max_speed_kmh: float
    max_distance_from_start_m: float
    movement_started: bool
    movement_start_frame_id: int | None
    movement_start_race_time_ms: int | None
    first_stall: StallEvent | None
    stall_count: int
    finished: bool


class JsonlEventWriter:
    def __init__(self) -> None:
        self._path: Path | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    def set_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if self._path is None:
            return
        row = {
            "timestamp": time.time(),
            "event": event,
            "payload": payload,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


class MovementMonitor:
    def __init__(
        self,
        *,
        stall_speed_kmh: float,
        stall_duration_seconds: float,
        stall_position_epsilon_m: float,
        movement_speed_kmh: float,
        movement_distance_m: float,
    ) -> None:
        self.stall_speed_kmh = float(stall_speed_kmh)
        self.stall_duration_seconds = float(stall_duration_seconds)
        self.stall_position_epsilon_m = float(stall_position_epsilon_m)
        self.movement_speed_kmh = float(movement_speed_kmh)
        self.movement_distance_m = float(movement_distance_m)
        self.current_run_id: str | None = None
        self._run_start_pos: tuple[float, float, float] | None = None
        self._movement_started = False
        self._movement_start_frame_id: int | None = None
        self._movement_start_race_time_ms: int | None = None
        self._candidate_stall_frame_id: int | None = None
        self._candidate_stall_race_time_ms: int | None = None
        self._candidate_stall_timestamp_ns: int | None = None
        self._candidate_stall_wall_time: float | None = None
        self._candidate_stall_pos: tuple[float, float, float] | None = None
        self._run_first_frame_id: int | None = None
        self._run_first_race_time_ms: int | None = None
        self._last_frame_id: int | None = None
        self._last_race_time_ms: int | None = None
        self._max_speed_kmh = 0.0
        self._max_distance_from_start_m = 0.0
        self._finished = False
        self._stalls_in_run: list[StallEvent] = []
        self._summaries: list[RunSummary] = []
        self._first_stall_overall: StallEvent | None = None

    @property
    def run_summaries(self) -> list[RunSummary]:
        return list(self._summaries)

    @property
    def first_stall_overall(self) -> StallEvent | None:
        return self._first_stall_overall

    def observe(
        self,
        frame: TelemetryFrame,
        *,
        wall_time: float,
        approx_env_step: int | None,
        event_writer: JsonlEventWriter,
    ) -> None:
        if not frame.run_id:
            return
        if self.current_run_id != frame.run_id:
            self._finalize_current_run(event_writer=event_writer)
            self.current_run_id = frame.run_id
            self._run_start_pos = frame.pos_xyz
            self._movement_started = False
            self._movement_start_frame_id = None
            self._movement_start_race_time_ms = None
            self._candidate_stall_frame_id = None
            self._candidate_stall_race_time_ms = None
            self._candidate_stall_timestamp_ns = None
            self._candidate_stall_wall_time = None
            self._candidate_stall_pos = None
            self._run_first_frame_id = frame.frame_id
            self._run_first_race_time_ms = frame.race_time_ms
            self._last_frame_id = frame.frame_id
            self._last_race_time_ms = frame.race_time_ms
            self._max_speed_kmh = float(frame.speed_kmh)
            self._max_distance_from_start_m = 0.0
            self._finished = False
            self._stalls_in_run = []
            event_writer.write(
                "run_started",
                {
                    "run_id": frame.run_id,
                    "frame_id": frame.frame_id,
                    "race_time_ms": frame.race_time_ms,
                    "approx_env_step": approx_env_step,
                    "pos_xyz": frame.pos_xyz,
                },
            )
            log(
                f"run_started run_id={frame.run_id} frame_id={frame.frame_id} "
                f"race_time_ms={frame.race_time_ms} approx_env_step={approx_env_step}"
            )

        self._last_frame_id = frame.frame_id
        self._last_race_time_ms = frame.race_time_ms
        self._finished = bool(frame.finished)
        self._max_speed_kmh = max(self._max_speed_kmh, float(frame.speed_kmh))
        dist_from_start = _distance(self._run_start_pos, frame.pos_xyz)
        if dist_from_start is not None:
            self._max_distance_from_start_m = max(self._max_distance_from_start_m, dist_from_start)

        if not self._movement_started:
            started_moving = float(frame.speed_kmh) >= self.movement_speed_kmh
            if dist_from_start is not None and dist_from_start >= self.movement_distance_m:
                started_moving = True
            if started_moving:
                self._movement_started = True
                self._movement_start_frame_id = frame.frame_id
                self._movement_start_race_time_ms = frame.race_time_ms
                event_writer.write(
                    "movement_started",
                    {
                        "run_id": frame.run_id,
                        "frame_id": frame.frame_id,
                        "race_time_ms": frame.race_time_ms,
                        "speed_kmh": float(frame.speed_kmh),
                        "distance_from_start_m": dist_from_start,
                        "approx_env_step": approx_env_step,
                    },
                )
                log(
                    f"movement_started run_id={frame.run_id} frame_id={frame.frame_id} "
                    f"race_time_ms={frame.race_time_ms} speed_kmh={frame.speed_kmh:.2f}"
                )

        if not self._movement_started or frame.finished:
            self._clear_stall_candidate()
            return

        if float(frame.speed_kmh) > self.stall_speed_kmh:
            self._clear_stall_candidate()
            return

        if self._candidate_stall_frame_id is None:
            self._candidate_stall_frame_id = frame.frame_id
            self._candidate_stall_race_time_ms = frame.race_time_ms
            self._candidate_stall_timestamp_ns = frame.timestamp_ns
            self._candidate_stall_wall_time = wall_time
            self._candidate_stall_pos = frame.pos_xyz
            return

        distance_in_candidate = _distance(self._candidate_stall_pos, frame.pos_xyz)
        if distance_in_candidate is not None and distance_in_candidate > self.stall_position_epsilon_m:
            self._candidate_stall_frame_id = frame.frame_id
            self._candidate_stall_race_time_ms = frame.race_time_ms
            self._candidate_stall_timestamp_ns = frame.timestamp_ns
            self._candidate_stall_wall_time = wall_time
            self._candidate_stall_pos = frame.pos_xyz
            return

        assert self._candidate_stall_timestamp_ns is not None
        assert self._candidate_stall_frame_id is not None
        assert self._candidate_stall_race_time_ms is not None
        assert self._candidate_stall_wall_time is not None
        stall_duration = max(0.0, (frame.timestamp_ns - self._candidate_stall_timestamp_ns) / 1_000_000_000.0)
        already_reported = any(event.frame_id == self._candidate_stall_frame_id for event in self._stalls_in_run)
        if not already_reported and stall_duration >= self.stall_duration_seconds:
            event = StallEvent(
                run_id=frame.run_id,
                frame_id=self._candidate_stall_frame_id,
                race_time_ms=self._candidate_stall_race_time_ms,
                timestamp_ns=self._candidate_stall_timestamp_ns,
                wall_time=self._candidate_stall_wall_time,
                speed_kmh=float(frame.speed_kmh),
                pos_xyz=self._candidate_stall_pos,
                approx_env_step=approx_env_step,
                duration_seconds=stall_duration,
            )
            self._stalls_in_run.append(event)
            if self._first_stall_overall is None:
                self._first_stall_overall = event
            payload = asdict(event)
            event_writer.write("stall_detected", payload)
            log(
                f"stall_detected run_id={event.run_id} frame_id={event.frame_id} "
                f"race_time_ms={event.race_time_ms} approx_env_step={event.approx_env_step} "
                f"duration_s={event.duration_seconds:.2f}"
            )

    def finish(self, *, event_writer: JsonlEventWriter) -> None:
        self._finalize_current_run(event_writer=event_writer)

    def _clear_stall_candidate(self) -> None:
        self._candidate_stall_frame_id = None
        self._candidate_stall_race_time_ms = None
        self._candidate_stall_timestamp_ns = None
        self._candidate_stall_wall_time = None
        self._candidate_stall_pos = None

    def _finalize_current_run(self, *, event_writer: JsonlEventWriter) -> None:
        if self.current_run_id is None:
            return
        summary = RunSummary(
            run_id=self.current_run_id,
            first_frame_id=int(self._run_first_frame_id or 0),
            last_frame_id=int(self._last_frame_id or 0),
            first_race_time_ms=int(self._run_first_race_time_ms or 0),
            last_race_time_ms=int(self._last_race_time_ms or 0),
            max_speed_kmh=float(self._max_speed_kmh),
            max_distance_from_start_m=float(self._max_distance_from_start_m),
            movement_started=bool(self._movement_started),
            movement_start_frame_id=self._movement_start_frame_id,
            movement_start_race_time_ms=self._movement_start_race_time_ms,
            first_stall=self._stalls_in_run[0] if self._stalls_in_run else None,
            stall_count=len(self._stalls_in_run),
            finished=self._finished,
        )
        self._summaries.append(summary)
        event_writer.write(
            "run_finished",
            {
                "run_id": summary.run_id,
                "movement_started": summary.movement_started,
                "movement_start_frame_id": summary.movement_start_frame_id,
                "movement_start_race_time_ms": summary.movement_start_race_time_ms,
                "max_speed_kmh": summary.max_speed_kmh,
                "max_distance_from_start_m": summary.max_distance_from_start_m,
                "stall_count": summary.stall_count,
                "first_stall": None if summary.first_stall is None else asdict(summary.first_stall),
                "finished": summary.finished,
            },
        )
        self.current_run_id = None
        self._clear_stall_candidate()


class TrainOutputReader:
    def __init__(self) -> None:
        self.run_dir: Path | None = None
        self.summary_path: Path | None = None
        self.latest_env_step: int | None = None
        self.final_checkpoint: str | None = None
        self.report_json: str | None = None
        self.report_markdown: str | None = None
        self.lines: list[str] = []

    def consume_stream(self, stream: TextIO) -> None:
        for raw_line in stream:
            line = raw_line.rstrip()
            self.lines.append(line)
            print(line, flush=True)
            if " run_dir=" in line:
                run_dir_text = line.split(" run_dir=", 1)[1].strip()
                self.run_dir = Path(run_dir_text)
                self.summary_path = self.run_dir / "summary.json"
            env_step = _parse_env_step(line)
            if env_step is not None:
                self.latest_env_step = env_step
            if " final_checkpoint=" in line:
                self.final_checkpoint = line.split(" final_checkpoint=", 1)[1].strip()
            if " report_json=" in line:
                self.report_json = line.split(" report_json=", 1)[1].strip()
            if " report_markdown=" in line:
                self.report_markdown = line.split(" report_markdown=", 1)[1].strip()


def _latest_env_step(reader: TrainOutputReader) -> int | None:
    if reader.summary_path is not None and reader.summary_path.exists():
        try:
            payload = json.loads(reader.summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            env_step = payload.get("env_step")
            if isinstance(env_step, int):
                return env_step
    return reader.latest_env_step


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run FULL SAC training while monitoring bridge telemetry for the first sustained no-movement stall."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_sac.yaml"))
    parser.add_argument("--max-env-steps", type=int, default=6000)
    parser.add_argument("--progress-log-interval", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--stall-speed-kmh", type=float, default=2.0)
    parser.add_argument("--stall-duration-seconds", type=float, default=1.0)
    parser.add_argument("--stall-position-epsilon-m", type=float, default=0.15)
    parser.add_argument("--movement-speed-kmh", type=float, default=5.0)
    parser.add_argument("--movement-distance-m", type=float, default=0.5)
    parser.add_argument("--post-exit-tail-seconds", type=float, default=2.0)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_tm20ai_config(config_path)
    bridge_client = BridgeClient(config.bridge)
    reader = TrainOutputReader()
    writer = JsonlEventWriter()
    monitor = MovementMonitor(
        stall_speed_kmh=args.stall_speed_kmh,
        stall_duration_seconds=args.stall_duration_seconds,
        stall_position_epsilon_m=args.stall_position_epsilon_m,
        movement_speed_kmh=args.movement_speed_kmh,
        movement_distance_m=args.movement_distance_m,
    )

    train_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_full_sac.py"),
        "--config",
        str(config_path),
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-env-steps",
        str(args.max_env_steps),
        "--progress-log-interval",
        str(args.progress_log_interval),
    ]
    log(f"train_command={' '.join(train_command)}")

    process: subprocess.Popen[str] | None = None
    reader_thread: threading.Thread | None = None
    exit_deadline: float | None = None
    summary_output_path: Path | None = None
    fallback_jsonl = ROOT / "artifacts" / "movement_monitor" / f"movement_monitor_{int(time.time())}.jsonl"
    writer.set_path(fallback_jsonl)
    try:
        bridge_client.start()
        initial_frame = bridge_client.wait_for_frame(timeout=config.bridge.initial_frame_timeout)
        writer.write(
            "initial_frame",
            {
                "run_id": initial_frame.run_id,
                "frame_id": initial_frame.frame_id,
                "race_time_ms": initial_frame.race_time_ms,
                "speed_kmh": float(initial_frame.speed_kmh),
            },
        )
        process = subprocess.Popen(
            train_command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        reader_thread = threading.Thread(
            target=reader.consume_stream,
            args=(process.stdout,),
            name="movement-monitor-train-stdout",
            daemon=True,
        )
        reader_thread.start()

        last_frame_id = initial_frame.frame_id
        while True:
            if reader.run_dir is not None and writer.path != reader.run_dir / "movement_monitor.jsonl":
                writer.set_path(reader.run_dir / "movement_monitor.jsonl")
                summary_output_path = reader.run_dir / "movement_monitor_summary.json"
            approx_env_step = _latest_env_step(reader)
            try:
                frame = bridge_client.wait_for_frame(after_frame_id=last_frame_id, timeout=0.25)
            except TimeoutError:
                if process.poll() is not None:
                    if exit_deadline is None:
                        exit_deadline = time.monotonic() + max(0.0, args.post_exit_tail_seconds)
                    elif time.monotonic() >= exit_deadline:
                        break
                continue
            last_frame_id = frame.frame_id
            monitor.observe(
                frame,
                wall_time=time.time(),
                approx_env_step=approx_env_step,
                event_writer=writer,
            )
            if process.poll() is not None and exit_deadline is None:
                exit_deadline = time.monotonic() + max(0.0, args.post_exit_tail_seconds)
        exit_code = int(process.wait())
    finally:
        monitor.finish(event_writer=writer)
        bridge_client.close()
        if reader_thread is not None:
            reader_thread.join(timeout=2.0)

    run_summaries = [asdict(item) for item in monitor.run_summaries]
    movement_runs = [item for item in run_summaries if bool(item["movement_started"])]
    no_movement_runs = [item for item in run_summaries if not bool(item["movement_started"])]
    summary_payload = {
        "config_path": str(config_path),
        "train_command": train_command,
        "run_dir": None if reader.run_dir is None else str(reader.run_dir),
        "summary_json": None if reader.summary_path is None else str(reader.summary_path),
        "event_log_path": None if writer.path is None else str(writer.path),
        "train_exit_code": exit_code,
        "max_env_steps": args.max_env_steps,
        "progress_log_interval": args.progress_log_interval,
        "runs_observed": len(run_summaries),
        "runs_with_motion": len(movement_runs),
        "runs_without_motion": len(no_movement_runs),
        "first_stall_overall": None if monitor.first_stall_overall is None else asdict(monitor.first_stall_overall),
        "first_run_without_motion": no_movement_runs[0] if no_movement_runs else None,
        "run_summaries": run_summaries,
        "final_checkpoint": reader.final_checkpoint,
        "report_json": reader.report_json,
        "report_markdown": reader.report_markdown,
    }
    if summary_output_path is None:
        summary_output_path = ROOT / "artifacts" / "movement_monitor" / f"movement_monitor_{int(time.time())}.json"
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    log(f"movement_summary={summary_output_path}")
    if monitor.first_stall_overall is not None:
        first = monitor.first_stall_overall
        log(
            f"first_stall_overall run_id={first.run_id} frame_id={first.frame_id} "
            f"race_time_ms={first.race_time_ms} approx_env_step={first.approx_env_step} "
            f"duration_s={first.duration_seconds:.2f}"
        )
    elif no_movement_runs:
        first_run = no_movement_runs[0]
        log(
            f"no_motion_run run_id={first_run['run_id']} "
            f"max_speed_kmh={first_run['max_speed_kmh']:.2f} "
            f"max_distance_from_start_m={first_run['max_distance_from_start_m']:.3f}"
        )
    else:
        log("No sustained stall was detected during the monitored run.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
