from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.config import load_tm20ai_config
from tm20ai.env import make_env


def log(message: str) -> None:
    print(f"[benchmark-env] {message}", flush=True)


def _avg(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _benchmark_to_json(payload: dict[str, tuple[float | None, float | None]]) -> dict[str, dict[str, float | None]]:
    return {
        key: {"average": average, "average_deviation": deviation}
        for key, (average, deviation) in payload.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the custom rtgym Trackmania environment.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--duration", type=float, default=60.0, help="Benchmark duration in seconds.")
    parser.add_argument(
        "--artifact-dir",
        default=str(ROOT / "artifacts" / "benchmarks"),
        help="Directory that receives the JSON benchmark report.",
    )
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.config, benchmark=True)
    step_durations: list[float] = []
    inference_durations: list[float] = []
    reset_durations: list[float] = []
    timeout_overruns = 0
    cumulative_drift = 0.0
    max_drift = 0.0
    episode_count = 0
    step_count = 0
    loop_start = time.perf_counter()

    try:
        reset_start = time.perf_counter()
        observation, info = env.reset()
        reset_durations.append(time.perf_counter() - reset_start)
        episode_count = 1
        del observation, info

        while (time.perf_counter() - loop_start) < args.duration:
            inference_start = time.perf_counter()
            action = env.default_action.copy()
            inference_durations.append(time.perf_counter() - inference_start)

            step_start = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action)
            step_duration = time.perf_counter() - step_start
            step_durations.append(step_duration)
            step_count += 1

            if step_duration > config.runtime.time_step_duration:
                timeout_overruns += 1
                cumulative_drift += step_duration - config.runtime.time_step_duration

            expected_elapsed = step_count * config.runtime.time_step_duration
            max_drift = max(max_drift, (time.perf_counter() - loop_start) - expected_elapsed)

            del observation, reward, info
            if terminated or truncated:
                reset_start = time.perf_counter()
                observation, info = env.reset()
                reset_durations.append(time.perf_counter() - reset_start)
                episode_count += 1
                del observation, info
    finally:
        benchmarks = env.benchmarks()
        env.close()

    report = {
        "config_path": str(Path(args.config).resolve()),
        "duration_seconds": args.duration,
        "episodes": episode_count,
        "steps": step_count,
        "averages": {
            "target_time_step_duration": config.runtime.time_step_duration,
            "step_duration": _avg(step_durations),
            "inference_duration": _avg(inference_durations),
            "reset_duration": _avg(reset_durations),
            "send_control_duration": benchmarks["send_control_duration"][0],
            "retrieve_observation_duration": benchmarks["get_obs_duration"][0],
            "rtgym_time_step_duration": benchmarks["time_step_duration"][0],
        },
        "timeout_overruns": timeout_overruns,
        "cumulative_drift_seconds": cumulative_drift,
        "max_drift_seconds": max_drift,
        "rtgym_benchmarks": _benchmark_to_json(benchmarks),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"benchmark_{timestamp}.json"
    artifact_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log(f"steps={step_count} episodes={episode_count} timeout_overruns={timeout_overruns}")
    log(f"artifact={artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
