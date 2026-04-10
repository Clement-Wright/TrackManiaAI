from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.config import load_tm20ai_config
from tm20ai.data.parquet_writer import ensure_directory, resolve_artifact_root, timestamp_tag
from tm20ai.env import make_env
from tm20ai.train.metrics import ActiveStepBenchmark


def log(message: str) -> None:
    print(f"[benchmark-env] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the custom rtgym Trackmania environment.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--duration", type=float, default=60.0, help="Benchmark duration in seconds.")
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional directory override for the JSON benchmark report. Defaults to <artifacts.root>/benchmarks.",
    )
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    default_artifact_dir = resolve_artifact_root(config) / "benchmarks"
    artifact_dir = ensure_directory(Path(args.artifact_dir).resolve()) if args.artifact_dir else ensure_directory(default_artifact_dir)

    env = make_env(args.config, benchmark=True)
    tracker = ActiveStepBenchmark(step_dt_seconds=config.runtime.time_step_duration)
    episode_count = 0
    loop_start = time.perf_counter()
    tracker.reanchor(loop_start)

    try:
        reset_start = time.perf_counter()
        observation, info = env.reset(seed=config.eval.seed_base)
        reset_end = time.perf_counter()
        tracker.record_reset(reset_end - reset_start, reanchor_time=reset_end)
        episode_count = 1
        del observation, info

        while (time.perf_counter() - loop_start) < args.duration:
            action = env.default_action.copy()
            step_start = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action)
            step_end = time.perf_counter()
            tracker.record_step(end_time=step_end, duration_seconds=step_end - step_start)

            del observation, reward, info
            if terminated or truncated:
                reset_start = time.perf_counter()
                observation, info = env.reset(seed=config.eval.seed_base + episode_count)
                reset_end = time.perf_counter()
                tracker.record_reset(reset_end - reset_start, reanchor_time=reset_end)
                episode_count += 1
                del observation, info
    finally:
        benchmarks = env.benchmarks()
        env.close()

    tm20ai_benchmarks = benchmarks["tm20ai"]
    rtgym_benchmarks = benchmarks["rtgym"]
    report = {
        "config_path": str(Path(args.config).resolve()),
        "duration_seconds": args.duration,
        **tracker.to_report(
            episodes=episode_count,
            wall_clock_total_seconds=time.perf_counter() - loop_start,
            avg_obs_retrieval_seconds=float(tm20ai_benchmarks["avg_obs_retrieval_seconds"]),
            avg_send_control_seconds=float(rtgym_benchmarks["send_control_duration"][0] or 0.0),
            avg_reward_compute_seconds=float(tm20ai_benchmarks["avg_reward_compute_seconds"]),
            avg_preprocess_seconds=float(tm20ai_benchmarks["avg_preprocess_seconds"]),
            raw_rtgym_benchmarks=rtgym_benchmarks,
        ),
    }

    artifact_path = artifact_dir / f"benchmark_{timestamp_tag()}.json"
    artifact_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log(f"active_steps={report['active_steps']} episodes={report['episodes']} reset_count={report['reset_count']}")
    log(f"artifact={artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
