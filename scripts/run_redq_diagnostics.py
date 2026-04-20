from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def log(message: str) -> None:
    print(f"[run-redq-diagnostics] {message}", flush=True)


class SummaryProgressMonitor:
    def __init__(
        self,
        *,
        summary_path: Path,
        progress_log_interval: int,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.summary_path = summary_path
        self.progress_log_interval = max(1, int(progress_log_interval))
        self.poll_interval_seconds = max(0.25, float(poll_interval_seconds))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tm20ai-redq-diagnostics-progress", daemon=True)
        self._last_logged_progress_step = -self.progress_log_interval

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._tick(force_progress_log=True)

    def _run(self) -> None:
        while not self._stop_event.wait(self.poll_interval_seconds):
            self._tick()

    def _read_summary(self) -> dict | None:
        if not self.summary_path.exists():
            return None
        try:
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _tick(self, *, force_progress_log: bool = False) -> None:
        summary = self._read_summary()
        if summary is None:
            return
        env_step = int(summary.get("env_step", 0))
        learner_step = int(summary.get("learner_step", 0))
        actor_step = int(summary.get("actor_step", 0))
        if force_progress_log or env_step >= self._last_logged_progress_step + self.progress_log_interval:
            actor_sync_profile = dict(summary.get("actor_sync_profile", {}))
            runtime_profile = dict(summary.get("runtime_profile", {}))
            bottleneck = dict(runtime_profile.get("bottleneck_verdict", {}))
            log(
                f"{'progress_final' if force_progress_log else 'progress'} "
                f"env_step={env_step} learner_step={learner_step} actor_step={actor_step} "
                f"policy_control_fraction={actor_sync_profile.get('policy_control_fraction')} "
                f"bottleneck={bottleneck.get('label')}"
            )
            self._last_logged_progress_step = env_step


def main() -> int:
    from tm20ai.train.learner import REDQLearner
    from tm20ai.train.reporting import write_training_report
    from tm20ai.train.worker import worker_entry

    parser = argparse.ArgumentParser(description="Run a live REDQ diagnostic training pass with detailed profiling.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq_diagnostic.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-actor", default=None)
    parser.add_argument(
        "--init-mode",
        choices=("scratch", "actor_only", "actor_plus_critic_encoders"),
        default="scratch",
    )
    parser.add_argument("--demo-root", default=None)
    parser.add_argument("--seed-demos", default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=500)
    parser.add_argument("--max-env-steps", type=int, default=5000)
    parser.add_argument("--max-wall-clock-minutes", type=float, default=None)
    args = parser.parse_args()

    if args.resume is not None and (
        args.init_actor is not None
        or args.init_mode != "scratch"
        or args.seed_demos is not None
        or args.demo_root is not None
    ):
        parser.error("--resume cannot be combined with BC warm-start or demo-root/replay-seeding options.")

    multiprocessing.freeze_support()
    run_name = args.run_name
    if run_name is None and args.resume is not None:
        run_name = Path(args.resume).resolve().parents[1].name

    learner = REDQLearner(
        config_path=args.config,
        run_name=run_name,
        max_env_steps=args.max_env_steps,
        init_actor=args.init_actor,
        init_mode=args.init_mode,
        demo_root=args.demo_root,
        seed_demos=args.seed_demos,
        eval_episodes_override=args.eval_episodes,
        diagnostics_enabled=True,
        detailed_cuda_timing=True,
        max_wall_clock_minutes=args.max_wall_clock_minutes,
    )
    if args.resume is not None:
        learner.load_checkpoint(args.resume)

    progress_monitor = SummaryProgressMonitor(
        summary_path=learner.paths.summary_json,
        progress_log_interval=args.progress_log_interval,
    )

    ctx = multiprocessing.get_context("spawn")
    command_queue = ctx.Queue(maxsize=learner.config.train.queue_capacity)
    output_queue = ctx.Queue(maxsize=learner.config.train.queue_capacity)
    eval_result_queue = ctx.Queue(maxsize=learner.config.train.queue_capacity)
    shutdown_event = ctx.Event()
    worker_done_event = ctx.Event()
    worker = ctx.Process(
        target=worker_entry,
        args=(
            str(Path(args.config).resolve()),
            command_queue,
            output_queue,
            eval_result_queue,
            shutdown_event,
            worker_done_event,
            str(learner.paths.run_dir / "worker_bootstrap.log"),
            learner.max_env_steps,
        ),
        name="tm20ai-redq-diagnostics-worker",
    )
    learner.attach_worker(
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
        worker_process=worker,
    )

    exit_code = 0
    final_checkpoint = None
    progress_monitor.start()
    worker.start()
    log(f"run_dir={learner.paths.run_dir}")
    try:
        learner.run()
    except KeyboardInterrupt:
        log("KeyboardInterrupt received, requesting graceful shutdown.")
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        log(f"ERROR: {exc}")
    finally:
        try:
            final_checkpoint = learner.finalize_run(timeout_seconds=30.0)
        except Exception as exc:  # noqa: BLE001
            exit_code = 1 if exit_code == 0 else exit_code
            log(f"ERROR during finalization: {exc}")
        finally:
            progress_monitor.stop()

        log(f"final_checkpoint={final_checkpoint}")
        try:
            report_paths = write_training_report(learner.paths.run_dir)
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: failed to generate training report: {exc}")
        else:
            report = json.loads(report_paths.json_path.read_text(encoding="utf-8"))
            bottleneck = dict(report.get("runtime_profile", {})).get("bottleneck_verdict", {})
            actor_sync = dict(report.get("actor_sync_profile", {}))
            log(f"report_json={report_paths.json_path}")
            log(f"report_markdown={report_paths.markdown_path}")
            log(
                f"diagnostics bottleneck={dict(bottleneck).get('label')} "
                f"time_to_first_policy_control_window_seconds={actor_sync.get('time_to_first_policy_control_window_seconds')}"
            )
        learner.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
