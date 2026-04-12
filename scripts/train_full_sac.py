from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def log(message: str) -> None:
    print(f"[train-full-sac] {message}", flush=True)


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
        self._thread = threading.Thread(target=self._run, name="tm20ai-full-progress", daemon=True)
        self._last_logged_progress_step = -self.progress_log_interval
        self._last_checkpoint_count = 0
        self._last_eval_count = 0
        self._last_eval_run_name: str | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._tick()

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

    def _tick(self) -> None:
        summary = self._read_summary()
        if summary is None:
            return

        env_step = int(summary.get("env_step", 0))
        learner_step = int(summary.get("learner_step", 0))
        replay_size = int(summary.get("replay_size", 0))
        episode_count = int(summary.get("episode_count", 0))
        if env_step >= self._last_logged_progress_step + self.progress_log_interval:
            log(
                "progress "
                f"env_step={env_step} learner_step={learner_step} "
                f"replay_size={replay_size} episodes={episode_count}"
            )
            self._last_logged_progress_step = env_step

        pending_eval = summary.get("pending_eval")
        eval_run_name = None if not isinstance(pending_eval, dict) else pending_eval.get("run_name")
        if summary.get("eval_in_flight") and eval_run_name and eval_run_name != self._last_eval_run_name:
            log(
                "eval_started "
                f"run={eval_run_name} env_step={pending_eval.get('env_step')} "
                f"episodes={pending_eval.get('episodes')}"
            )
            self._last_eval_run_name = str(eval_run_name)
        elif not summary.get("eval_in_flight"):
            self._last_eval_run_name = None

        checkpoint_history = list(summary.get("checkpoint_history", []))
        if len(checkpoint_history) > self._last_checkpoint_count:
            for checkpoint in checkpoint_history[self._last_checkpoint_count :]:
                log(
                    "checkpoint_written "
                    f"env_step={checkpoint.get('env_step')} "
                    f"path={checkpoint.get('path')}"
                )
            self._last_checkpoint_count = len(checkpoint_history)

        eval_history = list(summary.get("eval_history", []))
        if len(eval_history) > self._last_eval_count:
            for entry in eval_history[self._last_eval_count :]:
                eval_summary = dict(entry.get("summary", {}))
                log(
                    "eval_finished "
                    f"env_step={entry.get('env_step')} "
                    f"mean_progress={eval_summary.get('mean_final_progress_index')} "
                    f"completion_rate={eval_summary.get('completion_rate')} "
                    f"summary={entry.get('summary_path')}"
                )
            self._last_eval_count = len(eval_history)


def main() -> int:
    from tm20ai.train.learner import SACLearner
    from tm20ai.train.reporting import write_training_report
    from tm20ai.train.worker import worker_entry

    parser = argparse.ArgumentParser(description="Train the FULL-observation SAC baseline with optional BC warm start.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_sac.yaml"))
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
    parser.add_argument("--progress-log-interval", type=int, default=1000)
    parser.add_argument("--max-env-steps", type=int, default=None)
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

    learner = SACLearner(
        config_path=args.config,
        run_name=run_name,
        max_env_steps=args.max_env_steps,
        init_actor=args.init_actor,
        init_mode=args.init_mode,
        demo_root=args.demo_root,
        seed_demos=args.seed_demos,
        eval_episodes_override=args.eval_episodes,
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
        name="tm20ai-sac-worker",
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
    except Exception as exc:  # noqa: BLE001 - entrypoint should log the fatal error
        exit_code = 1
        log(f"ERROR: {exc}")
    finally:
        finalize_error = None
        try:
            final_checkpoint = learner.finalize_run(timeout_seconds=30.0)
        except Exception as exc:  # noqa: BLE001 - finalization errors should still surface in logs
            finalize_error = exc
            exit_code = 1 if exit_code == 0 else exit_code
            log(f"ERROR during finalization: {exc}")
        finally:
            progress_monitor.stop()
        if not learner.clean_shutdown:
            log("Worker did not exit cleanly; final summary recorded an unclean shutdown.")
        if learner.latest_eval_summary is not None:
            log(f"latest_eval_env_step={learner.latest_eval_summary.get('env_step')}")
        log(f"final_checkpoint={final_checkpoint}")
        try:
            report_paths = write_training_report(learner.paths.run_dir)
        except Exception as exc:  # noqa: BLE001 - reporting should not hide train results
            log(f"WARNING: failed to generate training report: {exc}")
        else:
            log(f"report_json={report_paths.json_path}")
            log(f"report_markdown={report_paths.markdown_path}")
        learner.close()
        if finalize_error is not None:
            return exit_code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
