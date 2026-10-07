from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def log(message: str) -> None:
    print(f"[train-lidar-sac] {message}", flush=True)


def main() -> int:
    from tm20ai.config import load_tm20ai_config
    from tm20ai.data.parquet_writer import resolve_artifact_root
    from tm20ai.train.artifact_retention import enforce_storage_preflight, format_bytes
    from tm20ai.train.learner import SACLearner
    from tm20ai.train.worker import worker_entry

    parser = argparse.ArgumentParser(description="Train the first single-machine LIDAR SAC baseline.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "lidar_sac.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=150.0)
    parser.add_argument("--max-artifact-gb", type=float, default=150.0)
    parser.add_argument("--disable-storage-preflight", action="store_true")
    args = parser.parse_args()

    long_run = args.max_env_steps is None or args.max_env_steps >= 10000
    if long_run and not args.disable_storage_preflight:
        loaded_config = load_tm20ai_config(args.config)
        max_artifact_gb = None if args.max_artifact_gb <= 0.0 else args.max_artifact_gb
        storage_report = enforce_storage_preflight(
            resolve_artifact_root(loaded_config),
            min_free_gb=args.min_free_gb,
            max_artifact_gb=max_artifact_gb,
        )
        log(
            "storage_preflight_ok "
            f"free={format_bytes(storage_report.free_bytes)} "
            f"artifact_root_size={format_bytes(storage_report.artifact_root_bytes)} "
            f"artifact_quota={format_bytes(storage_report.max_artifact_bytes)}"
        )

    multiprocessing.freeze_support()
    run_name = args.run_name
    if run_name is None and args.resume is not None:
        run_name = Path(args.resume).resolve().parents[1].name

    learner = SACLearner(
        config_path=args.config,
        run_name=run_name,
        max_env_steps=args.max_env_steps,
    )
    if args.resume is not None:
        learner.load_checkpoint(args.resume)

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
        final_checkpoint = learner.finalize_run(timeout_seconds=30.0)
        if not learner.clean_shutdown:
            log("Worker did not exit cleanly; final summary recorded an unclean shutdown.")
        if learner.latest_eval_summary is not None:
            log(f"latest_eval_env_step={learner.latest_eval_summary.get('env_step')}")
        log(f"final_checkpoint={final_checkpoint}")
        learner.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
