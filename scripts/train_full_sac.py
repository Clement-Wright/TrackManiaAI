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
    print(f"[train-full-sac] {message}", flush=True)


def main() -> int:
    from tm20ai.train.learner import SACLearner
    from tm20ai.train.worker import worker_entry

    parser = argparse.ArgumentParser(description="Train the first single-machine FULL-observation SAC baseline.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_sac.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-actor", default=None)
    parser.add_argument("--seed-demos", default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    args = parser.parse_args()

    multiprocessing.freeze_support()
    run_name = args.run_name
    if run_name is None and args.resume is not None:
        run_name = Path(args.resume).resolve().parents[1].name

    learner = SACLearner(
        config_path=args.config,
        run_name=run_name,
        max_env_steps=args.max_env_steps,
        init_actor=args.init_actor,
        seed_demos=args.seed_demos,
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
    except Exception as exc:  # noqa: BLE001 - entrypoint should log the fatal error
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
