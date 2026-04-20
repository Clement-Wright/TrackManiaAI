from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.action_space import ACTION_DIM, LEGACY_ACTION_DIM, clamp_action
from tm20ai.data.parquet_writer import sha256_file
from tm20ai.config import load_tm20ai_config
from tm20ai.train.evaluator import resolve_policy_adapter, run_policy_episodes


def log(message: str) -> None:
    print(f"[evaluate] {message}", flush=True)


def checkpoint_eval_worker_entry(
    config_path: str,
    command_queue,
    output_queue,
    eval_result_queue,
    shutdown_event,
    worker_done_event,
) -> None:
    from tm20ai.train.worker import SACWorker

    worker = SACWorker(
        config_path=config_path,
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
        bootstrap_log_path=None,
    )
    worker.run()


def run_checkpoint_eval_via_worker(
    *,
    config_path: str,
    checkpoint_path: Path,
    episodes: int,
    seed_base: int,
    record_video: bool,
    modes: list[str],
    run_name: str | None,
    trace_seconds: float,
    checkpoint_summary_extra: dict[str, object],
) -> dict[str, dict]:
    ctx = multiprocessing.get_context("spawn")
    command_queue = ctx.Queue(maxsize=8)
    output_queue = ctx.Queue(maxsize=8)
    eval_result_queue = ctx.Queue(maxsize=8)
    shutdown_event = ctx.Event()
    worker_done_event = ctx.Event()
    worker = ctx.Process(
        target=checkpoint_eval_worker_entry,
        args=(
            config_path,
            command_queue,
            output_queue,
            eval_result_queue,
            shutdown_event,
            worker_done_event,
        ),
        name="tm20ai-eval-worker",
    )
    worker.start()
    try:
        command_queue.put(
            {
                "type": "run_eval",
                "run_name": run_name,
                "env_step": int(checkpoint_summary_extra.get("eval_checkpoint_env_step", 0) or 0),
                "learner_step": int(checkpoint_summary_extra.get("eval_checkpoint_learner_step", 0) or 0),
                "episodes": int(episodes),
                "seed_base": int(seed_base),
                "record_video": bool(record_video),
                "modes": list(modes),
                "trace_seconds": float(trace_seconds),
                "scheduled_actor_version": None,
                **checkpoint_summary_extra,
            }
        )
        command_queue.put({"type": "shutdown"})

        deadline = time.time() + max(300.0, float(episodes) * 180.0)
        eval_result = None
        while time.time() < deadline:
            try:
                eval_result = eval_result_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not worker.is_alive() and eval_result_queue.empty():
                    break
        if eval_result is None:
            if worker.exitcode not in (None, 0):
                raise RuntimeError(f"Checkpoint eval worker exited with code {worker.exitcode} before producing a result.")
            raise RuntimeError("Checkpoint eval worker did not produce an eval result before the timeout.")

        summary = dict(eval_result.summary)
        mode_summaries = dict(summary.get("eval_mode_summaries") or {})
        mode_summary_paths = dict(summary.get("eval_mode_summary_paths") or {})
        results: dict[str, dict] = {}
        primary_mode = str(summary.get("eval_mode") or ("deterministic" if "deterministic" in modes else modes[0]))
        for mode_name in modes:
            mode_summary = dict(mode_summaries.get(mode_name) or (summary if mode_name == primary_mode else {}))
            mode_summary_path = mode_summary_paths.get(mode_name)
            if mode_summary_path is None and mode_name == primary_mode:
                mode_summary_path = str(eval_result.summary_path)
            results[mode_name] = {
                "summary": mode_summary,
                "summary_path": mode_summary_path,
            }
        return results
    finally:
        shutdown_event.set()
        worker.join(timeout=30.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)


def parse_action(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {ACTION_DIM, LEGACY_ACTION_DIM}:
        raise ValueError(
            "Fixed action must have either 2 values (throttle,steer) or 3 legacy values (gas,brake,steer)."
        )
    return clamp_action(np.asarray([float(part) for part in parts], dtype=np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation episodes against the custom env or a checkpoint policy.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument(
        "--modes",
        default="deterministic",
        help="Comma-separated eval modes. Valid values: deterministic, stochastic.",
    )
    parser.add_argument("--run-name", default=None, help="Base run name. Mode suffixes are appended automatically.")
    parser.add_argument("--policy", choices=("zero", "fixed", "scripted", "checkpoint"), default="zero")
    parser.add_argument(
        "--fixed-action",
        default="0.0,0.0",
        help="Fixed action as throttle,steer. Legacy gas,brake,steer input is also accepted.",
    )
    parser.add_argument("--script", default=None, help="Scripted policy in 'module:attr' or 'path.py:attr' form.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    episodes = config.eval.episodes if args.episodes is None else args.episodes
    seed_base = config.eval.seed_base if args.seed_base is None else args.seed_base
    record_video = config.eval.record_video or args.record_video
    fixed_action = parse_action(args.fixed_action)
    modes = [mode.strip().lower() for mode in str(args.modes).split(",") if mode.strip()]
    if not modes:
        parser.error("--modes must contain at least one of deterministic or stochastic.")
    invalid_modes = sorted({mode for mode in modes if mode not in {"deterministic", "stochastic"}})
    if invalid_modes:
        parser.error(f"--modes only accepts deterministic and stochastic, got {invalid_modes!r}.")

    checkpoint_summary_extra: dict[str, object] = {}
    if args.policy == "checkpoint":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when --policy checkpoint is selected.")
        import torch

        checkpoint_path = Path(args.checkpoint).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_summary_extra = {
            "eval_provenance_mode": "checkpoint_authoritative",
            "eval_checkpoint_path": str(checkpoint_path),
            "eval_checkpoint_sha256": sha256_file(checkpoint_path),
            "eval_checkpoint_env_step": int(payload.get("env_step", 0)),
            "eval_checkpoint_learner_step": int(payload.get("learner_step", 0)),
            "eval_checkpoint_actor_step": (
                int(payload["actor_step"]) if payload.get("actor_step") is not None else None
            ),
        }

    if args.policy == "checkpoint":
        results = run_checkpoint_eval_via_worker(
            config_path=str(Path(args.config).resolve()),
            checkpoint_path=checkpoint_path,
            episodes=int(episodes),
            seed_base=int(seed_base),
            record_video=bool(record_video),
            modes=modes,
            run_name=args.run_name,
            trace_seconds=float(config.eval.trace_seconds),
            checkpoint_summary_extra=checkpoint_summary_extra,
        )
    else:
        results = {}
        for mode_name in modes:
            deterministic = mode_name == "deterministic"
            policy = resolve_policy_adapter(
                policy=args.policy,
                fixed_action=fixed_action,
                script=args.script,
                checkpoint=args.checkpoint,
                deterministic=deterministic,
            )
            mode_run_name = None if args.run_name is None else f"{args.run_name}_{mode_name}"
            result = run_policy_episodes(
                config_path=args.config,
                mode="eval",
                policy=policy,
                episodes=episodes,
                seed_base=seed_base,
                record_video=record_video,
                checkpoint_path=args.checkpoint,
                run_name=mode_run_name,
                eval_mode=mode_name,
                deterministic=deterministic,
                trace_seconds=config.eval.trace_seconds,
                summary_extra=checkpoint_summary_extra,
            )
            results[mode_name] = result

    for mode_name in modes:
        result = results[mode_name]
        log(f"{mode_name}_summary={result['summary_path']}")
        log(json.dumps(result["summary"], indent=2))

    if len(results) > 1:
        log(
            json.dumps(
                {
                    "modes": list(results),
                    "summary_paths": {mode: result["summary_path"] for mode, result in results.items()},
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
