from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.data.parquet_writer import read_json, sha256_file, write_json
from tm20ai.config import load_tm20ai_config
from tm20ai.train.campaign import validate_campaign_run
from tm20ai.train.evaluator import (
    combine_mode_run_results,
    resolve_policy_adapter,
    run_checkpoint_eval_via_worker,
    run_policy_episodes,
)
from tm20ai.train.reporting import write_training_report


def log(message: str) -> None:
    print(f"[backfill-final-eval] {message}", flush=True)


def _resolve_default_run_name(summary: dict, checkpoint_path: Path) -> str:
    run_name = str(summary.get("run_name") or checkpoint_path.parents[1].name)
    env_step = int(summary.get("env_step", 0) or 0)
    return f"{run_name}_final_exact_step_{env_step:08d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill exact final deterministic+stochastic checkpoint eval for a run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary missing: {summary_path}")
    summary = read_json(summary_path)

    config_path = Path(args.config).resolve() if args.config is not None else Path(str(summary["config_path"])).resolve()
    config = load_tm20ai_config(config_path)

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint is not None
        else Path(str(summary.get("latest_checkpoint_path") or "")).resolve()
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Final checkpoint missing: {checkpoint_path}")

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    env_step = int(checkpoint_payload.get("env_step", summary.get("env_step", 0)) or 0)
    learner_step = int(checkpoint_payload.get("learner_step", summary.get("learner_step", 0)) or 0)
    actor_step = checkpoint_payload.get("actor_step", summary.get("actor_step"))
    actor_step = None if actor_step is None else int(actor_step)
    eval_run_name = args.run_name or _resolve_default_run_name(summary, checkpoint_path)

    checkpoint_metadata = {
        "eval_provenance_mode": "checkpoint_authoritative",
        "eval_checkpoint_path": str(checkpoint_path),
        "eval_checkpoint_sha256": sha256_file(checkpoint_path),
        "eval_checkpoint_env_step": env_step,
        "eval_checkpoint_learner_step": learner_step,
        "eval_checkpoint_actor_step": actor_step,
        "final_checkpoint_eval": True,
        "eval_actor_version": summary.get("applied_actor_version") or summary.get("desired_actor_version"),
        "eval_actor_source_learner_step": summary.get("applied_source_learner_step") or learner_step,
        "scheduled_actor_version": summary.get("applied_actor_version") or summary.get("desired_actor_version"),
        "applied_actor_version": summary.get("applied_actor_version"),
        "applied_actor_source_learner_step": summary.get("applied_source_learner_step"),
        "worker_env_step_at_eval_start": summary.get("env_step"),
    }
    log(f"run_dir={run_dir}")
    log(f"checkpoint={checkpoint_path}")
    log(f"run_name={eval_run_name}")
    try:
        results = run_checkpoint_eval_via_worker(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            episodes=config.eval.episodes,
            seed_base=config.eval.seed_base,
            record_video=config.eval.record_video,
            modes=list(config.eval.modes),
            run_name=eval_run_name,
            trace_seconds=float(config.eval.trace_seconds),
            checkpoint_summary_extra=checkpoint_metadata,
            timeout_seconds=max(300.0, float(config.eval.episodes) * 180.0),
        )
    except Exception as exc:  # noqa: BLE001
        log(f"worker_eval_failed={exc!r}")
        log("retrying_in_process_eval=true")
        results = {}
        for mode_name in config.eval.modes:
            deterministic = str(mode_name) == "deterministic"
            policy = resolve_policy_adapter(
                policy="checkpoint",
                fixed_action=None,
                script=None,
                checkpoint=checkpoint_path,
                deterministic=deterministic,
            )
            result = run_policy_episodes(
                config_path=config_path,
                mode="eval",
                policy=policy,
                episodes=config.eval.episodes,
                seed_base=config.eval.seed_base,
                record_video=config.eval.record_video,
                checkpoint_path=checkpoint_path,
                run_name=f"{eval_run_name}_{mode_name}",
                eval_mode=str(mode_name),
                deterministic=deterministic,
                trace_seconds=float(config.eval.trace_seconds),
                summary_extra=checkpoint_metadata,
            )
            results[str(mode_name)] = result
            log(f"{mode_name}_summary={result['summary_path']}")
    primary_result = combine_mode_run_results(results)
    combined_summary = dict(primary_result["summary"])
    mode_summaries = dict(combined_summary.get("eval_mode_summaries") or {})
    mode_summary_paths = dict(combined_summary.get("eval_mode_summary_paths") or {})
    mode_run_dirs = dict(combined_summary.get("eval_mode_run_dirs") or {})
    deterministic_summary = dict(mode_summaries.get("deterministic") or combined_summary)

    summary["latest_eval_summary"] = deterministic_summary
    summary["latest_eval_summary_path"] = str(mode_summary_paths.get("deterministic") or primary_result["summary_path"])
    summary["latest_eval_mode_summaries"] = mode_summaries
    summary["latest_eval_mode_summary_paths"] = mode_summary_paths
    summary["latest_eval_mode_run_dirs"] = mode_run_dirs
    summary["exact_final_eval_summary"] = combined_summary
    summary["exact_final_eval_summary_path"] = str(primary_result["summary_path"])
    summary["exact_final_eval_mode_summaries"] = mode_summaries
    summary["exact_final_eval_mode_summary_paths"] = mode_summary_paths
    summary["exact_final_eval_mode_run_dirs"] = mode_run_dirs
    summary["exact_final_eval_complete"] = True
    summary["incomplete_final_eval"] = False
    summary["final_eval_state"] = "complete"
    final_eval_status = dict(summary.get("final_eval_status") or {})
    final_eval_status.update(
        {
            "requested": True,
            "scheduled": True,
            "completed": True,
            "skipped_reason": None,
        }
    )
    summary["final_eval_status"] = final_eval_status

    eval_history = [entry for entry in list(summary.get("eval_history") or []) if not bool(entry.get("final_checkpoint_eval", False))]
    eval_history.append(
        {
            "checkpoint_step": env_step,
            "env_step": env_step,
            "learner_step": learner_step,
            "final_checkpoint_eval": True,
            "summary_path": summary["latest_eval_summary_path"],
            "summary": deterministic_summary,
            "mode_summaries": mode_summaries,
            "mode_summary_paths": mode_summary_paths,
            "mode_run_dirs": mode_run_dirs,
            "deterministic_collapse": combined_summary.get("deterministic_collapse"),
            "eval_actor_version": checkpoint_metadata.get("eval_actor_version"),
            "eval_actor_source_learner_step": checkpoint_metadata.get("eval_actor_source_learner_step"),
            "scheduled_actor_version": checkpoint_metadata.get("scheduled_actor_version"),
            "eval_provenance_mode": checkpoint_metadata.get("eval_provenance_mode"),
            "eval_checkpoint_path": checkpoint_metadata.get("eval_checkpoint_path"),
            "eval_checkpoint_sha256": checkpoint_metadata.get("eval_checkpoint_sha256"),
            "eval_checkpoint_env_step": env_step,
            "eval_checkpoint_learner_step": learner_step,
            "eval_checkpoint_actor_step": actor_step,
            "worker_env_step_at_eval_start": checkpoint_metadata.get("worker_env_step_at_eval_start"),
            "applied_actor_version": checkpoint_metadata.get("applied_actor_version"),
            "applied_actor_source_learner_step": checkpoint_metadata.get("applied_actor_source_learner_step"),
            "timestamp": time.time(),
        }
    )
    summary["eval_history"] = eval_history
    write_json(summary_path, summary)
    report_paths = write_training_report(run_dir)
    validation = validate_campaign_run(run_dir)
    log(f"deterministic_summary={mode_summary_paths.get('deterministic')}")
    log(f"stochastic_summary={mode_summary_paths.get('stochastic')}")
    log(f"report_json={report_paths.json_path}")
    log(f"report_markdown={report_paths.markdown_path}")
    log(f"validation={validation.valid} reasons={list(validation.reasons)}")
    return 0 if validation.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
