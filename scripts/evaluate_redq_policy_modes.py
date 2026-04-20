from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.data.parquet_writer import build_run_artifact_paths, sha256_file, write_json
from tm20ai.config import load_tm20ai_config
from tm20ai.train.evaluator import resolve_policy_adapter, run_policy_episodes


def _mode_specs(extraction_modes: list[str], temperatures: list[float], best_of_k: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for mode in extraction_modes:
        if mode in {"deterministic_mean", "clipped_mean"}:
            specs.append({"name": mode, "extraction_mode": mode, "temperature": 1.0, "best_of_k": 1})
        elif mode == "stochastic":
            for temperature in temperatures:
                specs.append(
                    {
                        "name": f"stochastic_temp_{temperature:g}",
                        "extraction_mode": "stochastic",
                        "temperature": float(temperature),
                        "best_of_k": 1,
                    }
                )
        elif mode == "sample_best_of_k":
            for temperature in temperatures:
                specs.append(
                    {
                        "name": f"best_of_{best_of_k}_temp_{temperature:g}",
                        "extraction_mode": "sample_best_of_k",
                        "temperature": float(temperature),
                        "best_of_k": int(best_of_k),
                    }
                )
        else:
            raise ValueError(f"Unsupported extraction mode: {mode}")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep REDQ checkpoint action-extraction modes without retraining.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--extraction-modes", default=None, help="Comma-separated modes; defaults to eval.extraction_modes.")
    parser.add_argument("--temperatures", default=None, help="Comma-separated stochastic temperatures.")
    parser.add_argument("--best-of-k", type=int, default=None)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    checkpoint_path = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    extraction_modes = (
        [part.strip().lower() for part in args.extraction_modes.split(",") if part.strip()]
        if args.extraction_modes is not None
        else list(config.eval.extraction_modes)
    )
    temperatures = (
        [float(part.strip()) for part in args.temperatures.split(",") if part.strip()]
        if args.temperatures is not None
        else list(config.eval.temperature_sweep)
    )
    best_of_k = args.best_of_k or config.eval.best_of_k
    base_run_name = args.run_name or f"redq_policy_modes_{checkpoint_path.stem}"
    checkpoint_summary_extra = {
        "eval_provenance_mode": "checkpoint_authoritative",
        "eval_checkpoint_path": str(checkpoint_path),
        "eval_checkpoint_sha256": sha256_file(checkpoint_path),
        "eval_checkpoint_env_step": int(payload.get("env_step", 0)),
        "eval_checkpoint_learner_step": int(payload.get("learner_step", 0)),
        "eval_checkpoint_actor_step": int(payload["actor_step"]) if payload.get("actor_step") is not None else None,
    }
    results: dict[str, dict] = {}
    for spec in _mode_specs(extraction_modes, temperatures, best_of_k):
        name = str(spec["name"])
        deterministic = spec["extraction_mode"] in {"deterministic_mean", "clipped_mean"}
        policy = resolve_policy_adapter(
            policy="checkpoint",
            checkpoint=checkpoint_path,
            deterministic=deterministic,
            extraction_mode=str(spec["extraction_mode"]),
            temperature=float(spec["temperature"]),
            best_of_k=int(spec["best_of_k"]),
        )
        result = run_policy_episodes(
            config_path=args.config,
            mode="eval",
            policy=policy,
            episodes=config.eval.episodes if args.episodes is None else args.episodes,
            seed_base=config.eval.seed_base if args.seed_base is None else args.seed_base,
            record_video=config.eval.record_video or args.record_video,
            checkpoint_path=checkpoint_path,
            run_name=f"{base_run_name}_{name}",
            eval_mode=name,
            deterministic=deterministic,
            trace_seconds=config.eval.trace_seconds,
            extraction_mode=str(spec["extraction_mode"]),
            temperature=float(spec["temperature"]),
            best_of_k=int(spec["best_of_k"]),
            summary_extra=checkpoint_summary_extra,
        )
        results[name] = {
            "summary_path": str(result["summary_path"]),
            "mean_final_progress_index": result["summary"].get("mean_final_progress_index"),
            "median_final_progress_index": result["summary"].get("median_final_progress_index"),
            "completion_rate": result["summary"].get("completion_rate"),
            "mean_abs_steer": result["summary"].get("mean_abs_steer"),
            "mean_abs_throttle": result["summary"].get("mean_abs_throttle"),
        }
        print(f"[evaluate-redq-policy-modes] {name}_summary={result['summary_path']}", flush=True)
    run_paths = build_run_artifact_paths(config, mode="eval", run_name=base_run_name)
    combined_path = run_paths.run_dir / "policy_mode_sweep.json"
    write_json(
        combined_path,
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "base_run_name": base_run_name,
            "results": results,
        },
    )
    print(f"[evaluate-redq-policy-modes] combined={combined_path}", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
