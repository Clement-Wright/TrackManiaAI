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

from tm20ai.data.parquet_writer import build_run_artifact_paths, read_json, sha256_file, write_json
from tm20ai.config import load_tm20ai_config
from tm20ai.train.campaign import analyze_policy_mode_sweep_results
from tm20ai.train.evaluator import resolve_policy_adapter, run_policy_episodes


def _mode_specs(extraction_modes: list[str], temperatures: list[float], best_of_k: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for mode in extraction_modes:
        if mode in {"deterministic_mean", "clipped_mean"}:
            specs.append(
                {
                    "name": mode,
                    "extraction_mode": mode,
                    "temperature": 1.0,
                    "best_of_k": 1,
                    "deployment_eligible": True,
                    "diagnostic_only": False,
                }
            )
        elif mode == "stochastic":
            for temperature in temperatures:
                specs.append(
                    {
                        "name": f"stochastic_temp_{temperature:g}",
                        "extraction_mode": "stochastic",
                        "temperature": float(temperature),
                        "best_of_k": 1,
                        "deployment_eligible": False,
                        "diagnostic_only": False,
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
                        "deployment_eligible": False,
                        "diagnostic_only": True,
                    }
                )
        else:
            raise ValueError(f"Unsupported extraction mode: {mode}")
    return specs


def _resolve_config_ghost_bundle(config, config_path: str | Path) -> Path:  # noqa: ANN001
    configured_bundle = config.ghosts.bundle_manifest
    if configured_bundle in (None, ""):
        raise RuntimeError(f"Config {config_path} does not define ghosts.bundle_manifest.")
    configured_path = Path(str(configured_bundle))
    if not configured_path.is_absolute():
        configured_path = ROOT / configured_path
    return configured_path.resolve()


def _preflight_policy_sweep(
    *,
    config,  # noqa: ANN001
    config_path: str | Path,
    required_ghost_bundle: str | Path | None,
    required_training_family: str | None,
    required_selected_count: int | None,
    forbid_mixed_fallback: bool,
) -> dict[str, object] | None:
    if (
        required_ghost_bundle is None
        and required_training_family is None
        and required_selected_count is None
        and not forbid_mixed_fallback
    ):
        return None

    configured_bundle = _resolve_config_ghost_bundle(config, config_path)
    if required_ghost_bundle is not None:
        required_bundle = Path(required_ghost_bundle)
        if not required_bundle.is_absolute():
            required_bundle = ROOT / required_bundle
        required_bundle = required_bundle.resolve()
        if configured_bundle != required_bundle:
            raise RuntimeError(
                f"Policy-mode sweep config points at {configured_bundle}; required ghost bundle is {required_bundle}."
            )
    if not configured_bundle.exists():
        raise RuntimeError(f"Policy-mode sweep ghost bundle does not exist: {configured_bundle}")

    manifest = read_json(configured_bundle)
    if forbid_mixed_fallback and bool(manifest.get("mixed_fallback", False)):
        raise RuntimeError(f"Policy-mode sweep refuses mixed fallback bundle: {configured_bundle}")
    if required_training_family is not None and str(manifest.get("selected_training_family") or "") != str(
        required_training_family
    ):
        raise RuntimeError(
            f"Policy-mode sweep bundle selected_training_family={manifest.get('selected_training_family')!r}; "
            f"expected {required_training_family!r}."
        )
    if required_selected_count is not None and int(manifest.get("selected_count", 0) or 0) != int(required_selected_count):
        raise RuntimeError(
            f"Policy-mode sweep bundle selected_count={manifest.get('selected_count')!r}; "
            f"expected {int(required_selected_count)}."
        )
    return {
        "ghost_bundle_manifest_path": str(configured_bundle),
        "selected_training_family": manifest.get("selected_training_family"),
        "selected_count": manifest.get("selected_count"),
        "mixed_fallback": bool(manifest.get("mixed_fallback", False)),
        "bundle_resolution_mode": manifest.get("bundle_resolution_mode"),
    }


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
    parser.add_argument("--required-ghost-bundle", default=None)
    parser.add_argument("--required-training-family", default=None)
    parser.add_argument("--required-selected-count", type=int, default=None)
    parser.add_argument("--allow-mixed-fallback", action="store_true")
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    target_preflight = _preflight_policy_sweep(
        config=config,
        config_path=args.config,
        required_ghost_bundle=args.required_ghost_bundle,
        required_training_family=args.required_training_family,
        required_selected_count=args.required_selected_count,
        forbid_mixed_fallback=not args.allow_mixed_fallback
        and (
            args.required_ghost_bundle is not None
            or args.required_training_family is not None
            or args.required_selected_count is not None
        ),
    )
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
    if target_preflight is not None:
        checkpoint_summary_extra.update(target_preflight)
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
            "eval_checkpoint_path": str(checkpoint_path),
            "eval_checkpoint_sha256": checkpoint_summary_extra["eval_checkpoint_sha256"],
            "eval_checkpoint_env_step": checkpoint_summary_extra["eval_checkpoint_env_step"],
            "eval_checkpoint_learner_step": checkpoint_summary_extra["eval_checkpoint_learner_step"],
            "eval_checkpoint_actor_step": checkpoint_summary_extra["eval_checkpoint_actor_step"],
            "extraction_mode": str(spec["extraction_mode"]),
            "temperature": float(spec["temperature"]),
            "best_of_k": int(spec["best_of_k"]),
            "deployment_eligible": bool(spec["deployment_eligible"]),
            "diagnostic_only": bool(spec["diagnostic_only"]),
            "mean_final_progress_index": result["summary"].get("mean_final_progress_index"),
            "median_final_progress_index": result["summary"].get("median_final_progress_index"),
            "mean_final_progress_meters": result["summary"].get("mean_final_progress_meters"),
            "mean_final_arc_length_m": result["summary"].get("mean_final_arc_length_m"),
            "mean_progress_fraction_of_reference": result["summary"].get("mean_progress_fraction_of_reference"),
            "mean_ghost_relative_time_delta_ms": result["summary"].get("mean_ghost_relative_time_delta_ms"),
            "completion_rate": result["summary"].get("completion_rate"),
            "mean_abs_steer": result["summary"].get("mean_abs_steer"),
            "mean_abs_throttle": result["summary"].get("mean_abs_throttle"),
        }
        print(f"[evaluate-redq-policy-modes] {name}_summary={result['summary_path']}", flush=True)
    analysis = analyze_policy_mode_sweep_results(results)
    run_paths = build_run_artifact_paths(config, mode="eval", run_name=base_run_name)
    combined_path = run_paths.run_dir / "policy_mode_sweep.json"
    write_json(
        combined_path,
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "base_run_name": base_run_name,
            "results": results,
            "analysis": analysis,
            "target_preflight": target_preflight,
        },
    )
    print(f"[evaluate-redq-policy-modes] combined={combined_path}", flush=True)
    print(json.dumps({"results": results, "analysis": analysis}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
