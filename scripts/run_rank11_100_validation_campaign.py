from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.data.parquet_writer import ensure_directory, read_json, write_json  # noqa: E402
from tm20ai.train.artifact_retention import cleanup_artifact_root  # noqa: E402
from tm20ai.train.campaign import (  # noqa: E402
    analyze_policy_mode_sweep_results,
    best_scheduled_deterministic_checkpoint,
    select_reward_winner,
    validate_campaign_run,
)
from tm20ai.train.research import append_results_entry, write_algorithm_comparison_report  # noqa: E402


def log(message: str) -> None:
    print(f"[rank11-validation-campaign] {message}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


REWARD_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "A0_baseline_current",
        "reward": {
            "corridor_soft_margin_m": 25.0,
            "corridor_hard_margin_m": 90.0,
            "corridor_patience_steps": 20,
            "corridor_penalty_scale": 0.03,
            "corridor_penalty_max": 8.0,
            "corridor_recovery_bonus": 1.0,
            "corridor_min_recovery_progress_m": 0.5,
            "corridor_min_recovery_speed_kmh": 8.0,
            "corridor_recovery_distance_delta_m": 1.0,
        },
    },
    {
        "label": "A1_wider_recovery",
        "reward": {
            "corridor_soft_margin_m": 25.0,
            "corridor_hard_margin_m": 90.0,
            "corridor_patience_steps": 40,
            "corridor_penalty_scale": 0.03,
            "corridor_penalty_max": 8.0,
            "corridor_recovery_bonus": 1.0,
            "corridor_min_recovery_progress_m": 0.25,
            "corridor_min_recovery_speed_kmh": 5.0,
            "corridor_recovery_distance_delta_m": 0.5,
        },
    },
    {
        "label": "A2_softer_wider_corridor",
        "reward": {
            "corridor_soft_margin_m": 35.0,
            "corridor_hard_margin_m": 120.0,
            "corridor_patience_steps": 40,
            "corridor_penalty_scale": 0.02,
            "corridor_penalty_max": 6.0,
            "corridor_recovery_bonus": 1.0,
            "corridor_min_recovery_progress_m": 0.25,
            "corridor_min_recovery_speed_kmh": 5.0,
            "corridor_recovery_distance_delta_m": 0.5,
        },
    },
    {
        "label": "A3_buffered_hard_boundary",
        "reward": {
            "corridor_soft_margin_m": 25.0,
            "corridor_hard_margin_m": 90.0,
            "corridor_patience_steps": 60,
            "corridor_penalty_scale": 0.03,
            "corridor_penalty_max": 8.0,
            "corridor_recovery_bonus": 1.0,
            "corridor_min_recovery_progress_m": 0.10,
            "corridor_min_recovery_speed_kmh": 3.0,
            "corridor_recovery_distance_delta_m": 0.25,
        },
    },
    {
        "label": "A4_hard_stray_control",
        "reward": {
            "corridor_soft_margin_m": 10.0,
            "corridor_hard_margin_m": 45.0,
            "corridor_patience_steps": 1,
            "corridor_penalty_scale": 0.05,
            "corridor_penalty_max": 12.0,
            "corridor_recovery_bonus": 0.0,
            "corridor_catastrophic_distance_m": 120.0,
            "corridor_min_recovery_progress_m": 1.0,
            "corridor_min_recovery_speed_kmh": 12.0,
            "corridor_recovery_distance_delta_m": 2.0,
        },
    },
)

REPAIRABLE_FINAL_EVAL_REASONS = {
    "exact_final_eval_complete_false",
    "incomplete_final_eval_true",
    "final_eval_state=exact_final_eval_missing",
    "deterministic_summary_path_missing",
    "deterministic_summary_missing_on_disk",
    "stochastic_summary_path_missing",
    "stochastic_summary_missing_on_disk",
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged.get(key) or {}), value)
        else:
            merged[key] = value
    return merged


def _run_command(command: list[str], *, dry_run: bool = False) -> None:
    log("command=" + " ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _remove_stale_live_lock() -> None:
    live_lock = ROOT / "artifacts" / "live_env.lock"
    if live_lock.exists():
        live_lock.unlink()
        log(f"removed_stale_lock={live_lock}")


def _write_override_config(*, source_path: Path, target_path: Path, overrides: dict[str, Any]) -> Path:
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(payload, overrides)
    ensure_directory(target_path.parent)
    target_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return target_path.resolve()


def _latest_resume_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if not candidates:
        return None
    valid_candidates = [candidate for candidate in candidates if candidate.is_file() and candidate.stat().st_size > 0]
    if not valid_candidates:
        return None
    return valid_candidates[-1].resolve()


def _write_status(status_path: Path, payload: dict[str, Any]) -> None:
    body = dict(payload)
    body["updated_at"] = now_iso()
    write_json(status_path, body)


def _is_repairable_final_eval_gap(validation) -> bool:  # noqa: ANN001
    reasons = set(validation.reasons)
    return bool(reasons) and reasons.issubset(REPAIRABLE_FINAL_EVAL_REASONS)


def _preflight(*, base_config_path: Path, bundle_manifest_path: Path, dry_run: bool) -> None:
    _run_command([sys.executable, "-m", "pytest", "-q"], dry_run=dry_run)
    if not bundle_manifest_path.exists():
        raise RuntimeError(f"rank11_100_bundle manifest is missing: {bundle_manifest_path}")
    bundle_manifest = read_json(bundle_manifest_path)
    if int(bundle_manifest.get("selected_count", 0)) != 90:
        raise RuntimeError(
            f"rank11_100_bundle manifest must contain 90 selected trajectories, found {bundle_manifest.get('selected_count')}."
        )
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    configured_bundle_raw = Path(str(((config.get("ghosts") or {}).get("bundle_manifest") or "")))
    configured_bundle = (
        configured_bundle_raw.resolve()
        if configured_bundle_raw.is_absolute()
        else (ROOT / configured_bundle_raw).resolve()
    )
    if configured_bundle != bundle_manifest_path.resolve():
        raise RuntimeError(
            f"Base config {base_config_path} does not point at {bundle_manifest_path}; got {configured_bundle}."
        )


def _gate_environment(*, config_path: Path, dry_run: bool) -> None:
    _remove_stale_live_lock()
    _run_command(
        [sys.executable, str(ROOT / "scripts" / "force_window_size.py"), "--config", str(config_path)],
        dry_run=dry_run,
    )


def _repair_missing_exact_final_eval(
    *,
    run_dir: Path,
    config_path: Path,
    dry_run: bool,
    status_path: Path,
    leg_label: str,
) -> None:
    _write_status(
        status_path,
        {
            "state": "repairing_exact_final_eval",
            "current_leg": leg_label,
            "current_run_name": run_dir.name,
            "current_config_path": str(config_path),
        },
    )
    _gate_environment(config_path=config_path, dry_run=dry_run)
    _run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "backfill_final_checkpoint_eval.py"),
            "--run-dir",
            str(run_dir),
            "--config",
            str(config_path),
        ],
        dry_run=dry_run,
    )
    _run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_environment.py"),
            "--config",
            str(config_path),
            "--require-reward",
        ],
        dry_run=dry_run,
    )


def _run_training_leg(
    *,
    run_name: str,
    config_path: Path,
    artifact_root: Path,
    wall_clock_minutes: float,
    dry_run: bool,
    status_path: Path,
    leg_label: str,
    extra_args: list[str] | None = None,
) -> Path:
    run_dir = artifact_root / "train" / run_name
    if run_dir.exists():
        validation = validate_campaign_run(run_dir)
        if validation.valid:
            log(f"leg_reused_valid={run_name}")
            return run_dir
        if _is_repairable_final_eval_gap(validation):
            log(f"leg_repair_exact_final_eval={run_name}")
            _repair_missing_exact_final_eval(
                run_dir=run_dir,
                config_path=config_path,
                dry_run=dry_run,
                status_path=status_path,
                leg_label=leg_label,
            )
            validation = validate_campaign_run(run_dir)
            if validation.valid:
                log(f"leg_reused_repaired={run_name}")
                return run_dir

    _gate_environment(config_path=config_path, dry_run=dry_run)
    _write_status(
        status_path,
        {
            "state": "running_leg",
            "current_leg": leg_label,
            "current_run_name": run_name,
            "current_config_path": str(config_path),
        },
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_full_redq.py"),
        "--config",
        str(config_path),
        "--max-wall-clock-minutes",
        str(wall_clock_minutes),
    ]
    resume_checkpoint = _latest_resume_checkpoint(run_dir)
    if resume_checkpoint is not None and run_dir.exists():
        command.extend(["--resume", str(resume_checkpoint)])
    else:
        command.extend(["--run-name", run_name])
    if extra_args and resume_checkpoint is None:
        command.extend(extra_args)
    _run_command(command, dry_run=dry_run)
    if dry_run:
        return run_dir

    validation = validate_campaign_run(run_dir)
    if not validation.valid and _is_repairable_final_eval_gap(validation):
        log(f"leg_backfill_exact_final_eval={run_name}")
        _repair_missing_exact_final_eval(
            run_dir=run_dir,
            config_path=config_path,
            dry_run=dry_run,
            status_path=status_path,
            leg_label=leg_label,
        )
        validation = validate_campaign_run(run_dir)
    if not validation.valid:
        raise RuntimeError(
            f"Run {run_name} is not valid for campaign ranking: {', '.join(validation.reasons)}"
        )
    return run_dir


def _render_block_a_summary(
    *,
    winner_name: str,
    ordered_candidates: list[dict[str, Any]],
) -> str:
    lines = [
        "# Block A Reward Stability",
        "",
        f"- winner={winner_name}",
        "",
        "## Ranked Runs",
    ]
    for candidate in ordered_candidates:
        lines.append(
            f"- {candidate['run_name']}: exact_final_progress={candidate['exact_final_progress']} "
            f"ghost_delta_ms={candidate['exact_final_ghost_delta_ms']} "
            f"progress_fraction={candidate['exact_final_progress_fraction']} "
            f"corridor_violation_truncation_rate={candidate['exact_final_corridor_violation_truncation_rate']} "
            f"corridor_nonrecovering_p95={candidate['corridor_nonrecovering_p95']}"
        )
    return "\n".join(lines) + "\n"


def _render_block_b_summary(*, analysis_rows: list[dict[str, Any]], deployment_choice: str | None) -> str:
    lines = [
        "# Block B Deterministic Extraction",
        "",
        f"- recommended_deployment={deployment_choice}",
        "",
        "## Checkpoints",
    ]
    for row in analysis_rows:
        lines.append(
            f"- {row['checkpoint_label']}: deployment_choice={row.get('deployment_choice')} "
            f"deployment_meets_target={row.get('deployment_choice_meets_target')} "
            f"stochastic_reference_mode={row.get('stochastic_reference_mode')}"
        )
        for mode_name, payload in sorted(dict(row.get("per_mode") or {}).items()):
            lines.append(
                f"-   {mode_name}: progress={payload.get('mean_final_progress_index')} "
                f"progress_fraction={payload.get('mean_progress_fraction_of_reference')} "
                f"ghost_delta_ms={payload.get('mean_ghost_relative_time_delta_ms')} "
                f"dcs={payload.get('determinism_conversion_score')}"
            )
    return "\n".join(lines) + "\n"


def _render_block_c_summary(*, winner_name: str, baseline_progress: float, offline_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Block C Offline Leverage",
        "",
        f"- baseline_run={winner_name}",
        f"- baseline_exact_final_progress={baseline_progress}",
        "",
        "## Offline Runs",
    ]
    for row in offline_rows:
        lines.append(
            f"- {row['run_name']}: exact_final_progress={row['exact_final_progress']} "
            f"progress_25k={row.get('progress_25k')} progress_50k={row.get('progress_50k')} "
            f"offline_init={row.get('offline_init_checkpoint_path')} "
            f"offline_strategy={row.get('offline_pretrain_strategy')}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the rank11_100 REDQ validation campaign.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq_top100_tmrl_test.yaml"))
    parser.add_argument("--session-name", default="rank11_100_validation_campaign")
    parser.add_argument("--wall-clock-minutes", type=float, default=90.0)
    parser.add_argument("--results-file", default="rank11_100_validation_campaign.md")
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--blocks", default="A,B,C")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config).resolve()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    bundle_manifest_path = (ROOT / str(((base_config.get("ghosts") or {}).get("bundle_manifest") or ""))).resolve()
    session_root = ensure_directory(ROOT / ".tmp" / "live_rank11_100_validation_campaign" / args.session_name)
    config_dir = ensure_directory(session_root / "configs")
    reports_dir = ensure_directory(session_root / "reports")
    results_root = ensure_directory(ROOT / "results")
    artifact_root = ensure_directory(
        Path(args.artifact_root).resolve() if args.artifact_root is not None else session_root / "artifacts"
    )
    status_path = session_root / "status.json"

    blocks = {part.strip().upper() for part in args.blocks.split(",") if part.strip()}
    invalid_blocks = sorted(blocks - {"A", "B", "C"})
    if invalid_blocks:
        parser.error(f"Unsupported blocks: {invalid_blocks!r}")

    _write_status(
        status_path,
        {
            "state": "starting",
            "session_name": args.session_name,
            "artifact_root": str(artifact_root),
            "base_config_path": str(base_config_path),
            "bundle_manifest_path": str(bundle_manifest_path),
            "blocks": sorted(blocks),
        },
    )

    _preflight(base_config_path=base_config_path, bundle_manifest_path=bundle_manifest_path, dry_run=args.dry_run)
    common_config_overrides: dict[str, Any] = {
        "artifacts": {"root": str(artifact_root)},
        "ghosts": {"bundle_manifest": str(bundle_manifest_path.resolve())},
    }

    reward_run_dirs: list[Path] = []
    winner_run_dir: Path | None = None
    winner_report: dict[str, Any] | None = None
    winner_config_path: Path | None = None
    block_a_runner_up_dirs: list[Path] = []
    if "A" in blocks:
        for variant in REWARD_VARIANTS:
            run_name = f"{args.session_name}_{variant['label']}"
            override_config_path = _write_override_config(
                source_path=base_config_path,
                target_path=config_dir / f"{variant['label']}.yaml",
                overrides={
                    **common_config_overrides,
                    "reward": dict(variant["reward"]),
                },
            )
            reward_run_dirs.append(
                _run_training_leg(
                    run_name=run_name,
                    config_path=override_config_path,
                    artifact_root=artifact_root,
                    wall_clock_minutes=args.wall_clock_minutes,
                    dry_run=args.dry_run,
                    status_path=status_path,
                    leg_label=variant["label"],
                )
            )

        if args.dry_run:
            winner_run_dir = reward_run_dirs[0]
            winner_config_path = config_dir / f"{REWARD_VARIANTS[0]['label']}.yaml"
            _write_status(
                status_path,
                {
                    "state": "dry_run_block_a_complete",
                    "winner_run_name": winner_run_dir.name,
                    "winner_variant": REWARD_VARIANTS[0]["label"],
                },
            )
        else:
            winner_candidate, ordered_candidates = select_reward_winner(reward_run_dirs)
            winner_run_dir = winner_candidate.run_dir
            winner_report = dict(winner_candidate.report)
            winner_config_path = config_dir / f"{winner_candidate.run_name.replace(args.session_name + '_', '')}.yaml"
            winner_progress = winner_candidate.exact_final_progress
            block_a_runner_up_dirs = [
                candidate.run_dir
                for candidate in ordered_candidates
                if candidate.run_dir != winner_run_dir and candidate.exact_final_progress >= winner_progress * 0.95
            ]
            block_a_payload = {
                "winner_run_name": winner_candidate.run_name,
                "winner_exact_final_progress": winner_candidate.exact_final_progress,
                "ordered_candidates": [
                    {
                        "run_name": candidate.run_name,
                        "run_dir": str(candidate.run_dir),
                        "exact_final_progress": candidate.exact_final_progress,
                        "exact_final_progress_fraction": candidate.exact_final_progress_fraction,
                        "exact_final_ghost_delta_ms": candidate.exact_final_ghost_delta_ms,
                        "exact_final_corridor_violation_truncation_rate": candidate.exact_final_corridor_violation_truncation_rate,
                        "corridor_nonrecovering_p95": candidate.corridor_nonrecovering_p95,
                    }
                    for candidate in ordered_candidates
                ],
            }
            write_json(reports_dir / "block_a_reward_summary.json", block_a_payload)
            (reports_dir / "block_a_reward_summary.md").write_text(
                _render_block_a_summary(
                    winner_name=winner_candidate.run_name,
                    ordered_candidates=block_a_payload["ordered_candidates"],
                ),
                encoding="utf-8",
            )
            write_algorithm_comparison_report(
                reward_run_dirs,
                output_dir=reports_dir / "block_a_comparison",
                wall_clock_budget_minutes=args.wall_clock_minutes,
            )
            _write_status(
                status_path,
                {
                    "state": "block_a_complete",
                    "winner_run_name": winner_candidate.run_name,
                    "winner_run_dir": str(winner_run_dir),
                    "winner_variant": winner_candidate.run_name.replace(args.session_name + "_", ""),
                },
            )

    if winner_run_dir is None:
        raise RuntimeError("Campaign could not resolve a Block A winner.")

    if "B" in blocks:
        checkpoint_targets: list[tuple[str, Path]] = []
        if args.dry_run:
            checkpoint_targets.append(("final", winner_run_dir / "checkpoints" / "checkpoint_fake_final.pt"))
        else:
            winner_summary = read_json(winner_run_dir / "summary.json")
            final_checkpoint_path = Path(str(winner_summary.get("latest_checkpoint_path"))).resolve()
            checkpoint_targets.append(("final", final_checkpoint_path))
            final_progress = float(
                dict(winner_summary.get("exact_final_eval_summary") or {}).get("mean_final_progress_index", 0.0) or 0.0
            )
            scheduled_candidate = best_scheduled_deterministic_checkpoint(winner_run_dir)
            if scheduled_candidate is not None and scheduled_candidate.progress >= final_progress * 1.10:
                checkpoint_targets.append(("best_scheduled", Path(scheduled_candidate.checkpoint_path).resolve()))

        extraction_rows: list[dict[str, Any]] = []
        for checkpoint_label, checkpoint_path in checkpoint_targets:
            run_name = f"{args.session_name}_{winner_run_dir.name}_{checkpoint_label}_policy_modes"
            _run_command(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_redq_policy_modes.py"),
                    "--config",
                    str(winner_config_path or base_config_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--run-name",
                    run_name,
                ],
                dry_run=args.dry_run,
            )
            if args.dry_run:
                extraction_rows.append(
                    {
                        "checkpoint_label": checkpoint_label,
                        "policy_mode_sweep_path": None,
                        "deployment_choice": "deterministic_mean",
                        "deployment_choice_meets_target": False,
                        "stochastic_reference_mode": "stochastic_temp_1",
                        "per_mode": {},
                    }
                )
                continue
            sweep_run_dir = artifact_root / "eval" / run_name
            sweep_payload = read_json(sweep_run_dir / "policy_mode_sweep.json")
            analysis = analyze_policy_mode_sweep_results(dict(sweep_payload.get("results") or {}))
            extraction_rows.append(
                {
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_path": str(checkpoint_path),
                    "policy_mode_sweep_path": str((sweep_run_dir / "policy_mode_sweep.json").resolve()),
                    **analysis,
                }
            )

        deployment_choice = next(
            (
                str(row.get("deployment_choice"))
                for row in extraction_rows
                if row.get("checkpoint_label") == "final" and row.get("deployment_choice") is not None
            ),
            None,
        )
        write_json(reports_dir / "block_b_extraction_summary.json", {"rows": extraction_rows, "deployment_choice": deployment_choice})
        (reports_dir / "block_b_extraction_summary.md").write_text(
            _render_block_b_summary(analysis_rows=extraction_rows, deployment_choice=deployment_choice),
            encoding="utf-8",
        )
        _write_status(
            status_path,
            {
                "state": "block_b_complete",
                "deployment_choice": deployment_choice,
                "checkpoint_targets": [label for label, _path in checkpoint_targets],
            },
        )

    offline_rows: list[dict[str, Any]] = []
    if "C" in blocks:
        offline_pretrain_dir = ensure_directory(session_root / "pretrain")
        offline_checkpoint_path = offline_pretrain_dir / "ghost_redq_pretrain.pt"
        if not args.dry_run and not offline_checkpoint_path.exists():
            _run_command(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "pretrain_ghost_redq.py"),
                    "--config",
                    str(winner_config_path or base_config_path),
                    "--ghost-bundle",
                    str(bundle_manifest_path),
                    "--output-dir",
                    str(offline_pretrain_dir),
                    "--run-name",
                    f"{args.session_name}_offline_pretrain",
                ],
                dry_run=False,
            )
        elif args.dry_run:
            _run_command(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "pretrain_ghost_redq.py"),
                    "--config",
                    str(winner_config_path or base_config_path),
                    "--ghost-bundle",
                    str(bundle_manifest_path),
                    "--output-dir",
                    str(offline_pretrain_dir),
                    "--run-name",
                    f"{args.session_name}_offline_pretrain",
                ],
                dry_run=True,
            )

        c1_config_path = _write_override_config(
            source_path=winner_config_path or base_config_path,
            target_path=config_dir / "C1_weight_init_only.yaml",
            overrides={
                **common_config_overrides,
                "balanced_replay": {"enabled": False},
                "offline_pretrain": {"seed_replay_buffer": False},
            },
        )
        c2_config_path = _write_override_config(
            source_path=winner_config_path or base_config_path,
            target_path=config_dir / "C2_full_offline_to_online.yaml",
            overrides={
                **common_config_overrides,
                "balanced_replay": {
                    "enabled": True,
                    "offline_initial_fraction": 0.75,
                    "offline_final_fraction": 0.10,
                    "decay_env_steps": 50000,
                },
                "offline_pretrain": {"seed_replay_buffer": True},
            },
        )

        c1_run_dir = _run_training_leg(
            run_name=f"{args.session_name}_C1_weight_init_only",
            config_path=c1_config_path,
            artifact_root=artifact_root,
            wall_clock_minutes=args.wall_clock_minutes,
            dry_run=args.dry_run,
            status_path=status_path,
            leg_label="C1_weight_init_only",
            extra_args=["--offline-init-checkpoint", str(offline_checkpoint_path)],
        )
        c2_run_dir = _run_training_leg(
            run_name=f"{args.session_name}_C2_full_offline_to_online",
            config_path=c2_config_path,
            artifact_root=artifact_root,
            wall_clock_minutes=args.wall_clock_minutes,
            dry_run=args.dry_run,
            status_path=status_path,
            leg_label="C2_full_offline_to_online",
            extra_args=["--offline-init-checkpoint", str(offline_checkpoint_path), "--ghost-bundle", str(bundle_manifest_path)],
        )

        if not args.dry_run:
            baseline_report = winner_report or read_json(winner_run_dir / "report.json")
            for run_dir in (c1_run_dir, c2_run_dir):
                report = read_json(run_dir / "report.json")
                exact_final = dict(read_json(run_dir / "summary.json").get("exact_final_eval_summary") or {})
                progress_25k = next(
                    (
                        row.get("mean_final_progress_index")
                        for row in report.get("eval_history_table", [])
                        if int(row.get("env_step", 0)) >= 25000
                    ),
                    None,
                )
                progress_50k = next(
                    (
                        row.get("mean_final_progress_index")
                        for row in report.get("eval_history_table", [])
                        if int(row.get("env_step", 0)) >= 50000
                    ),
                    None,
                )
                offline_rows.append(
                    {
                        "run_name": str(report.get("run_name")),
                        "run_dir": str(run_dir),
                        "exact_final_progress": float(exact_final.get("mean_final_progress_index", 0.0) or 0.0),
                        "progress_25k": progress_25k,
                        "progress_50k": progress_50k,
                        "offline_init_checkpoint_path": report.get("offline_init_checkpoint_path"),
                        "offline_pretrain_strategy": dict(report.get("offline_pretrain_metadata") or {}).get(
                            "offline_pretrain_strategy"
                        ),
                    }
                )
            baseline_progress = float(
                dict(read_json(winner_run_dir / "summary.json").get("exact_final_eval_summary") or {}).get(
                    "mean_final_progress_index",
                    0.0,
                )
                or 0.0
            )
            write_json(reports_dir / "block_c_offline_summary.json", {"rows": offline_rows, "baseline_progress": baseline_progress})
            (reports_dir / "block_c_offline_summary.md").write_text(
                _render_block_c_summary(
                    winner_name=winner_run_dir.name,
                    baseline_progress=baseline_progress,
                    offline_rows=offline_rows,
                ),
                encoding="utf-8",
            )
            write_algorithm_comparison_report(
                [winner_run_dir, c1_run_dir, c2_run_dir],
                output_dir=reports_dir / "block_c_comparison",
                wall_clock_budget_minutes=args.wall_clock_minutes,
            )
        _write_status(
            status_path,
            {
                "state": "block_c_complete",
                "offline_checkpoint_path": str(offline_checkpoint_path),
                "offline_runs": [str(c1_run_dir), str(c2_run_dir)],
            },
        )

    if not args.dry_run:
        summary_lines = [
            f"session={args.session_name}",
            f"wall_clock_budget_minutes={args.wall_clock_minutes}",
            f"bundle={bundle_manifest_path}",
            f"winner_run={winner_run_dir.name}",
        ]
        if winner_report is not None:
            summary_lines.append(
                "winner_exact_final_progress="
                + str(dict(read_json(winner_run_dir / "summary.json").get("exact_final_eval_summary") or {}).get("mean_final_progress_index"))
            )
        artifact_links = {
            "block_a_reward_summary": reports_dir / "block_a_reward_summary.md",
        }
        if "B" in blocks:
            artifact_links["block_b_extraction_summary"] = reports_dir / "block_b_extraction_summary.md"
        if "C" in blocks:
            artifact_links["block_c_offline_summary"] = reports_dir / "block_c_offline_summary.md"
        append_results_entry(
            results_root=results_root,
            filename=args.results_file,
            title="rank11_100 Validation Campaign",
            summary_lines=summary_lines,
            artifact_links=artifact_links,
        )

        if not args.keep_artifacts:
            keeper_run_dirs = [winner_run_dir, *block_a_runner_up_dirs]
            if offline_rows:
                best_offline = max(offline_rows, key=lambda row: float(row.get("exact_final_progress", 0.0) or 0.0))
                keeper_run_dirs.append(Path(str(best_offline["run_dir"])).resolve())
            cleanup_artifact_root(artifact_root, keep_run_dirs=keeper_run_dirs, dry_run=False)

    _write_status(
        status_path,
        {
            "state": "complete_dry_run" if args.dry_run else "complete",
            "winner_run_dir": None if winner_run_dir is None else str(winner_run_dir),
        },
    )
    log("dry_run_complete" if args.dry_run else "campaign_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
