from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..data.parquet_writer import ensure_directory, read_json, sha256_file, timestamp_tag, write_json


DEFAULT_PROGRESS_THRESHOLDS = (50.0, 100.0, 200.0, 400.0)


@dataclass(slots=True, frozen=True)
class ReportPaths:
    json_path: Path
    markdown_path: Path


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Training run summary is missing: {summary_path}")
    return read_json(summary_path)


def _discover_video_paths(run_dir: Path, run_name: str, extra_video_paths: Iterable[str | Path]) -> list[str]:
    discovered: set[str] = set()
    for video_path in extra_video_paths:
        resolved = str(Path(video_path).resolve())
        if Path(resolved).exists():
            discovered.add(resolved)

    for candidate in run_dir.rglob("*.mp4"):
        discovered.add(str(candidate.resolve()))

    artifacts_root = run_dir.parent.parent
    eval_root = artifacts_root / "eval"
    if eval_root.exists():
        for eval_dir in eval_root.glob(f"{run_name}_step_*"):
            for candidate in eval_dir.rglob("*.mp4"):
                discovered.add(str(candidate.resolve()))

    return sorted(discovered)[:2]


def _build_eval_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in summary.get("eval_history", []):
        eval_summary = dict(entry.get("summary", {}))
        mode_summaries = {
            str(mode): dict(payload)
            for mode, payload in dict(entry.get("mode_summaries") or {}).items()
            if isinstance(payload, dict)
        }
        rows.append(
            {
                "env_step": int(entry.get("env_step", eval_summary.get("env_step", 0))),
                "learner_step": int(entry.get("learner_step", eval_summary.get("learner_step", 0))),
                "mean_final_progress_index": float(eval_summary.get("mean_final_progress_index", 0.0) or 0.0),
                "median_final_progress_index": eval_summary.get("median_final_progress_index"),
                "completion_rate": float(eval_summary.get("completion_rate", 0.0) or 0.0),
                "determinism_conversion_score": eval_summary.get("determinism_conversion_score"),
                "deterministic_stochastic_progress_gap": eval_summary.get("deterministic_stochastic_progress_gap"),
                "deterministic_stochastic_completion_gap": eval_summary.get("deterministic_stochastic_completion_gap"),
                "best_reward": eval_summary.get("best_reward"),
                "summary_path": entry.get("summary_path"),
                "mode_summaries": mode_summaries,
                "mode_summary_paths": dict(entry.get("mode_summary_paths") or {}),
                "deterministic_collapse": entry.get("deterministic_collapse"),
                "eval_provenance_mode": entry.get("eval_provenance_mode", eval_summary.get("eval_provenance_mode")),
                "eval_checkpoint_path": entry.get("eval_checkpoint_path", eval_summary.get("eval_checkpoint_path")),
                "eval_checkpoint_sha256": entry.get("eval_checkpoint_sha256", eval_summary.get("eval_checkpoint_sha256")),
                "eval_checkpoint_env_step": entry.get("eval_checkpoint_env_step", eval_summary.get("eval_checkpoint_env_step")),
                "eval_checkpoint_learner_step": entry.get(
                    "eval_checkpoint_learner_step",
                    eval_summary.get("eval_checkpoint_learner_step"),
                ),
                "eval_checkpoint_actor_step": entry.get(
                    "eval_checkpoint_actor_step",
                    eval_summary.get("eval_checkpoint_actor_step"),
                ),
                "scheduled_actor_version": entry.get("scheduled_actor_version", eval_summary.get("scheduled_actor_version")),
                "applied_actor_version": entry.get("applied_actor_version", eval_summary.get("applied_actor_version")),
                "applied_actor_source_learner_step": entry.get(
                    "applied_actor_source_learner_step",
                    eval_summary.get("applied_actor_source_learner_step"),
                ),
                "worker_env_step_at_eval_start": entry.get(
                    "worker_env_step_at_eval_start",
                    eval_summary.get("worker_env_step_at_eval_start"),
                ),
            }
        )
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _summarize_event_log(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    counts: dict[str, int] = {}
    for row in rows:
        event = str(row.get("event", "unknown"))
        counts[event] = counts.get(event, 0) + 1
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "event_count": len(rows),
        "event_type_counts": dict(sorted(counts.items())),
    }


def build_training_report(
    run_dir: str | Path,
    *,
    extra_video_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).resolve()
    summary = _load_summary(resolved_run_dir)
    config_path = None if summary.get("config_path") is None else Path(str(summary.get("config_path"))).resolve()
    eval_rows = _build_eval_rows(summary)
    checkpoint_rows = list(summary.get("checkpoint_history", []))
    replay_growth = [
        {
            "env_step": int(entry.get("env_step", 0)),
            "replay_size": int(entry.get("replay_size", entry.get("env_step", 0))),
            "learner_step": int(entry.get("learner_step", 0)),
            "path": entry.get("path"),
            "final": bool(entry.get("final", False)),
        }
        for entry in checkpoint_rows
    ]
    if not replay_growth:
        replay_growth.append(
            {
                "env_step": int(summary.get("env_step", 0)),
                "replay_size": int(summary.get("replay_size", 0)),
                "learner_step": int(summary.get("learner_step", 0)),
                "path": summary.get("latest_checkpoint_path"),
                "final": True,
            }
        )
    best_deterministic_eval = max(
        eval_rows,
        key=lambda row: float(row.get("mean_final_progress_index", 0.0) or 0.0),
        default=None,
    )

    failure_notes: list[str] = []
    termination_reason = summary.get("termination_reason")
    if termination_reason not in {None, "completed", "max_env_steps"}:
        failure_notes.append(f"termination_reason={termination_reason}")
    if summary.get("clean_shutdown") is False:
        failure_notes.append("run did not shut down cleanly")
    worker_exit = summary.get("worker_exit", {})
    if worker_exit.get("terminated"):
        failure_notes.append("worker process was force-terminated")

    report = {
        "run_name": summary.get("run_name"),
        "run_dir": str(resolved_run_dir),
        "config_path": summary.get("config_path"),
        "config_sha256": None if config_path is None or not config_path.exists() else sha256_file(config_path),
        "observation_mode": summary.get("observation_mode"),
        "algorithm": summary.get("algorithm"),
        "primary_metric": summary.get("primary_metric", "mean_final_progress_index"),
        "metric_version": summary.get("metric_version"),
        "init_mode": summary.get("init_mode", "scratch"),
        "bc_checkpoint_path": summary.get("bc_checkpoint_path"),
        "demo_root": summary.get("demo_root"),
        "replay_seeded": bool(summary.get("replay_seeded", False)),
        "device": summary.get("device"),
        "run_start_timestamp": summary.get("run_start_timestamp"),
        "run_end_timestamp": summary.get("run_end_timestamp"),
        "training_duration_seconds": summary.get("wall_clock_elapsed_seconds"),
        "env_step": int(summary.get("env_step", 0)),
        "learner_step": int(summary.get("learner_step", 0)),
        "achieved_utd_1k": summary.get("achieved_utd_1k"),
        "cumulative_utd": summary.get("cumulative_utd"),
        "current_actor_staleness": summary.get("current_actor_staleness"),
        "episode_count": int(summary.get("episode_count", 0)),
        "replay_size": int(summary.get("replay_size", 0)),
        "latest_checkpoint_path": summary.get("latest_checkpoint_path"),
        "latest_eval_summary_path": summary.get("latest_eval_summary_path"),
        "latest_eval_mode_summaries": dict(summary.get("latest_eval_mode_summaries") or {}),
        "latest_eval_mode_summary_paths": dict(summary.get("latest_eval_mode_summary_paths") or {}),
        "checkpoint_list": checkpoint_rows,
        "replay_growth": replay_growth,
        "eval_history_table": eval_rows,
        "best_deterministic_eval": best_deterministic_eval,
        "best_progress_over_time": [
            {"env_step": row["env_step"], "mean_final_progress_index": row["mean_final_progress_index"]}
            for row in eval_rows
        ],
        "median_progress_over_time": [
            {"env_step": row["env_step"], "median_final_progress_index": row["median_final_progress_index"]}
            for row in eval_rows
        ],
        "videos": _discover_video_paths(resolved_run_dir, str(summary.get("run_name", "")), extra_video_paths),
        "failure_notes": failure_notes,
        "runtime_profile": dict(summary.get("runtime_profile", {})),
        "queue_profile": dict(summary.get("queue_profile", {})),
        "actor_sync_profile": dict(summary.get("actor_sync_profile", {})),
        "episode_diagnostics": dict(summary.get("episode_diagnostics", {})),
        "movement_diagnostics": dict(summary.get("movement_diagnostics", {})),
        "resource_profile": dict(summary.get("resource_profile", {})),
        "event_logs": {
            "learner": _summarize_event_log(resolved_run_dir / "learner_events.log"),
            "worker": _summarize_event_log(resolved_run_dir / "worker_events.log"),
        },
    }
    return report


def _threshold_crossings(
    eval_rows: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float],
) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for threshold in thresholds:
        crossing = next(
            (int(row["env_step"]) for row in eval_rows if float(row["mean_final_progress_index"]) >= float(threshold)),
            None,
        )
        result[str(float(threshold))] = crossing
    return result


def build_comparison_report(
    run_dirs: Sequence[str | Path],
    *,
    progress_thresholds: Sequence[float] = DEFAULT_PROGRESS_THRESHOLDS,
) -> dict[str, Any]:
    reports = [build_training_report(run_dir) for run_dir in run_dirs]
    if not reports:
        raise RuntimeError("At least one run_dir is required to build a comparison report.")

    labeled_reports: list[tuple[str, dict[str, Any]]] = []
    for report in reports:
        init_mode = str(report.get("init_mode", "scratch"))
        label = init_mode if init_mode != "scratch" else "scratch"
        if any(existing_label == label for existing_label, _ in labeled_reports):
            label = str(report.get("run_name"))
        labeled_reports.append((label, report))

    all_steps = sorted(
        {
            int(row["env_step"])
            for _label, report in labeled_reports
            for row in report.get("eval_history_table", [])
        }
    )
    matched_step_table: list[dict[str, Any]] = []
    for step in all_steps:
        row: dict[str, Any] = {"env_step": step}
        for label, report in labeled_reports:
            progress = next(
                (
                    float(entry["mean_final_progress_index"])
                    for entry in report.get("eval_history_table", [])
                    if int(entry["env_step"]) == step
                ),
                None,
            )
            row[label] = progress
        matched_step_table.append(row)

    threshold_table = {
        label: _threshold_crossings(report.get("eval_history_table", []), thresholds=progress_thresholds)
        for label, report in labeled_reports
    }
    best_progress_by_run = {
        label: max(
            (float(entry["mean_final_progress_index"]) for entry in report.get("eval_history_table", [])),
            default=0.0,
        )
        for label, report in labeled_reports
    }
    best_label = max(best_progress_by_run, key=best_progress_by_run.get)

    conclusion = f"{best_label} currently has the highest best mean final progress."
    scratch_thresholds = threshold_table.get("scratch", {})
    bc_better = []
    for label, _report in labeled_reports:
        if label == "scratch":
            continue
        faster = [
            threshold
            for threshold, env_step in threshold_table[label].items()
            if env_step is not None
            and (
                scratch_thresholds.get(threshold) is None
                or env_step < scratch_thresholds.get(threshold)
            )
        ]
        if faster:
            bc_better.append(label)
    if bc_better:
        conclusion = (
            ", ".join(bc_better)
            + " reached at least one progress threshold faster than scratch, which suggests a BC warm-start gain."
        )

    return {
        "run_count": len(labeled_reports),
        "progress_thresholds": [float(value) for value in progress_thresholds],
        "runs": {
            label: {
                "run_name": report.get("run_name"),
                "run_dir": report.get("run_dir"),
                "init_mode": report.get("init_mode"),
                "best_progress": best_progress_by_run[label],
                "latest_checkpoint_path": report.get("latest_checkpoint_path"),
                "demo_root": report.get("demo_root"),
                "bc_checkpoint_path": report.get("bc_checkpoint_path"),
            }
            for label, report in labeled_reports
        },
        "matched_step_table": matched_step_table,
        "time_to_progress_thresholds": threshold_table,
        "best_progress_by_run": best_progress_by_run,
        "conclusion": conclusion,
    }


def _render_run_report_markdown(report: dict[str, Any]) -> str:
    runtime_profile = dict(report.get("runtime_profile", {}))
    bottleneck = dict(runtime_profile.get("bottleneck_verdict", {}))
    actor_sync_profile = dict(report.get("actor_sync_profile", {}))
    episode_diagnostics = dict(report.get("episode_diagnostics", {}))
    movement_diagnostics = dict(report.get("movement_diagnostics", {}))
    resource_profile = dict(report.get("resource_profile", {}))
    event_logs = dict(report.get("event_logs", {}))
    lines = [
        f"# Training Report: {report['run_name']}",
        "",
        "## Summary",
        f"- Observation mode: {report['observation_mode']}",
        f"- Algorithm: {report.get('algorithm')}",
        f"- Init mode: {report['init_mode']}",
        f"- Primary metric: {report.get('primary_metric')}",
        f"- Env steps: {report['env_step']}",
        f"- Learner steps: {report['learner_step']}",
        f"- Achieved UTD (1k window): {report.get('achieved_utd_1k')}",
        f"- Cumulative UTD: {report.get('cumulative_utd')}",
        f"- Current actor staleness: {report.get('current_actor_staleness')}",
        f"- Replay size: {report['replay_size']}",
        f"- Training duration (s): {report.get('training_duration_seconds')}",
        "",
        "## Checkpoints",
    ]
    for checkpoint in report.get("checkpoint_list", []):
        lines.append(
            f"- env_step={checkpoint.get('env_step')} replay_size={checkpoint.get('replay_size')} path={checkpoint.get('path')}"
        )
    lines.extend(["", "## Eval History"])
    for row in report.get("eval_history_table", []):
        lines.append(
            f"- env_step={row['env_step']} mean_progress={row['mean_final_progress_index']} "
            f"median_progress={row['median_final_progress_index']} completion_rate={row['completion_rate']} "
            f"dcs={row.get('determinism_conversion_score')}"
        )
        if row.get("eval_checkpoint_path") is not None:
            lines.append(
                f"-   provenance: mode={row.get('eval_provenance_mode')} "
                f"checkpoint={row.get('eval_checkpoint_path')} "
                f"checkpoint_env_step={row.get('eval_checkpoint_env_step')} "
                f"checkpoint_learner_step={row.get('eval_checkpoint_learner_step')}"
            )
        for mode_name, mode_summary in sorted(dict(row.get("mode_summaries", {})).items()):
            if mode_name == "deterministic":
                continue
            lines.append(
                f"-   {mode_name}: mean_progress={mode_summary.get('mean_final_progress_index')} "
                f"completion_rate={mode_summary.get('completion_rate')} "
                f"summary_path={dict(row.get('mode_summary_paths', {})).get(mode_name)}"
            )
        collapse = row.get("deterministic_collapse")
        if isinstance(collapse, dict):
            lines.append(
                f"-   deterministic_collapse: meaningfully_outperformed={collapse.get('meaningfully_outperformed')} "
                f"progress_delta={collapse.get('progress_delta')} "
                f"completion_rate_delta={collapse.get('completion_rate_delta')} "
                f"dcs={row.get('determinism_conversion_score')}"
            )
    lines.extend(["", "## Diagnostics"])
    if bottleneck:
        lines.append(f"- Bottleneck verdict: {bottleneck.get('label')}")
        for key, value in dict(bottleneck.get("breakdown_seconds", {})).items():
            lines.append(f"- {key}_seconds={value}")
    if actor_sync_profile:
        lines.append(
            f"- achieved_utd_1k={report.get('achieved_utd_1k')} cumulative_utd={report.get('cumulative_utd')} "
            f"current_actor_staleness={report.get('current_actor_staleness')}"
        )
        lines.append(
            f"- time_to_first_ready_actor_seconds={actor_sync_profile.get('time_to_first_ready_actor_seconds')} "
            f"time_to_first_applied_ready_actor_seconds={actor_sync_profile.get('time_to_first_applied_ready_actor_seconds')} "
            f"time_to_first_policy_control_window_seconds={actor_sync_profile.get('time_to_first_policy_control_window_seconds')}"
        )
        lines.append(
            f"- policy_control_fraction={actor_sync_profile.get('policy_control_fraction')} "
            f"current_versions_behind={actor_sync_profile.get('current_versions_behind')} "
            f"applied_lag_p50={dict(actor_sync_profile.get('time_to_applied_seconds', {})).get('p50')} "
            f"applied_lag_p95={dict(actor_sync_profile.get('time_to_applied_seconds', {})).get('p95')}"
        )
    if episode_diagnostics:
        lines.append(
            f"- positive_progress_mean={dict(episode_diagnostics.get('positive_progress_fraction', {})).get('mean')} "
            f"nonpositive_progress_mean={dict(episode_diagnostics.get('nonpositive_progress_fraction', {})).get('mean')} "
            f"max_no_progress_p95={dict(episode_diagnostics.get('max_no_progress_streak', {})).get('p95')}"
        )
    if movement_diagnostics:
        lines.append(
            f"- no_movement_episode_count={movement_diagnostics.get('no_movement_episode_count')} "
            f"stall_episode_rate={movement_diagnostics.get('stall_episode_rate')} "
            f"first_stall_delay_p95_ms={dict(movement_diagnostics.get('first_stall_delay_ms', {})).get('p95')}"
        )
    if resource_profile:
        lines.append(
            f"- actor_params={resource_profile.get('actor_parameter_count')} "
            f"critic_params={resource_profile.get('critic_parameter_count')} "
            f"unique_critic_encoder_params={resource_profile.get('unique_critic_encoder_parameter_count')}"
        )
    lines.extend(["", "## Event Logs"])
    for label, payload in event_logs.items():
        lines.append(
            f"- {label}: events={payload.get('event_count')} path={payload.get('path')}"
        )
    lines.extend(["", "## Videos"])
    videos = report.get("videos", [])
    if videos:
        for video in videos:
            lines.append(f"- {video}")
    else:
        lines.append("- No rollout videos were discovered for this run.")
    lines.extend(["", "## Failure Notes"])
    if report.get("failure_notes"):
        for note in report["failure_notes"]:
            lines.append(f"- {note}")
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def _render_comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Training Comparison Report",
        "",
        "## Runs",
    ]
    for label, run in report.get("runs", {}).items():
        lines.append(
            f"- {label}: init_mode={run.get('init_mode')} best_progress={run.get('best_progress')} "
            f"checkpoint={run.get('latest_checkpoint_path')}"
        )
    lines.extend(["", "## Matched Steps"])
    for row in report.get("matched_step_table", []):
        metrics = ", ".join(f"{key}={value}" for key, value in row.items() if key != "env_step")
        lines.append(f"- env_step={row['env_step']}: {metrics}")
    lines.extend(["", "## Threshold Crossings"])
    for label, thresholds in report.get("time_to_progress_thresholds", {}).items():
        formatted = ", ".join(f"{threshold}->{step}" for threshold, step in thresholds.items())
        lines.append(f"- {label}: {formatted}")
    lines.extend(["", "## Conclusion", report.get("conclusion", "")])
    return "\n".join(lines) + "\n"


def write_training_report(
    run_dir: str | Path,
    *,
    extra_video_paths: Sequence[str | Path] = (),
) -> ReportPaths:
    resolved_run_dir = Path(run_dir).resolve()
    report = build_training_report(resolved_run_dir, extra_video_paths=extra_video_paths)
    json_path = resolved_run_dir / "report.json"
    markdown_path = resolved_run_dir / "report.md"
    write_json(json_path, report)
    markdown_path.write_text(_render_run_report_markdown(report), encoding="utf-8")
    return ReportPaths(json_path=json_path, markdown_path=markdown_path)


def write_comparison_report(
    run_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path | None = None,
    progress_thresholds: Sequence[float] = DEFAULT_PROGRESS_THRESHOLDS,
) -> ReportPaths:
    report = build_comparison_report(run_dirs, progress_thresholds=progress_thresholds)
    base_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else ensure_directory(Path(run_dirs[0]).resolve().parent.parent / "reports" / f"comparison_{timestamp_tag()}")
    )
    ensure_directory(base_dir)
    json_path = base_dir / "comparison_report.json"
    markdown_path = base_dir / "comparison_report.md"
    write_json(json_path, report)
    markdown_path.write_text(_render_comparison_markdown(report), encoding="utf-8")
    return ReportPaths(json_path=json_path, markdown_path=markdown_path)
