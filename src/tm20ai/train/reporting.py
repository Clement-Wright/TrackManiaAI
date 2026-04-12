from __future__ import annotations

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
        rows.append(
            {
                "env_step": int(entry.get("env_step", eval_summary.get("env_step", 0))),
                "learner_step": int(entry.get("learner_step", eval_summary.get("learner_step", 0))),
                "mean_final_progress_index": float(eval_summary.get("mean_final_progress_index", 0.0) or 0.0),
                "median_final_progress_index": eval_summary.get("median_final_progress_index"),
                "completion_rate": float(eval_summary.get("completion_rate", 0.0) or 0.0),
                "best_reward": eval_summary.get("best_reward"),
                "summary_path": entry.get("summary_path"),
            }
        )
    return rows


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
        "episode_count": int(summary.get("episode_count", 0)),
        "replay_size": int(summary.get("replay_size", 0)),
        "latest_checkpoint_path": summary.get("latest_checkpoint_path"),
        "latest_eval_summary_path": summary.get("latest_eval_summary_path"),
        "checkpoint_list": checkpoint_rows,
        "replay_growth": replay_growth,
        "eval_history_table": eval_rows,
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
    lines = [
        f"# Training Report: {report['run_name']}",
        "",
        "## Summary",
        f"- Observation mode: {report['observation_mode']}",
        f"- Init mode: {report['init_mode']}",
        f"- Env steps: {report['env_step']}",
        f"- Learner steps: {report['learner_step']}",
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
            f"median_progress={row['median_final_progress_index']} completion_rate={row['completion_rate']}"
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
