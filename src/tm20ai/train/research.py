from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data.parquet_writer import ensure_directory, read_json, write_json
from .reporting import build_training_report


def _utc_date_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_eval_rows_for_run(run_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    run_name = str(run_report.get("run_name") or "")
    run_dir = Path(str(run_report.get("run_dir"))).resolve()
    eval_root = run_dir.parent.parent / "eval"
    rows: list[dict[str, Any]] = []
    if not eval_root.exists():
        return rows
    for eval_dir in eval_root.iterdir():
        if not eval_dir.is_dir() or not eval_dir.name.startswith(run_name):
            continue
        summary_path = eval_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        rows.append(
            {
                "run_dir": str(eval_dir.resolve()),
                "summary_path": str(summary_path.resolve()),
                "eval_mode": summary.get("eval_mode"),
                "mean_final_progress_index": float(summary.get("mean_final_progress_index", 0.0) or 0.0),
                "completion_rate": float(summary.get("completion_rate", 0.0) or 0.0),
                "summary": summary,
            }
        )
    return rows


@dataclass(slots=True, frozen=True)
class ResearchReportPaths:
    json_path: Path
    markdown_path: Path


def build_algorithm_comparison_report(
    run_dirs: Sequence[str | Path],
    *,
    wall_clock_budget_minutes: float | None = None,
) -> dict[str, Any]:
    reports = [build_training_report(run_dir) for run_dir in run_dirs]
    if not reports:
        raise RuntimeError("At least one run_dir is required to build an algorithm comparison report.")

    scoreboard: list[dict[str, Any]] = []
    for report in reports:
        all_eval_rows = _collect_eval_rows_for_run(report)
        deterministic_rows = [row for row in all_eval_rows if row.get("eval_mode") == "deterministic"]
        stochastic_rows = [row for row in all_eval_rows if row.get("eval_mode") == "stochastic"]
        best_eval = (
            max(deterministic_rows, key=lambda row: row["mean_final_progress_index"])
            if deterministic_rows
            else dict(report.get("best_deterministic_eval") or {})
        )
        best_eval_summary = dict(best_eval.get("summary") or best_eval)
        stochastic_summary = (
            dict(max(stochastic_rows, key=lambda row: row["mean_final_progress_index"]).get("summary") or {})
            if stochastic_rows
            else dict(best_eval_summary.get("mode_summaries", {}).get("stochastic") or {})
        )
        scoreboard.append(
            {
                "algorithm": report.get("algorithm"),
                "run_name": report.get("run_name"),
                "run_dir": report.get("run_dir"),
                "env_steps_reached": int(report.get("env_step", 0)),
                "learner_steps_reached": int(report.get("learner_step", 0)),
                "training_duration_seconds": report.get("training_duration_seconds"),
                "best_deterministic_mean_final_progress_index": float(
                    best_eval_summary.get("mean_final_progress_index", 0.0) or 0.0
                ),
                "best_deterministic_eval_env_step": best_eval_summary.get("env_step"),
                "best_deterministic_summary_path": best_eval.get("summary_path"),
                "best_deterministic_checkpoint_path": best_eval_summary.get("eval_checkpoint_path"),
                "best_deterministic_checkpoint_sha256": best_eval_summary.get("eval_checkpoint_sha256"),
                "best_deterministic_checkpoint_env_step": best_eval_summary.get("eval_checkpoint_env_step"),
                "best_deterministic_checkpoint_learner_step": best_eval_summary.get("eval_checkpoint_learner_step"),
                "best_deterministic_checkpoint_actor_step": best_eval_summary.get("eval_checkpoint_actor_step"),
                "best_stochastic_mean_final_progress_index": float(
                    stochastic_summary.get("mean_final_progress_index", 0.0) or 0.0
                ),
                "best_stochastic_completion_rate": float(stochastic_summary.get("completion_rate", 0.0) or 0.0),
                "determinism_conversion_score": best_eval_summary.get("determinism_conversion_score"),
                "deterministic_stochastic_progress_gap": best_eval_summary.get(
                    "deterministic_stochastic_progress_gap"
                ),
                "achieved_utd_1k": report.get("achieved_utd_1k"),
                "cumulative_utd": report.get("cumulative_utd"),
                "current_actor_staleness": report.get("current_actor_staleness"),
                "latest_checkpoint_path": report.get("latest_checkpoint_path"),
            }
        )
    scoreboard.sort(
        key=lambda row: (
            float(row["best_deterministic_mean_final_progress_index"]),
            int(row["env_steps_reached"]),
        ),
        reverse=True,
    )
    winner = scoreboard[0]
    return {
        "generated_at": _utc_timestamp(),
        "wall_clock_budget_minutes": None if wall_clock_budget_minutes is None else float(wall_clock_budget_minutes),
        "run_count": len(scoreboard),
        "scoreboard": scoreboard,
        "winner": winner,
        "headline_metric": "checkpoint-backed deterministic mean_final_progress_index",
    }


def _render_algorithm_comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Algorithm Comparison Report",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Wall-clock budget (minutes): {report.get('wall_clock_budget_minutes')}",
        f"- Headline metric: {report.get('headline_metric')}",
        "",
        "## Scoreboard",
    ]
    for row in report.get("scoreboard", []):
        lines.append(
            f"- {row.get('algorithm')}: best_deterministic_progress={row.get('best_deterministic_mean_final_progress_index')} "
            f"env_steps={row.get('env_steps_reached')} learner_steps={row.get('learner_steps_reached')} "
            f"best_checkpoint={row.get('best_deterministic_checkpoint_path')}"
        )
        lines.append(
            f"-   stochastic_progress={row.get('best_stochastic_mean_final_progress_index')} "
            f"dcs={row.get('determinism_conversion_score')} "
            f"achieved_utd_1k={row.get('achieved_utd_1k')} cumulative_utd={row.get('cumulative_utd')} "
            f"current_actor_staleness={row.get('current_actor_staleness')}"
        )
    winner = dict(report.get("winner") or {})
    if winner:
        lines.extend(
            [
                "",
                "## Winner",
                f"- algorithm={winner.get('algorithm')}",
                f"- best_deterministic_progress={winner.get('best_deterministic_mean_final_progress_index')}",
                f"- checkpoint={winner.get('best_deterministic_checkpoint_path')}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_algorithm_comparison_report(
    run_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    wall_clock_budget_minutes: float | None = None,
) -> ResearchReportPaths:
    resolved_output_dir = ensure_directory(Path(output_dir).resolve())
    report = build_algorithm_comparison_report(
        run_dirs,
        wall_clock_budget_minutes=wall_clock_budget_minutes,
    )
    json_path = resolved_output_dir / "algorithm_comparison_report.json"
    markdown_path = resolved_output_dir / "algorithm_comparison_report.md"
    write_json(json_path, report)
    markdown_path.write_text(_render_algorithm_comparison_markdown(report), encoding="utf-8")
    return ResearchReportPaths(json_path=json_path, markdown_path=markdown_path)


def append_results_entry(
    *,
    results_root: str | Path,
    filename: str,
    title: str,
    summary_lines: Sequence[str],
    artifact_links: Mapping[str, str | Path] | None = None,
) -> Path:
    resolved_root = ensure_directory(Path(results_root).resolve())
    results_path = resolved_root / filename
    entry_lines = [
        f"## {_utc_date_stamp()} - {title}",
        "",
    ]
    entry_lines.extend(f"- {line}" for line in summary_lines)
    if artifact_links:
        entry_lines.extend(["", "Artifacts:"])
        entry_lines.extend(f"- {label}: {Path(path).resolve()}" for label, path in artifact_links.items())
    entry_lines.append("")
    with results_path.open("a", encoding="utf-8") as handle:
        if results_path.stat().st_size == 0:
            handle.write(f"# {title}\n\n")
        handle.write("\n".join(entry_lines))
        handle.write("\n")
    return results_path


def read_algorithm_comparison_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))
