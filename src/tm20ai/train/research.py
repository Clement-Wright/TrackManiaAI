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


def _is_exact_final_eval_row(row: Mapping[str, Any]) -> bool:
    summary = dict(row.get("summary") or {})
    if bool(summary.get("final_checkpoint_eval", False)):
        return True
    summary_path = str(row.get("summary_path") or "")
    run_dir = str(row.get("run_dir") or "")
    return "_final_exact_step_" in summary_path or "_final_exact_step_" in run_dir


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
        exact_final_deterministic_row = next((row for row in deterministic_rows if _is_exact_final_eval_row(row)), None)
        exact_final_stochastic_row = next((row for row in stochastic_rows if _is_exact_final_eval_row(row)), None)
        exact_final_eval = dict(report.get("exact_final_eval") or {})
        if not exact_final_eval and exact_final_deterministic_row is not None:
            exact_final_eval = dict(exact_final_deterministic_row.get("summary") or {})
            exact_final_eval["summary_path"] = exact_final_deterministic_row.get("summary_path")
            exact_final_eval["mode_summaries"] = {
                "deterministic": dict(exact_final_deterministic_row.get("summary") or {}),
                **(
                    {}
                    if exact_final_stochastic_row is None
                    else {"stochastic": dict(exact_final_stochastic_row.get("summary") or {})}
                ),
            }
        exact_final_eval_complete = bool(
            report.get("exact_final_eval_complete", False)
            or (exact_final_deterministic_row is not None and exact_final_stochastic_row is not None)
        )
        best_eval = max(deterministic_rows, key=lambda row: row["mean_final_progress_index"]) if deterministic_rows else dict(report.get("best_deterministic_eval") or {})
        best_eval_summary = dict(best_eval.get("summary") or best_eval)
        exact_stochastic_summary = dict((exact_final_eval.get("mode_summaries") or {}).get("stochastic") or {})
        stochastic_summary = (
            exact_stochastic_summary
            if exact_final_eval_complete and exact_stochastic_summary
            else dict(max(stochastic_rows, key=lambda row: row["mean_final_progress_index"]).get("summary") or {})
            if stochastic_rows
            else dict(best_eval_summary.get("mode_summaries", {}).get("stochastic") or {})
        )
        scoreboard.append(
            {
                "algorithm": report.get("algorithm"),
                "run_name": report.get("run_name"),
                "run_dir": report.get("run_dir"),
                "final_eval_state": report.get(
                    "final_eval_state",
                    "complete" if exact_final_eval_complete else "exact_final_eval_missing",
                ),
                "exact_final_eval_complete": exact_final_eval_complete,
                "incomplete_final_eval": bool(report.get("incomplete_final_eval", not exact_final_eval_complete)),
                "env_steps_reached": int(report.get("env_step", 0)),
                "learner_steps_reached": int(report.get("learner_step", 0)),
                "training_duration_seconds": report.get("training_duration_seconds"),
                "exact_final_deterministic_mean_final_progress_index": float(
                    exact_final_eval.get("mean_final_progress_index", 0.0) or 0.0
                ),
                "best_deterministic_mean_final_progress_index": float(
                    exact_final_eval.get("mean_final_progress_index", 0.0) or 0.0
                ),
                "exact_final_deterministic_mean_final_progress_meters": exact_final_eval.get(
                    "mean_final_progress_meters"
                ),
                "best_deterministic_mean_final_progress_meters": exact_final_eval.get(
                    "mean_final_progress_meters"
                ),
                "exact_final_deterministic_mean_progress_fraction_of_reference": exact_final_eval.get(
                    "mean_progress_fraction_of_reference"
                ),
                "best_deterministic_mean_progress_fraction_of_reference": exact_final_eval.get(
                    "mean_progress_fraction_of_reference"
                ),
                "exact_final_deterministic_mean_ghost_relative_time_delta_ms": exact_final_eval.get(
                    "mean_ghost_relative_time_delta_ms"
                ),
                "best_deterministic_mean_ghost_relative_time_delta_ms": exact_final_eval.get(
                    "mean_ghost_relative_time_delta_ms"
                ),
                "exact_final_deterministic_progress_index_semantics": exact_final_eval.get(
                    "progress_index_semantics"
                ),
                "best_deterministic_progress_index_semantics": exact_final_eval.get(
                    "progress_index_semantics"
                ),
                "exact_final_deterministic_progress_spacing_meters": exact_final_eval.get(
                    "progress_spacing_meters"
                ),
                "best_deterministic_progress_spacing_meters": exact_final_eval.get(
                    "progress_spacing_meters"
                ),
                "exact_final_deterministic_eval_env_step": exact_final_eval.get("env_step"),
                "exact_final_deterministic_summary_path": exact_final_eval.get("summary_path"),
                "exact_final_deterministic_checkpoint_path": exact_final_eval.get("eval_checkpoint_path"),
                "best_deterministic_checkpoint_path": exact_final_eval.get("eval_checkpoint_path"),
                "exact_final_deterministic_checkpoint_sha256": exact_final_eval.get("eval_checkpoint_sha256"),
                "exact_final_deterministic_checkpoint_env_step": exact_final_eval.get("eval_checkpoint_env_step"),
                "exact_final_deterministic_checkpoint_learner_step": exact_final_eval.get("eval_checkpoint_learner_step"),
                "exact_final_deterministic_checkpoint_actor_step": exact_final_eval.get("eval_checkpoint_actor_step"),
                "best_deterministic_checkpoint_progress": float(
                    best_eval_summary.get("mean_final_progress_index", 0.0) or 0.0
                ),
                "best_stochastic_mean_final_progress_index": float(
                    stochastic_summary.get("mean_final_progress_index", 0.0) or 0.0
                ),
                "best_stochastic_mean_final_progress_meters": stochastic_summary.get("mean_final_progress_meters"),
                "best_stochastic_mean_progress_fraction_of_reference": stochastic_summary.get(
                    "mean_progress_fraction_of_reference"
                ),
                "best_stochastic_mean_ghost_relative_time_delta_ms": stochastic_summary.get(
                    "mean_ghost_relative_time_delta_ms"
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
            bool(row["exact_final_eval_complete"]),
            float(row["exact_final_deterministic_mean_final_progress_index"]),
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
        "headline_metric": "exact final checkpoint-backed deterministic mean_final_progress_index",
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
            f"- {row.get('algorithm')}: exact_final_complete={row.get('exact_final_eval_complete')} "
            f"final_eval_state={row.get('final_eval_state')} "
            f"exact_final_progress={row.get('exact_final_deterministic_mean_final_progress_index')} "
            f"exact_final_m={row.get('exact_final_deterministic_mean_final_progress_meters')} "
            f"exact_final_fraction={row.get('exact_final_deterministic_mean_progress_fraction_of_reference')} "
            f"env_steps={row.get('env_steps_reached')} learner_steps={row.get('learner_steps_reached')} "
            f"final_checkpoint={row.get('exact_final_deterministic_checkpoint_path')}"
        )
        lines.append(
            f"-   stochastic_progress={row.get('best_stochastic_mean_final_progress_index')} "
            f"stochastic_m={row.get('best_stochastic_mean_final_progress_meters')} "
            f"stochastic_fraction={row.get('best_stochastic_mean_progress_fraction_of_reference')} "
            f"ghost_delta_ms={row.get('exact_final_deterministic_mean_ghost_relative_time_delta_ms')} "
            f"dcs={row.get('determinism_conversion_score')} "
            f"semantics={row.get('exact_final_deterministic_progress_index_semantics')} "
            f"spacing_m={row.get('exact_final_deterministic_progress_spacing_meters')} "
            f"achieved_utd_1k={row.get('achieved_utd_1k')} cumulative_utd={row.get('cumulative_utd')} "
            f"current_actor_staleness={row.get('current_actor_staleness')} "
            f"best_checkpoint_progress={row.get('best_deterministic_checkpoint_progress')} "
            f"incomplete_final_eval={row.get('incomplete_final_eval')}"
        )
    winner = dict(report.get("winner") or {})
    if winner:
        lines.extend(
            [
                "",
                "## Winner",
                f"- algorithm={winner.get('algorithm')}",
                f"- exact_final_progress={winner.get('exact_final_deterministic_mean_final_progress_index')}",
                f"- exact_final_progress_meters={winner.get('exact_final_deterministic_mean_final_progress_meters')}",
                f"- checkpoint={winner.get('exact_final_deterministic_checkpoint_path')}",
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
