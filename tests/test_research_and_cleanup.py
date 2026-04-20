from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tm20ai.train.artifact_retention import cleanup_artifact_root, select_keeper_run_dirs
from tm20ai.train.research import append_results_entry, write_algorithm_comparison_report


ROOT = Path(__file__).resolve().parents[1]


def _write_train_run(
    artifact_root: Path,
    *,
    algorithm: str,
    run_name: str,
    best_progress: float,
    run_end_timestamp: str,
) -> Path:
    run_dir = artifact_root / "train" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = artifact_root / "eval" / f"{run_name}_final_exact_step_00001000_deterministic"
    eval_dir.mkdir(parents=True, exist_ok=True)
    deterministic_summary = {
        "eval_mode": "deterministic",
        "mean_final_progress_index": best_progress,
        "completion_rate": 0.0,
        "eval_checkpoint_path": str(run_dir / "checkpoints" / "checkpoint_00001000_final.pt"),
        "eval_checkpoint_sha256": f"sha-{run_name}",
        "eval_checkpoint_env_step": 1000,
        "eval_checkpoint_learner_step": 2000,
        "eval_checkpoint_actor_step": 100,
        "env_step": 1000,
    }
    (eval_dir / "summary.json").write_text(json.dumps(deterministic_summary, indent=2), encoding="utf-8")
    stochastic_dir = artifact_root / "eval" / f"{run_name}_final_exact_step_00001000_stochastic"
    stochastic_dir.mkdir(parents=True, exist_ok=True)
    (stochastic_dir / "summary.json").write_text(
        json.dumps(
            {
                **deterministic_summary,
                "eval_mode": "stochastic",
                "mean_final_progress_index": best_progress - 10.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "run_name": run_name,
        "algorithm": algorithm,
        "config_path": "C:/config.yaml",
        "device": "cpu",
        "run_start_timestamp": "2026-04-19T00:00:00+00:00",
        "run_end_timestamp": run_end_timestamp,
        "wall_clock_elapsed_seconds": 3600.0,
        "env_step": 1000,
        "learner_step": 2000,
        "episode_count": 10,
        "replay_size": 1000,
        "init_mode": "scratch",
        "eval_episodes": 5,
        "latest_checkpoint_path": str(run_dir / "checkpoints" / "checkpoint_00001000_final.pt"),
        "latest_eval_summary": deterministic_summary,
        "latest_eval_summary_path": str(eval_dir / "summary.json"),
        "latest_eval_mode_summaries": {
            "deterministic": deterministic_summary,
            "stochastic": {**deterministic_summary, "eval_mode": "stochastic", "mean_final_progress_index": best_progress - 10.0},
        },
        "latest_eval_mode_summary_paths": {
            "deterministic": str(eval_dir / "summary.json"),
            "stochastic": str(stochastic_dir / "summary.json"),
        },
        "eval_in_flight": False,
        "pending_eval": None,
        "checkpoint_history": [
            {
                "path": str(run_dir / "checkpoints" / "checkpoint_00001000_final.pt"),
                "env_step": 1000,
                "learner_step": 2000,
                "algorithm": algorithm,
                "replay_size": 1000,
                "timestamp": run_end_timestamp,
                "final": True,
            }
        ],
        "eval_history": [],
        "termination_reason": "max_wall_clock_minutes",
        "clean_shutdown": True,
        "worker_exit": {"done_event_set": True, "exitcode": 0, "terminated": False, "timeout": False},
        "timestamp": run_end_timestamp,
        "observation_mode": "full",
        "primary_metric": "mean_final_progress_index",
        "achieved_utd_1k": 1.0,
        "cumulative_utd": 2.0,
        "current_actor_staleness": 3,
        "runtime_profile": {},
        "queue_profile": {},
        "actor_sync_profile": {},
        "episode_diagnostics": {},
        "movement_diagnostics": {},
        "resource_profile": {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def test_write_algorithm_comparison_report_and_results_log(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    redq_dir = _write_train_run(
        artifact_root,
        algorithm="redq",
        run_name="session_redq",
        best_progress=1500.0,
        run_end_timestamp="2026-04-19T00:30:00+00:00",
    )
    droq_dir = _write_train_run(
        artifact_root,
        algorithm="droq",
        run_name="session_droq",
        best_progress=1800.0,
        run_end_timestamp="2026-04-19T00:40:00+00:00",
    )

    report_paths = write_algorithm_comparison_report(
        [redq_dir, droq_dir],
        output_dir=tmp_path / "results" / "comparison",
        wall_clock_budget_minutes=90.0,
    )
    report = json.loads(report_paths.json_path.read_text(encoding="utf-8"))
    assert report["winner"]["algorithm"] == "droq"
    assert report["winner"]["best_deterministic_mean_final_progress_index"] == 1800.0

    results_path = append_results_entry(
        results_root=tmp_path / "results",
        filename="algorithm_ladder.md",
        title="Algorithm Ladder",
        summary_lines=["winner=droq", "wall_clock_budget_minutes=90.0"],
        artifact_links={"comparison_markdown": report_paths.markdown_path},
    )
    body = results_path.read_text(encoding="utf-8")
    assert "winner=droq" in body
    assert "comparison_markdown" in body


def test_artifact_cleanup_keeps_best_and_latest_runs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    best_redq = _write_train_run(
        artifact_root,
        algorithm="redq",
        run_name="best_redq",
        best_progress=2000.0,
        run_end_timestamp="2026-04-19T02:00:00+00:00",
    )
    stale_redq = _write_train_run(
        artifact_root,
        algorithm="redq",
        run_name="stale_redq",
        best_progress=500.0,
        run_end_timestamp="2026-04-19T01:00:00+00:00",
    )
    latest_crossq = _write_train_run(
        artifact_root,
        algorithm="crossq",
        run_name="latest_crossq",
        best_progress=300.0,
        run_end_timestamp="2026-04-19T03:00:00+00:00",
    )

    keepers = select_keeper_run_dirs(artifact_root, keep_best_per_algorithm=1, keep_latest_per_algorithm=1)
    assert best_redq in keepers
    assert latest_crossq in keepers

    cleanup_result = cleanup_artifact_root(artifact_root, keep_run_dirs=keepers, dry_run=False)
    assert str(stale_redq.resolve()) in cleanup_result.removed_paths
    assert not stale_redq.exists()
    assert best_redq.exists()


def test_run_algorithm_ladder_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_algorithm_ladder.py"),
            "--algorithms",
            "redq,droq",
            "--wall-clock-minutes",
            "1",
            "--artifact-root",
            str(tmp_path / ".tmp" / "artifacts" / "ladder"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "dry_run_complete" in result.stdout
