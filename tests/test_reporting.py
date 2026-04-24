from __future__ import annotations

import json
from pathlib import Path

from tm20ai.train.reporting import write_comparison_report, write_training_report


def _write_summary(
    run_dir: Path,
    *,
    run_name: str,
    init_mode: str,
    env_step: int,
    learner_step: int,
    replay_size: int,
    eval_rows: list[dict],
    checkpoint_rows: list[dict],
    termination_reason: str = "max_env_steps",
    clean_shutdown: bool = True,
    extra_summary: dict | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": run_name,
        "config_path": "C:/config.yaml",
        "device": "cpu",
        "run_start_timestamp": "2026-04-11T00:00:00+00:00",
        "run_end_timestamp": "2026-04-11T00:10:00+00:00",
        "wall_clock_elapsed_seconds": 600.0,
        "env_step": env_step,
        "learner_step": learner_step,
        "episode_count": 12,
        "replay_size": replay_size,
        "init_mode": init_mode,
        "bc_checkpoint_path": None if init_mode == "scratch" else f"C:/checkpoints/{run_name}.pt",
        "bc_checkpoint_metadata": None if init_mode == "scratch" else {"map_uid": "test-map"},
        "demo_root": None if init_mode == "scratch" else "C:/demos/test-map",
        "replay_seeded": False,
        "eval_episodes": 2,
        "latest_checkpoint_path": checkpoint_rows[-1]["path"] if checkpoint_rows else None,
        "latest_eval_summary": eval_rows[-1]["summary"] if eval_rows else None,
        "latest_eval_summary_path": eval_rows[-1]["summary_path"] if eval_rows else None,
        "eval_in_flight": False,
        "pending_eval": None,
        "checkpoint_history": checkpoint_rows,
        "eval_history": eval_rows,
        "last_worker_heartbeat": None,
        "termination_reason": termination_reason,
        "clean_shutdown": clean_shutdown,
        "worker_exit": {
            "done_event_set": True,
            "exitcode": 0,
            "terminated": False,
            "timeout": False,
        },
        "timestamp": "2026-04-11T00:10:00+00:00",
        "observation_mode": "full",
    }
    if extra_summary:
        summary.update(extra_summary)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def test_write_training_report_handles_interrupted_run(tmp_path) -> None:
    run_dir = tmp_path / "artifacts" / "train" / "full_sac_unit"
    eval_rows = [
        {
            "checkpoint_step": 5000,
            "env_step": 5000,
            "learner_step": 20000,
            "summary_path": str(run_dir.parent.parent / "eval" / "full_sac_unit_step_00005000" / "summary.json"),
            "summary": {
                "env_step": 5000,
                "mean_final_progress_index": 150.0,
                "median_final_progress_index": 120.0,
                "mean_final_progress_meters": 75.0,
                "median_final_progress_meters": 60.0,
                "mean_final_arc_length_m": 75.0,
                "median_final_arc_length_m": 60.0,
                "mean_progress_fraction_of_reference": 0.6,
                "median_progress_fraction_of_reference": 0.55,
                "reference_total_arc_length_m": 125.0,
                "mean_ghost_relative_time_delta_ms": 320.0,
                "median_ghost_relative_time_delta_ms": 300.0,
                "best_ghost_relative_time_delta_ms": 250.0,
                "progress_spacing_meters": 0.5,
                "progress_index_semantics": "fixed_spacing_meters",
                "completion_rate": 0.0,
                "best_reward": 300.0,
            },
            "timestamp": 1_700_000_000.0,
        }
    ]
    checkpoint_rows = [
        {
            "path": str(run_dir / "checkpoints" / "checkpoint_00005000.pt"),
            "env_step": 5000,
            "learner_step": 20000,
            "replay_size": 5000,
            "timestamp": "2026-04-11T00:05:00+00:00",
            "final": False,
        }
    ]
    _write_summary(
        run_dir,
        run_name="full_sac_unit",
        init_mode="scratch",
        env_step=5200,
        learner_step=20800,
        replay_size=5200,
        eval_rows=eval_rows,
        checkpoint_rows=checkpoint_rows,
        termination_reason="fatal_error",
        clean_shutdown=False,
        extra_summary={
            "ghost_bundle_manifest_path": "C:/ghosts/test-map/ghost_bundle_manifest.json",
            "canonical_reference_source": "author_reference_manifest",
            "canonical_reference_path": "C:/ghosts/test-map/author_reference.json",
            "strategy_classification_status": "classified",
            "selected_training_family": "intended_route",
            "mixed_fallback": False,
            "bundle_resolution_mode": "intended_route",
            "selected_ghost_selector": None,
            "resolved_selected_ghost_rank": None,
            "resolved_selected_ghost_name": None,
            "author_fallback_used": False,
            "intended_bundle_manifest_path": "C:/ghosts/test-map/ghost_bundle_intended.json",
            "exploit_bundle_manifest_path": "C:/ghosts/test-map/ghost_bundle_exploit.json",
            "selected_override_manifest_path": None,
            "author_fallback_manifest_path": None,
            "strategy_family_counts": {
                "intended_route": 15,
                "shortcut_or_exploit": 10,
                "unclassified": 3,
            },
        },
    )
    video_path = run_dir / "rollout.mp4"
    video_path.write_bytes(b"fake")

    report_paths = write_training_report(run_dir)
    report = json.loads(report_paths.json_path.read_text(encoding="utf-8"))
    assert report_paths.markdown_path.exists()
    assert report["run_name"] == "full_sac_unit"
    assert report["failure_notes"]
    assert str(video_path.resolve()) in report["videos"]
    assert report["eval_history_table"][0]["mean_final_progress_index"] == 150.0
    assert report["eval_history_table"][0]["mean_final_progress_meters"] == 75.0
    assert report["eval_history_table"][0]["mean_progress_fraction_of_reference"] == 0.6
    assert report["eval_history_table"][0]["mean_ghost_relative_time_delta_ms"] == 320.0
    assert report["selected_training_family"] == "intended_route"
    assert report["mixed_fallback"] is False
    assert report["bundle_resolution_mode"] == "intended_route"
    assert report["exact_final_eval_complete"] is False
    assert report["incomplete_final_eval"] is True
    markdown = report_paths.markdown_path.read_text(encoding="utf-8")
    assert "Strategy selection: status=classified family=intended_route mixed_fallback=False" in markdown
    assert "Bundle resolution: mode=intended_route selector=None resolved_rank=None resolved_name=None author_fallback_used=False" in markdown
    assert "Canonical reference: source=author_reference_manifest path=C:/ghosts/test-map/author_reference.json" in markdown
    assert "Exact final eval complete: False" in markdown


def test_write_comparison_report_builds_bc_comparison(tmp_path) -> None:
    train_root = tmp_path / "artifacts" / "train"
    scratch_dir = train_root / "scratch_run"
    actor_only_dir = train_root / "actor_only_run"
    encoder_dir = train_root / "encoder_run"

    scratch_eval = [
        {
            "checkpoint_step": 5000,
            "env_step": 5000,
            "learner_step": 20000,
            "summary_path": str(scratch_dir / "eval_5000.json"),
            "summary": {"env_step": 5000, "mean_final_progress_index": 80.0, "median_final_progress_index": 75.0},
            "timestamp": 1.0,
        },
        {
            "checkpoint_step": 10000,
            "env_step": 10000,
            "learner_step": 40000,
            "summary_path": str(scratch_dir / "eval_10000.json"),
            "summary": {"env_step": 10000, "mean_final_progress_index": 180.0, "median_final_progress_index": 170.0},
            "timestamp": 2.0,
        },
    ]
    actor_only_eval = [
        {
            "checkpoint_step": 5000,
            "env_step": 5000,
            "learner_step": 20000,
            "summary_path": str(actor_only_dir / "eval_5000.json"),
            "summary": {"env_step": 5000, "mean_final_progress_index": 140.0, "median_final_progress_index": 130.0},
            "timestamp": 1.0,
        },
        {
            "checkpoint_step": 10000,
            "env_step": 10000,
            "learner_step": 40000,
            "summary_path": str(actor_only_dir / "eval_10000.json"),
            "summary": {"env_step": 10000, "mean_final_progress_index": 240.0, "median_final_progress_index": 230.0},
            "timestamp": 2.0,
        },
    ]
    encoder_eval = [
        {
            "checkpoint_step": 5000,
            "env_step": 5000,
            "learner_step": 20000,
            "summary_path": str(encoder_dir / "eval_5000.json"),
            "summary": {"env_step": 5000, "mean_final_progress_index": 120.0, "median_final_progress_index": 110.0},
            "timestamp": 1.0,
        }
    ]
    checkpoint_rows = [
        {
            "path": "checkpoint.pt",
            "env_step": 10000,
            "learner_step": 40000,
            "replay_size": 10000,
            "timestamp": "2026-04-11T00:10:00+00:00",
            "final": True,
        }
    ]
    _write_summary(
        scratch_dir,
        run_name="scratch_run",
        init_mode="scratch",
        env_step=10000,
        learner_step=40000,
        replay_size=10000,
        eval_rows=scratch_eval,
        checkpoint_rows=checkpoint_rows,
    )
    _write_summary(
        actor_only_dir,
        run_name="actor_only_run",
        init_mode="actor_only",
        env_step=10000,
        learner_step=40000,
        replay_size=10000,
        eval_rows=actor_only_eval,
        checkpoint_rows=checkpoint_rows,
    )
    _write_summary(
        encoder_dir,
        run_name="encoder_run",
        init_mode="actor_plus_critic_encoders",
        env_step=5000,
        learner_step=20000,
        replay_size=5000,
        eval_rows=encoder_eval,
        checkpoint_rows=checkpoint_rows,
    )

    report_paths = write_comparison_report([scratch_dir, actor_only_dir, encoder_dir], output_dir=tmp_path / "reports")
    report = json.loads(report_paths.json_path.read_text(encoding="utf-8"))
    assert report_paths.markdown_path.exists()
    assert report["matched_step_table"]
    assert report["best_progress_by_run"]["actor_only"] == 240.0
    assert "actor_only" in report["time_to_progress_thresholds"]
    assert "BC warm-start gain" in report["conclusion"]


def test_write_training_report_includes_diagnostics_sections_and_event_logs(tmp_path) -> None:
    run_dir = tmp_path / "artifacts" / "train" / "full_redq_diag"
    checkpoint_rows = [
        {
            "path": str(run_dir / "checkpoints" / "checkpoint_00002500.pt"),
            "env_step": 2500,
            "learner_step": 5000,
            "replay_size": 2500,
            "timestamp": "2026-04-11T00:05:00+00:00",
            "final": False,
        }
    ]
    _write_summary(
        run_dir,
        run_name="full_redq_diag",
        init_mode="scratch",
        env_step=2500,
        learner_step=5000,
        replay_size=2500,
        eval_rows=[],
        checkpoint_rows=checkpoint_rows,
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "algorithm": "redq",
            "primary_metric": "mean_final_progress_index",
            "achieved_utd_1k": 1.75,
            "cumulative_utd": 2.0,
            "current_actor_staleness": 12,
            "runtime_profile": {
                "bottleneck_verdict": {
                    "label": "worker_env",
                    "breakdown_seconds": {
                        "learner_backprop": 10.0,
                        "worker_env": 20.0,
                        "ipc_backpressure": 1.0,
                        "actor_sync": 0.5,
                    },
                }
            },
            "queue_profile": {"learner": {"command_put": {"attempts": 2}}, "worker": {"output_put": {"attempts": 3}}},
            "actor_sync_profile": {
                "policy_control_fraction": 0.75,
                "current_versions_behind": 2,
                "time_to_first_ready_actor_seconds": 12.0,
                "time_to_first_applied_ready_actor_seconds": 13.0,
                "time_to_first_policy_control_window_seconds": 15.0,
                "time_to_applied_seconds": {"p50": 0.25, "p95": 0.5},
            },
            "episode_diagnostics": {
                "positive_progress_fraction": {"mean": 0.6},
                "nonpositive_progress_fraction": {"mean": 0.4},
                "max_no_progress_streak": {"p95": 18.0},
                "final_arc_length_m": {"mean": 75.0},
                "progress_fraction_of_reference": {"mean": 0.6},
                "ghost_relative_time_delta_ms": {"mean": 320.0},
                "corridor_violation_fraction": {"mean": 0.25},
                "corridor_distance_m": {"p95": 42.0},
                "max_corridor_distance_m": {"p95": 80.0},
                "termination_reason_counts": {"corridor_violation": 2},
            },
            "movement_diagnostics": {
                "no_movement_episode_count": 2,
                "stall_episode_rate": 0.4,
                "first_stall_delay_ms": {"p95": 800.0},
            },
            "resource_profile": {
                "actor_parameter_count": 100,
                "critic_parameter_count": 200,
                "unique_critic_encoder_parameter_count": 50,
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "learner_events.log").write_text(
        json.dumps({"event": "actor_broadcast", "payload": {}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "worker_events.log").write_text(
        json.dumps({"event": "movement_episode_summary", "payload": {}}) + "\n",
        encoding="utf-8",
    )

    report_paths = write_training_report(run_dir)
    report = json.loads(report_paths.json_path.read_text(encoding="utf-8"))

    assert report["primary_metric"] == "mean_final_progress_index"
    assert report["achieved_utd_1k"] == 1.75
    assert report["cumulative_utd"] == 2.0
    assert report["current_actor_staleness"] == 12
    assert report["runtime_profile"]["bottleneck_verdict"]["label"] == "worker_env"
    assert report["actor_sync_profile"]["policy_control_fraction"] == 0.75
    assert report["event_logs"]["learner"]["event_count"] == 1
    markdown = report_paths.markdown_path.read_text(encoding="utf-8")
    assert "## Diagnostics" in markdown
    assert "Bottleneck verdict: worker_env" in markdown
    assert "achieved_utd_1k=1.75 cumulative_utd=2.0 current_actor_staleness=12" in markdown
    assert "final_arc_length_mean=75.0" in markdown
    assert "progress_fraction_mean=0.6" in markdown
    assert "ghost_delta_mean_ms=320.0" in markdown
    assert "corridor_violation_fraction_mean=0.25" in markdown
    assert "corridor_truncations=2" in markdown


def test_write_training_report_preserves_per_mode_eval_rows(tmp_path) -> None:
    run_dir = tmp_path / "artifacts" / "train" / "full_redq_modes"
    eval_rows = [
        {
            "checkpoint_step": 5000,
            "env_step": 5000,
            "learner_step": 9000,
            "summary_path": str(run_dir.parent.parent / "eval" / "full_redq_modes_step_00005000_deterministic" / "summary.json"),
            "summary": {
                "env_step": 5000,
                "mean_final_progress_index": 80.0,
                "median_final_progress_index": 75.0,
                "completion_rate": 0.0,
            },
            "mode_summaries": {
                "deterministic": {
                    "env_step": 5000,
                    "mean_final_progress_index": 80.0,
                    "completion_rate": 0.0,
                },
                "stochastic": {
                    "env_step": 5000,
                    "mean_final_progress_index": 120.0,
                    "completion_rate": 0.2,
                },
            },
            "mode_summary_paths": {
                "deterministic": "C:/eval/deterministic.json",
                "stochastic": "C:/eval/stochastic.json",
            },
            "deterministic_collapse": {
                "meaningfully_outperformed": True,
                "progress_delta": 40.0,
                "completion_rate_delta": 0.2,
            },
            "timestamp": 1_700_000_000.0,
        }
    ]
    checkpoint_rows = [
        {
            "path": str(run_dir / "checkpoints" / "checkpoint_00005000.pt"),
            "env_step": 5000,
            "learner_step": 9000,
            "replay_size": 5000,
            "timestamp": "2026-04-11T00:05:00+00:00",
            "final": False,
        }
    ]
    _write_summary(
        run_dir,
        run_name="full_redq_modes",
        init_mode="scratch",
        env_step=5000,
        learner_step=9000,
        replay_size=5000,
        eval_rows=eval_rows,
        checkpoint_rows=checkpoint_rows,
    )

    report_paths = write_training_report(run_dir)
    report = json.loads(report_paths.json_path.read_text(encoding="utf-8"))
    row = report["eval_history_table"][0]

    assert row["mode_summaries"]["stochastic"]["mean_final_progress_index"] == 120.0
    assert row["deterministic_collapse"]["meaningfully_outperformed"] is True
    markdown = report_paths.markdown_path.read_text(encoding="utf-8")
    assert "stochastic: mean_progress=120.0" in markdown
    assert "deterministic_collapse: meaningfully_outperformed=True" in markdown
