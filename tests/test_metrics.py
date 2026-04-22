from __future__ import annotations

from tm20ai.train.metrics import ActiveStepBenchmark, aggregate_episode_summaries, mode_comparison_metrics


def test_active_step_benchmark_excludes_reset_time_from_drift() -> None:
    tracker = ActiveStepBenchmark(step_dt_seconds=0.05)
    tracker.reanchor(0.0)
    tracker.record_step(end_time=0.05, duration_seconds=0.05)
    tracker.record_step(end_time=0.11, duration_seconds=0.06)
    tracker.record_reset(1.5, reanchor_time=10.0)
    tracker.record_step(end_time=10.05, duration_seconds=0.05)

    assert tracker.active_steps == 3
    assert tracker.reset_count == 1
    assert tracker.cumulative_reset_time_seconds == 1.5
    assert tracker.max_active_drift_seconds < 0.02


def test_active_step_benchmark_report_has_frozen_keys() -> None:
    tracker = ActiveStepBenchmark(step_dt_seconds=0.05)
    tracker.reanchor(0.0)
    tracker.record_step(end_time=0.05, duration_seconds=0.05)

    report = tracker.to_report(
        episodes=1,
        wall_clock_total_seconds=0.1,
        avg_obs_retrieval_seconds=0.01,
        avg_send_control_seconds=0.001,
        avg_reward_compute_seconds=0.002,
        avg_preprocess_seconds=0.003,
        raw_rtgym_benchmarks={"send_control_duration": (0.001, 0.0)},
    )

    expected = {
        "active_steps",
        "episodes",
        "step_dt_seconds",
        "avg_step_duration_seconds",
        "max_step_duration_seconds",
        "avg_step_overrun_seconds",
        "max_step_overrun_seconds",
        "avg_obs_retrieval_seconds",
        "avg_send_control_seconds",
        "avg_reward_compute_seconds",
        "avg_preprocess_seconds",
        "reset_count",
        "avg_reset_duration_seconds",
        "max_reset_duration_seconds",
        "cumulative_reset_time_seconds",
        "cumulative_active_drift_seconds",
        "max_active_drift_seconds",
        "wall_clock_total_seconds",
        "rtgym_benchmarks",
    }
    assert expected.issubset(report.keys())


def test_mode_comparison_metrics_compute_determinism_conversion_score() -> None:
    metrics = mode_comparison_metrics(
        {
            "deterministic": {
                "mean_final_progress_index": 50.0,
                "completion_rate": 0.25,
                "median_final_progress_index": 40.0,
            },
            "stochastic": {
                "mean_final_progress_index": 100.0,
                "completion_rate": 0.5,
                "median_final_progress_index": 90.0,
            },
        }
    )

    assert metrics["determinism_conversion_score"] == 0.5
    assert metrics["deterministic_stochastic_progress_gap"] == -50.0
    assert metrics["deterministic_stochastic_completion_gap"] == -0.25


def test_aggregate_episode_summaries_reports_progress_meters_and_corridor_rate() -> None:
    summary = aggregate_episode_summaries(
        [
            {
                "completion_flag": False,
                "final_progress_index": 10,
                "final_progress_meters": 5.0,
                "final_arc_length_m": 5.0,
                "progress_fraction_of_reference": 0.25,
                "reference_total_arc_length_m": 20.0,
                "progress_spacing_meters": 0.5,
                "progress_index_semantics": "fixed_spacing_meters",
                "ghost_relative_time_delta_ms": 120.0,
                "termination_reason": "corridor_violation",
                "episode_reward_total": 1.0,
            },
            {
                "completion_flag": False,
                "final_progress_index": 20,
                "final_progress_meters": 10.0,
                "final_arc_length_m": 10.0,
                "progress_fraction_of_reference": 0.5,
                "reference_total_arc_length_m": 20.0,
                "progress_spacing_meters": 0.5,
                "progress_index_semantics": "fixed_spacing_meters",
                "ghost_relative_time_delta_ms": 80.0,
                "termination_reason": "no_progress",
                "episode_reward_total": 2.0,
            },
        ],
        sector_count=2,
    )

    assert summary["mean_final_progress_index"] == 15.0
    assert summary["mean_final_progress_meters"] == 7.5
    assert summary["mean_final_arc_length_m"] == 7.5
    assert summary["median_final_progress_meters"] == 7.5
    assert summary["mean_progress_fraction_of_reference"] == 0.375
    assert summary["reference_total_arc_length_m"] == 20.0
    assert summary["progress_spacing_meters"] == 0.5
    assert summary["progress_index_semantics"] == "fixed_spacing_meters"
    assert summary["mean_ghost_relative_time_delta_ms"] == 100.0
    assert summary["best_ghost_relative_time_delta_ms"] == 80.0
    assert summary["corridor_violation_truncation_rate"] == 0.5
