from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tm20ai.train.campaign import (
    analyze_policy_mode_sweep_results,
    select_reward_winner,
    validate_campaign_run,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SCRIPT_PATH = ROOT / "scripts" / "run_rank11_100_validation_campaign.py"


def _load_campaign_script_module():
    spec = importlib.util.spec_from_file_location("rank11_validation_campaign_script", CAMPAIGN_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_campaign_run(
    artifact_root: Path,
    *,
    run_name: str,
    exact_progress: float,
    ghost_delta_ms: float,
    progress_fraction: float,
    corridor_truncation_rate: float,
    corridor_nonrecovering_p95: float,
    exact_complete: bool = True,
    final_eval_state: str = "complete",
) -> Path:
    run_dir = artifact_root / "train" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    deterministic_eval_dir = artifact_root / "eval" / f"{run_name}_final_exact_step_00010000_deterministic"
    stochastic_eval_dir = artifact_root / "eval" / f"{run_name}_final_exact_step_00010000_stochastic"
    deterministic_eval_dir.mkdir(parents=True, exist_ok=True)
    stochastic_eval_dir.mkdir(parents=True, exist_ok=True)
    deterministic_summary_path = deterministic_eval_dir / "summary.json"
    stochastic_summary_path = stochastic_eval_dir / "summary.json"
    deterministic_summary = {
        "eval_mode": "deterministic",
        "final_checkpoint_eval": True,
        "mean_final_progress_index": exact_progress,
        "mean_ghost_relative_time_delta_ms": ghost_delta_ms,
        "mean_progress_fraction_of_reference": progress_fraction,
        "corridor_violation_truncation_rate": corridor_truncation_rate,
    }
    stochastic_summary = {
        "eval_mode": "stochastic",
        "final_checkpoint_eval": True,
        "mean_final_progress_index": exact_progress + 10.0,
        "mean_ghost_relative_time_delta_ms": ghost_delta_ms - 10.0,
        "mean_progress_fraction_of_reference": progress_fraction,
        "corridor_violation_truncation_rate": corridor_truncation_rate,
    }
    deterministic_summary_path.write_text(json.dumps(deterministic_summary, indent=2), encoding="utf-8")
    stochastic_summary_path.write_text(json.dumps(stochastic_summary, indent=2), encoding="utf-8")
    summary = {
        "run_name": run_name,
        "algorithm": "redq",
        "config_path": "C:/config.yaml",
        "device": "cpu",
        "run_start_timestamp": "2026-04-23T00:00:00+00:00",
        "run_end_timestamp": "2026-04-23T01:30:00+00:00",
        "wall_clock_elapsed_seconds": 5400.0,
        "env_step": 10000,
        "learner_step": 20000,
        "episode_count": 100,
        "replay_size": 10000,
        "eval_episodes": 5,
        "latest_checkpoint_path": str(run_dir / "checkpoints" / "checkpoint_00010000_final.pt"),
        "latest_eval_summary": deterministic_summary,
        "latest_eval_summary_path": str(deterministic_summary_path.resolve()),
        "latest_eval_mode_summaries": {
            "deterministic": deterministic_summary,
            "stochastic": stochastic_summary,
        },
        "latest_eval_mode_summary_paths": {
            "deterministic": str(deterministic_summary_path.resolve()),
            "stochastic": str(stochastic_summary_path.resolve()),
        },
        "latest_eval_mode_run_dirs": {
            "deterministic": str(deterministic_eval_dir.resolve()),
            "stochastic": str(stochastic_eval_dir.resolve()),
        },
        "exact_final_eval_summary": {
            **deterministic_summary,
            "eval_mode_summaries": {
                "deterministic": deterministic_summary,
                "stochastic": stochastic_summary,
            },
            "eval_mode_summary_paths": {
                "deterministic": str(deterministic_summary_path.resolve()),
                "stochastic": str(stochastic_summary_path.resolve()),
            },
        },
        "exact_final_eval_summary_path": str(deterministic_summary_path.resolve()),
        "exact_final_eval_mode_summaries": {
            "deterministic": deterministic_summary,
            "stochastic": stochastic_summary,
        },
        "exact_final_eval_mode_summary_paths": {
            "deterministic": str(deterministic_summary_path.resolve()),
            "stochastic": str(stochastic_summary_path.resolve()),
        },
        "exact_final_eval_mode_run_dirs": {
            "deterministic": str(deterministic_eval_dir.resolve()),
            "stochastic": str(stochastic_eval_dir.resolve()),
        },
        "exact_final_eval_complete": exact_complete,
        "incomplete_final_eval": not exact_complete,
        "final_eval_state": final_eval_state,
        "final_checkpoint_eval_enabled": True,
        "eval_history": [],
        "checkpoint_history": [],
        "clean_shutdown": True,
        "worker_exit": {"done_event_set": True, "exitcode": 0, "terminated": False, "timeout": False},
        "observation_mode": "full",
        "primary_metric": "mean_final_progress_index",
        "episode_diagnostics": {
            "corridor_nonrecovering_steps": {"p95": corridor_nonrecovering_p95},
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def test_validate_campaign_run_requires_exact_final_eval_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    run_dir = _write_campaign_run(
        artifact_root,
        run_name="campaign_invalid",
        exact_progress=100.0,
        ghost_delta_ms=200.0,
        progress_fraction=0.8,
        corridor_truncation_rate=0.1,
        corridor_nonrecovering_p95=5.0,
        exact_complete=False,
        final_eval_state="exact_final_eval_missing",
    )
    validation = validate_campaign_run(run_dir)
    assert validation.valid is False
    assert "exact_final_eval_complete_false" in validation.reasons
    assert "incomplete_final_eval_true" in validation.reasons


def test_select_reward_winner_uses_tiebreakers(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    run_a = _write_campaign_run(
        artifact_root,
        run_name="campaign_a",
        exact_progress=1000.0,
        ghost_delta_ms=450.0,
        progress_fraction=0.81,
        corridor_truncation_rate=0.20,
        corridor_nonrecovering_p95=8.0,
    )
    run_b = _write_campaign_run(
        artifact_root,
        run_name="campaign_b",
        exact_progress=980.0,
        ghost_delta_ms=300.0,
        progress_fraction=0.85,
        corridor_truncation_rate=0.05,
        corridor_nonrecovering_p95=2.0,
    )
    winner, ordered = select_reward_winner([run_a, run_b])
    assert len(ordered) == 2
    assert winner.run_name == "campaign_b"


def test_analyze_policy_mode_sweep_results_prefers_clipped_when_it_clearly_wins() -> None:
    analysis = analyze_policy_mode_sweep_results(
        {
            "deterministic_mean": {
                "mean_final_progress_index": 80.0,
                "mean_progress_fraction_of_reference": 0.70,
            },
            "clipped_mean": {
                "mean_final_progress_index": 90.0,
                "mean_progress_fraction_of_reference": 0.72,
            },
            "stochastic_temp_1": {
                "mean_final_progress_index": 100.0,
                "mean_progress_fraction_of_reference": 0.75,
            },
            "best_of_8_temp_1": {
                "mean_final_progress_index": 200.0,
                "mean_progress_fraction_of_reference": 0.95,
            },
        }
    )
    assert analysis["deployment_choice"] == "clipped_mean"
    assert analysis["per_mode"]["deterministic_mean"]["determinism_conversion_score"] == 0.8
    assert analysis["per_mode"]["clipped_mean"]["determinism_conversion_score"] == 0.9
    assert analysis["deployment_choice_meets_target"] is True
    assert analysis["per_mode"]["clipped_mean"]["deployment_eligible"] is True
    assert analysis["per_mode"]["stochastic_temp_1"]["deployment_eligible"] is False
    assert analysis["per_mode"]["best_of_8_temp_1"]["diagnostic_only"] is True
    assert analysis["per_mode"]["best_of_8_temp_1"]["deployment_eligible"] is False


def test_policy_mode_specs_mark_best_of_k_diagnostic_only() -> None:
    module = importlib.util.spec_from_file_location("evaluate_redq_policy_modes", ROOT / "scripts" / "evaluate_redq_policy_modes.py")
    assert module is not None and module.loader is not None
    evaluate_module = importlib.util.module_from_spec(module)
    module.loader.exec_module(evaluate_module)

    specs = evaluate_module._mode_specs(
        ["deterministic_mean", "clipped_mean", "stochastic", "sample_best_of_k"],
        [0.5, 1.0],
        8,
    )

    by_name = {spec["name"]: spec for spec in specs}
    assert by_name["deterministic_mean"]["deployment_eligible"] is True
    assert by_name["clipped_mean"]["deployment_eligible"] is True
    assert by_name["stochastic_temp_1"]["deployment_eligible"] is False
    assert by_name["best_of_8_temp_1"]["diagnostic_only"] is True
    assert by_name["best_of_8_temp_1"]["deployment_eligible"] is False


def test_policy_mode_preflight_requires_rank11_bundle(tmp_path: Path) -> None:
    module = importlib.util.spec_from_file_location("evaluate_redq_policy_modes", ROOT / "scripts" / "evaluate_redq_policy_modes.py")
    assert module is not None and module.loader is not None
    evaluate_module = importlib.util.module_from_spec(module)
    module.loader.exec_module(evaluate_module)

    bundle_path = tmp_path / "ghost_bundle_rank_011_100.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "ghost_bundle_v1",
                "selected_training_family": "rank11_100_bundle",
                "selected_count": 90,
                "mixed_fallback": False,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "train": {"algorithm": "redq"},
                "ghosts": {
                    "bundle_manifest": str(bundle_path),
                    "ambiguous_family_policy": "hard_stop",
                },
            }
        ),
        encoding="utf-8",
    )
    config = evaluate_module.load_tm20ai_config(config_path)

    evaluate_module._preflight_policy_sweep(
        config=config,
        config_path=config_path,
        required_ghost_bundle=str(bundle_path),
        required_training_family="rank11_100_bundle",
        required_selected_count=90,
        forbid_mixed_fallback=True,
    )

    with pytest.raises(RuntimeError, match="required ghost bundle"):
        evaluate_module._preflight_policy_sweep(
            config=config,
            config_path=config_path,
            required_ghost_bundle=str(tmp_path / "other_bundle.json"),
            required_training_family="rank11_100_bundle",
            required_selected_count=90,
            forbid_mixed_fallback=True,
        )


def test_worker_eval_uses_configured_deployment_extraction_mode() -> None:
    from tm20ai.config import EvalConfig
    from tm20ai.train.worker import _eval_extraction_mode_for_mode

    eval_config = EvalConfig(deployment_extraction_mode="clipped_mean")

    assert _eval_extraction_mode_for_mode("deterministic", eval_config) == "clipped_mean"
    assert _eval_extraction_mode_for_mode("stochastic", eval_config) == "stochastic"


def test_run_rank11_validation_campaign_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_rank11_100_validation_campaign.py"),
            "--dry-run",
            "--session-name",
            "pytest_rank11_validation",
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "dry_run_complete" in result.stdout
    assert "check_environment.py" in result.stdout
    assert "--require-reward" in result.stdout
    generated_config = (
        ROOT
        / ".tmp"
        / "live_rank11_100_validation_campaign"
        / "pytest_rank11_validation"
        / "configs"
        / "A0_baseline_current.yaml"
    )
    payload = yaml.safe_load(generated_config.read_text(encoding="utf-8"))
    assert Path(payload["ghosts"]["bundle_manifest"]).is_absolute()
    assert payload["ghosts"]["ambiguous_family_policy"] == "hard_stop"
    assert payload["ghosts"]["bundle_manifest"].endswith("ghost_bundle_rank_011_100.json")


def test_run_rank11_validation_campaign_can_resume_phase4_from_existing_winner(tmp_path: Path) -> None:
    source_artifact_root = tmp_path / "source_artifacts"
    winner_run_dir = _write_campaign_run(
        source_artifact_root,
        run_name="rank11_existing_A3_buffered_hard_boundary",
        exact_progress=1612.8,
        ghost_delta_ms=-14717.18,
        progress_fraction=0.40,
        corridor_truncation_rate=0.0,
        corridor_nonrecovering_p95=0.0,
    )
    winner_config = tmp_path / "winner_config.yaml"
    winner_config.write_text(
        yaml.safe_dump(
            {
                "reward": {"corridor_patience_steps": 60},
                "ghosts": {
                    "bundle_manifest": str(
                        ROOT
                        / "data"
                        / "ghosts"
                        / "oqIJ5rQDRrNwLPTh9H2p_W4tLof"
                        / "ghost_bundle_rank_011_100.json"
                    ),
                    "ambiguous_family_policy": "hard_stop",
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_rank11_100_validation_campaign.py"),
            "--dry-run",
            "--blocks",
            "B",
            "--session-name",
            "pytest_rank11_phase4",
            "--artifact-root",
            str(tmp_path / "campaign_artifacts"),
            "--winner-run-dir",
            str(winner_run_dir),
            "--winner-config",
            str(winner_config),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "evaluate_redq_policy_modes.py" in result.stdout
    assert "--required-ghost-bundle" in result.stdout
    assert "--required-training-family rank11_100_bundle" in result.stdout
    assert "train_full_redq.py" not in result.stdout
    status_path = ROOT / ".tmp" / "live_rank11_100_validation_campaign" / "pytest_rank11_phase4" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "complete_dry_run"
    assert status["winner_run_dir"] == str(winner_run_dir.resolve())


def test_run_rank11_validation_campaign_phase5_uses_weight_only_offline_init(tmp_path: Path) -> None:
    campaign_script = _load_campaign_script_module()
    source_artifact_root = tmp_path / "source_artifacts"
    winner_run_dir = _write_campaign_run(
        source_artifact_root,
        run_name="rank11_existing_A3_buffered_hard_boundary",
        exact_progress=1612.8,
        ghost_delta_ms=-14717.18,
        progress_fraction=0.40,
        corridor_truncation_rate=0.0,
        corridor_nonrecovering_p95=0.0,
    )
    winner_config = tmp_path / "winner_config.yaml"
    winner_config.write_text(
        yaml.safe_dump(
            {
                "ghosts": {
                    "bundle_manifest": str(
                        ROOT
                        / "data"
                        / "ghosts"
                        / "oqIJ5rQDRrNwLPTh9H2p_W4tLof"
                        / "ghost_bundle_rank_011_100.json"
                    ),
                    "ambiguous_family_policy": "hard_stop",
                },
                "offline_pretrain": {"require_actions": True},
            }
        ),
        encoding="utf-8",
    )
    bundle_path = tmp_path / "ghost_bundle_rank_011_100.json"
    offline_npz = tmp_path / "offline_transitions.npz"
    offline_npz.write_bytes(b"npz")
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "ghost_bundle_v1",
                "selected_count": 90,
                "selected_training_family": "rank11_100_bundle",
                "mixed_fallback": False,
                "action_channel_valid": True,
                "offline_transition_count": 10,
                "offline_transition_npz_path": str(offline_npz),
            }
        ),
        encoding="utf-8",
    )
    campaign_script._preflight_block_c_offline_bundle(bundle_manifest_path=bundle_path, dry_run=False)

    # Exercise the C1/C2 command construction through the public dry-run path.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_rank11_100_validation_campaign.py"),
            "--dry-run",
            "--blocks",
            "C",
            "--session-name",
            "pytest_rank11_phase5",
            "--artifact-root",
            str(tmp_path / "campaign_artifacts"),
            "--winner-run-dir",
            str(winner_run_dir),
            "--winner-config",
            str(winner_config),
            "--disable-storage-preflight",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "--offline-init-mode weights_only" in result.stdout


def test_rank11_campaign_preflight_requires_hard_stop_rank_bundle(tmp_path: Path) -> None:
    campaign_script = _load_campaign_script_module()
    bundle_path = tmp_path / "ghost_bundle_rank_011_100.json"
    bundle_path.write_text(
        json.dumps(
            {
                "selected_count": 90,
                "selected_training_family": "rank11_100_bundle",
                "mixed_fallback": False,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ghosts": {
                    "bundle_manifest": str(bundle_path),
                    "ambiguous_family_policy": "hard_stop",
                }
            }
        ),
        encoding="utf-8",
    )

    campaign_script._preflight(base_config_path=config_path, bundle_manifest_path=bundle_path, dry_run=True)

    config_path.write_text(
        yaml.safe_dump(
            {
                "ghosts": {
                    "bundle_manifest": str(bundle_path),
                    "ambiguous_family_policy": "mixed_with_warning",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="ambiguous_family_policy='hard_stop'"):
        campaign_script._preflight(base_config_path=config_path, bundle_manifest_path=bundle_path, dry_run=True)

    config_path.write_text(
        yaml.safe_dump(
            {
                "ghosts": {
                    "bundle_manifest": str(bundle_path),
                    "ambiguous_family_policy": "hard_stop",
                }
            }
        ),
        encoding="utf-8",
    )
    bundle_path.write_text(
        json.dumps(
            {
                "selected_count": 90,
                "selected_training_family": "rank11_100_bundle",
                "mixed_fallback": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="must not be a mixed fallback"):
        campaign_script._preflight(base_config_path=config_path, bundle_manifest_path=bundle_path, dry_run=True)


def test_block_c_preflight_requires_offline_transition_sidecar(tmp_path: Path) -> None:
    campaign_script = _load_campaign_script_module()
    bundle_path = tmp_path / "ghost_bundle_rank_011_100.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "ghost_bundle_v1",
                "selected_count": 90,
                "selected_training_family": "rank11_100_bundle",
                "mixed_fallback": False,
                "action_channel_valid": True,
                "offline_transition_count": 0,
                "offline_transition_npz_path": None,
            }
        ),
        encoding="utf-8",
    )

    campaign_script._preflight_block_c_offline_bundle(bundle_manifest_path=bundle_path, dry_run=True)
    with pytest.raises(RuntimeError, match="offline_transition_count"):
        campaign_script._preflight_block_c_offline_bundle(bundle_manifest_path=bundle_path, dry_run=False)

    offline_npz = tmp_path / "offline_transitions.npz"
    offline_npz.write_bytes(b"npz")
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "ghost_bundle_v1",
                "selected_count": 90,
                "selected_training_family": "rank11_100_bundle",
                "mixed_fallback": False,
                "action_channel_valid": True,
                "offline_transition_count": 10,
                "offline_transition_npz_path": str(offline_npz),
            }
        ),
        encoding="utf-8",
    )

    campaign_script._preflight_block_c_offline_bundle(bundle_manifest_path=bundle_path, dry_run=False)


def test_rank11_campaign_reward_variants_are_locked_to_corridor_only() -> None:
    campaign_script = _load_campaign_script_module()
    expected = {
        "A0_baseline_current": {
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
        "A1_wider_recovery": {
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
        "A2_softer_wider_corridor": {
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
        "A3_buffered_hard_boundary": {
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
        "A4_hard_stray_control": {
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
    }

    variants = {variant["label"]: variant["reward"] for variant in campaign_script.REWARD_VARIANTS}
    assert variants == expected
    assert all(set(variant) == {"label", "reward"} for variant in campaign_script.REWARD_VARIANTS)


def test_run_training_leg_repairs_missing_exact_final_eval_before_retry(tmp_path: Path, monkeypatch) -> None:
    campaign_script = _load_campaign_script_module()
    artifact_root = tmp_path / "artifacts"
    run_name = "campaign_repairable"
    run_dir = _write_campaign_run(
        artifact_root,
        run_name=run_name,
        exact_progress=100.0,
        ghost_delta_ms=200.0,
        progress_fraction=0.8,
        corridor_truncation_rate=0.1,
        corridor_nonrecovering_p95=5.0,
        exact_complete=False,
        final_eval_state="exact_final_eval_missing",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ghosts:\n  bundle_manifest: C:/bundle.json\n", encoding="utf-8")
    status_path = tmp_path / "status.json"

    repaired: list[Path] = []

    def fake_gate_environment(*, config_path: Path, dry_run: bool) -> None:
        return None

    def fake_repair_missing_exact_final_eval(
        *, run_dir: Path, config_path: Path, dry_run: bool, status_path: Path, leg_label: str
    ) -> None:
        repaired.append(run_dir)
        _write_campaign_run(
            artifact_root,
            run_name=run_name,
            exact_progress=150.0,
            ghost_delta_ms=150.0,
            progress_fraction=0.85,
            corridor_truncation_rate=0.05,
            corridor_nonrecovering_p95=2.0,
            exact_complete=True,
            final_eval_state="complete",
        )

    monkeypatch.setattr(campaign_script, "_gate_environment", fake_gate_environment)
    monkeypatch.setattr(campaign_script, "_repair_missing_exact_final_eval", fake_repair_missing_exact_final_eval)

    repaired_run_dir = campaign_script._run_training_leg(
        run_name=run_name,
        config_path=config_path,
        artifact_root=artifact_root,
        wall_clock_minutes=90.0,
        dry_run=False,
        status_path=status_path,
        leg_label="A0_baseline_current",
    )

    assert repaired_run_dir == run_dir
    assert repaired == [run_dir]
    assert validate_campaign_run(repaired_run_dir).valid is True


def test_latest_resume_checkpoint_skips_zero_byte_partial_files(tmp_path: Path) -> None:
    campaign_script = _load_campaign_script_module()
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    valid_checkpoint = checkpoint_dir / "checkpoint_00010009.pt"
    valid_checkpoint.write_bytes(b"not-empty")
    zero_byte_partial = checkpoint_dir / "checkpoint_00010026_final.pt"
    zero_byte_partial.write_bytes(b"")

    resume_checkpoint = campaign_script._latest_resume_checkpoint(tmp_path / "run")

    assert resume_checkpoint == valid_checkpoint.resolve()
