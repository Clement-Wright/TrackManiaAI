from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from tm20ai.train.features import TELEMETRY_DIM
from tm20ai.train.learner import REDQLearner


def write_redq_train_config(
    path: Path,
    artifacts_root: Path,
    *,
    q_updates_per_policy_update: int = 4,
    max_training_steps_per_environment_step: float = 1.0,
    update_model_interval: int = 1,
    update_buffer_interval: int = 1,
    broadcast_after_actor_update: bool = False,
    actor_publish_every: int = 1,
) -> None:
    path.write_text(
        f"""
runtime:
  time_step_duration: 0.05
  start_obs_capture: 0.04
  time_step_timeout_factor: 1.0
  act_buf_len: 2
  wait_on_done: true
  ep_max_length: 1000
  sleep_time_at_reset: 1.5
bridge:
  host: "127.0.0.1"
  telemetry_port: 9100
  command_port: 9101
  connect_timeout: 5.0
  command_timeout: 5.0
  initial_frame_timeout: 10.0
  reconnect_delay: 1.0
  stale_timeout: 0.25
  reset_timeout: 5.0
observation:
  mode: full
capture:
  window_title: "Trackmania"
  target_fps: 60
  max_buffer_len: 64
  latest_frame_only: true
  frame_timeout: 1.0
  post_reset_flush_seconds: 0.25
  invalid_frame_limit: 3
  region_change_tolerance_pixels: 4
full_observation:
  window_width: 256
  window_height: 128
  output_width: 64
  output_height: 64
  grayscale: true
  frame_stack: 4
lidar_observation:
  window_width: 958
  window_height: 488
  ray_count: 19
  lidar_hist_len: 4
  prev_action_hist_len: 2
  fixed_crop: [0.18, 0.34, 0.82, 0.96]
  border_threshold: 48
  ray_min_angle_degrees: -80.0
  ray_max_angle_degrees: 80.0
  max_ray_length_fraction: 1.0
reward:
  mode: trajectory_progress
  spacing_meters: 0.5
  end_of_track: 100.0
  constant_penalty: 0.0
  check_forward: 500
  check_backward: 10
  failure_countdown: 10
  min_steps: 70
  max_stray: 100.0
eval:
  episodes: 1
  seed_base: 12345
  sector_count: 4
  record_video: false
  video_fps: 20
  modes: ["deterministic", "stochastic"]
  trace_seconds: 3.0
train:
  algorithm: redq
  seed: 12345
  memory_size: 64
  batch_size: 2
  environment_steps_before_training: 2
  max_training_steps_per_environment_step: {max_training_steps_per_environment_step}
  update_model_interval: {update_model_interval}
  update_buffer_interval: {update_buffer_interval}
  eval_interval_steps: 2
  checkpoint_interval_steps: 2
  queue_capacity: 64
  max_env_steps: 4
  cuda_training: false
  cuda_inference: false
  single_live_env: true
  broadcast_after_actor_update: {str(broadcast_after_actor_update).lower()}
  actor_publish_every: {actor_publish_every}
sac:
  gamma: 0.995
  polyak: 0.995
  actor_lr: 3.0e-4
  critic_lr: 3.0e-4
  alpha_lr: 3.0e-4
  learn_entropy_coef: false
  alpha: 0.01
  target_entropy: -2.0
redq:
  n_critics: 4
  m_subset: 2
  q_updates_per_policy_update: {q_updates_per_policy_update}
  share_encoders: true
bc:
  epochs: 2
  learning_rate: 3.0e-4
  weight_decay: 0.0
  validation_fraction: 0.2
artifacts:
  root: "{artifacts_root.as_posix()}"
""".strip(),
        encoding="utf-8",
    )


def _transition(step: int) -> dict:
    return {
        "obs_uint8": np.full((4, 64, 64), step, dtype=np.uint8),
        "telemetry_float": np.full((TELEMETRY_DIM,), 0.1 * step, dtype=np.float32),
        "action": np.asarray([0.1, 0.0], dtype=np.float32),
        "reward": 1.0,
        "next_obs_uint8": np.full((4, 64, 64), step + 1, dtype=np.uint8),
        "next_telemetry_float": np.full((TELEMETRY_DIM,), 0.1 * (step + 1), dtype=np.float32),
        "terminated": False,
        "truncated": False,
        "episode_id": "episode",
        "map_uid": "test-map",
        "step_idx": step,
    }


def test_redq_learner_marks_actor_ready_only_after_actor_update(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    write_redq_train_config(
        config_path,
        artifacts_root,
        q_updates_per_policy_update=5,
        max_training_steps_per_environment_step=4.0,
        update_model_interval=16,
    )

    learner = REDQLearner(config_path=config_path, run_name="unit_redq", eval_episodes_override=1)
    command_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    eval_result_queue: queue.Queue = queue.Queue()
    shutdown_event = threading.Event()
    worker_done_event = threading.Event()
    learner.attach_worker(
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
    )

    learner.broadcast_actor(force=True)
    startup_actor_payload = json.loads(learner.paths.desired_actor_path.read_text(encoding="utf-8"))
    assert startup_actor_payload["ready_for_control"] is False
    assert startup_actor_payload["control_ready_reason"] == "startup_untrained"
    assert startup_actor_payload["actor_step"] == 0

    for step in range(3):
        output_queue.put(
            {
                "type": "transition_batch",
                "env_step": step + 1,
                "transitions": [_transition(step)],
            }
        )

    assert learner.drain_messages(timeout=0.01) == 3
    assert learner.maybe_train() == 0
    assert learner.learner_step == 4
    assert learner.actor_step == 0

    learner.broadcast_actor(force=True)
    critic_only_actor_payload = json.loads(learner.paths.desired_actor_path.read_text(encoding="utf-8"))
    assert critic_only_actor_payload["ready_for_control"] is False
    assert critic_only_actor_payload["actor_step"] == 0
    learner.maybe_schedule_eval()
    assert command_queue.empty()

    output_queue.put(
        {
            "type": "transition_batch",
            "env_step": 4,
            "transitions": [_transition(3)],
        }
    )
    assert learner.drain_messages(timeout=0.01) == 1
    assert learner.maybe_train() == 1
    assert learner.learner_step == 8
    assert learner.actor_step == 1

    learner.broadcast_actor(force=True)
    desired_actor_payload = json.loads(learner.paths.desired_actor_path.read_text(encoding="utf-8"))
    assert desired_actor_payload["ready_for_control"] is True
    assert desired_actor_payload["control_ready_reason"] == "trained_updates_available"
    assert desired_actor_payload["actor_step"] == 1

    learner.maybe_schedule_eval()
    assert command_queue.empty()

    output_queue.put(
        {
            "type": "actor_sync_desired_seen",
            "desired_actor_version": desired_actor_payload["desired_actor_version"],
            "desired_actor_ready_for_control": True,
            "seen_actor_version": desired_actor_payload["desired_actor_version"],
            "ready_for_control_seen": True,
            "env_step": learner.env_step,
            "learner_step": learner.learner_step,
            "actor_state_path": desired_actor_payload["actor_state_path"],
            "written_at": desired_actor_payload["written_at"],
            "control_ready_reason": desired_actor_payload["control_ready_reason"],
        }
    )
    output_queue.put(
        {
            "type": "actor_sync_applied",
            "desired_actor_version": desired_actor_payload["desired_actor_version"],
            "desired_actor_ready_for_control": True,
            "seen_actor_version": desired_actor_payload["desired_actor_version"],
            "ready_for_control_seen": True,
            "applied_actor_version": desired_actor_payload["desired_actor_version"],
            "actor_ready_for_control": True,
            "applied_source_learner_step": learner.learner_step,
            "env_step": learner.env_step,
            "learner_step": learner.learner_step,
            "actor_state_path": desired_actor_payload["actor_state_path"],
            "written_at": desired_actor_payload["written_at"],
            "last_actor_apply_env_step": learner.env_step,
            "last_actor_apply_episode_index": 1,
            "control_ready_reason": desired_actor_payload["control_ready_reason"],
        }
    )
    assert learner.drain_messages(timeout=0.01) == 0

    learner.maybe_schedule_eval()
    scheduled_commands = []
    while not command_queue.empty():
        scheduled_commands.append(command_queue.get())
    run_eval_command = next(command for command in scheduled_commands if command["type"] == "run_eval")
    assert run_eval_command["eval_actor_version"] == desired_actor_payload["desired_actor_version"]
    assert run_eval_command["eval_actor_source_learner_step"] == learner.learner_step
    assert run_eval_command["modes"] == ["deterministic", "stochastic"]
    assert run_eval_command["trace_seconds"] == 3.0
    assert run_eval_command["eval_provenance_mode"] == "checkpoint_authoritative"
    assert Path(str(run_eval_command["eval_checkpoint_path"])).exists()
    assert run_eval_command["eval_checkpoint_env_step"] == learner.env_step
    assert run_eval_command["eval_checkpoint_learner_step"] == learner.learner_step
    assert run_eval_command["scheduled_actor_version"] == desired_actor_payload["desired_actor_version"]

    checkpoint_path = learner.save_checkpoint()
    restored = REDQLearner(config_path=config_path, run_name="unit_redq_restored")
    restored.load_checkpoint(checkpoint_path)
    assert restored.env_step == learner.env_step
    assert restored.learner_step == learner.learner_step
    assert restored.actor_step == learner.actor_step

    summary_payload = json.loads(learner.paths.summary_json.read_text(encoding="utf-8"))
    assert summary_payload["actor_step"] == 1
    assert summary_payload["desired_actor_ready_for_control"] is True
    checkpoint_sidecar = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert checkpoint_sidecar["actor_step"] == 1
    assert checkpoint_sidecar["desired_actor_ready_for_control"] is True

    learner.close()
    restored.close()


def test_redq_learner_collects_diagnostics_profiles(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    write_redq_train_config(
        config_path,
        artifacts_root,
        q_updates_per_policy_update=1,
        update_model_interval=4,
    )

    learner = REDQLearner(
        config_path=config_path,
        run_name="unit_redq_diagnostics",
        eval_episodes_override=1,
        diagnostics_enabled=True,
    )
    command_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    eval_result_queue: queue.Queue = queue.Queue()
    shutdown_event = threading.Event()
    worker_done_event = threading.Event()
    learner.attach_worker(
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
    )

    for step in range(4):
        output_queue.put(
            {
                "type": "transition_batch",
                "env_step": step + 1,
                "transitions": [_transition(step)],
            }
        )
    assert learner.drain_messages(timeout=0.01) == 4
    assert learner.maybe_train() == 2
    learner.broadcast_actor(force=True)
    desired_actor_payload = json.loads(learner.paths.desired_actor_path.read_text(encoding="utf-8"))

    output_queue.put(
        {
            "type": "actor_sync_desired_seen",
            "desired_actor_version": desired_actor_payload["desired_actor_version"],
            "desired_actor_ready_for_control": True,
            "seen_actor_version": desired_actor_payload["desired_actor_version"],
            "ready_for_control_seen": True,
            "requested_env_step": 1,
            "control_ready_reason": desired_actor_payload["control_ready_reason"],
        }
    )
    output_queue.put(
        {
            "type": "actor_sync_applied",
            "desired_actor_version": desired_actor_payload["desired_actor_version"],
            "desired_actor_ready_for_control": True,
            "seen_actor_version": desired_actor_payload["desired_actor_version"],
            "ready_for_control_seen": True,
            "applied_actor_version": desired_actor_payload["desired_actor_version"],
            "actor_ready_for_control": True,
            "applied_source_learner_step": 1,
            "requested_env_step": 1,
            "apply_duration_seconds": 0.02,
            "last_actor_apply_env_step": 2,
            "last_actor_apply_episode_index": 1,
            "control_ready_reason": desired_actor_payload["control_ready_reason"],
        }
    )
    output_queue.put(
        {
            "type": "action_stats",
            "env_step": 4,
            "desired_actor_version": desired_actor_payload["desired_actor_version"],
            "desired_actor_ready_for_control": True,
            "seen_actor_version": desired_actor_payload["desired_actor_version"],
            "ready_for_control_seen": True,
            "applied_actor_version": desired_actor_payload["desired_actor_version"],
            "actor_ready_for_control": True,
            "applied_source_learner_step": 1,
            "last_actor_apply_env_step": 2,
            "last_actor_apply_episode_index": 1,
            "control_source": "policy",
            "policy_control_fraction": 0.5,
            "versions_behind": 0,
            "window_size": 10,
        }
    )
    output_queue.put(
        {
            "type": "movement_episode_summary",
            "summary": {
                "run_id": "run-0",
                "episode_index": 1,
                "movement_started": False,
                "stall_count": 1,
                "first_stall": {
                    "race_time_ms": 400,
                    "movement_start_race_time_ms": 200,
                },
                "termination_reason": "no_progress",
            },
        }
    )
    output_queue.put(
        {
            "type": "heartbeat",
            "env_step": 4,
            "desired_actor_version": desired_actor_payload["desired_actor_version"],
            "desired_actor_ready_for_control": True,
            "seen_actor_version": desired_actor_payload["desired_actor_version"],
            "ready_for_control_seen": True,
            "applied_actor_version": desired_actor_payload["desired_actor_version"],
            "actor_ready_for_control": True,
            "applied_source_learner_step": 1,
            "last_actor_apply_env_step": 2,
            "last_actor_apply_episode_index": 1,
            "latest_action_stats": {
                "control_source": "policy",
                "policy_control_fraction": 0.5,
                "desired_actor_version": desired_actor_payload["desired_actor_version"],
                "applied_actor_version": desired_actor_payload["desired_actor_version"],
                "applied_source_learner_step": 1,
                "last_actor_apply_env_step": 2,
            },
            "runtime_profile": {
                "env_loop_total_seconds": 1.25,
                "env_step": {"total_seconds": 1.0},
                "env_reset": {"total_seconds": 0.25},
                "actor_apply": {"total_seconds": 0.02},
            },
            "queue_profile": {
                "output_put": {"total_wait_seconds": 0.01},
                "eval_result_put": {"total_wait_seconds": 0.0},
            },
        }
    )
    assert learner.drain_messages(timeout=0.01) == 0

    learner.maybe_schedule_eval()
    scheduled_commands = []
    while not command_queue.empty():
        scheduled_commands.append(command_queue.get())
    assert any(command["type"] == "run_eval" for command in scheduled_commands)

    summary_payload = json.loads(learner.paths.summary_json.read_text(encoding="utf-8"))
    assert summary_payload["primary_metric"] == "mean_final_progress_index"
    assert summary_payload["runtime_profile"]["learner"]["replay_sample"]["count"] > 0
    assert summary_payload["runtime_profile"]["worker"]["env_loop_total_seconds"] == 1.25
    assert summary_payload["queue_profile"]["learner"]["command_put"]["attempts"] >= 1
    assert summary_payload["actor_sync_profile"]["time_to_applied_seconds"]["count"] >= 1
    assert summary_payload["actor_sync_profile"]["time_to_first_policy_control_window_seconds"] is not None
    assert summary_payload["episode_diagnostics"]["episode_count"] == 0
    assert summary_payload["movement_diagnostics"]["no_movement_episode_count"] == 1
    assert summary_payload["resource_profile"]["n_critics"] == 4
    assert summary_payload["current_actor_staleness"] == 1
    assert summary_payload["actor_sync_profile"]["current_actor_staleness"] == 1

    learner.close()


def test_redq_learner_broadcasts_on_actor_update_cadence_and_flushes_latest_actor(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    write_redq_train_config(
        config_path,
        artifacts_root,
        q_updates_per_policy_update=1,
        max_training_steps_per_environment_step=1.0,
        update_model_interval=4,
        broadcast_after_actor_update=True,
        actor_publish_every=3,
    )

    learner = REDQLearner(config_path=config_path, run_name="unit_redq_publish", eval_episodes_override=0)
    command_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    eval_result_queue: queue.Queue = queue.Queue()
    shutdown_event = threading.Event()
    worker_done_event = threading.Event()
    learner.attach_worker(
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
    )
    learner.broadcast_actor(force=True)
    startup_payload = json.loads(learner.paths.desired_actor_path.read_text(encoding="utf-8"))
    assert startup_payload["actor_step"] == 0

    for step in range(4):
        output_queue.put({"type": "transition_batch", "env_step": step + 1, "transitions": [_transition(step)]})
    assert learner.drain_messages(timeout=0.01) == 4

    actor_updates = learner.maybe_train()
    desired_actor_payload = json.loads(learner.paths.desired_actor_path.read_text(encoding="utf-8"))

    assert actor_updates == 2
    assert learner.actor_step == 2
    assert desired_actor_payload["actor_step"] == 2
    assert desired_actor_payload["ready_for_control"] is True
    assert desired_actor_payload["desired_actor_version"] == startup_payload["desired_actor_version"] + 1
    learner.close()


def test_redq_learner_caps_updates_per_call_while_preserving_utd_target(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    write_redq_train_config(
        config_path,
        artifacts_root,
        q_updates_per_policy_update=100,
        max_training_steps_per_environment_step=8.0,
        update_model_interval=8,
        update_buffer_interval=8,
    )

    learner = REDQLearner(config_path=config_path, run_name="unit_redq_utd_cap", eval_episodes_override=0)
    for step in range(10):
        learner.replay.add(_transition(step))
    learner.env_step = 10
    learner._record_progress_diagnostics()

    assert learner._target_critic_steps() == 64
    assert learner.maybe_train() == 0
    assert learner.learner_step == 8
    assert learner.achieved_utd_1k == 0.8
    learner.close()


def test_redq_finalize_run_preserves_inflight_eval_result(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    write_redq_train_config(config_path, artifacts_root, q_updates_per_policy_update=1)

    learner = REDQLearner(config_path=config_path, run_name="unit_redq_finalize_eval")
    command_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    eval_result_queue: queue.Queue = queue.Queue()
    shutdown_event = threading.Event()
    worker_done_event = threading.Event()
    learner.attach_worker(
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
    )
    learner.eval_in_flight = True
    learner.pending_eval = {"run_name": "eval_pending", "env_step": 4}

    def publish_eval() -> None:
        time.sleep(0.05)
        eval_result_queue.put(
            {
                "checkpoint_step": 4,
                "env_step": 4,
                "learner_step": 4,
                "summary_path": str(tmp_path / "eval" / "deterministic" / "summary.json"),
                "summary": {
                    "env_step": 4,
                    "mean_final_progress_index": 12.0,
                    "eval_mode_summaries": {
                        "deterministic": {
                            "env_step": 4,
                            "mean_final_progress_index": 12.0,
                            "completion_rate": 0.0,
                        },
                        "stochastic": {
                            "env_step": 4,
                            "mean_final_progress_index": 18.0,
                            "completion_rate": 0.0,
                        },
                    },
                    "eval_mode_summary_paths": {
                        "deterministic": str(tmp_path / "eval" / "deterministic" / "summary.json"),
                        "stochastic": str(tmp_path / "eval" / "stochastic" / "summary.json"),
                    },
                    "eval_mode_run_dirs": {
                        "deterministic": str(tmp_path / "eval" / "deterministic"),
                        "stochastic": str(tmp_path / "eval" / "stochastic"),
                    },
                },
                "timestamp": time.time(),
            }
        )
        worker_done_event.set()

    thread = threading.Thread(target=publish_eval, daemon=True)
    thread.start()
    learner.finalize_run(timeout_seconds=1.0)
    thread.join(timeout=1.0)

    assert learner.latest_eval_summary is not None
    assert learner.latest_eval_summary["mean_final_progress_index"] == 12.0
    assert learner.latest_eval_mode_summaries is not None
    assert learner.latest_eval_mode_summaries["stochastic"]["mean_final_progress_index"] == 18.0
    learner.close()
