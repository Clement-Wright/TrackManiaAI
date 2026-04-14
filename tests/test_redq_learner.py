from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import numpy as np

from tm20ai.train.features import TELEMETRY_DIM
from tm20ai.train.learner import REDQLearner


def write_redq_train_config(
    path: Path,
    artifacts_root: Path,
    *,
    q_updates_per_policy_update: int = 4,
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
train:
  algorithm: redq
  seed: 12345
  memory_size: 64
  batch_size: 2
  environment_steps_before_training: 2
  max_training_steps_per_environment_step: 1.0
  update_model_interval: 1
  update_buffer_interval: 1
  eval_interval_steps: 2
  checkpoint_interval_steps: 2
  queue_capacity: 64
  max_env_steps: 4
  cuda_training: false
  cuda_inference: false
  single_live_env: true
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
  share_encoders: false
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
    write_redq_train_config(config_path, artifacts_root, q_updates_per_policy_update=4)

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
    assert learner.learner_step == 3
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
    assert learner.learner_step == 4
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
