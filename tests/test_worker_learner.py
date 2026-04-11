from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from tm20ai.train.evaluator import resolve_policy_adapter
from tm20ai.train.learner import SACLearner
from tm20ai.train.worker import SACWorker


class FakeTrainingEnv:
    def __init__(self) -> None:
        self.default_action = np.zeros(3, dtype=np.float32)
        self._episode = -1
        self._step = 0

    def reset(self, *, seed=None, options=None):  # noqa: ANN001
        del seed, options
        self._episode += 1
        self._step = 0
        return np.zeros((4, 64, 64), dtype=np.uint8), {
            "session_id": "session",
            "run_id": f"run-{self._episode}",
            "map_uid": "test-map",
            "frame_id": 0,
            "timestamp_ns": 0,
            "race_time_ms": 0,
            "terminal_reason": None,
            "progress_index": 0,
            "progress_delta": 0,
            "no_progress_steps": 0,
            "stray_distance": 0.0,
            "trajectory_arc_length_m": 0.0,
            "reward_reason": None,
            "tm20ai_done_type": None,
            "speed_kmh": 0.0,
            "gear": 0,
            "rpm": 0.0,
            "pos_xyz": (0.0, 0.0, 0.0),
            "vel_xyz": (0.0, 0.0, 0.0),
            "yaw_pitch_roll": (0.0, 0.0, 0.0),
        }

    def step(self, action):  # noqa: ANN001
        del action
        self._step += 1
        terminated = self._step >= 2
        reward = 1.0
        info = {
            "session_id": "session",
            "run_id": f"run-{self._episode}",
            "map_uid": "test-map",
            "frame_id": self._step,
            "timestamp_ns": self._step * 1_000,
            "race_time_ms": self._step * 50,
            "terminal_reason": "finished" if terminated else None,
            "progress_index": self._step,
            "progress_delta": 1,
            "no_progress_steps": 0,
            "stray_distance": 0.0,
            "trajectory_arc_length_m": float(self._step),
            "reward_reason": "finished" if terminated else None,
            "tm20ai_done_type": "terminated" if terminated else None,
            "speed_kmh": 100.0,
            "gear": 3,
            "rpm": 5000.0,
            "pos_xyz": (1.0, 2.0, 3.0),
            "vel_xyz": (0.0, 0.0, 0.0),
            "yaw_pitch_roll": (0.0, 0.0, 0.0),
        }
        return np.zeros((4, 64, 64), dtype=np.uint8), reward, terminated, False, info

    def close(self):
        return None


def write_train_config(path: Path, artifacts_root: Path, trajectory_path: Path) -> None:
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
  algorithm: sac
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
  target_entropy: -3.0
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
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trajectory_path,
        map_uid=np.asarray(["test-map"]),
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        tangents=np.asarray([[1.0, 0.0, 0.0]] * 3, dtype=np.float32),
        arc_length=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        race_time_ms=np.asarray([0.0, 100.0, 200.0], dtype=np.float32),
    )


def test_worker_emits_transition_batches_and_eval_results(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    monkeypatch.setattr("tm20ai.train.evaluator.runtime_trajectory_path_for_map", lambda *args, **kwargs: trajectory_path)

    command_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    shutdown_event = threading.Event()
    worker = SACWorker(
        config_path=str(config_path),
        command_queue=command_queue,
        output_queue=output_queue,
        shutdown_event=shutdown_event,
        env_factory=lambda _path, _benchmark=False: FakeTrainingEnv(),
    )
    command_queue.put({"type": "run_eval", "episodes": 1, "seed_base": 7, "run_name": "worker_eval"})

    thread = threading.Thread(target=worker.run, kwargs={"max_env_steps": 4}, daemon=True)
    thread.start()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    message_types: list[str] = []
    while not output_queue.empty():
        message_types.append(output_queue.get()["type"])
    assert "transition_batch" in message_types
    assert "episode_summary" in message_types
    assert "eval_result" in message_types


def test_learner_checkpoint_roundtrip_and_command_scheduling(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    learner = SACLearner(config_path=config_path, run_name="unit_train")
    command_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    shutdown_event = threading.Event()
    learner.attach_worker(command_queue=command_queue, output_queue=output_queue, shutdown_event=shutdown_event)

    for step in range(3):
        output_queue.put(
            {
                "type": "transition_batch",
                "env_step": step + 1,
                "transitions": [
                    {
                        "obs_uint8": np.full((4, 64, 64), step, dtype=np.uint8),
                        "telemetry_float": np.full((14,), 0.1 * step, dtype=np.float32),
                        "action": np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
                        "reward": 1.0,
                        "next_obs_uint8": np.full((4, 64, 64), step + 1, dtype=np.uint8),
                        "next_telemetry_float": np.full((14,), 0.1 * (step + 1), dtype=np.float32),
                        "terminated": step == 2,
                        "truncated": False,
                        "episode_id": "episode",
                        "map_uid": "test-map",
                        "step_idx": step,
                    }
                ],
            }
        )

    drained = learner.drain_messages(timeout=0.01)
    assert drained == 3
    assert learner.replay.size == 3
    assert learner.maybe_train() == 3
    assert learner.learner_step == 3

    learner.broadcast_actor(force=True)
    learner.maybe_schedule_eval()

    commands = []
    while not command_queue.empty():
        commands.append(command_queue.get()["type"])
    assert "set_actor" in commands
    assert "run_eval" in commands

    checkpoint_path = learner.save_checkpoint()
    assert checkpoint_path.exists()

    checkpoint_policy = resolve_policy_adapter(policy="checkpoint", checkpoint=checkpoint_path)
    action = checkpoint_policy.act(
        np.zeros((4, 64, 64), dtype=np.uint8),
        {
            "run_id": "checkpoint-run",
            "speed_kmh": 0.0,
            "rpm": 0.0,
            "gear": 0,
        },
    )
    assert action.shape == (3,)

    restored = SACLearner(config_path=config_path, run_name="unit_train_restored")
    restored.load_checkpoint(checkpoint_path)
    assert restored.env_step == learner.env_step
    assert restored.learner_step == learner.learner_step
    learner.close()
    restored.close()
