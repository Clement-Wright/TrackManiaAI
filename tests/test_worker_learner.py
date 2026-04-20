from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from tm20ai.action_space import ACTION_DIM
from tm20ai.train.features import TELEMETRY_DIM
from tm20ai.train.evaluator import resolve_policy_adapter
from tm20ai.train.learner import SACLearner
from tm20ai.train.protocol import EvalResult
from tm20ai.train.worker import SACWorker


class FakeTrainingEnv:
    def __init__(self) -> None:
        self.default_action = np.zeros(ACTION_DIM, dtype=np.float32)
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


class FakeWorkerProcess:
    def __init__(self, *, alive: bool, exitcode: int) -> None:
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None) -> None:  # noqa: ANN001
        del timeout
        return None

    def terminate(self) -> None:
        self._alive = False


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
  target_entropy: -2.0
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

    learner = SACLearner(config_path=config_path, run_name="worker_sync_case")
    learner.env_step = 3
    learner.learner_step = 5
    learner.broadcast_actor(force=True)
    learner.env_step = 9
    learner.learner_step = 11
    learner.broadcast_actor(force=True)
    checkpoint_path = learner.save_checkpoint()

    command_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    eval_result_queue: queue.Queue = queue.Queue()
    shutdown_event = threading.Event()
    worker_done_event = threading.Event()
    bootstrap_log_path = learner.paths.run_dir / "worker_bootstrap.log"
    worker = SACWorker(
        config_path=str(config_path),
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
        bootstrap_log_path=str(bootstrap_log_path),
        env_factory=lambda _path, _benchmark=False: FakeTrainingEnv(),
    )
    command_queue.put(
        {
            "type": "run_eval",
            "episodes": 1,
            "seed_base": 7,
            "run_name": "worker_eval",
            "modes": ["deterministic", "stochastic"],
            "trace_seconds": 0.1,
            "eval_provenance_mode": "checkpoint_authoritative",
            "eval_checkpoint_path": str(checkpoint_path),
            "eval_checkpoint_sha256": "unit-sha",
            "eval_checkpoint_env_step": 9,
            "eval_checkpoint_learner_step": 11,
            "eval_checkpoint_actor_step": None,
            "scheduled_actor_version": 2,
        }
    )

    thread = threading.Thread(target=worker.run, kwargs={"max_env_steps": 4}, daemon=True)
    thread.start()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    message_types: list[str] = []
    while not output_queue.empty():
        message_types.append(output_queue.get()["type"])
    eval_results: list[EvalResult] = []
    while not eval_result_queue.empty():
        eval_results.append(eval_result_queue.get())
    assert "transition_batch" in message_types
    assert "episode_summary" in message_types
    assert "actor_sync_desired_seen" in message_types
    assert "eval_started" in message_types
    assert "actor_sync_applied" in message_types
    assert eval_results
    assert eval_results[-1].summary["eval_actor_version"] == 2
    assert eval_results[-1].summary["eval_actor_source_learner_step"] == 11
    assert eval_results[-1].summary["eval_checkpoint_path"] == str(checkpoint_path.resolve())
    assert eval_results[-1].summary["eval_checkpoint_sha256"] == "unit-sha"
    assert "deterministic" in eval_results[-1].summary["eval_mode_summaries"]
    assert "stochastic" in eval_results[-1].summary["eval_mode_summaries"]
    assert worker_done_event.is_set()
    worker_events = [json.loads(line) for line in bootstrap_log_path.with_name("worker_events.log").read_text(encoding="utf-8").splitlines()]
    events = [entry["event"] for entry in worker_events]
    assert "actor_sync_desired_seen" in events
    assert "actor_sync_applied" in events
    assert "eval_begin" in events
    assert "eval_end" in events
    assert "eval_result_published" in events
    actor_sync_events = [entry for entry in worker_events if entry["event"] == "actor_sync_applied"]
    assert actor_sync_events[-1]["payload"]["applied_actor_version"] == 2
    desired_seen_events = [entry for entry in worker_events if entry["event"] == "actor_sync_desired_seen"]
    assert desired_seen_events[-1]["payload"]["desired_actor_version"] == 2
    status_payload = json.loads(learner.paths.worker_actor_status_path.read_text(encoding="utf-8"))
    assert status_payload["desired_actor_version"] == 2
    assert status_payload["desired_actor_ready_for_control"] is True
    assert status_payload["seen_actor_version"] == 2
    assert status_payload["ready_for_control_seen"] is True
    assert status_payload["applied_actor_version"] == 2
    assert status_payload["actor_ready_for_control"] is True
    assert status_payload["applied_source_learner_step"] == 11
    learner.close()


def test_worker_preserves_pending_eval_when_shutdown_is_requested(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    worker = SACWorker(
        config_path=str(config_path),
        command_queue=queue.Queue(),
        output_queue=queue.Queue(),
        eval_result_queue=queue.Queue(),
        shutdown_event=threading.Event(),
        worker_done_event=threading.Event(),
        bootstrap_log_path=str(artifacts_root / "worker_bootstrap.log"),
        env_factory=lambda _path, _benchmark=False: FakeTrainingEnv(),
    )
    worker.command_queue.put({"type": "run_eval", "episodes": 1, "run_name": "pending_eval"})
    worker.command_queue.put({"type": "shutdown"})

    pending = worker._drain_commands()

    assert pending is not None
    assert pending["run_name"] == "pending_eval"
    assert worker._shutdown_requested is True


def test_learner_checkpoint_roundtrip_and_command_scheduling(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    demo_root = tmp_path / "demo_root"
    learner = SACLearner(
        config_path=config_path,
        run_name="unit_train",
        demo_root=demo_root,
        eval_episodes_override=3,
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

    learner.broadcast_actor(force=True)
    startup_actor_payload = json.loads(learner.paths.desired_actor_path.read_text(encoding="utf-8"))
    assert startup_actor_payload["ready_for_control"] is False
    assert startup_actor_payload["control_ready_reason"] == "startup_untrained"
    assert startup_actor_payload["actor_state_path"].endswith("actor_v000001.pt")

    for step in range(3):
        output_queue.put(
            {
                "type": "transition_batch",
                "env_step": step + 1,
                "transitions": [
                    {
                        "obs_uint8": np.full((4, 64, 64), step, dtype=np.uint8),
                        "telemetry_float": np.full((TELEMETRY_DIM,), 0.1 * step, dtype=np.float32),
                        "action": np.asarray([0.1, 0.0], dtype=np.float32),
                        "reward": 1.0,
                        "next_obs_uint8": np.full((4, 64, 64), step + 1, dtype=np.uint8),
                        "next_telemetry_float": np.full((TELEMETRY_DIM,), 0.1 * (step + 1), dtype=np.float32),
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
    desired_actor_payload = json.loads(learner.paths.desired_actor_path.read_text(encoding="utf-8"))
    assert desired_actor_payload["desired_actor_version"] >= 2
    assert desired_actor_payload["ready_for_control"] is True
    assert desired_actor_payload["control_ready_reason"] == "trained_updates_available"
    assert desired_actor_payload["actor_state_path"].endswith(
        f"actor_v{desired_actor_payload['desired_actor_version']:06d}.pt"
    )
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
    output_queue.put(
        {
            "type": "heartbeat",
            "env_step": learner.env_step,
            "episode_index": 1,
            "desired_actor_version": desired_actor_payload["desired_actor_version"],
            "desired_actor_ready_for_control": True,
            "seen_actor_version": desired_actor_payload["desired_actor_version"],
            "ready_for_control_seen": True,
            "applied_actor_version": desired_actor_payload["desired_actor_version"],
            "actor_ready_for_control": True,
            "applied_source_learner_step": learner.learner_step,
            "last_actor_apply_env_step": learner.env_step,
            "last_actor_apply_episode_index": 1,
            "control_ready_reason": desired_actor_payload["control_ready_reason"],
        }
    )
    assert learner.drain_messages(timeout=0.01) == 0

    learner.maybe_schedule_eval()

    commands = []
    while not command_queue.empty():
        commands.append(command_queue.get())
    assert any(command["type"] == "run_eval" for command in commands)
    run_eval_command = next(command for command in commands if command["type"] == "run_eval")
    assert run_eval_command["eval_actor_version"] == desired_actor_payload["desired_actor_version"]
    assert run_eval_command["eval_actor_source_learner_step"] == learner.learner_step
    assert run_eval_command["eval_provenance_mode"] == "checkpoint_authoritative"
    assert Path(str(run_eval_command["eval_checkpoint_path"])).exists()
    assert run_eval_command["eval_checkpoint_env_step"] == learner.env_step
    assert run_eval_command["scheduled_actor_version"] == desired_actor_payload["desired_actor_version"]

    output_queue.put(
        {
            "type": "eval_started",
            "run_name": f"unit_train_step_{learner.env_step:08d}",
            "env_step": learner.env_step,
            "learner_step": learner.learner_step,
            "episodes": 3,
            "timestamp": time.time(),
            "eval_actor_version": desired_actor_payload["desired_actor_version"],
            "eval_actor_source_learner_step": learner.learner_step,
            "scheduled_actor_version": desired_actor_payload["desired_actor_version"],
            "eval_provenance_mode": "checkpoint_authoritative",
            "eval_checkpoint_path": run_eval_command["eval_checkpoint_path"],
            "eval_checkpoint_sha256": run_eval_command["eval_checkpoint_sha256"],
            "eval_checkpoint_env_step": run_eval_command["eval_checkpoint_env_step"],
            "eval_checkpoint_learner_step": run_eval_command["eval_checkpoint_learner_step"],
            "eval_checkpoint_actor_step": run_eval_command["eval_checkpoint_actor_step"],
        }
    )
    assert learner.drain_messages(timeout=0.01) == 0
    assert learner.started_eval is not None
    assert learner.started_eval["run_name"] == "unit_train_step_00000003"
    assert learner.desired_actor_ready_for_control is True
    assert learner.ready_for_control_seen is True
    assert learner.applied_actor_version == desired_actor_payload["desired_actor_version"]
    assert learner.seen_actor_version == desired_actor_payload["desired_actor_version"]
    assert learner.actor_ready_for_control is True
    assert learner.applied_source_learner_step == learner.learner_step
    assert learner.last_actor_apply_env_step == learner.env_step
    assert learner.last_actor_apply_episode_index == 1
    assert learner.eval_actor_version == desired_actor_payload["desired_actor_version"]
    assert learner.eval_actor_source_learner_step == learner.learner_step

    eval_result_queue.put(
        EvalResult(
            checkpoint_step=learner.env_step,
            env_step=learner.env_step,
            learner_step=learner.learner_step,
            summary_path=str(tmp_path / "eval_summary.json"),
            summary={
                "env_step": learner.env_step,
                "mean_final_progress_index": 12.0,
                "eval_actor_version": desired_actor_payload["desired_actor_version"],
                "eval_actor_source_learner_step": learner.learner_step,
                "eval_provenance_mode": "checkpoint_authoritative",
                "eval_checkpoint_path": run_eval_command["eval_checkpoint_path"],
                "eval_checkpoint_sha256": run_eval_command["eval_checkpoint_sha256"],
                "eval_checkpoint_env_step": run_eval_command["eval_checkpoint_env_step"],
                "eval_checkpoint_learner_step": run_eval_command["eval_checkpoint_learner_step"],
                "eval_checkpoint_actor_step": run_eval_command["eval_checkpoint_actor_step"],
            },
            timestamp=time.time(),
        )
    )
    assert learner.drain_eval_results(timeout=0.01) == 1
    assert learner.latest_eval_summary is not None
    assert learner.latest_eval_summary["mean_final_progress_index"] == 12.0
    assert learner.eval_history
    assert learner.started_eval is None
    assert learner.latest_checkpoint_path is not None

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
    assert action.shape == (ACTION_DIM,)

    restored = SACLearner(config_path=config_path, run_name="unit_train_restored")
    restored.load_checkpoint(checkpoint_path)
    assert restored.env_step == learner.env_step
    assert restored.learner_step == learner.learner_step
    worker_done_event.set()
    final_checkpoint = learner.finalize_run(timeout_seconds=0.1)
    assert final_checkpoint.exists()
    assert learner.clean_shutdown is True
    assert learner.latest_eval_summary is not None
    summary_payload = json.loads(learner.paths.summary_json.read_text(encoding="utf-8"))
    assert summary_payload["latest_eval_summary"] is not None
    assert summary_payload["started_eval"] is None
    assert summary_payload["desired_actor_version"] == desired_actor_payload["desired_actor_version"]
    assert summary_payload["desired_actor_ready_for_control"] is True
    assert summary_payload["desired_actor_control_ready_reason"] == "trained_updates_available"
    assert summary_payload["seen_actor_version"] == desired_actor_payload["desired_actor_version"]
    assert summary_payload["ready_for_control_seen"] is True
    assert summary_payload["applied_actor_version"] == desired_actor_payload["desired_actor_version"]
    assert summary_payload["actor_ready_for_control"] is True
    assert summary_payload["applied_source_learner_step"] == learner.learner_step
    assert summary_payload["last_actor_apply_env_step"] == learner.env_step
    assert summary_payload["last_actor_apply_episode_index"] == 1
    assert summary_payload["eval_actor_version"] == desired_actor_payload["desired_actor_version"]
    assert summary_payload["eval_actor_source_learner_step"] == learner.learner_step
    assert summary_payload["eval_history"]
    assert summary_payload["checkpoint_history"]
    assert summary_payload["init_mode"] == "scratch"
    assert summary_payload["demo_root"] == str(demo_root.resolve())
    assert summary_payload["eval_episodes"] == 3
    assert summary_payload["last_worker_heartbeat"] is not None
    checkpoint_sidecar = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert checkpoint_sidecar["desired_actor_ready_for_control"] is True
    assert checkpoint_sidecar["ready_for_control_seen"] is True
    assert checkpoint_sidecar["actor_ready_for_control"] is True
    assert checkpoint_sidecar["applied_source_learner_step"] == learner.learner_step
    assert checkpoint_sidecar["init_mode"] == "scratch"
    assert checkpoint_sidecar["demo_root"] == str(demo_root.resolve())
    assert checkpoint_sidecar["replay_seeded"] is False
    learner.close()
    restored.close()


def test_learner_treats_clean_worker_exit_at_env_limit_as_normal_completion(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    learner = SACLearner(config_path=config_path, run_name="unit_train_exit_ok", max_env_steps=1)
    command_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    eval_result_queue: queue.Queue = queue.Queue()
    shutdown_event = threading.Event()
    worker_done_event = threading.Event()
    worker_done_event.set()
    output_queue.put(
        {
            "type": "transition_batch",
            "env_step": 1,
            "transitions": [
                {
                    "obs_uint8": np.zeros((4, 64, 64), dtype=np.uint8),
                    "telemetry_float": np.zeros((TELEMETRY_DIM,), dtype=np.float32),
                    "action": np.zeros((ACTION_DIM,), dtype=np.float32),
                    "reward": 0.0,
                    "next_obs_uint8": np.zeros((4, 64, 64), dtype=np.uint8),
                    "next_telemetry_float": np.zeros((TELEMETRY_DIM,), dtype=np.float32),
                    "terminated": False,
                    "truncated": False,
                    "episode_id": "episode",
                    "map_uid": "test-map",
                    "step_idx": 0,
                }
            ],
        }
    )
    learner.attach_worker(
        command_queue=command_queue,
        output_queue=output_queue,
        eval_result_queue=eval_result_queue,
        shutdown_event=shutdown_event,
        worker_done_event=worker_done_event,
        worker_process=FakeWorkerProcess(alive=False, exitcode=0),
    )

    learner.run()

    assert learner.env_step == 1
    assert learner.termination_reason == "max_env_steps"
    learner.close()


def test_learner_disables_eval_when_eval_episodes_is_zero(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    learner = SACLearner(config_path=config_path, run_name="unit_train_no_eval", eval_episodes_override=0)
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
    learner.env_step = 10
    learner.maybe_schedule_eval()
    assert command_queue.empty()
    assert learner.pending_eval is None
    assert learner.eval_in_flight is False
    summary_payload = json.loads(learner.paths.summary_json.read_text(encoding="utf-8"))
    assert summary_payload["pending_eval"] is None
    assert summary_payload["eval_history"] == []
    learner.close()


def test_worker_action_stats_derive_exclusive_gas_and_brake_from_throttle(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    worker = SACWorker(
        config_path=str(config_path),
        command_queue=queue.Queue(),
        output_queue=queue.Queue(),
        eval_result_queue=queue.Queue(),
        shutdown_event=threading.Event(),
        worker_done_event=threading.Event(),
        bootstrap_log_path=str(artifacts_root / "worker_bootstrap.log"),
        env_factory=lambda _path, _benchmark=False: FakeTrainingEnv(),
    )
    worker._has_policy_weights = True
    worker._status_mode = "train"
    worker._env_step = worker.config.train.environment_steps_before_training
    worker._applied_actor_version = 3
    worker._seen_actor_version = 3
    worker._desired_actor_version = 3
    worker._desired_actor_ready_for_control = True
    worker._ready_for_control_seen = True
    worker._actor_ready_for_control = True
    worker._applied_source_learner_step = 10
    worker._control_ready_reason = "trained_updates_available"

    for index in range(10):
        throttle = 0.8 if index % 2 == 0 else -0.6
        worker._execute_training_action(np.asarray([throttle, 0.25], dtype=np.float32))
    worker._env_step = worker._action_stats_interval_steps
    worker._record_action_stats(
        raw_action=np.asarray([0.0, 0.0], dtype=np.float32),
        executed_action=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    assert worker._last_action_stats is not None
    assert worker._last_action_stats["action_guard_active"] is False
    assert worker._last_action_stats["control_source"] == "policy"
    assert worker._last_action_stats["mean_gas"] > 0.0
    assert worker._last_action_stats["mean_brake"] > 0.0
    assert worker._last_action_stats["fraction_gas_and_brake_gt_0_1"] == 0.0
    assert worker._last_action_stats["raw_fraction_gas_and_brake_gt_0_1"] == 0.0


def test_worker_select_action_uses_exploration_until_actor_ready(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    worker = SACWorker(
        config_path=str(config_path),
        command_queue=queue.Queue(),
        output_queue=queue.Queue(),
        eval_result_queue=queue.Queue(),
        shutdown_event=threading.Event(),
        worker_done_event=threading.Event(),
        bootstrap_log_path=str(artifacts_root / "worker_bootstrap.log"),
        env_factory=lambda _path, _benchmark=False: FakeTrainingEnv(),
    )
    worker.actor = object()
    worker._torch = object()
    worker._has_policy_weights = True
    worker._actor_ready_for_control = False
    worker._env_step = worker.config.train.environment_steps_before_training

    action = worker._select_action(np.zeros((4, 64, 64), dtype=np.uint8), np.zeros((TELEMETRY_DIM,), dtype=np.float32))
    assert action.shape == (ACTION_DIM,)
    assert -1.0 <= float(action[0]) <= 1.0
    assert -1.0 <= float(action[1]) <= 1.0


def test_worker_retries_same_desired_actor_after_transient_load_failure(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    learner = SACLearner(config_path=config_path, run_name="worker_retry_case")
    learner.env_step = 3
    learner.learner_step = 5
    learner.broadcast_actor(force=True)

    worker = SACWorker(
        config_path=str(config_path),
        command_queue=queue.Queue(),
        output_queue=queue.Queue(),
        eval_result_queue=queue.Queue(),
        shutdown_event=threading.Event(),
        worker_done_event=threading.Event(),
        bootstrap_log_path=str(learner.paths.run_dir / "worker_bootstrap.log"),
        env_factory=lambda _path, _benchmark=False: FakeTrainingEnv(),
    )
    worker._initialize_policy_components()
    real_load = worker._torch.load
    calls = {"count": 0}

    def flaky_load(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("transient load failure")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(worker._torch, "load", flaky_load)

    worker._apply_latest_desired_actor(mode="train")
    assert worker._applied_actor_version is None
    worker._apply_latest_desired_actor(mode="train")

    assert calls["count"] == 2
    assert worker._applied_actor_version == 1
    assert worker._actor_ready_for_control is True
    worker_events = [
        json.loads(line)
        for line in (learner.paths.run_dir / "worker_events.log").read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event"] == "actor_sync_error" for entry in worker_events)
    assert any(entry["event"] == "actor_sync_applied" for entry in worker_events)
    learner.close()


def test_worker_logs_first_sustained_movement_stall(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "reward" / "trajectory_0p5m.npz"
    write_train_config(config_path, artifacts_root, trajectory_path)

    worker = SACWorker(
        config_path=str(config_path),
        command_queue=queue.Queue(),
        output_queue=queue.Queue(),
        eval_result_queue=queue.Queue(),
        shutdown_event=threading.Event(),
        worker_done_event=threading.Event(),
        bootstrap_log_path=str(artifacts_root / "worker_bootstrap.log"),
        env_factory=lambda _path, _benchmark=False: FakeTrainingEnv(),
    )
    worker._episode_state = worker._episode_state.__class__(episode_index=1)
    worker._begin_movement_episode(
        {
            "run_id": "run-stall",
            "pos_xyz": (0.0, 0.0, 0.0),
        }
    )
    worker._env_step = 1
    worker._observe_movement(
        {
            "run_id": "run-stall",
            "frame_id": 1,
            "timestamp_ns": 0,
            "race_time_ms": 0,
            "speed_kmh": 25.0,
            "pos_xyz": (1.0, 0.0, 0.0),
        }
    )
    worker._env_step = 2
    worker._observe_movement(
        {
            "run_id": "run-stall",
            "frame_id": 2,
            "timestamp_ns": 500_000_000,
            "race_time_ms": 500,
            "speed_kmh": 0.0,
            "pos_xyz": (1.0, 0.0, 0.0),
        }
    )
    worker._env_step = 3
    worker._observe_movement(
        {
            "run_id": "run-stall",
            "frame_id": 3,
            "timestamp_ns": 1_600_000_000,
            "race_time_ms": 1600,
            "speed_kmh": 0.0,
            "pos_xyz": (1.0, 0.0, 0.0),
        }
    )
    worker._finalize_movement_episode({"run_id": "run-stall", "frame_id": 3, "race_time_ms": 1600, "reward_reason": "no_progress"})

    worker_events = [
        json.loads(line)
        for line in (artifacts_root / "worker_events.log").read_text(encoding="utf-8").splitlines()
    ]
    events = [entry["event"] for entry in worker_events]
    output_message_types: list[str] = []
    while not worker.output_queue.empty():
        output_message_types.append(worker.output_queue.get()["type"])
    assert "movement_started" in events
    assert "movement_stall_detected" in events
    assert "movement_episode_summary" in output_message_types
    summary = next(entry for entry in worker_events if entry["event"] == "movement_episode_summary")
    assert summary["payload"]["movement_started"] is True
    assert summary["payload"]["stall_count"] == 1
    assert summary["payload"]["first_stall"]["race_time_ms"] == 500
