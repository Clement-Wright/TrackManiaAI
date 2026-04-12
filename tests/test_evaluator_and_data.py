from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tm20ai.data.dataset import (
    FullBehaviorCloningDataset,
    split_demo_dataset,
    validate_full_demo_dataset,
)
from tm20ai.train.evaluator import (
    FixedActionPolicy,
    KeyboardTeleopPolicy,
    ScriptedPolicyAdapter,
    VK_A,
    VK_D,
    VK_W,
    ZeroPolicy,
    run_policy_episodes,
)


class FakeEnv:
    def __init__(self) -> None:
        self.default_action = np.zeros(3, dtype=np.float32)
        self._episode_index = -1
        self._step_index = 0

    def reset(self, *, seed=None, options=None):  # noqa: ANN001
        del seed, options
        self._episode_index += 1
        self._step_index = 0
        return np.zeros((4, 64, 64), dtype=np.uint8), {"map_uid": "test-map", "run_id": f"run-{self._episode_index}"}

    def step(self, action):  # noqa: ANN001
        self._step_index += 1
        terminated = self._step_index >= 2
        info = {
            "session_id": "session",
            "run_id": f"run-{self._episode_index}",
            "map_uid": "test-map",
            "frame_id": self._step_index,
            "timestamp_ns": self._step_index * 1000,
            "race_time_ms": self._step_index * 50,
            "terminal_reason": "finished" if terminated else None,
            "progress_index": self._step_index,
            "progress_delta": 1,
            "no_progress_steps": 0,
            "stray_distance": 0.0,
            "trajectory_arc_length_m": float(self._step_index),
            "reward_reason": "finished" if terminated else None,
            "tm20ai_done_type": "terminated" if terminated else None,
            "speed_kmh": 100.0,
            "gear": 3,
            "rpm": 5000.0,
            "pos_xyz": (1.0, 2.0, 3.0),
            "vel_xyz": (0.0, 0.0, 0.0),
            "yaw_pitch_roll": (0.0, 0.0, 0.0),
        }
        return np.zeros((4, 64, 64), dtype=np.uint8), 1.0, terminated, False, info

    def close(self):
        return None

    def benchmarks(self):
        return {
            "tm20ai": {
                "avg_obs_retrieval_seconds": 0.01,
                "avg_preprocess_seconds": 0.002,
                "avg_reward_compute_seconds": 0.003,
            },
            "rtgym": {"send_control_duration": (0.001, 0.0)},
        }


def write_test_trajectory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        map_uid=np.asarray(["test-map"]),
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        tangents=np.asarray([[1.0, 0.0, 0.0]] * 3, dtype=np.float32),
        arc_length=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        race_time_ms=np.asarray([0.0, 100.0, 200.0], dtype=np.float32),
    )


def write_test_config(path: Path, artifacts_root: Path) -> None:
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
  episodes: 2
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
  single_live_env: false
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
  validation_fraction: 0.5
artifacts:
  root: "{artifacts_root.as_posix()}"
""".strip(),
        encoding="utf-8",
    )


def test_run_policy_episodes_writes_scalar_artifacts(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "trajectory_0p5m.npz"
    write_test_config(config_path, artifacts_root)
    write_test_trajectory(trajectory_path)

    monkeypatch.setattr("tm20ai.train.evaluator.runtime_trajectory_path_for_map", lambda *args, **kwargs: trajectory_path)
    env_factory = lambda _path, _benchmark=False: FakeEnv()

    for policy in (
        ZeroPolicy(),
        FixedActionPolicy(action=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        ScriptedPolicyAdapter(callback=lambda obs, info: np.asarray([0.5, 0.0, 0.0], dtype=np.float32)),
    ):
        result = run_policy_episodes(
            config_path=config_path,
            mode="eval",
            policy=policy,
            episodes=2,
            seed_base=123,
            record_video=False,
            env_factory=env_factory,
        )
        assert Path(result["summary_path"]).exists()
        assert Path(result["episode_index_path"]).exists()
        assert result["summary"]["completion_rate"] == 1.0


def test_demo_runs_write_sidecars_and_dataset_split(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "trajectory_0p5m.npz"
    write_test_config(config_path, artifacts_root)
    write_test_trajectory(trajectory_path)

    monkeypatch.setattr("tm20ai.train.evaluator.runtime_trajectory_path_for_map", lambda *args, **kwargs: trajectory_path)
    env_factory = lambda _path, _benchmark=False: FakeEnv()

    result = run_policy_episodes(
        config_path=config_path,
        mode="demos",
        policy=FixedActionPolicy(action=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        episodes=2,
        seed_base=123,
        record_video=False,
        env_factory=env_factory,
    )
    episodes_dir = Path(result["run_dir"]) / "episodes"
    sidecars = sorted(episodes_dir.glob("*_observations.npz"))
    assert sidecars

    split = split_demo_dataset(Path(result["run_dir"]), validation_fraction=0.5, seed=7)
    assert split.train_episodes
    validation = validate_full_demo_dataset(Path(result["run_dir"]))
    assert validation.total_nonzero_action_steps > 0
    dataset = FullBehaviorCloningDataset(split.train_episodes)
    assert len(dataset) > 0
    observation, telemetry, action = dataset[0]
    assert observation.shape == (4, 64, 64)
    assert telemetry.shape == (14,)
    assert action.shape == (3,)


def test_keyboard_teleop_policy_maps_keys() -> None:
    pressed = {VK_W, VK_D}
    policy = KeyboardTeleopPolicy(key_state_reader=lambda key: 0x8000 if key in pressed else 0)
    action = policy.act(np.zeros((4, 64, 64), dtype=np.uint8), {})
    assert np.allclose(action, np.asarray([1.0, 0.0, 1.0], dtype=np.float32))

    conflicting = KeyboardTeleopPolicy(key_state_reader=lambda key: 0x8000 if key in {VK_A, VK_D} else 0)
    conflicting_action = conflicting.act(np.zeros((4, 64, 64), dtype=np.uint8), {})
    assert np.allclose(conflicting_action, np.asarray([0.0, 0.0, 0.0], dtype=np.float32))


def test_validate_full_demo_dataset_rejects_missing_sidecar(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "trajectory_0p5m.npz"
    write_test_config(config_path, artifacts_root)
    write_test_trajectory(trajectory_path)

    monkeypatch.setattr("tm20ai.train.evaluator.runtime_trajectory_path_for_map", lambda *args, **kwargs: trajectory_path)
    env_factory = lambda _path, _benchmark=False: FakeEnv()
    result = run_policy_episodes(
        config_path=config_path,
        mode="demos",
        policy=FixedActionPolicy(action=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        episodes=2,
        seed_base=123,
        record_video=False,
        env_factory=env_factory,
    )
    sidecar = next((Path(result["run_dir"]) / "episodes").glob("*_observations.npz"))
    sidecar.unlink()

    with pytest.raises(RuntimeError, match="missing the observation sidecar"):
        validate_full_demo_dataset(Path(result["run_dir"]))


def test_validate_full_demo_dataset_rejects_mixed_map_uids_without_override(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "trajectory_0p5m.npz"
    write_test_config(config_path, artifacts_root)
    write_test_trajectory(trajectory_path)

    monkeypatch.setattr("tm20ai.train.evaluator.runtime_trajectory_path_for_map", lambda *args, **kwargs: trajectory_path)
    env_factory = lambda _path, _benchmark=False: FakeEnv()
    result = run_policy_episodes(
        config_path=config_path,
        mode="demos",
        policy=FixedActionPolicy(action=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        episodes=2,
        seed_base=123,
        record_video=False,
        env_factory=env_factory,
    )
    metadata_paths = sorted((Path(result["run_dir"]) / "episodes").glob("*.json"))
    metadata = json.loads(metadata_paths[-1].read_text(encoding="utf-8"))
    metadata["map_uid"] = "other-map"
    metadata_paths[-1].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="multiple map_uids"):
        validate_full_demo_dataset(Path(result["run_dir"]))

    validation = validate_full_demo_dataset(Path(result["run_dir"]), allow_mixed_map_uids=True)
    assert validation.valid_episode_count == 2
    assert validation.sample_count > 0


def test_human_demo_run_rejects_all_zero_actions(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "trajectory_0p5m.npz"
    write_test_config(config_path, artifacts_root)
    write_test_trajectory(trajectory_path)

    monkeypatch.setattr("tm20ai.train.evaluator.runtime_trajectory_path_for_map", lambda *args, **kwargs: trajectory_path)
    env_factory = lambda _path, _benchmark=False: FakeEnv()
    policy = KeyboardTeleopPolicy(key_state_reader=lambda _key: 0)

    with pytest.raises(RuntimeError, match="did not record any non-zero actions"):
        run_policy_episodes(
            config_path=config_path,
            mode="demos",
            policy=policy,
            episodes=1,
            seed_base=123,
            record_video=False,
            env_factory=env_factory,
        )

    run_dir = artifacts_root / "demos"
    summaries = sorted(run_dir.glob("*/summary.json"))
    assert summaries
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    assert summary["valid_for_bc"] is False
    assert summary["invalid_reason"] == "all_zero_actions"


def test_validate_full_demo_dataset_rejects_zero_action_demos(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "trajectory_0p5m.npz"
    write_test_config(config_path, artifacts_root)
    write_test_trajectory(trajectory_path)

    monkeypatch.setattr("tm20ai.train.evaluator.runtime_trajectory_path_for_map", lambda *args, **kwargs: trajectory_path)
    env_factory = lambda _path, _benchmark=False: FakeEnv()
    result = run_policy_episodes(
        config_path=config_path,
        mode="demos",
        policy=FixedActionPolicy(action=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        episodes=1,
        seed_base=123,
        record_video=False,
        env_factory=env_factory,
    )
    sidecar = next((Path(result["run_dir"]) / "episodes").glob("*_observations.npz"))
    payload = dict(np.load(sidecar))
    payload["action"] = np.zeros_like(payload["action"])
    np.savez_compressed(sidecar, **payload)

    with pytest.raises(RuntimeError, match="contain no non-zero recorded actions"):
        validate_full_demo_dataset(Path(result["run_dir"]))


def test_pretrain_bc_writes_best_and_final_checkpoints(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    artifacts_root = tmp_path / "artifacts"
    trajectory_path = tmp_path / "trajectory_0p5m.npz"
    write_test_config(config_path, artifacts_root)
    write_test_trajectory(trajectory_path)

    monkeypatch.setattr("tm20ai.train.evaluator.runtime_trajectory_path_for_map", lambda *args, **kwargs: trajectory_path)
    env_factory = lambda _path, _benchmark=False: FakeEnv()
    demo_result = run_policy_episodes(
        config_path=config_path,
        mode="demos",
        policy=FixedActionPolicy(action=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        episodes=2,
        seed_base=123,
        record_video=False,
        env_factory=env_factory,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "pretrain_bc.py"),
            "--config",
            str(config_path),
            "--demos-root",
            str(demo_result["run_dir"]),
            "--run-name",
            "unit_bc",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    summary_path = artifacts_root / "bc" / "unit_bc" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["observation_mode"] == "full"
    assert summary["map_uid"] == "test-map"
    assert summary["best_checkpoint_path"].endswith("actor_checkpoint_best.pt")
    assert summary["final_checkpoint_path"].endswith("actor_checkpoint_final.pt")
    assert Path(summary["best_checkpoint_path"]).exists()
    assert Path(summary["final_checkpoint_path"]).exists()


def test_export_video_fails_clearly_without_ffmpeg(tmp_path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "frame_000000.png"
    frame_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D49484452000000010000000108000000003A7E9B550000000A49444154789C6360000000020001E221BC330000000049454E44AE426082"
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "export_video.py"),
            "--frames-dir",
            str(frames_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PATH": ""},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "ffmpeg is not available on PATH" in (result.stderr + result.stdout)
