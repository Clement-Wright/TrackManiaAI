from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces

from tm20ai.env import FROZEN_STEP_INFO_KEYS, build_rtgym_config
from tm20ai.env.gym_env import TM20AIGymEnv


ROOT = Path(__file__).resolve().parents[1]


def write_temp_config(tmp_path, source_name: str) -> Path:  # noqa: ANN001
    source_path = ROOT / "configs" / source_name
    config_path = tmp_path / source_name
    config_path.write_text(
        source_path.read_text(encoding="utf-8").replace("single_live_env: true", "single_live_env: false"),
        encoding="utf-8",
    )
    return config_path


class FakeFullInterface:
    def get_observation_space(self):
        return spaces.Tuple((spaces.Box(low=0, high=255, shape=(4, 64, 64), dtype=np.uint8),))

    def get_default_action(self):
        return np.zeros(3, dtype=np.float32)

    def get_runtime_metrics(self):
        return {
            "avg_obs_retrieval_seconds": 0.01,
            "avg_preprocess_seconds": 0.002,
            "avg_reward_compute_seconds": 0.003,
        }


class FakeLidarInterface:
    def get_observation_space(self):
        return spaces.Tuple((spaces.Box(low=0.0, high=1.0, shape=(83,), dtype=np.float32),))

    def get_default_action(self):
        return np.zeros(3, dtype=np.float32)

    def get_runtime_metrics(self):
        return {
            "avg_obs_retrieval_seconds": 0.01,
            "avg_preprocess_seconds": 0.001,
            "avg_reward_compute_seconds": 0.002,
        }


class FakeFullRealTimeEnv:
    def __init__(self, _config):
        self.interface = FakeFullInterface()
        self.action_space = spaces.Box(
            low=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):  # noqa: ANN001
        del seed, options
        return (np.zeros((4, 64, 64), dtype=np.uint8),), {"map_uid": "test-map", "run_id": "run"}

    def step(self, action):  # noqa: ANN001
        del action
        return (
            (np.zeros((4, 64, 64), dtype=np.uint8),),
            0.0,
            False,
            True,
            {
                "session_id": "session",
                "run_id": "run",
                "map_uid": "test-map",
                "frame_id": 1,
                "timestamp_ns": 10,
                "race_time_ms": 20,
                "terminal_reason": None,
                "progress_index": 0,
                "progress_delta": 0,
                "no_progress_steps": 0,
                "stray_distance": 0.0,
                "trajectory_arc_length_m": 0.0,
            },
        )

    def wait(self):
        return None

    def benchmarks(self):
        return {"send_control_duration": (0.001, 0.0)}

    def stop(self):
        return None


class FakeLidarRealTimeEnv(FakeFullRealTimeEnv):
    def __init__(self, _config):
        self.interface = FakeLidarInterface()
        self.action_space = spaces.Box(
            low=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):  # noqa: ANN001
        del seed, options
        return (np.zeros((83,), dtype=np.float32),), {"map_uid": "test-map", "run_id": "run"}

    def step(self, action):  # noqa: ANN001
        del action
        return (
            (np.zeros((83,), dtype=np.float32),),
            1.0,
            True,
            False,
            {
                "session_id": "session",
                "run_id": "run",
                "map_uid": "test-map",
                "frame_id": 2,
                "timestamp_ns": 20,
                "race_time_ms": 40,
                "terminal_reason": "finished",
                "progress_index": 10,
                "progress_delta": 2,
                "no_progress_steps": 0,
                "stray_distance": 0.0,
                "trajectory_arc_length_m": 10.0,
                "reward_reason": "finished",
                "tm20ai_done_type": "terminated",
            },
        )


def test_build_rtgym_config_freezes_public_contract_for_full() -> None:
    config = build_rtgym_config(ROOT / "configs" / "full_sac.yaml", benchmark=True)

    assert config["act_in_obs"] is False
    assert config["reset_act_buf"] is True
    assert config["time_step_duration"] == 0.05
    assert config["start_obs_capture"] == 0.04


def test_build_rtgym_config_freezes_public_contract_for_lidar() -> None:
    config = build_rtgym_config(ROOT / "configs" / "lidar_sac.yaml", benchmark=False)

    assert config["act_in_obs"] is False
    assert config["reset_act_buf"] is True
    assert config["time_step_duration"] == 0.05
    assert config["start_obs_capture"] == 0.04


def test_gym_env_step_normalizes_frozen_info_for_full(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("tm20ai.env.gym_env.RealTimeEnv", FakeFullRealTimeEnv)
    config_path = write_temp_config(tmp_path, "full_sac.yaml")

    env = TM20AIGymEnv(config_path, benchmark=True)
    try:
        observation, info = env.reset()
        assert observation.shape == (4, 64, 64)
        assert info["map_uid"] == "test-map"

        observation, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert observation.shape == (4, 64, 64)
        assert reward == 0.0
        assert terminated is False
        assert truncated is True
        assert info["tm20ai_done_type"] == "truncated"
        assert info["reward_reason"] == "ep_max_length"
        for key in FROZEN_STEP_INFO_KEYS:
            assert key in info
    finally:
        env.close()


def test_gym_env_step_normalizes_frozen_info_for_lidar(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("tm20ai.env.gym_env.RealTimeEnv", FakeLidarRealTimeEnv)
    config_path = write_temp_config(tmp_path, "lidar_sac.yaml")

    env = TM20AIGymEnv(config_path, benchmark=True)
    try:
        observation, info = env.reset()
        assert observation.shape == (83,)
        assert observation.dtype == np.float32
        assert info["map_uid"] == "test-map"

        observation, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert observation.shape == (83,)
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert info["tm20ai_done_type"] == "terminated"
        assert info["reward_reason"] == "finished"
        for key in FROZEN_STEP_INFO_KEYS:
            assert key in info

        benchmarks = env.benchmarks()
        assert "tm20ai" in benchmarks
        assert "rtgym" in benchmarks
    finally:
        env.close()
