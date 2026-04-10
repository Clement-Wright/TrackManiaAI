from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from rtgym.envs.real_time_env import RealTimeEnv

from ..config import load_tm20ai_config
from .rt_interface import TM20AIRtInterface


def build_rtgym_config(config_path: str | Path, *, benchmark: bool = False) -> dict[str, Any]:
    config = load_tm20ai_config(config_path)
    if config.observation.mode != "full":
        raise NotImplementedError(
            f"Observation mode {config.observation.mode!r} is planned but not implemented in Phase 3-4."
        )
    return {
        "interface": TM20AIRtInterface,
        "interface_kwargs": {"config_path": str(Path(config_path).resolve())},
        "time_step_duration": config.runtime.time_step_duration,
        "start_obs_capture": config.runtime.start_obs_capture,
        "time_step_timeout_factor": config.runtime.time_step_timeout_factor,
        "act_buf_len": config.runtime.act_buf_len,
        "wait_on_done": config.runtime.wait_on_done,
        "ep_max_length": config.runtime.ep_max_length,
        "act_in_obs": False,
        "reset_act_buf": True,
        "benchmark": benchmark,
    }


class TM20AIGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path, *, benchmark: bool = False):
        self.config_path = Path(config_path).resolve()
        self._rt_env = RealTimeEnv(build_rtgym_config(self.config_path, benchmark=benchmark))
        self.action_space = self._rt_env.action_space
        self.observation_space = self._rt_env.interface.get_observation_space().spaces[0]

    @property
    def interface(self) -> TM20AIRtInterface:
        return self._rt_env.interface

    @property
    def default_action(self):
        return self.interface.get_default_action()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self._rt_env.reset(seed=seed, options=options)
        return self._unwrap_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._rt_env.step(action)
        done_type = info.get("tm20ai_done_type")
        if done_type == "truncated":
            terminated = False
            truncated = True
        elif done_type == "terminated":
            terminated = True
        return self._unwrap_observation(observation), float(reward), bool(terminated), bool(truncated), info

    def wait(self):
        self._rt_env.wait()

    def benchmarks(self) -> dict[str, Any]:
        return self._rt_env.benchmarks()

    def close(self) -> None:
        try:
            self._rt_env.stop()
        finally:
            close = getattr(self.interface, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _unwrap_observation(observation):
        if not isinstance(observation, tuple) or len(observation) != 1:
            raise RuntimeError("Expected a single observation tensor from the rtgym interface.")
        return observation[0]


def make_env(config_path: str | Path, *, benchmark: bool = False) -> TM20AIGymEnv:
    return TM20AIGymEnv(config_path, benchmark=benchmark)
