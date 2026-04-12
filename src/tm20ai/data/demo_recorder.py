from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..env.trajectory import RuntimeTrajectory
from ..train.metrics import summarize_episode_trace
from .parquet_writer import EpisodeArtifactPaths, RunArtifactPaths, build_episode_artifact_paths, write_json, write_parquet_rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class EpisodeWriteResult:
    summary: dict[str, Any]
    paths: EpisodeArtifactPaths


class DemoRecorder:
    """Collects one episode at a time and writes scalar-first artifacts."""

    def __init__(
        self,
        *,
        run_paths: RunArtifactPaths,
        trajectory: RuntimeTrajectory,
        sector_count: int,
        record_video: bool = False,
        observation_mode: str = "full",
        record_observation_sidecar: bool = False,
    ) -> None:
        self._run_paths = run_paths
        self._trajectory = trajectory
        self._sector_count = sector_count
        self._record_video = record_video
        self._observation_mode = observation_mode
        self._record_observation_sidecar = record_observation_sidecar
        self._episode_index_rows: list[dict[str, Any]] = []
        self._current_episode_id: str | None = None
        self._current_episode_paths: EpisodeArtifactPaths | None = None
        self._current_metadata: dict[str, Any] | None = None
        self._current_rows: list[dict[str, Any]] = []
        self._current_step_index = 0
        self._current_observations: list[np.ndarray] = []
        self._current_telemetry: list[np.ndarray] = []
        self._current_actions: list[np.ndarray] = []
        self._current_action_abs_sum = np.zeros(3, dtype=np.float64)
        self._current_nonzero_action_steps = 0
        self._run_action_abs_sum = np.zeros(3, dtype=np.float64)
        self._run_action_step_count = 0
        self._run_nonzero_action_steps = 0

    @property
    def episode_index_rows(self) -> list[dict[str, Any]]:
        return list(self._episode_index_rows)

    def start_episode(
        self,
        *,
        episode_id: str,
        map_uid: str,
        run_id: str,
        episode_seed: int,
    ) -> EpisodeArtifactPaths:
        self._current_episode_id = episode_id
        self._current_episode_paths = build_episode_artifact_paths(
            self._run_paths.episodes_dir,
            episode_id=episode_id,
            record_video=self._record_video,
            record_observation_sidecar=self._record_observation_sidecar,
        )
        self._current_metadata = {
            "episode_id": episode_id,
            "map_uid": map_uid,
            "run_id": run_id,
            "episode_seed": episode_seed,
            "start_timestamp": _utc_now(),
        }
        self._current_rows = []
        self._current_step_index = 0
        self._current_observations = []
        self._current_telemetry = []
        self._current_actions = []
        self._current_action_abs_sum = np.zeros(3, dtype=np.float64)
        self._current_nonzero_action_steps = 0
        return self._current_episode_paths

    def record_step(
        self,
        *,
        observation: np.ndarray,
        action: Sequence[float],
        reward: float,
        info: Mapping[str, Any],
        policy_observation: np.ndarray | None = None,
        policy_telemetry: np.ndarray | None = None,
    ) -> None:
        if self._current_episode_id is None or self._current_episode_paths is None or self._current_metadata is None:
            raise RuntimeError("start_episode() must be called before record_step().")

        latest_frame = (
            np.asarray(observation[-1], dtype=np.uint8)
            if isinstance(observation, np.ndarray) and observation.ndim >= 3
            else None
        )
        progress_index = int(info["progress_index"])
        sector_index = self._trajectory.sector_index_for_progress(progress_index, self._sector_count)
        row = {
            "episode_id": self._current_episode_id,
            "step_index": self._current_step_index,
            "timestamp_ns": int(info["timestamp_ns"]),
            "race_time_ms": int(info["race_time_ms"]),
            "action_gas": float(action[0]),
            "action_brake": float(action[1]),
            "action_steer": float(action[2]),
            "reward": float(reward),
            "progress_index": progress_index,
            "progress_delta": int(info["progress_delta"]),
            "trajectory_arc_length_m": float(info["trajectory_arc_length_m"]),
            "sector_index": sector_index,
            "speed_kmh": float(info.get("speed_kmh", 0.0)),
            "gear": int(info.get("gear", 0)),
            "rpm": float(info.get("rpm", 0.0)),
            "pos_x": None if info.get("pos_xyz") is None else float(info["pos_xyz"][0]),
            "pos_y": None if info.get("pos_xyz") is None else float(info["pos_xyz"][1]),
            "pos_z": None if info.get("pos_xyz") is None else float(info["pos_xyz"][2]),
            "vel_x": None if info.get("vel_xyz") is None else float(info["vel_xyz"][0]),
            "vel_y": None if info.get("vel_xyz") is None else float(info["vel_xyz"][1]),
            "vel_z": None if info.get("vel_xyz") is None else float(info["vel_xyz"][2]),
            "yaw": None if info.get("yaw_pitch_roll") is None else float(info["yaw_pitch_roll"][0]),
            "pitch": None if info.get("yaw_pitch_roll") is None else float(info["yaw_pitch_roll"][1]),
            "roll": None if info.get("yaw_pitch_roll") is None else float(info["yaw_pitch_roll"][2]),
            "terminal_reason": info.get("terminal_reason"),
            "done_type": info.get("tm20ai_done_type"),
            "stray_distance": info.get("stray_distance"),
        }
        self._current_rows.append(row)
        action_array = np.asarray(action, dtype=np.float32)
        abs_action = np.abs(action_array)
        self._current_action_abs_sum += abs_action
        self._run_action_abs_sum += abs_action
        self._run_action_step_count += 1
        if bool(np.any(abs_action > 1.0e-6)):
            self._current_nonzero_action_steps += 1
            self._run_nonzero_action_steps += 1

        if self._current_episode_paths.frames_dir is not None and latest_frame is not None:
            frame_path = self._current_episode_paths.frames_dir / f"frame_{self._current_step_index:06d}.png"
            cv2.imwrite(str(frame_path), latest_frame)

        if self._current_episode_paths.observation_npz is not None:
            if policy_observation is None:
                raise RuntimeError("policy_observation is required when observation sidecars are enabled.")
            self._current_observations.append(np.asarray(policy_observation))
            if policy_telemetry is None:
                raise RuntimeError("policy_telemetry is required when observation sidecars are enabled.")
            self._current_telemetry.append(np.asarray(policy_telemetry, dtype=np.float32))
            self._current_actions.append(np.asarray(action, dtype=np.float32))

        self._current_step_index += 1

    def finish_episode(
        self,
        *,
        terminated: bool,
        truncated: bool,
        final_info: Mapping[str, Any],
    ) -> EpisodeWriteResult:
        if self._current_episode_id is None or self._current_episode_paths is None or self._current_metadata is None:
            raise RuntimeError("start_episode() must be called before finish_episode().")
        if not self._current_rows:
            raise RuntimeError("Cannot finish an episode without recorded steps.")

        termination_reason = final_info.get("reward_reason") or final_info.get("terminal_reason")
        completion_flag = bool(terminated and termination_reason == "finished")
        metadata = {
            **self._current_metadata,
            "end_timestamp": _utc_now(),
            "step_count": len(self._current_rows),
            "completion_flag": completion_flag,
            "done_type": "terminated" if terminated else "truncated" if truncated else None,
            "termination_reason": termination_reason,
            "final_progress_index": int(self._current_rows[-1]["progress_index"]),
            "completion_time_ms": int(final_info["race_time_ms"]) if completion_flag else None,
            "sampled_frames_dir": str(self._current_episode_paths.frames_dir) if self._current_episode_paths.frames_dir else None,
            "video_path": None,
            "observation_mode": self._observation_mode,
            "observation_sidecar_path": (
                str(self._current_episode_paths.observation_npz) if self._current_episode_paths.observation_npz else None
            ),
            "mean_abs_action": (
                self._current_action_abs_sum / max(1, len(self._current_rows))
            ).astype(float).tolist(),
            "nonzero_action_steps": int(self._current_nonzero_action_steps),
            "nonzero_action_fraction": float(self._current_nonzero_action_steps / max(1, len(self._current_rows))),
            "has_nonzero_actions": bool(self._current_nonzero_action_steps > 0),
        }
        summary = summarize_episode_trace(
            episode_id=self._current_episode_id,
            metadata=metadata,
            step_rows=self._current_rows,
            trajectory=self._trajectory,
            sector_count=self._sector_count,
        )
        summary.update(
            {
                "mean_abs_action": list(metadata["mean_abs_action"]),
                "nonzero_action_steps": metadata["nonzero_action_steps"],
                "nonzero_action_fraction": metadata["nonzero_action_fraction"],
                "has_nonzero_actions": metadata["has_nonzero_actions"],
            }
        )
        metadata["furthest_sector_reached"] = summary["furthest_sector_reached"]
        metadata["best_progress_index"] = summary["best_progress_index"]

        write_parquet_rows(self._current_episode_paths.steps_parquet, self._current_rows)
        if self._current_episode_paths.observation_npz is not None:
            np.savez_compressed(
                self._current_episode_paths.observation_npz,
                obs_uint8=np.stack(self._current_observations).astype(np.uint8, copy=False),
                telemetry_float=np.stack(self._current_telemetry).astype(np.float32, copy=False),
                action=np.stack(self._current_actions).astype(np.float32, copy=False),
            )
        write_json(self._current_episode_paths.metadata_json, metadata)
        self._episode_index_rows.append(summary)

        result = EpisodeWriteResult(summary=summary, paths=self._current_episode_paths)
        self._current_episode_id = None
        self._current_episode_paths = None
        self._current_metadata = None
        self._current_rows = []
        self._current_step_index = 0
        self._current_observations = []
        self._current_telemetry = []
        self._current_actions = []
        self._current_action_abs_sum = np.zeros(3, dtype=np.float64)
        self._current_nonzero_action_steps = 0
        return result

    def write_episode_index(self) -> Path:
        write_parquet_rows(self._run_paths.episode_index_parquet, self._episode_index_rows)
        return self._run_paths.episode_index_parquet

    def run_action_metrics(self) -> dict[str, Any]:
        mean_abs_action = (
            self._run_action_abs_sum / max(1, self._run_action_step_count)
        ).astype(float).tolist()
        return {
            "total_action_steps": int(self._run_action_step_count),
            "total_nonzero_action_steps": int(self._run_nonzero_action_steps),
            "mean_abs_action": mean_abs_action,
            "nonzero_action_fraction": float(self._run_nonzero_action_steps / max(1, self._run_action_step_count)),
        }
