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
    ) -> None:
        self._run_paths = run_paths
        self._trajectory = trajectory
        self._sector_count = sector_count
        self._record_video = record_video
        self._episode_index_rows: list[dict[str, Any]] = []
        self._current_episode_id: str | None = None
        self._current_episode_paths: EpisodeArtifactPaths | None = None
        self._current_metadata: dict[str, Any] | None = None
        self._current_rows: list[dict[str, Any]] = []
        self._current_step_index = 0

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
        return self._current_episode_paths

    def record_step(
        self,
        *,
        observation: np.ndarray,
        action: Sequence[float],
        reward: float,
        info: Mapping[str, Any],
    ) -> None:
        if self._current_episode_id is None or self._current_episode_paths is None or self._current_metadata is None:
            raise RuntimeError("start_episode() must be called before record_step().")

        latest_frame = np.asarray(observation[-1], dtype=np.uint8)
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

        if self._current_episode_paths.frames_dir is not None:
            frame_path = self._current_episode_paths.frames_dir / f"frame_{self._current_step_index:06d}.png"
            cv2.imwrite(str(frame_path), latest_frame)

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
        }
        summary = summarize_episode_trace(
            episode_id=self._current_episode_id,
            metadata=metadata,
            step_rows=self._current_rows,
            trajectory=self._trajectory,
            sector_count=self._sector_count,
        )
        metadata["furthest_sector_reached"] = summary["furthest_sector_reached"]
        metadata["best_progress_index"] = summary["best_progress_index"]

        write_parquet_rows(self._current_episode_paths.steps_parquet, self._current_rows)
        write_json(self._current_episode_paths.metadata_json, metadata)
        self._episode_index_rows.append(summary)

        result = EpisodeWriteResult(summary=summary, paths=self._current_episode_paths)
        self._current_episode_id = None
        self._current_episode_paths = None
        self._current_metadata = None
        self._current_rows = []
        self._current_step_index = 0
        return result

    def write_episode_index(self) -> Path:
        write_parquet_rows(self._run_paths.episode_index_parquet, self._episode_index_rows)
        return self._run_paths.episode_index_parquet
