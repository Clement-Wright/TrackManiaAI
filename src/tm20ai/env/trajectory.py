from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def reward_root() -> Path:
    return _repo_root() / "data" / "reward"


def _spacing_tag(spacing_meters: float) -> str:
    raw = f"{spacing_meters:g}".replace(".", "p")
    return f"{raw}m"


def _map_dir(map_uid: str) -> Path:
    if not map_uid or re.search(r"[<>:\"/\\\\|?*]", map_uid):
        raise ValueError(f"map_uid {map_uid!r} is not usable as a reward artifact folder name.")
    return reward_root() / map_uid


def raw_lap_path_for_map(map_uid: str) -> Path:
    return _map_dir(map_uid) / "raw_lap.parquet"


def runtime_trajectory_path_for_map(map_uid: str, spacing_meters: float = 0.5) -> Path:
    return _map_dir(map_uid) / f"trajectory_{_spacing_tag(spacing_meters)}.npz"


@dataclass(slots=True)
class RuntimeTrajectory:
    map_uid: str
    points: np.ndarray
    tangents: np.ndarray
    arc_length: np.ndarray
    race_time_ms: np.ndarray

    @property
    def total_length(self) -> float:
        return float(self.arc_length[-1]) if len(self.arc_length) else 0.0

    def nearest_index(
        self,
        position: Sequence[float],
        *,
        reference_index: int | None = None,
        check_backward: int = 10,
        check_forward: int = 500,
    ) -> tuple[int, float]:
        pos = np.asarray(position, dtype=np.float32)
        if pos.shape != (3,):
            raise ValueError("Reward position must be a 3D vector.")
        if reference_index is None:
            start = 0
            end = len(self.points)
        else:
            start = max(0, int(reference_index) - int(check_backward))
            end = min(len(self.points), int(reference_index) + int(check_forward) + 1)
        window = self.points[start:end]
        if window.size == 0:
            raise RuntimeError("Trajectory search window is empty.")
        distances = np.linalg.norm(window - pos, axis=1)
        local_index = int(np.argmin(distances))
        return start + local_index, float(distances[local_index])

    def sector_edges(self, sector_count: int) -> np.ndarray:
        if sector_count <= 0:
            raise ValueError("sector_count must be positive.")
        if self.total_length <= 0.0:
            return np.zeros(sector_count + 1, dtype=np.float32)
        return np.linspace(0.0, self.total_length, sector_count + 1, dtype=np.float32)

    def sector_index_for_arc_length(self, arc_length_value: float, sector_count: int) -> int:
        if sector_count <= 0:
            raise ValueError("sector_count must be positive.")
        edges = self.sector_edges(sector_count)
        if arc_length_value <= 0.0:
            return 0
        if arc_length_value >= float(edges[-1]):
            return sector_count - 1
        return max(0, min(sector_count - 1, int(np.searchsorted(edges, arc_length_value, side="right") - 1)))

    def sector_index_for_progress(self, progress_index: int, sector_count: int) -> int:
        bounded_index = max(0, min(len(self.arc_length) - 1, int(progress_index)))
        return self.sector_index_for_arc_length(float(self.arc_length[bounded_index]), sector_count)


def save_raw_lap_records(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        raise ValueError("Cannot save an empty reward recording.")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, path)


def load_raw_lap(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    if table.num_rows < 2:
        raise ValueError(f"Reward recording at {path} must contain at least two rows.")
    payload = table.to_pydict()
    map_values = payload.get("map_uid")
    if not map_values:
        raise ValueError(f"Reward recording at {path} is missing map_uid.")
    map_uid = str(map_values[0])
    positions = np.column_stack(
        (
            np.asarray(payload["x"], dtype=np.float32),
            np.asarray(payload["y"], dtype=np.float32),
            np.asarray(payload["z"], dtype=np.float32),
        )
    )
    race_time_ms = np.asarray(payload["race_time_ms"], dtype=np.float32)
    return map_uid, positions, race_time_ms


def _deduplicate_points(points: np.ndarray, race_time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep_indices = [0]
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[keep_indices[-1]])) > 1e-6:
            keep_indices.append(index)
    dedup_points = points[keep_indices]
    dedup_times = race_time_ms[keep_indices]
    if len(dedup_points) < 2:
        raise ValueError("Reward trajectory collapsed to fewer than two unique points.")
    return dedup_points, dedup_times


def _resample_positions(points: np.ndarray, race_time_ms: np.ndarray, spacing_meters: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate((np.array([0.0], dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)))
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError("Reward trajectory length must be positive.")
    sample_arc = np.arange(0.0, total_length, spacing_meters, dtype=np.float32)
    if sample_arc.size == 0 or not np.isclose(sample_arc[-1], total_length):
        sample_arc = np.concatenate((sample_arc, np.asarray([total_length], dtype=np.float32)))
    x = np.interp(sample_arc, cumulative, points[:, 0]).astype(np.float32)
    y = np.interp(sample_arc, cumulative, points[:, 1]).astype(np.float32)
    z = np.interp(sample_arc, cumulative, points[:, 2]).astype(np.float32)
    t = np.interp(sample_arc, cumulative, race_time_ms).astype(np.float32)
    resampled = np.column_stack((x, y, z)).astype(np.float32)
    return resampled, sample_arc, t


def _compute_tangents(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    zero_norms = norms.squeeze(-1) <= 1e-6
    tangents[zero_norms] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return (tangents / norms).astype(np.float32)


def build_runtime_trajectory(map_uid: str, positions: np.ndarray, race_time_ms: np.ndarray, spacing_meters: float) -> RuntimeTrajectory:
    dedup_points, dedup_times = _deduplicate_points(
        np.asarray(positions, dtype=np.float32),
        np.asarray(race_time_ms, dtype=np.float32),
    )
    resampled_points, arc_length, resampled_times = _resample_positions(dedup_points, dedup_times, spacing_meters)
    tangents = _compute_tangents(resampled_points)
    return RuntimeTrajectory(
        map_uid=map_uid,
        points=resampled_points.astype(np.float32, copy=False),
        tangents=tangents.astype(np.float32, copy=False),
        arc_length=arc_length.astype(np.float32, copy=False),
        race_time_ms=resampled_times.astype(np.float32, copy=False),
    )


def save_runtime_trajectory(trajectory: RuntimeTrajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        map_uid=np.asarray([trajectory.map_uid]),
        points=trajectory.points,
        tangents=trajectory.tangents,
        arc_length=trajectory.arc_length,
        race_time_ms=trajectory.race_time_ms,
    )


def load_runtime_trajectory(path: Path) -> RuntimeTrajectory:
    data = np.load(path, allow_pickle=False)
    map_uid = str(data["map_uid"][0])
    return RuntimeTrajectory(
        map_uid=map_uid,
        points=np.asarray(data["points"], dtype=np.float32),
        tangents=np.asarray(data["tangents"], dtype=np.float32),
        arc_length=np.asarray(data["arc_length"], dtype=np.float32),
        race_time_ms=np.asarray(data["race_time_ms"], dtype=np.float32),
    )


def build_runtime_trajectory_from_raw_lap(raw_lap_path: Path, spacing_meters: float) -> RuntimeTrajectory:
    map_uid, positions, race_time_ms = load_raw_lap(raw_lap_path)
    return build_runtime_trajectory(map_uid, positions, race_time_ms, spacing_meters)
