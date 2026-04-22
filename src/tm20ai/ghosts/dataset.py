from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from ..action_space import ACTION_DIM, LEGACY_ACTION_DIM, clamp_action
from ..data.parquet_writer import read_json, sha256_file, write_json, write_parquet_rows


TRAJECTORY_SCHEMA_VERSION = "ghost_trajectory_v1"
GHOST_BUNDLE_SCHEMA_VERSION = "ghost_bundle_v1"


@dataclass(frozen=True, slots=True)
class GhostBundleBuildResult:
    manifest_path: Path
    selected_count: int
    trajectory_count: int
    action_channel_valid: bool
    offline_transition_count: int
    bundle_resolution_mode: str | None = None
    strategy_manifest_path: Path | None = None
    intended_manifest_path: Path | None = None
    exploit_manifest_path: Path | None = None
    selected_override_manifest_path: Path | None = None
    author_fallback_manifest_path: Path | None = None
    mixed_fallback_manifest_path: Path | None = None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise RuntimeError(f"{path} must contain a list or a mapping with a rows list.")
        return [dict(row) for row in rows]
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".parquet":
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    raise RuntimeError(
        f"Unsupported Openplanet export format for {path}. "
        "Use .json, .jsonl, or .parquet exports produced by the Openplanet replay extractor."
    )


def _vector(value: Any, *, length: int, default: Sequence[float] | None = None) -> tuple[float, ...]:
    if value is None:
        if default is None:
            return tuple(0.0 for _ in range(length))
        return tuple(float(item) for item in default)
    if isinstance(value, Mapping):
        candidates = [value.get(key) for key in ("x", "y", "z", "w")][:length]
        return tuple(float(0.0 if item is None else item) for item in candidates)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        padded = [float(part) for part in parts[:length]]
        while len(padded) < length:
            padded.append(0.0)
        return tuple(padded)
    values = list(value) if isinstance(value, Iterable) else [value]
    padded = [float(item) for item in values[:length]]
    while len(padded) < length:
        padded.append(0.0)
    return tuple(padded)


def _position(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return _vector(
        row.get("position", row.get("pos_xyz", row.get("pos", row.get("Position")))),
        length=3,
    )


def _velocity(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return _vector(
        row.get("velocity", row.get("vel_xyz", row.get("world_velocity", row.get("WorldVel")))),
        length=3,
    )


def _forward(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return _vector(
        row.get("forward", row.get("dir", row.get("orientation_forward", row.get("Dir")))),
        length=3,
        default=(0.0, 0.0, 1.0),
    )


def _action(row: Mapping[str, Any]) -> tuple[float, float] | None:
    action_value = row.get("action")
    if action_value is not None:
        values = list(action_value) if not isinstance(action_value, str) else [
            float(part.strip()) for part in action_value.split(",") if part.strip()
        ]
        if len(values) in {ACTION_DIM, LEGACY_ACTION_DIM}:
            return tuple(float(item) for item in clamp_action(np.asarray(values, dtype=np.float32)))
    throttle = row.get("throttle", row.get("gas", row.get("InputGasPedal")))
    brake = row.get("brake", row.get("InputBrakePedal"))
    steer = row.get("steer", row.get("InputSteer"))
    if throttle is None and brake is None and steer is None:
        return None
    gas_value = float(0.0 if throttle is None else throttle)
    brake_value = float(0.0 if brake is None else brake)
    steer_value = float(0.0 if steer is None else steer)
    return tuple(float(item) for item in clamp_action(np.asarray([gas_value - brake_value, steer_value], dtype=np.float32)))


def _normalize_rows(rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    normalized: list[dict[str, Any]] = []
    action_count = 0
    cumulative_arc = 0.0
    previous_position: np.ndarray | None = None
    for index, row in enumerate(rows):
        position = np.asarray(_position(row), dtype=np.float64)
        if previous_position is not None:
            cumulative_arc += float(np.linalg.norm(position - previous_position))
        previous_position = position
        action = _action(row)
        if action is not None:
            action_count += 1
        velocity = _velocity(row)
        speed = row.get("speed", row.get("speed_kmh", row.get("FrontSpeed")))
        if speed is None:
            speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)) * 3.6)
        normalized.append(
            {
                "step_index": int(row.get("step_index", row.get("index", index))),
                "timestamp": row.get("timestamp"),
                "race_time_ms": int(row.get("race_time_ms", row.get("time_ms", row.get("RaceTime", index * 50)))),
                "position_x": float(position[0]),
                "position_y": float(position[1]),
                "position_z": float(position[2]),
                "velocity_x": float(velocity[0]),
                "velocity_y": float(velocity[1]),
                "velocity_z": float(velocity[2]),
                "forward_x": float(_forward(row)[0]),
                "forward_y": float(_forward(row)[1]),
                "forward_z": float(_forward(row)[2]),
                "speed_kmh": float(speed),
                "progress_index": int(row.get("progress_index", index)),
                "arc_length": float(row.get("arc_length", row.get("arc_length_m", cumulative_arc))),
                "sector": int(row.get("sector", 0)),
                "throttle": None if action is None else float(action[0]),
                "brake": float(row.get("brake", row.get("InputBrakePedal", 0.0)) or 0.0),
                "steer": None if action is None else float(action[1]),
                "gear": None if row.get("gear", row.get("CurGear")) is None else int(row.get("gear", row.get("CurGear"))),
                "ground_material": row.get("ground_material", row.get("GroundContactMaterial")),
            }
        )
    action_channel_valid = bool(normalized) and action_count == len(normalized)
    return normalized, action_channel_valid


def extract_openplanet_export(
    export_path: str | Path,
    *,
    output_dir: str | Path,
    replay_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Normalize an Openplanet replay/ghost export into repo trajectory artifacts."""

    source_path = Path(export_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = dict(replay_metadata or {})
    record_filename = metadata.get("record_filename")
    if record_filename in (None, "") and isinstance(metadata.get("record_metadata"), Mapping):
        record_filename = metadata.get("record_metadata", {}).get("filename")
    downloaded_replay_path = metadata.get("downloaded_replay_path")
    if downloaded_replay_path in (None, "") and isinstance(metadata.get("fetch_status"), Mapping):
        downloaded_replay_path = metadata.get("fetch_status", {}).get("path")
    record_filename_basename = None if record_filename in (None, "") else Path(str(record_filename)).name
    downloaded_replay_basename = (
        None if downloaded_replay_path in (None, "") else Path(str(downloaded_replay_path)).name
    )
    rows, action_channel_valid = _normalize_rows(_load_rows(source_path))
    trajectory_id = str(metadata.get("trajectory_id") or source_path.stem)
    parquet_path = output_root / f"{trajectory_id}.parquet"
    metadata_path = output_root / f"{trajectory_id}.json"
    write_parquet_rows(parquet_path, rows)
    offline_transition_npz_path = None
    offline_transition_count = 0
    offline_dataset_hash = None
    observation_npz_value = metadata.get("observation_npz_path")
    if action_channel_valid and observation_npz_value not in (None, ""):
        observation_npz_path = Path(str(observation_npz_value)).resolve()
        observation_payload = np.load(observation_npz_path)
        observations = np.asarray(observation_payload["obs_uint8"], dtype=np.uint8)
        telemetry = np.asarray(observation_payload["telemetry_float"], dtype=np.float32)
        count = min(len(rows), int(observations.shape[0]), int(telemetry.shape[0]))
        if count >= 2:
            actions = np.asarray([[row["throttle"], row["steer"]] for row in rows[:count]], dtype=np.float32)
            rewards = np.asarray(
                [
                    float(rows[min(index + 1, count - 1)]["progress_index"]) - float(rows[index]["progress_index"])
                    for index in range(count)
                ],
                dtype=np.float32,
            )
            offline_transition_npz_path = str(output_root / f"{trajectory_id}_offline_transitions.npz")
            np.savez_compressed(
                offline_transition_npz_path,
                obs_uint8=observations[:count],
                next_obs_uint8=observations[np.minimum(np.arange(count) + 1, count - 1)],
                telemetry_float=telemetry[:count],
                next_telemetry_float=telemetry[np.minimum(np.arange(count) + 1, count - 1)],
                action=actions,
                reward=rewards,
                terminated=np.zeros((count,), dtype=np.bool_),
                truncated=np.zeros((count,), dtype=np.bool_),
                step_idx=np.arange(count, dtype=np.int32),
                episode_id=np.asarray([trajectory_id for _ in range(count)], dtype=object),
                map_uid=np.asarray([metadata.get("map_uid") for _ in range(count)], dtype=object),
            )
            offline_transition_count = count
            offline_dataset_hash = sha256_file(Path(offline_transition_npz_path))
    write_json(
        metadata_path,
        {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "trajectory_id": trajectory_id,
            "source_export_path": str(source_path),
            "source_export_sha256": sha256_file(source_path),
            "trajectory_parquet_path": str(parquet_path),
            "trajectory_parquet_sha256": sha256_file(parquet_path),
            "row_count": len(rows),
            "action_channel_valid": action_channel_valid,
            "offline_transition_npz_path": offline_transition_npz_path,
            "offline_transition_count": offline_transition_count,
            "offline_dataset_hash": offline_dataset_hash,
            "map_uid": metadata.get("map_uid"),
            "account_id": metadata.get("account_id"),
            "rank": metadata.get("rank"),
            "record_time_ms": metadata.get("record_time_ms"),
            "replay_source": metadata.get("replay_source", "openplanet_export"),
            **metadata,
            "record_filename": None if record_filename in (None, "") else str(record_filename),
            "record_filename_basename": record_filename_basename,
            "downloaded_replay_path": None if downloaded_replay_path in (None, "") else str(downloaded_replay_path),
            "downloaded_replay_basename": downloaded_replay_basename,
            "source_export_basename": source_path.name,
            "source_export_stem": source_path.stem,
        },
    )
    return metadata_path


def load_ghost_bundle_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = read_json(manifest_path)
    if payload.get("schema_version") != GHOST_BUNDLE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{manifest_path} is not a {GHOST_BUNDLE_SCHEMA_VERSION} manifest "
            f"(schema_version={payload.get('schema_version')!r})."
        )
    return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _reward_trajectory_path_for_map(map_uid: str, spacing_meters: float) -> Path:
    spacing_tag = f"{spacing_meters:g}".replace(".", "p")
    return _repo_root() / "data" / "reward" / map_uid / f"trajectory_{spacing_tag}m.npz"


def _deduplicate_positions(points: np.ndarray, race_time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep_indices = [0]
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[keep_indices[-1]])) > 1.0e-6:
            keep_indices.append(index)
    deduped = points[keep_indices]
    deduped_times = race_time_ms[keep_indices]
    if len(deduped) < 2:
        raise ValueError("Trajectory collapsed to fewer than two unique points.")
    return deduped, deduped_times


def _resample_positions_fixed_spacing(
    points: np.ndarray,
    race_time_ms: np.ndarray,
    spacing_meters: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deduped_points, deduped_times = _deduplicate_positions(
        np.asarray(points, dtype=np.float64),
        np.asarray(race_time_ms, dtype=np.float64),
    )
    segment_lengths = np.linalg.norm(np.diff(deduped_points, axis=0), axis=1)
    cumulative = np.concatenate((np.asarray([0.0], dtype=np.float64), np.cumsum(segment_lengths, dtype=np.float64)))
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError("Trajectory length must be positive.")
    spacing = max(1.0e-6, float(spacing_meters))
    sample_arc = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if sample_arc.size == 0 or not np.isclose(sample_arc[-1], total_length):
        sample_arc = np.concatenate((sample_arc, np.asarray([total_length], dtype=np.float64)))
    x = np.interp(sample_arc, cumulative, deduped_points[:, 0])
    y = np.interp(sample_arc, cumulative, deduped_points[:, 1])
    z = np.interp(sample_arc, cumulative, deduped_points[:, 2])
    t = np.interp(sample_arc, cumulative, deduped_times)
    return (
        np.column_stack((x, y, z)).astype(np.float64, copy=False),
        sample_arc.astype(np.float64, copy=False),
        t.astype(np.float64, copy=False),
    )


def _load_runtime_reference(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=False)
    points = np.asarray(payload["points"], dtype=np.float64)
    arc_length = np.asarray(payload["arc_length"], dtype=np.float64)
    race_time_ms = np.asarray(payload["race_time_ms"], dtype=np.float64)
    return {
        "source": "reward_trajectory_fallback",
        "path": str(path.resolve()),
        "points": points,
        "arc_length": arc_length,
        "race_time_ms": race_time_ms,
        "total_length_m": float(arc_length[-1]) if arc_length.size else 0.0,
    }


def _rank_band(rank: int, bands: Sequence[str]) -> str:
    for band in bands:
        left, _, right = str(band).partition("-")
        if not right:
            continue
        if int(left) <= rank <= int(right):
            return str(band)
    return "unbanded"


def _load_trajectory_metadata(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise RuntimeError(f"{path} is not a normalized ghost trajectory metadata file.")
    return payload


def _load_trajectory_arrays(metadata: Mapping[str, Any]) -> dict[str, Any]:
    rows = pq.read_table(Path(str(metadata["trajectory_parquet_path"]))).to_pylist()
    positions = np.asarray(
        [
            [
                float(row.get("position_x") or 0.0),
                float(row.get("position_y") or 0.0),
                float(row.get("position_z") or 0.0),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    race_time_ms = np.asarray(
        [float(row.get("race_time_ms") or index * 50) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    speed_kmh = np.asarray([float(row.get("speed_kmh") or 0.0) for row in rows], dtype=np.float64)
    steer = np.asarray(
        [
            0.0 if row.get("steer") is None else float(row.get("steer") or 0.0)
            for row in rows
        ],
        dtype=np.float64,
    )
    throttle = np.asarray(
        [
            0.0 if row.get("throttle") is None else float(row.get("throttle") or 0.0)
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "rows": rows,
        "positions": positions,
        "race_time_ms": race_time_ms,
        "speed_kmh": speed_kmh,
        "steer": steer,
        "throttle": throttle,
    }


def _string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _name_candidates(metadata: Mapping[str, Any]) -> list[str]:
    values = [
        metadata.get("trajectory_id"),
        metadata.get("source_export_basename"),
        metadata.get("source_export_stem"),
        metadata.get("record_filename_basename"),
        metadata.get("downloaded_replay_basename"),
    ]
    names: list[str] = []
    for value in values:
        string_value = _string_or_none(value)
        if string_value is None:
            continue
        names.append(string_value)
        names.append(Path(string_value).stem)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _trajectory_features(metadata: Mapping[str, Any], arrays: Mapping[str, Any] | None = None) -> dict[str, float]:
    rows = list((arrays or {}).get("rows", []))
    if not rows:
        rows = pq.read_table(Path(str(metadata["trajectory_parquet_path"]))).to_pylist()
    if not rows:
        return {"mean_speed_kmh": 0.0, "finish_arc_length": 0.0, "path_x_mean": 0.0, "path_z_mean": 0.0}
    speeds = np.asarray([float(row.get("speed_kmh") or 0.0) for row in rows], dtype=np.float64)
    xs = np.asarray([float(row.get("position_x") or 0.0) for row in rows], dtype=np.float64)
    zs = np.asarray([float(row.get("position_z") or 0.0) for row in rows], dtype=np.float64)
    return {
        "mean_speed_kmh": float(np.mean(speeds)),
        "finish_arc_length": float(rows[-1].get("arc_length") or 0.0),
        "path_x_mean": float(np.mean(xs)),
        "path_z_mean": float(np.mean(zs)),
    }


def _cluster_key(metadata: Mapping[str, Any], features: Mapping[str, float]) -> str:
    rank = int(metadata.get("rank") or 999)
    speed_bucket = int(float(features["mean_speed_kmh"]) // 25.0)
    x_bucket = int(float(features["path_x_mean"]) // 10.0)
    z_bucket = int(float(features["path_z_mean"]) // 10.0)
    return f"rank{rank // 10:02d}_speed{speed_bucket:02d}_x{x_bucket:03d}_z{z_bucket:03d}"


def _bundle_candidate(
    metadata: Mapping[str, Any],
    *,
    metadata_path: Path,
    canonical_reference: Mapping[str, Any] | None,
    anchor_payload: Mapping[str, Any] | None,
    ghost_config,  # noqa: ANN001
    bands: Sequence[str],
) -> dict[str, Any]:
    rank = int(metadata.get("rank") or 999)
    arrays = _load_trajectory_arrays(metadata)
    features = _trajectory_features(metadata, arrays)
    route_features: dict[str, Any] = {}
    strategy_family = "unclassified"
    if canonical_reference is not None and anchor_payload is not None and ghost_config is not None:
        anchor_features = _anchor_route_features(
            anchor_points=np.asarray(anchor_payload["points"], dtype=np.float64),
            anchor_arc_length=np.asarray(anchor_payload["arc_length"], dtype=np.float64),
            positions=np.asarray(arrays["positions"], dtype=np.float64),
            speed_kmh=np.asarray(arrays["speed_kmh"], dtype=np.float64),
            steer=np.asarray(arrays["steer"], dtype=np.float64),
            anchor_radius_m=float(ghost_config.anchor_radius_m),
            canonical_divergence_radius_m=float(ghost_config.canonical_divergence_radius_m),
        )
        canonical_features = _canonical_projection_features(
            reference_points=np.asarray(canonical_reference["points"], dtype=np.float64),
            reference_arc_length=np.asarray(canonical_reference["arc_length"], dtype=np.float64),
            positions=np.asarray(arrays["positions"], dtype=np.float64),
            race_time_ms=np.asarray(arrays["race_time_ms"], dtype=np.float64),
            early_reverse_window_ms=int(ghost_config.early_reverse_window_ms),
        )
        route_features = {
            **anchor_features,
            **canonical_features,
            "finished_run": metadata.get("record_time_ms") is not None,
            "record_time_ms": metadata.get("record_time_ms"),
        }
        strategy_family = _strategy_family(route_features, ghost_config)
    return {
        **metadata,
        "rank": rank,
        "band": _rank_band(rank, bands),
        "features": features,
        "route_features": route_features,
        "strategy_family": strategy_family,
        "cluster_key": _cluster_key(metadata, features),
        "selection_signature": (
            []
            if not route_features
            else _selection_signature(route_features).astype(float, copy=False).tolist()
        ),
        "selection_names": _name_candidates(metadata),
        "metadata_path": str(metadata_path.resolve()),
    }


def _resolve_reference_metadata_path(path: Path) -> Path | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".npz":
        return path
    payload = read_json(path)
    schema_version = payload.get("schema_version")
    if schema_version == TRAJECTORY_SCHEMA_VERSION:
        return path
    if schema_version == "ghost_extraction_manifest_v1":
        items = list(payload.get("trajectory_metadata_paths") or [])
        if not items:
            return None
        return Path(str(items[0])).resolve()
    return None


def _load_author_reference(path: Path, spacing_meters: float) -> dict[str, Any] | None:
    resolved = _resolve_reference_metadata_path(path.resolve())
    if resolved is None:
        return None
    if resolved.suffix.lower() == ".npz":
        reference = _load_runtime_reference(resolved)
        reference["source"] = "author_reference_manifest"
        return reference
    metadata = _load_trajectory_metadata(resolved)
    arrays = _load_trajectory_arrays(metadata)
    positions, arc_length, race_time_ms = _resample_positions_fixed_spacing(
        arrays["positions"],
        arrays["race_time_ms"],
        spacing_meters,
    )
    return {
        "source": "author_reference_manifest",
        "path": str(resolved),
        "points": positions,
        "arc_length": arc_length,
        "race_time_ms": race_time_ms,
        "total_length_m": float(arc_length[-1]) if arc_length.size else 0.0,
    }


def _load_author_reference_metadata(path: Path) -> tuple[dict[str, Any], Path] | None:
    resolved = _resolve_reference_metadata_path(path.resolve())
    if resolved is None or resolved.suffix.lower() == ".npz":
        return None
    return _load_trajectory_metadata(resolved), resolved


def _canonical_reference(
    *,
    map_uid: str,
    spacing_meters: float,
    author_reference_manifest: str | None,
) -> tuple[dict[str, Any] | None, str]:
    if author_reference_manifest is not None:
        author_reference = _load_author_reference(Path(author_reference_manifest), spacing_meters)
        if author_reference is not None:
            return author_reference, "classified"
    reward_path = _reward_trajectory_path_for_map(map_uid, spacing_meters)
    if reward_path.exists():
        return _load_runtime_reference(reward_path), "classified_with_reward_fallback"
    return None, "unavailable_canonical_reference"


def _anchor_payload(reference: Mapping[str, Any], anchor_count: int) -> dict[str, np.ndarray]:
    points = np.asarray(reference["points"], dtype=np.float64)
    arc_length = np.asarray(reference["arc_length"], dtype=np.float64)
    anchor_indices = np.linspace(0, max(0, len(points) - 1), max(2, int(anchor_count)), dtype=np.int32)
    return {
        "indices": anchor_indices,
        "points": points[anchor_indices],
        "arc_length": arc_length[anchor_indices],
    }


def _anchor_route_features(
    *,
    anchor_points: np.ndarray,
    anchor_arc_length: np.ndarray,
    positions: np.ndarray,
    speed_kmh: np.ndarray,
    steer: np.ndarray,
    anchor_radius_m: float,
    canonical_divergence_radius_m: float,
) -> dict[str, Any]:
    if positions.size == 0:
        count = int(anchor_points.shape[0])
        return {
            "anchor_coverage_fraction": 0.0,
            "anchor_covered_mask": [False] * count,
            "anchor_distances_m": [float("inf")] * count,
            "anchor_nearest_speed_kmh": [0.0] * count,
            "anchor_nearest_steer": [0.0] * count,
            "first_major_divergence_arc_length_m": None,
        }
    nearest_indices: list[int] = []
    nearest_distances: list[float] = []
    nearest_speeds: list[float] = []
    nearest_steers: list[float] = []
    for anchor in anchor_points:
        distances = np.linalg.norm(positions - anchor[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_indices.append(nearest_index)
        nearest_distances.append(float(distances[nearest_index]))
        nearest_speeds.append(float(speed_kmh[nearest_index]))
        nearest_steers.append(float(steer[nearest_index]))
    covered_mask: list[bool] = []
    last_index = -1
    first_major_divergence_arc_length_m: float | None = None
    for anchor_idx, (nearest_index, distance) in enumerate(zip(nearest_indices, nearest_distances)):
        ordered = nearest_index >= last_index
        covered = distance <= anchor_radius_m and ordered
        covered_mask.append(covered)
        if covered:
            last_index = nearest_index
        elif first_major_divergence_arc_length_m is None and distance > canonical_divergence_radius_m:
            first_major_divergence_arc_length_m = float(anchor_arc_length[anchor_idx])
        elif first_major_divergence_arc_length_m is None and not ordered:
            first_major_divergence_arc_length_m = float(anchor_arc_length[anchor_idx])
    coverage_fraction = float(np.mean(np.asarray(covered_mask, dtype=np.float64))) if covered_mask else 0.0
    return {
        "anchor_coverage_fraction": coverage_fraction,
        "anchor_covered_mask": list(covered_mask),
        "anchor_distances_m": list(nearest_distances),
        "anchor_nearest_speed_kmh": list(nearest_speeds),
        "anchor_nearest_steer": list(nearest_steers),
        "first_major_divergence_arc_length_m": first_major_divergence_arc_length_m,
    }


def _sample_indices(length: int, max_samples: int = 200) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int32)
    if length <= max_samples:
        return np.arange(length, dtype=np.int32)
    return np.linspace(0, length - 1, max_samples, dtype=np.int32)


def _canonical_projection_features(
    *,
    reference_points: np.ndarray,
    reference_arc_length: np.ndarray,
    positions: np.ndarray,
    race_time_ms: np.ndarray,
    early_reverse_window_ms: int,
) -> dict[str, Any]:
    sample_indices = _sample_indices(len(positions))
    if sample_indices.size <= 0:
        return {
            "final_canonical_progress_fraction": 0.0,
            "max_reverse_canonical_progress_m": 0.0,
            "mean_distance_to_canonical_route_m": None,
            "max_distance_to_canonical_route_m": None,
        }
    sampled_positions = positions[sample_indices]
    sampled_times = race_time_ms[sample_indices]
    projected_arc: list[float] = []
    projected_distances: list[float] = []
    for position in sampled_positions:
        distances = np.linalg.norm(reference_points - position[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        projected_arc.append(float(reference_arc_length[nearest_index]))
        projected_distances.append(float(distances[nearest_index]))
    total_length = float(reference_arc_length[-1]) if reference_arc_length.size else 0.0
    furthest_arc = max(projected_arc, default=0.0)
    running_max = 0.0
    max_reverse = 0.0
    for arc, race_time in zip(projected_arc, sampled_times):
        if float(race_time) > float(early_reverse_window_ms):
            break
        running_max = max(running_max, float(arc))
        max_reverse = max(max_reverse, running_max - float(arc))
    return {
        "final_canonical_progress_fraction": (
            None if total_length <= 0.0 else float(np.clip(furthest_arc / total_length, 0.0, 1.0))
        ),
        "max_reverse_canonical_progress_m": float(max_reverse),
        "mean_distance_to_canonical_route_m": float(np.mean(projected_distances)),
        "max_distance_to_canonical_route_m": float(np.max(projected_distances)),
    }


def _strategy_family(route_features: Mapping[str, Any], ghost_config) -> str:  # noqa: ANN001
    anchor_fraction = float(route_features.get("anchor_coverage_fraction") or 0.0)
    final_progress_fraction = float(route_features.get("final_canonical_progress_fraction") or 0.0)
    reverse_progress = float(route_features.get("max_reverse_canonical_progress_m") or 0.0)
    if (
        anchor_fraction >= float(ghost_config.intended_anchor_fraction_min)
        and final_progress_fraction >= 0.85
        and reverse_progress <= float(ghost_config.exploit_reverse_progress_threshold_m)
    ):
        return "intended_route"
    if (
        anchor_fraction <= float(ghost_config.exploit_anchor_fraction_max)
        or final_progress_fraction < 0.70
        or reverse_progress > float(ghost_config.exploit_reverse_progress_threshold_m)
    ):
        return "shortcut_or_exploit"
    return "unclassified"


def _selection_signature(route_features: Mapping[str, Any]) -> np.ndarray:
    anchor_mask = np.asarray(route_features.get("anchor_covered_mask") or [], dtype=np.float64)
    anchor_speeds = np.asarray(route_features.get("anchor_nearest_speed_kmh") or [], dtype=np.float64) / 300.0
    anchor_steers = np.asarray(route_features.get("anchor_nearest_steer") or [], dtype=np.float64)
    anchor_distances = np.asarray(route_features.get("anchor_distances_m") or [], dtype=np.float64)
    if anchor_distances.size:
        scale = max(1.0, float(np.max(anchor_distances)))
        anchor_distances = anchor_distances / scale
    return np.concatenate((anchor_mask, anchor_speeds, anchor_steers, anchor_distances), axis=0)


def _selection_distance(candidate: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> float:
    candidate_signature = np.asarray(candidate.get("selection_signature") or [], dtype=np.float64)
    if candidate_signature.size <= 0 or not selected:
        return float("inf")
    distances = [
        float(
            np.linalg.norm(
                candidate_signature - np.asarray(item.get("selection_signature") or [], dtype=np.float64)
            )
        )
        for item in selected
    ]
    return min(distances, default=float("inf"))


def _diverse_select(candidates: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = [dict(item) for item in candidates]
    if len(ordered) <= int(limit):
        return ordered
    selected = [ordered.pop(0)]
    while ordered and len(selected) < int(limit):
        best_index = 0
        best_score = -1.0
        for index, candidate in enumerate(ordered):
            score = _selection_distance(candidate, selected)
            rank_key = int(candidate.get("rank") or 999)
            time_key = float(candidate.get("record_time_ms") or rank_key)
            if score > best_score:
                best_score = score
                best_index = index
            elif np.isclose(score, best_score):
                incumbent = ordered[best_index]
                incumbent_time = float(incumbent.get("record_time_ms") or int(incumbent.get("rank") or 999))
                if time_key < incumbent_time or (
                    np.isclose(time_key, incumbent_time) and rank_key < int(incumbent.get("rank") or 999)
                ):
                    best_index = index
        selected.append(ordered.pop(best_index))
    return selected


def _select_mixed_band_cluster(
    candidates: Sequence[Mapping[str, Any]],
    *,
    bands: Sequence[str],
    max_representatives_per_band: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_band_counts: dict[str, int] = {}
    seen_cluster_by_band: set[tuple[str, str]] = set()
    for item in sorted(candidates, key=lambda entry: (str(entry["band"]), str(entry["cluster_key"]), int(entry["rank"]))):
        band = str(item["band"])
        if band == "unbanded":
            continue
        if per_band_counts.get(band, 0) >= max_representatives_per_band:
            continue
        cluster_key = str(item["cluster_key"])
        if (band, cluster_key) in seen_cluster_by_band:
            continue
        selected.append(dict(item))
        per_band_counts[band] = per_band_counts.get(band, 0) + 1
        seen_cluster_by_band.add((band, cluster_key))
    if selected:
        return selected
    fallback_limit = max(1, min(len(candidates), int(max_representatives_per_band)))
    return [dict(item) for item in list(candidates)[:fallback_limit]]


def _family_counts(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "intended_route": 0,
        "shortcut_or_exploit": 0,
        "unclassified": 0,
    }
    for item in candidates:
        family = str(item.get("strategy_family") or "unclassified")
        counts[family] = counts.get(family, 0) + 1
    return counts


def _selector_payload(selector) -> dict[str, Any]:  # noqa: ANN001
    if selector is None:
        return {}
    if hasattr(selector, "to_mapping"):
        return dict(selector.to_mapping())
    if isinstance(selector, Mapping):
        payload: dict[str, Any] = {}
        if selector.get("ghost_name_contains") not in (None, ""):
            payload["ghost_name_contains"] = str(selector["ghost_name_contains"])
        if selector.get("rank") not in (None, ""):
            payload["rank"] = int(selector["rank"])
        return payload
    payload = {}
    ghost_name_contains = getattr(selector, "ghost_name_contains", None)
    rank = getattr(selector, "rank", None)
    if ghost_name_contains not in (None, ""):
        payload["ghost_name_contains"] = str(ghost_name_contains)
    if rank not in (None, ""):
        payload["rank"] = int(rank)
    return payload


def _resolve_selected_ghost_override(
    *,
    candidates: Sequence[Mapping[str, Any]],
    selector,
) -> dict[str, Any] | None:  # noqa: ANN001
    selector_payload = _selector_payload(selector)
    if not selector_payload:
        return None
    requested_name = selector_payload.get("ghost_name_contains")
    requested_rank = selector_payload.get("rank")
    if requested_name is not None:
        exact_matches = [
            dict(item)
            for item in candidates
            if any(str(name).casefold() == str(requested_name).casefold() for name in item.get("selection_names") or [])
        ]
        if len(exact_matches) > 1:
            raise RuntimeError(
                f"Selected ghost override {requested_name!r} matched multiple ghosts exactly; "
                "preserve a more specific name or use rank."
            )
        if len(exact_matches) == 1:
            return exact_matches[0]
        substring_matches = [
            dict(item)
            for item in candidates
            if any(str(requested_name).casefold() in str(name).casefold() for name in item.get("selection_names") or [])
        ]
        if len(substring_matches) > 1:
            raise RuntimeError(
                f"Selected ghost override {requested_name!r} matched multiple ghosts by substring; "
                "preserve a more specific name or use rank."
            )
        if len(substring_matches) == 1:
            return substring_matches[0]
    if requested_rank is not None:
        for item in candidates:
            if int(item.get("rank") or 0) == int(requested_rank):
                return dict(item)
    return None


def _maybe_build_offline_npz(
    *,
    selected: list[dict[str, Any]],
    output_root: Path,
) -> tuple[str | None, int, bool, str | None]:
    output_root.mkdir(parents=True, exist_ok=True)
    transition_paths = [
        Path(str(item["offline_transition_npz_path"]))
        for item in selected
        if item.get("offline_transition_npz_path")
    ]
    if not transition_paths:
        return None, 0, all(bool(item.get("action_channel_valid")) for item in selected), None
    arrays: dict[str, list[np.ndarray]] = {}
    for path in transition_paths:
        payload = np.load(path, allow_pickle=True)
        for key in payload.files:
            arrays.setdefault(key, []).append(np.asarray(payload[key]))
    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items() if value}
    transition_count = int(next(iter(merged.values())).shape[0]) if merged else 0
    if transition_count <= 0:
        return None, 0, all(bool(item.get("action_channel_valid")) for item in selected), None
    output_path = output_root / "offline_transitions.npz"
    np.savez_compressed(output_path, **merged)
    return (
        str(output_path),
        transition_count,
        all(bool(item.get("action_channel_valid")) for item in selected),
        sha256_file(output_path),
    )


def _bundle_payload(
    *,
    map_uid: str,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    canonical_reference: Mapping[str, Any] | None,
    strategy_classification_status: str,
    selected_training_family: str,
    mixed_fallback: bool,
    bundle_resolution_mode: str,
    selected_ghost_selector: Mapping[str, Any] | None,
    resolved_selected_ghost_rank: int | None,
    resolved_selected_ghost_name: str | None,
    author_fallback_used: bool,
    intended_bundle_manifest_path: str | None,
    exploit_bundle_manifest_path: str | None,
    mixed_fallback_manifest_path: str | None,
    selected_override_manifest_path: str | None,
    author_fallback_manifest_path: str | None,
    output_root: Path,
    bands: Sequence[str],
    max_representatives_per_band: int,
) -> dict[str, Any]:
    offline_npz, offline_count, action_valid, offline_hash = _maybe_build_offline_npz(
        selected=selected,
        output_root=output_root,
    )
    return {
        "schema_version": GHOST_BUNDLE_SCHEMA_VERSION,
        "map_uid": map_uid,
        "bands": list(bands),
        "max_representatives_per_band": int(max_representatives_per_band),
        "trajectory_count": len(candidates),
        "selected_count": len(selected),
        "selected_trajectories": selected,
        "all_trajectories": candidates,
        "action_channel_valid": action_valid,
        "offline_transition_npz_path": offline_npz,
        "offline_transition_count": offline_count,
        "offline_dataset_hash": offline_hash,
        "canonical_reference_source": None if canonical_reference is None else canonical_reference.get("source"),
        "canonical_reference_path": None if canonical_reference is None else canonical_reference.get("path"),
        "canonical_reference_total_length_m": None if canonical_reference is None else canonical_reference.get("total_length_m"),
        "strategy_classification_status": strategy_classification_status,
        "selected_training_family": selected_training_family,
        "strategy_family_counts": _family_counts(candidates),
        "mixed_fallback": bool(mixed_fallback),
        "bundle_resolution_mode": bundle_resolution_mode,
        "selected_ghost_selector": None if selected_ghost_selector is None else dict(selected_ghost_selector),
        "resolved_selected_ghost_rank": resolved_selected_ghost_rank,
        "resolved_selected_ghost_name": resolved_selected_ghost_name,
        "author_fallback_used": bool(author_fallback_used),
        "intended_bundle_manifest_path": intended_bundle_manifest_path,
        "exploit_bundle_manifest_path": exploit_bundle_manifest_path,
        "mixed_fallback_manifest_path": mixed_fallback_manifest_path,
        "selected_override_manifest_path": selected_override_manifest_path,
        "author_fallback_manifest_path": author_fallback_manifest_path,
    }


def _strategy_manifest_payload(
    *,
    map_uid: str,
    candidates: list[dict[str, Any]],
    canonical_reference: Mapping[str, Any] | None,
    strategy_classification_status: str,
    default_manifest_path: Path | None,
    intended_manifest_path: Path | None,
    exploit_manifest_path: Path | None,
    selected_override_manifest_path: Path | None,
    author_fallback_manifest_path: Path | None,
    default_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "ghost_strategy_manifest_v1",
        "map_uid": map_uid,
        "canonical_reference_source": None if canonical_reference is None else canonical_reference.get("source"),
        "canonical_reference_path": None if canonical_reference is None else canonical_reference.get("path"),
        "canonical_reference_total_length_m": None if canonical_reference is None else canonical_reference.get("total_length_m"),
        "strategy_classification_status": strategy_classification_status,
        "selected_training_family": None if default_manifest is None else default_manifest.get("selected_training_family"),
        "mixed_fallback": False if default_manifest is None else bool(default_manifest.get("mixed_fallback", False)),
        "bundle_resolution_mode": None if default_manifest is None else default_manifest.get("bundle_resolution_mode"),
        "selected_ghost_selector": None if default_manifest is None else default_manifest.get("selected_ghost_selector"),
        "resolved_selected_ghost_rank": None if default_manifest is None else default_manifest.get("resolved_selected_ghost_rank"),
        "resolved_selected_ghost_name": None if default_manifest is None else default_manifest.get("resolved_selected_ghost_name"),
        "author_fallback_used": False if default_manifest is None else bool(default_manifest.get("author_fallback_used", False)),
        "strategy_family_counts": _family_counts(candidates),
        "default_bundle_manifest_path": None if default_manifest_path is None else str(default_manifest_path),
        "intended_bundle_manifest_path": None if intended_manifest_path is None else str(intended_manifest_path),
        "exploit_bundle_manifest_path": None if exploit_manifest_path is None else str(exploit_manifest_path),
        "mixed_fallback_manifest_path": None,
        "selected_override_manifest_path": None if selected_override_manifest_path is None else str(selected_override_manifest_path),
        "author_fallback_manifest_path": None if author_fallback_manifest_path is None else str(author_fallback_manifest_path),
        "all_trajectories": candidates,
    }


def build_ghost_bundle(
    *,
    map_uid: str,
    trajectory_metadata_paths: Sequence[str | Path],
    output_dir: str | Path,
    spacing_meters: float = 0.5,
    ghost_config=None,  # noqa: ANN001
    author_reference_manifest: str | Path | None = None,
    selected_ghost_selector: Mapping[str, Any] | None = None,
    bands: Sequence[str],
    max_representatives_per_band: int,
) -> GhostBundleBuildResult:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    canonical_reference, strategy_classification_status = _canonical_reference(
        map_uid=map_uid,
        spacing_meters=spacing_meters,
        author_reference_manifest=(
            None
            if author_reference_manifest is None
            else str(Path(author_reference_manifest).resolve())
        ),
    )
    anchor_payload = (
        None
        if canonical_reference is None or ghost_config is None
        else _anchor_payload(canonical_reference, int(ghost_config.anchor_count))
    )
    candidates: list[dict[str, Any]] = []
    for metadata_path in trajectory_metadata_paths:
        resolved_metadata_path = Path(metadata_path).resolve()
        metadata = _load_trajectory_metadata(resolved_metadata_path)
        candidates.append(
            _bundle_candidate(
                metadata,
                metadata_path=resolved_metadata_path,
                canonical_reference=canonical_reference,
                anchor_payload=anchor_payload,
                ghost_config=ghost_config,
                bands=bands,
            )
        )
    candidates.sort(key=lambda item: int(item["rank"]))

    selected_override_selector = _selector_payload(selected_ghost_selector)
    if not selected_override_selector and ghost_config is not None:
        selected_override_selector = _selector_payload(getattr(ghost_config, "selected_ghost_overrides", {}).get(map_uid))

    training_family = "legacy_rank_band_selection" if ghost_config is None else str(ghost_config.training_family)
    intended_candidates = [
        dict(item)
        for item in candidates
        if str(item.get("strategy_family")) == "intended_route"
    ]
    exploit_candidates = [
        dict(item)
        for item in candidates
        if str(item.get("strategy_family")) == "shortcut_or_exploit"
    ]

    intended_manifest_path = output_root / "ghost_bundle_intended.json"
    exploit_manifest_path = output_root / "ghost_bundle_exploit.json"
    default_manifest_path = output_root / "ghost_bundle_manifest.json"
    strategy_manifest_path = output_root / "ghost_strategy_manifest.json"
    selected_override_manifest_path = output_root / "ghost_bundle_selected_override.json"
    author_fallback_manifest_path = output_root / "ghost_bundle_author_reference.json"
    mixed_fallback_manifest_path = output_root / "ghost_bundle_mixed_fallback.json"

    intended_selected: list[dict[str, Any]] = []
    exploit_selected: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    default_manifest: dict[str, Any] | None = None
    bundle_resolution_mode: str | None = None
    resolved_selected_ghost_rank: int | None = None
    resolved_selected_ghost_name: str | None = None
    author_fallback_used = False
    selected_override_manifest_result: Path | None = None
    author_fallback_manifest_result: Path | None = None
    mixed_fallback_manifest_result: Path | None = None

    if ghost_config is None:
        selected = _select_mixed_band_cluster(
            candidates,
            bands=bands,
            max_representatives_per_band=max_representatives_per_band,
        )
        bundle_resolution_mode = "legacy_rank_band_selection"
        default_manifest = _bundle_payload(
            map_uid=map_uid,
            selected=selected,
            candidates=candidates,
            canonical_reference=canonical_reference,
            strategy_classification_status="legacy_rank_band_selection",
            selected_training_family=training_family,
            mixed_fallback=False,
            bundle_resolution_mode=bundle_resolution_mode,
            selected_ghost_selector=None,
            resolved_selected_ghost_rank=None,
            resolved_selected_ghost_name=None,
            author_fallback_used=False,
            intended_bundle_manifest_path=None,
            exploit_bundle_manifest_path=None,
            mixed_fallback_manifest_path=None,
            selected_override_manifest_path=None,
            author_fallback_manifest_path=None,
            output_root=output_root / "legacy_bundle",
            bands=bands,
            max_representatives_per_band=max_representatives_per_band,
        )
        write_json(default_manifest_path, default_manifest)
        intended_manifest_result = None
        exploit_manifest_result = None
    else:
        intended_candidates.sort(key=lambda item: (float(item.get("record_time_ms") or 1.0e18), int(item["rank"])))
        exploit_candidates.sort(key=lambda item: (float(item.get("record_time_ms") or 1.0e18), int(item["rank"])))
        intended_pool = intended_candidates[: int(ghost_config.intended_candidate_pool)]
        intended_selected = _diverse_select(intended_pool, int(ghost_config.intended_bundle_size))
        exploit_selected = _diverse_select(exploit_candidates, int(ghost_config.exploit_bundle_size))
        if not intended_selected and canonical_reference is not None:
            strategy_classification_status = "no_intended_route_candidates"

        exploit_manifest_result = None
        intended_manifest_result = None

        if intended_selected:
            bundle_resolution_mode = "intended_route"
            intended_manifest = _bundle_payload(
                map_uid=map_uid,
                selected=intended_selected,
                candidates=candidates,
                canonical_reference=canonical_reference,
                strategy_classification_status=strategy_classification_status,
                selected_training_family=training_family,
                mixed_fallback=False,
                bundle_resolution_mode=bundle_resolution_mode,
                selected_ghost_selector=None,
                resolved_selected_ghost_rank=None,
                resolved_selected_ghost_name=None,
                author_fallback_used=False,
                intended_bundle_manifest_path=str(intended_manifest_path),
                exploit_bundle_manifest_path=str(exploit_manifest_path) if exploit_selected else None,
                mixed_fallback_manifest_path=None,
                selected_override_manifest_path=None,
                author_fallback_manifest_path=None,
                output_root=output_root / "intended_bundle",
                bands=bands,
                max_representatives_per_band=max_representatives_per_band,
            )
            write_json(intended_manifest_path, intended_manifest)
            default_manifest = dict(intended_manifest)
            default_manifest["default_bundle_alias_of"] = str(intended_manifest_path)
            write_json(default_manifest_path, default_manifest)
            selected = intended_selected
            intended_manifest_result = intended_manifest_path
        else:
            error_message: str | None = None
            selected_override_candidate: dict[str, Any] | None = None
            try:
                selected_override_candidate = _resolve_selected_ghost_override(
                    candidates=candidates,
                    selector=selected_override_selector,
                )
            except RuntimeError as exc:
                error_message = str(exc)

            if error_message is not None:
                default_manifest = {
                    "bundle_resolution_mode": "error_no_default_bundle",
                    "selected_ghost_selector": selected_override_selector or None,
                    "resolved_selected_ghost_rank": None,
                    "resolved_selected_ghost_name": None,
                    "author_fallback_used": False,
                }
                write_json(
                    strategy_manifest_path,
                    _strategy_manifest_payload(
                        map_uid=map_uid,
                        candidates=candidates,
                        canonical_reference=canonical_reference,
                        strategy_classification_status=strategy_classification_status,
                        default_manifest_path=None,
                        intended_manifest_path=None,
                        exploit_manifest_path=None,
                        selected_override_manifest_path=None,
                        author_fallback_manifest_path=None,
                        default_manifest=default_manifest,
                    ),
                )
                raise RuntimeError(error_message)

            if selected_override_candidate is not None:
                selected = [selected_override_candidate]
                bundle_resolution_mode = "selected_ghost_override"
                resolved_selected_ghost_rank = int(selected_override_candidate.get("rank") or 0) or None
                resolved_selected_ghost_name = next(
                    (
                        str(name)
                        for name in selected_override_candidate.get("selection_names") or []
                        if str(name).casefold()
                        == str((selected_override_selector or {}).get("ghost_name_contains", "")).casefold()
                    ),
                    next(
                        (
                            str(name)
                            for name in selected_override_candidate.get("selection_names") or []
                            if str((selected_override_selector or {}).get("ghost_name_contains", "")).casefold()
                            in str(name).casefold()
                        ),
                        str(
                            selected_override_candidate.get("trajectory_id")
                            or selected_override_candidate.get("record_filename_basename")
                            or selected_override_candidate.get("downloaded_replay_basename")
                            or selected_override_candidate.get("source_export_basename")
                            or selected_override_candidate.get("metadata_path")
                        ),
                    ),
                )
                selected_training_family = str(
                    selected_override_candidate.get("strategy_family") or "selected_ghost_override"
                )
                selected_override_manifest = _bundle_payload(
                    map_uid=map_uid,
                    selected=selected,
                    candidates=candidates,
                    canonical_reference=canonical_reference,
                    strategy_classification_status=strategy_classification_status,
                    selected_training_family=selected_training_family,
                    mixed_fallback=False,
                    bundle_resolution_mode=bundle_resolution_mode,
                    selected_ghost_selector=selected_override_selector,
                    resolved_selected_ghost_rank=resolved_selected_ghost_rank,
                    resolved_selected_ghost_name=resolved_selected_ghost_name,
                    author_fallback_used=False,
                    intended_bundle_manifest_path=None,
                    exploit_bundle_manifest_path=str(exploit_manifest_path) if exploit_selected else None,
                    mixed_fallback_manifest_path=None,
                    selected_override_manifest_path=str(selected_override_manifest_path),
                    author_fallback_manifest_path=None,
                    output_root=output_root / "selected_override_bundle",
                    bands=bands,
                    max_representatives_per_band=max_representatives_per_band,
                )
                write_json(selected_override_manifest_path, selected_override_manifest)
                default_manifest = dict(selected_override_manifest)
                default_manifest["default_bundle_alias_of"] = str(selected_override_manifest_path)
                write_json(default_manifest_path, default_manifest)
                selected_override_manifest_result = selected_override_manifest_path
            else:
                author_reference_metadata = (
                    None
                    if author_reference_manifest is None
                    else _load_author_reference_metadata(Path(author_reference_manifest))
                )
                if author_reference_metadata is not None:
                    author_metadata, author_metadata_path = author_reference_metadata
                    author_candidate = next(
                        (
                            dict(item)
                            for item in candidates
                            if str(item.get("metadata_path")) == str(author_metadata_path.resolve())
                        ),
                        _bundle_candidate(
                            author_metadata,
                            metadata_path=author_metadata_path,
                            canonical_reference=canonical_reference,
                            anchor_payload=anchor_payload,
                            ghost_config=ghost_config,
                            bands=bands,
                        ),
                    )
                    selected = [author_candidate]
                    bundle_resolution_mode = "author_reference_fallback"
                    author_fallback_used = True
                    author_manifest = _bundle_payload(
                        map_uid=map_uid,
                        selected=selected,
                        candidates=candidates,
                        canonical_reference=canonical_reference,
                        strategy_classification_status=strategy_classification_status,
                        selected_training_family="author_reference_fallback",
                        mixed_fallback=False,
                        bundle_resolution_mode=bundle_resolution_mode,
                        selected_ghost_selector=selected_override_selector or None,
                        resolved_selected_ghost_rank=None,
                        resolved_selected_ghost_name=None,
                        author_fallback_used=True,
                        intended_bundle_manifest_path=None,
                        exploit_bundle_manifest_path=str(exploit_manifest_path) if exploit_selected else None,
                        mixed_fallback_manifest_path=None,
                        selected_override_manifest_path=None,
                        author_fallback_manifest_path=str(author_fallback_manifest_path),
                        output_root=output_root / "author_reference_bundle",
                        bands=bands,
                        max_representatives_per_band=max_representatives_per_band,
                    )
                    write_json(author_fallback_manifest_path, author_manifest)
                    default_manifest = dict(author_manifest)
                    default_manifest["default_bundle_alias_of"] = str(author_fallback_manifest_path)
                    write_json(default_manifest_path, default_manifest)
                    author_fallback_manifest_result = author_fallback_manifest_path
                else:
                    default_manifest = {
                        "bundle_resolution_mode": "error_no_default_bundle",
                        "selected_ghost_selector": selected_override_selector or None,
                        "resolved_selected_ghost_rank": None,
                        "resolved_selected_ghost_name": None,
                        "author_fallback_used": False,
                    }
                    write_json(
                        strategy_manifest_path,
                        _strategy_manifest_payload(
                            map_uid=map_uid,
                            candidates=candidates,
                            canonical_reference=canonical_reference,
                            strategy_classification_status=strategy_classification_status,
                            default_manifest_path=None,
                            intended_manifest_path=None,
                            exploit_manifest_path=None,
                            selected_override_manifest_path=None,
                            author_fallback_manifest_path=None,
                            default_manifest=default_manifest,
                        ),
                    )
                    if error_message is None:
                        error_message = (
                            "No intended-route bundle could be built, no selected ghost override resolved, "
                            "and no author/reference fallback was available."
                        )
                    raise RuntimeError(error_message)

        if exploit_selected:
            exploit_manifest = _bundle_payload(
                map_uid=map_uid,
                selected=exploit_selected,
                candidates=candidates,
                canonical_reference=canonical_reference,
                strategy_classification_status=strategy_classification_status,
                selected_training_family="shortcut_or_exploit",
                mixed_fallback=False,
                bundle_resolution_mode="shortcut_or_exploit_bundle",
                selected_ghost_selector=None,
                resolved_selected_ghost_rank=None,
                resolved_selected_ghost_name=None,
                author_fallback_used=False,
                intended_bundle_manifest_path=None if intended_manifest_result is None else str(intended_manifest_result),
                exploit_bundle_manifest_path=str(exploit_manifest_path),
                mixed_fallback_manifest_path=None,
                selected_override_manifest_path=(
                    None if selected_override_manifest_result is None else str(selected_override_manifest_result)
                ),
                author_fallback_manifest_path=(
                    None if author_fallback_manifest_result is None else str(author_fallback_manifest_result)
                ),
                output_root=output_root / "exploit_bundle",
                bands=bands,
                max_representatives_per_band=max_representatives_per_band,
            )
            write_json(exploit_manifest_path, exploit_manifest)
            exploit_manifest_result = exploit_manifest_path

    write_json(
        strategy_manifest_path,
        _strategy_manifest_payload(
            map_uid=map_uid,
            candidates=candidates,
            canonical_reference=canonical_reference,
            strategy_classification_status=strategy_classification_status,
            default_manifest_path=default_manifest_path if default_manifest is not None else None,
            intended_manifest_path=intended_manifest_result,
            exploit_manifest_path=exploit_manifest_result,
            selected_override_manifest_path=selected_override_manifest_result,
            author_fallback_manifest_path=author_fallback_manifest_result,
            default_manifest=default_manifest,
        ),
    )
    return GhostBundleBuildResult(
        manifest_path=default_manifest_path,
        selected_count=len(selected),
        trajectory_count=len(candidates),
        action_channel_valid=bool(default_manifest.get("action_channel_valid")) if default_manifest is not None else False,
        offline_transition_count=(
            int(default_manifest.get("offline_transition_count", 0) or 0) if default_manifest is not None else 0
        ),
        bundle_resolution_mode=bundle_resolution_mode,
        strategy_manifest_path=strategy_manifest_path,
        intended_manifest_path=intended_manifest_result,
        exploit_manifest_path=exploit_manifest_result,
        selected_override_manifest_path=selected_override_manifest_result,
        author_fallback_manifest_path=author_fallback_manifest_result,
        mixed_fallback_manifest_path=mixed_fallback_manifest_result,
    )
