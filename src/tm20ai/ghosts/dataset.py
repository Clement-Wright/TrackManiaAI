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


def _trajectory_features(metadata: Mapping[str, Any]) -> dict[str, float]:
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


def _maybe_build_offline_npz(
    *,
    selected: list[dict[str, Any]],
    output_root: Path,
) -> tuple[str | None, int, bool, str | None]:
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


def build_ghost_bundle(
    *,
    map_uid: str,
    trajectory_metadata_paths: Sequence[str | Path],
    output_dir: str | Path,
    bands: Sequence[str],
    max_representatives_per_band: int,
) -> GhostBundleBuildResult:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for metadata_path in trajectory_metadata_paths:
        metadata = _load_trajectory_metadata(Path(metadata_path).resolve())
        rank = int(metadata.get("rank") or 999)
        features = _trajectory_features(metadata)
        candidates.append(
            {
                **metadata,
                "rank": rank,
                "band": _rank_band(rank, bands),
                "features": features,
                "cluster_key": _cluster_key(metadata, features),
                "metadata_path": str(Path(metadata_path).resolve()),
            }
        )
    candidates.sort(key=lambda item: (str(item["band"]), str(item["cluster_key"]), int(item["rank"])))
    selected: list[dict[str, Any]] = []
    per_band_counts: dict[str, int] = {}
    seen_cluster_by_band: set[tuple[str, str]] = set()
    for item in candidates:
        band = str(item["band"])
        if band == "unbanded":
            continue
        if per_band_counts.get(band, 0) >= max_representatives_per_band:
            continue
        cluster_key = str(item["cluster_key"])
        if (band, cluster_key) in seen_cluster_by_band:
            continue
        selected.append(item)
        per_band_counts[band] = per_band_counts.get(band, 0) + 1
        seen_cluster_by_band.add((band, cluster_key))
    if not selected:
        selected = candidates[: max(1, min(len(candidates), max_representatives_per_band))]
    offline_npz, offline_count, action_valid, offline_hash = _maybe_build_offline_npz(
        selected=selected,
        output_root=output_root,
    )
    manifest_path = output_root / "ghost_bundle_manifest.json"
    manifest = {
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
    }
    write_json(manifest_path, manifest)
    return GhostBundleBuildResult(
        manifest_path=manifest_path,
        selected_count=len(selected),
        trajectory_count=len(candidates),
        action_channel_valid=action_valid,
        offline_transition_count=offline_count,
    )
