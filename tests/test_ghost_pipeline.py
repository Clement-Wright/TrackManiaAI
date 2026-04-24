from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import tm20ai.ghosts.dataset as ghost_dataset
from tm20ai.bridge import TelemetryFrame
from tm20ai.config import GhostConfig, RewardConfig
from tm20ai.data.parquet_writer import read_json, write_json
from tm20ai.env.rt_interface import TM20AIRtInterface
from tm20ai.ghosts.dataset import (
    build_ghost_bundle,
    build_reference_target_bundle,
    extract_openplanet_export,
    load_ghost_bundle_manifest,
)
from tm20ai.ghosts.nadeo import NadeoCredentials, NadeoServicesClient, fetch_top100_ghost_manifest
from tm20ai.ghosts.offline import seed_replay_from_ghost_bundle
from tm20ai.ghosts.reward import GhostBundleReward
from tm20ai.train.features import TELEMETRY_DIM
from tm20ai.train.replay import BalancedReplayBuffer, ReplayBuffer


def _rows(offset: float = 0.0) -> list[dict[str, object]]:
    rows = []
    for index in range(5):
        rows.append(
            {
                "race_time_ms": index * 50,
                "position": [float(index), 0.0, offset],
                "velocity": [1.0, 0.0, 0.0],
                "forward": [1.0, 0.0, 0.0],
                "speed_kmh": 50.0 + index,
                "progress_index": index,
                "arc_length": float(index),
                "throttle": 1.0,
                "brake": 0.0,
                "steer": 0.1 * offset,
                "gear": 2,
            }
        )
    return rows


def _frame(x: float, z: float = 0.0, *, finished: bool = False, speed_kmh: float = 50.0) -> TelemetryFrame:
    return TelemetryFrame(
        session_id="session",
        run_id="run",
        frame_id=int(x),
        timestamp_ns=0,
        map_uid="map",
        race_time_ms=int(x * 50),
        cp_count=0,
        cp_target=0,
        speed_kmh=speed_kmh,
        gear=2,
        rpm=1000.0,
        pos_xyz=(x, 0.0, z),
        vel_xyz=(1.0, 0.0, 0.0),
        yaw_pitch_roll=(0.0, 0.0, 0.0),
        finished=finished,
        terminal_reason="finished" if finished else None,
    )


def _path_rows(
    points: list[tuple[float, float, float]],
    *,
    speed_kmh: float = 80.0,
    steer: float = 0.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous = None
    arc_length = 0.0
    for index, point in enumerate(points):
        position = np.asarray(point, dtype=np.float64)
        if previous is not None:
            arc_length += float(np.linalg.norm(position - previous))
        previous = position
        rows.append(
            {
                "race_time_ms": index * 100,
                "position": [float(position[0]), float(position[1]), float(position[2])],
                "velocity": [1.0, 0.0, 0.0],
                "forward": [1.0, 0.0, 0.0],
                "speed_kmh": speed_kmh,
                "arc_length": arc_length,
                "throttle": 1.0,
                "brake": 0.0,
                "steer": steer,
                "gear": 3,
            }
        )
    return rows


def _straight_points(
    *,
    start_x: float = 0.0,
    stop_x: float = 40.0,
    step_x: float = 2.0,
    z_offset: float = 0.0,
) -> list[tuple[float, float, float]]:
    xs = np.arange(start_x, stop_x + 1.0e-6, step_x, dtype=np.float64)
    return [(float(x), 0.0, float(z_offset)) for x in xs]


def _write_route_metadata(
    tmp_path: Path,
    *,
    name: str,
    rank: int,
    points: list[tuple[float, float, float]],
    record_time_ms: int,
) -> Path:
    export_path = tmp_path / f"{name}.json"
    write_json(export_path, {"rows": _path_rows(points, steer=0.02 * rank)})
    return extract_openplanet_export(
        export_path,
        output_dir=tmp_path / "route_trajectories",
        replay_metadata={
            "trajectory_id": name,
            "map_uid": "map",
            "rank": rank,
            "record_time_ms": record_time_ms,
            "record_filename": f"Replays\\Downloaded\\{name}.replay.gbx",
            "downloaded_replay_path": str((tmp_path / "downloads" / f"{name}.replay.gbx").resolve()),
        },
    )


def _write_reward_reference_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(_straight_points(), dtype=np.float32)
    arc_length = np.asarray([float(index * 2.0) for index in range(len(points))], dtype=np.float32)
    race_time_ms = np.asarray([float(index * 100.0) for index in range(len(points))], dtype=np.float32)
    np.savez_compressed(
        path,
        map_uid=np.asarray(["map"]),
        points=points,
        tangents=np.asarray([[1.0, 0.0, 0.0]] * len(points), dtype=np.float32),
        arc_length=arc_length,
        race_time_ms=race_time_ms,
    )
    return path


def _route_family_bundle(
    tmp_path: Path,
    *,
    author_reference_manifest: Path | None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    reward_reference_path: Path | None = None,
    selected_ghost_selector: dict[str, Any] | None = None,
) -> tuple[object, dict[str, Any]]:
    if monkeypatch is not None and reward_reference_path is not None:
        monkeypatch.setattr(
            ghost_dataset,
            "_reward_trajectory_path_for_map",
            lambda map_uid, spacing_meters: reward_reference_path,
        )

    intended_paths = [
        _write_route_metadata(
            tmp_path,
            name="intended_rank_015",
            rank=15,
            points=_straight_points(z_offset=0.0),
            record_time_ms=2100,
        ),
        _write_route_metadata(
            tmp_path,
            name="intended_rank_022",
            rank=22,
            points=_straight_points(z_offset=1.5),
            record_time_ms=2200,
        ),
        _write_route_metadata(
            tmp_path,
            name="intended_rank_028",
            rank=28,
            points=_straight_points(z_offset=-1.5),
            record_time_ms=2300,
        ),
    ]
    exploit_paths = [
        _write_route_metadata(
            tmp_path,
            name="exploit_rank_001",
            rank=1,
            points=[
                (0.0, 0.0, 0.0),
                (-6.0, 0.0, 0.0),
                (-12.0, 0.0, 0.0),
                (-18.0, 0.0, 0.0),
                (-6.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (8.0, 0.0, 0.0),
            ],
            record_time_ms=1500,
        ),
        _write_route_metadata(
            tmp_path,
            name="exploit_rank_005",
            rank=5,
            points=[
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (6.0, 0.0, 45.0),
                (8.0, 0.0, 65.0),
                (10.0, 0.0, 80.0),
            ],
            record_time_ms=1600,
        ),
    ]
    ambiguous_path = _write_route_metadata(
        tmp_path,
        name="ambiguous_rank_040",
        rank=40,
        points=_straight_points(stop_x=30.0, z_offset=0.0),
        record_time_ms=2600,
    )

    result = build_ghost_bundle(
        map_uid="map",
        trajectory_metadata_paths=[*intended_paths, *exploit_paths, ambiguous_path],
        output_dir=tmp_path / "route_bundle",
        spacing_meters=0.5,
        ghost_config=GhostConfig(
            anchor_count=8,
            anchor_radius_m=6.0,
            canonical_divergence_radius_m=12.0,
            intended_candidate_pool=5,
            intended_bundle_size=2,
            exploit_bundle_size=2,
        ),
        author_reference_manifest=None if author_reference_manifest is None else str(author_reference_manifest),
        selected_ghost_selector=selected_ghost_selector,
        bands=("1-10", "11-30", "31-60"),
        max_representatives_per_band=2,
    )
    manifest = load_ghost_bundle_manifest(result.manifest_path)
    return result, manifest


def test_ghost_extract_build_seed_and_reward(tmp_path: Path) -> None:
    export_path = tmp_path / "rank_001.json"
    write_json(export_path, {"rows": _rows()})
    obs_npz = tmp_path / "rank_001_observations.npz"
    np.savez_compressed(
        obs_npz,
        obs_uint8=np.zeros((5, 4, 64, 64), dtype=np.uint8),
        telemetry_float=np.zeros((5, TELEMETRY_DIM), dtype=np.float32),
    )

    metadata_path = extract_openplanet_export(
        export_path,
        output_dir=tmp_path / "trajectories",
        replay_metadata={
            "trajectory_id": "rank_001",
            "map_uid": "map",
            "rank": 1,
            "record_time_ms": 1234,
            "record_filename": r"Replays\Downloaded\rank_001.replay.gbx",
            "downloaded_replay_path": str((tmp_path / "downloads" / "rank_001.replay.gbx").resolve()),
            "observation_npz_path": str(obs_npz),
        },
    )
    metadata = read_json(metadata_path)
    assert metadata["action_channel_valid"] is True
    assert metadata["offline_transition_count"] == 5
    assert metadata["record_filename_basename"] == "rank_001.replay.gbx"
    assert metadata["downloaded_replay_basename"] == "rank_001.replay.gbx"
    assert metadata["source_export_basename"] == "rank_001.json"
    assert metadata["source_export_stem"] == "rank_001"

    result = build_ghost_bundle(
        map_uid="map",
        trajectory_metadata_paths=[metadata_path],
        output_dir=tmp_path / "bundle",
        bands=("1-10",),
        max_representatives_per_band=1,
    )
    assert result.selected_count == 1
    assert result.offline_transition_count == 5

    replay = BalancedReplayBuffer(
        online=ReplayBuffer(mode="full", capacity=8, observation_shape=(4, 64, 64), telemetry_dim=TELEMETRY_DIM),
        offline=ReplayBuffer(mode="full", capacity=8, observation_shape=(4, 64, 64), telemetry_dim=TELEMETRY_DIM),
        offline_initial_fraction=1.0,
        offline_final_fraction=1.0,
        decay_env_steps=1,
    )
    seed = seed_replay_from_ghost_bundle(replay, result.manifest_path)
    assert seed["seeded"] == 5
    assert replay.offline_size == 5

    reward = GhostBundleReward(
        manifest_path=result.manifest_path,
        reward_config=RewardConfig(failure_countdown=10, min_steps=1),
        ghost_config=GhostConfig(line_switch_hysteresis=0),
    )
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))
    step = reward.evaluate(_frame(3.0))
    assert step.progress_delta > 0
    assert step.info["ghost_line_rank"] == 1


def _build_reward_bundle(tmp_path: Path, offsets: tuple[float, ...] = (0.0,)) -> Path:
    metadata_paths = []
    for index, offset in enumerate(offsets):
        rank = 1 + index * 10
        export_path = tmp_path / f"rank_{rank:03d}.json"
        write_json(export_path, {"rows": _rows(offset)})
        metadata_paths.append(
            extract_openplanet_export(
                export_path,
                output_dir=tmp_path / "trajectories",
                replay_metadata={
                    "trajectory_id": f"rank_{rank:03d}",
                    "map_uid": "map",
                    "rank": rank,
                    "record_time_ms": 1234 + rank,
                },
            )
        )
    result = build_ghost_bundle(
        map_uid="map",
        trajectory_metadata_paths=metadata_paths,
        output_dir=tmp_path / "bundle",
        bands=("1-10", "11-30", "31-60"),
        max_representatives_per_band=1,
    )
    return result.manifest_path


def _build_variable_density_bundle(tmp_path: Path) -> Path:
    metadata_paths = []
    for rank, step in ((1, 1.0), (11, 0.25)):
        rows = []
        index = 0
        x = 0.0
        while x <= 4.0 + 1.0e-6:
            rows.append(
                {
                    "race_time_ms": index * 50,
                    "position": [x, 0.0, 0.0],
                    "velocity": [1.0, 0.0, 0.0],
                    "forward": [1.0, 0.0, 0.0],
                    "speed_kmh": 50.0,
                    "throttle": 1.0,
                    "brake": 0.0,
                    "steer": 0.0,
                    "gear": 2,
                }
            )
            index += 1
            x += step
        export_path = tmp_path / f"density_rank_{rank:03d}.json"
        write_json(export_path, {"rows": rows})
        metadata_paths.append(
            extract_openplanet_export(
                export_path,
                output_dir=tmp_path / "density_trajectories",
                replay_metadata={
                    "trajectory_id": f"density_rank_{rank:03d}",
                    "map_uid": "map",
                    "rank": rank,
                    "record_time_ms": 1234 + rank,
                },
            )
        )
    result = build_ghost_bundle(
        map_uid="map",
        trajectory_metadata_paths=metadata_paths,
        output_dir=tmp_path / "density_bundle",
        bands=("1-10", "11-30"),
        max_representatives_per_band=1,
    )
    return result.manifest_path


def test_ghost_progress_uses_fixed_spacing_not_source_row_density(tmp_path: Path) -> None:
    reward = GhostBundleReward(
        manifest_path=_build_variable_density_bundle(tmp_path),
        reward_config=RewardConfig(failure_countdown=100, min_steps=100, spacing_meters=0.5),
        ghost_config=GhostConfig(line_switch_hysteresis=0),
    )
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    step = reward.evaluate(_frame(2.4))

    assert step.progress_index == 5
    assert step.progress_delta == 5
    assert step.info["progress_index_semantics"] == "fixed_spacing_meters"
    assert step.info["progress_spacing_meters"] == 0.5
    assert step.info["final_arc_length_m"] >= 2.0
    assert 0.0 <= step.info["progress_fraction_of_reference"] <= 1.0
    assert step.info["ghost_reference_time_ms"] is not None
    assert step.info["ghost_relative_time_delta_ms"] is not None
    assert step.info["ghost_source_row_index"] != step.progress_index


def test_ghost_corridor_soft_violation_penalizes_without_reset(tmp_path: Path) -> None:
    reward = GhostBundleReward(
        manifest_path=_build_reward_bundle(tmp_path),
        reward_config=RewardConfig(
            failure_countdown=100,
            min_steps=100,
            corridor_soft_margin_m=1.0,
            corridor_hard_margin_m=5.0,
            corridor_penalty_scale=1.0,
            corridor_penalty_max=10.0,
            corridor_catastrophic_distance_m=50.0,
        ),
        ghost_config=GhostConfig(line_switch_hysteresis=0),
    )
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    step = reward.evaluate(_frame(1.0, z=2.5))

    assert step.done_type is None
    assert step.info["corridor_soft_violation"] is True
    assert step.info["corridor_hard_violation"] is False
    assert step.info["corridor_penalty"] > 0.0
    assert step.reward < float(step.progress_delta)


def test_ghost_corridor_hard_violation_uses_patience(tmp_path: Path) -> None:
    reward = GhostBundleReward(
        manifest_path=_build_reward_bundle(tmp_path),
        reward_config=RewardConfig(
            failure_countdown=100,
            min_steps=100,
            corridor_soft_margin_m=0.5,
            corridor_hard_margin_m=1.0,
            corridor_patience_steps=2,
            corridor_catastrophic_distance_m=50.0,
        ),
        ghost_config=GhostConfig(line_switch_hysteresis=0),
    )
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    first = reward.evaluate(_frame(0.0, z=2.0, speed_kmh=0.0))
    second = reward.evaluate(_frame(0.0, z=2.0, speed_kmh=0.0))

    assert first.done_type is None
    assert first.info["corridor_violation_steps"] == 1
    assert first.info["corridor_nonrecovering_steps"] == 1
    assert second.done_type == "truncated"
    assert second.done_reason == "corridor_violation"
    assert second.info["corridor_truncation_count"] == 1


def test_ghost_corridor_hard_violation_does_not_truncate_while_recovering(tmp_path: Path) -> None:
    reward = GhostBundleReward(
        manifest_path=_build_reward_bundle(tmp_path),
        reward_config=RewardConfig(
            failure_countdown=100,
            min_steps=100,
            corridor_soft_margin_m=0.5,
            corridor_hard_margin_m=1.0,
            corridor_patience_steps=2,
            corridor_min_recovery_progress_m=0.5,
            corridor_min_recovery_speed_kmh=100.0,
            corridor_recovery_distance_delta_m=1.0,
            corridor_catastrophic_distance_m=50.0,
        ),
        ghost_config=GhostConfig(line_switch_hysteresis=0),
    )
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    first = reward.evaluate(_frame(1.0, z=2.0, speed_kmh=0.0))
    second = reward.evaluate(_frame(2.0, z=2.0, speed_kmh=0.0))

    assert first.done_type is None
    assert second.done_type is None
    assert second.info["corridor_recovering"] is True
    assert second.info["corridor_nonrecovering_steps"] == 0


def test_ghost_corridor_recovery_bonus_after_violation(tmp_path: Path) -> None:
    reward = GhostBundleReward(
        manifest_path=_build_reward_bundle(tmp_path),
        reward_config=RewardConfig(
            failure_countdown=100,
            min_steps=100,
            corridor_soft_margin_m=1.0,
            corridor_hard_margin_m=5.0,
            corridor_recovery_bonus=2.0,
            corridor_catastrophic_distance_m=50.0,
        ),
        ghost_config=GhostConfig(line_switch_hysteresis=0),
    )
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    reward.evaluate(_frame(1.0, z=2.0))
    recovered = reward.evaluate(_frame(2.0, z=0.0))

    assert recovered.done_type is None
    assert recovered.info["corridor_recovery_count"] == 1
    assert recovered.info["corridor_recovery_bonus"] == 2.0
    assert recovered.reward >= recovered.progress_delta + 2.0


def test_ghost_corridor_catastrophic_distance_is_diagnostic_not_terminal(tmp_path: Path) -> None:
    reward = GhostBundleReward(
        manifest_path=_build_reward_bundle(tmp_path),
        reward_config=RewardConfig(
            failure_countdown=100,
            min_steps=100,
            corridor_soft_margin_m=1.0,
            corridor_hard_margin_m=5.0,
            corridor_catastrophic_distance_m=10.0,
        ),
        ghost_config=GhostConfig(line_switch_hysteresis=0),
    )
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    step = reward.evaluate(_frame(1.0, z=20.0))

    assert step.done_type is None
    assert step.done_reason is None
    assert step.info["corridor_catastrophic"] is True


def test_ghost_line_switch_rejects_distant_candidate(tmp_path: Path) -> None:
    reward = GhostBundleReward(
        manifest_path=_build_reward_bundle(tmp_path, offsets=(0.0, 100.0)),
        reward_config=RewardConfig(
            failure_countdown=100,
            min_steps=100,
            corridor_mode="map_calibrated",
            corridor_soft_margin_m=25.0,
            corridor_hard_margin_m=90.0,
            corridor_catastrophic_distance_m=350.0,
            line_switch_max_distance_m=10.0,
        ),
        ghost_config=GhostConfig(line_switch_hysteresis=0),
    )
    reward.reset(run_id="run", initial_position=(0.0, 0.0, 0.0))

    step = reward.evaluate(_frame(2.0, z=80.0))

    assert step.info["ghost_line_rank"] == 1
    assert step.info["ghost_line_switch_count"] == 0
    assert step.info["corridor_distance_m"] > 70.0


def test_ghost_action_missing_fails_closed_for_offline_seed(tmp_path: Path) -> None:
    export_path = tmp_path / "rank_002.json"
    rows_without_actions = []
    for row in _rows():
        copy = dict(row)
        copy.pop("throttle", None)
        copy.pop("brake", None)
        copy.pop("steer", None)
        rows_without_actions.append(copy)
    write_json(export_path, {"rows": rows_without_actions})
    metadata_path = extract_openplanet_export(
        export_path,
        output_dir=tmp_path / "trajectories",
        replay_metadata={"trajectory_id": "rank_002", "map_uid": "map", "rank": 2},
    )
    result = build_ghost_bundle(
        map_uid="map",
        trajectory_metadata_paths=[metadata_path],
        output_dir=tmp_path / "bundle",
        bands=("1-10",),
        max_representatives_per_band=1,
    )
    replay = ReplayBuffer(mode="full", capacity=8, observation_shape=(4, 64, 64), telemetry_dim=TELEMETRY_DIM)
    with pytest.raises(RuntimeError, match="validated action channels"):
        seed_replay_from_ghost_bundle(replay, result.manifest_path, require_actions=True)


class _FakeNadeoClient(NadeoServicesClient):
    def __init__(self, output: Path) -> None:
        super().__init__(
            NadeoCredentials(
                dedicated_login="login",
                dedicated_password="password",
                user_agent="tm20ai-tests",
                core_token="core",
                live_token="live",
            )
        )
        self.output = output

    def resolve_map_uid(self, map_uid: str) -> dict:
        return {"uid": map_uid, "mapId": "map-id"}

    def leaderboard_top(self, *, map_uid: str, group_uid: str, length: int, only_world: bool) -> list[dict]:
        return [{"rank": 1, "accountId": "account", "score": 1234}]

    def map_records_by_accounts(self, *, map_id: str, account_ids: list[str]) -> list[dict]:
        return [{"accountId": "account", "replayUrl": "https://example.invalid/replay.gbx"}]

    def download_replay(self, url: str, destination: str | Path) -> dict:
        path = Path(destination)
        path.write_bytes(b"gbx")
        return {"ok": True, "status": 200, "path": str(path), "sha256": "hash", "bytes": 3}


def test_fetch_top100_ghost_manifest_with_mock_client(tmp_path: Path) -> None:
    manifest_path = fetch_top100_ghost_manifest(
        map_uid="map",
        output_dir=tmp_path,
        leaderboard_length=1,
        client=_FakeNadeoClient(tmp_path),
    )
    manifest = read_json(manifest_path)

    assert manifest["map_uid"] == "map"
    assert manifest["entries"][0]["rank"] == 1
    assert manifest["entries"][0]["fetch_status"]["ok"] is True


def test_route_aware_bundle_selects_intended_family_and_emits_exploit_bundle(tmp_path: Path) -> None:
    author_reference = _write_route_metadata(
        tmp_path,
        name="author_reference",
        rank=999,
        points=_straight_points(),
        record_time_ms=2500,
    )

    result, manifest = _route_family_bundle(tmp_path, author_reference_manifest=author_reference)

    assert result.intended_manifest_path is not None
    assert result.exploit_manifest_path is not None
    assert result.mixed_fallback_manifest_path is None
    assert manifest["selected_training_family"] == "intended_route"
    assert manifest["mixed_fallback"] is False
    assert manifest["bundle_resolution_mode"] == "intended_route"
    assert manifest["canonical_reference_source"] == "author_reference_manifest"
    selected_ranks = {int(item["rank"]) for item in manifest["selected_trajectories"]}
    assert selected_ranks <= {15, 22, 28}
    assert 1 not in selected_ranks
    assert 5 not in selected_ranks

    strategy_manifest = read_json(result.strategy_manifest_path)
    assert strategy_manifest["selected_training_family"] == "intended_route"
    assert strategy_manifest["bundle_resolution_mode"] == "intended_route"
    assert strategy_manifest["strategy_family_counts"]["intended_route"] == 3
    assert strategy_manifest["strategy_family_counts"]["shortcut_or_exploit"] == 2
    assert strategy_manifest["strategy_family_counts"]["unclassified"] == 1

    exploit_manifest = load_ghost_bundle_manifest(result.exploit_manifest_path)
    exploit_ranks = {int(item["rank"]) for item in exploit_manifest["selected_trajectories"]}
    assert exploit_ranks <= {1, 5}

    seeded = seed_replay_from_ghost_bundle(
        ReplayBuffer(mode="full", capacity=32, observation_shape=(4, 64, 64), telemetry_dim=TELEMETRY_DIM),
        result.manifest_path,
        require_actions=False,
    )
    assert seeded["selected_training_family"] == "intended_route"
    assert seeded["mixed_fallback"] is False


def test_route_aware_bundle_uses_reward_trajectory_fallback_when_author_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reward_reference_path = _write_reward_reference_npz(tmp_path / "reward_reference" / "trajectory_0p5m.npz")
    _result, manifest = _route_family_bundle(
        tmp_path,
        author_reference_manifest=None,
        monkeypatch=monkeypatch,
        reward_reference_path=reward_reference_path,
    )

    assert manifest["canonical_reference_source"] == "reward_trajectory_fallback"
    assert manifest["canonical_reference_path"] == str(reward_reference_path.resolve())
    assert manifest["strategy_classification_status"] == "classified_with_reward_fallback"
    assert manifest["mixed_fallback"] is False
    assert manifest["selected_training_family"] == "intended_route"
    assert manifest["bundle_resolution_mode"] == "intended_route"


def test_route_aware_bundle_uses_selected_ghost_override_when_intended_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_reference_path = tmp_path / "missing_reward_reference.npz"
    result, manifest = _route_family_bundle(
        tmp_path,
        author_reference_manifest=None,
        monkeypatch=monkeypatch,
        reward_reference_path=missing_reference_path,
        selected_ghost_selector={"ghost_name_contains": "exploit_rank_001"},
    )

    assert result.selected_override_manifest_path is not None
    assert manifest["bundle_resolution_mode"] == "selected_ghost_override"
    assert manifest["selected_training_family"] == "unclassified"
    assert manifest["mixed_fallback"] is False
    assert manifest["strategy_classification_status"] == "unavailable_canonical_reference"
    assert manifest["resolved_selected_ghost_rank"] == 1
    assert manifest["resolved_selected_ghost_name"] == "exploit_rank_001"
    assert manifest["selected_ghost_selector"] == {"ghost_name_contains": "exploit_rank_001"}
    assert [int(item["rank"]) for item in manifest["selected_trajectories"]] == [1]

    seeded = seed_replay_from_ghost_bundle(
        ReplayBuffer(mode="full", capacity=64, observation_shape=(4, 64, 64), telemetry_dim=TELEMETRY_DIM),
        tmp_path / "route_bundle" / "ghost_bundle_manifest.json",
        require_actions=False,
    )
    assert seeded["selected_training_family"] == "unclassified"
    assert seeded["bundle_resolution_mode"] == "selected_ghost_override"
    assert seeded["resolved_selected_ghost_rank"] == 1


def test_route_aware_bundle_uses_selected_ghost_rank_fallback_when_name_does_not_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_reference_path = tmp_path / "missing_reward_reference.npz"
    _result, manifest = _route_family_bundle(
        tmp_path,
        author_reference_manifest=None,
        monkeypatch=monkeypatch,
        reward_reference_path=missing_reference_path,
        selected_ghost_selector={"ghost_name_contains": "does_not_exist", "rank": 5},
    )

    assert manifest["bundle_resolution_mode"] == "selected_ghost_override"
    assert manifest["resolved_selected_ghost_rank"] == 5
    assert manifest["selected_ghost_selector"] == {
        "ghost_name_contains": "does_not_exist",
        "rank": 5,
    }
    assert [int(item["rank"]) for item in manifest["selected_trajectories"]] == [5]


def test_route_aware_bundle_resolves_selected_ghost_by_substring_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_reference_path = tmp_path / "missing_reward_reference.npz"
    _result, manifest = _route_family_bundle(
        tmp_path,
        author_reference_manifest=None,
        monkeypatch=monkeypatch,
        reward_reference_path=missing_reference_path,
        selected_ghost_selector={"ghost_name_contains": "rank_001"},
    )

    assert manifest["bundle_resolution_mode"] == "selected_ghost_override"
    assert manifest["resolved_selected_ghost_rank"] == 1
    assert [int(item["rank"]) for item in manifest["selected_trajectories"]] == [1]


def test_route_aware_bundle_fails_on_ambiguous_selected_ghost_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_reference_path = tmp_path / "missing_reward_reference.npz"

    with pytest.raises(RuntimeError, match="matched multiple ghosts by substring"):
        _route_family_bundle(
            tmp_path,
            author_reference_manifest=None,
            monkeypatch=monkeypatch,
            reward_reference_path=missing_reference_path,
            selected_ghost_selector={"ghost_name_contains": "rank_"},
        )

    strategy_manifest = read_json(tmp_path / "route_bundle" / "ghost_strategy_manifest.json")
    assert strategy_manifest["bundle_resolution_mode"] == "error_no_default_bundle"
    assert strategy_manifest["selected_ghost_selector"] == {"ghost_name_contains": "rank_"}


def test_route_aware_bundle_uses_author_reference_fallback_when_selected_override_is_unavailable(tmp_path: Path) -> None:
    author_reference = _write_route_metadata(
        tmp_path,
        name="author_reference",
        rank=999,
        points=_straight_points(),
        record_time_ms=2500,
    )
    exploit_paths = [
        _write_route_metadata(
            tmp_path,
            name="exploit_rank_001",
            rank=1,
            points=[(0.0, 0.0, 0.0), (-5.0, 0.0, 0.0), (-10.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
            record_time_ms=1500,
        ),
        _write_route_metadata(
            tmp_path,
            name="exploit_rank_005",
            rank=5,
            points=[(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 0.0, 40.0), (6.0, 0.0, 70.0)],
            record_time_ms=1600,
        ),
    ]

    result = build_ghost_bundle(
        map_uid="map",
        trajectory_metadata_paths=exploit_paths,
        output_dir=tmp_path / "author_fallback_bundle",
        spacing_meters=0.5,
        ghost_config=GhostConfig(
            anchor_count=8,
            anchor_radius_m=6.0,
            canonical_divergence_radius_m=12.0,
            intended_candidate_pool=3,
            intended_bundle_size=2,
            exploit_bundle_size=1,
        ),
        author_reference_manifest=str(author_reference),
        selected_ghost_selector={"ghost_name_contains": "missing_ghost"},
        bands=("1-10", "11-30", "31-60"),
        max_representatives_per_band=2,
    )
    manifest = load_ghost_bundle_manifest(result.manifest_path)

    assert result.author_fallback_manifest_path is not None
    assert manifest["bundle_resolution_mode"] == "author_reference_fallback"
    assert manifest["author_fallback_used"] is True
    assert manifest["selected_training_family"] == "author_reference_fallback"
    assert manifest["selected_ghost_selector"] == {"ghost_name_contains": "missing_ghost"}
    assert [int(item["rank"]) for item in manifest["selected_trajectories"]] == [999]


def test_route_aware_bundle_errors_when_no_default_bundle_can_be_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_reference_path = tmp_path / "missing_reward_reference.npz"

    with pytest.raises(RuntimeError, match="No intended-route bundle could be built"):
        _route_family_bundle(
            tmp_path,
            author_reference_manifest=None,
            monkeypatch=monkeypatch,
            reward_reference_path=missing_reference_path,
            selected_ghost_selector=None,
        )

    strategy_manifest = read_json(tmp_path / "route_bundle" / "ghost_strategy_manifest.json")
    assert strategy_manifest["bundle_resolution_mode"] == "error_no_default_bundle"
    assert strategy_manifest["default_bundle_manifest_path"] is None


def test_build_reference_target_bundle_selects_rank_range(tmp_path: Path) -> None:
    author_reference = _write_route_metadata(
        tmp_path,
        name="author_reference",
        rank=999,
        points=_straight_points(),
        record_time_ms=2500,
    )
    rank_paths = [
        _write_route_metadata(
            tmp_path,
            name=f"rank_{rank:03d}",
            rank=rank,
            points=_straight_points(z_offset=float(rank)),
            record_time_ms=2_000 + rank,
        )
        for rank in range(1, 13)
    ]

    result = build_reference_target_bundle(
        map_uid="map",
        trajectory_metadata_paths=rank_paths,
        output_dir=tmp_path / "rank_bundle",
        manifest_name="ghost_bundle_rank_011_100.json",
        selected_training_family="rank_011_100_bundle",
        bundle_resolution_mode="manual_rank_range_11_100",
        strategy_classification_status="manual_rank_range_11_100",
        rank_min=11,
        rank_max=100,
        spacing_meters=0.5,
        ghost_config=GhostConfig(
            anchor_count=8,
            anchor_radius_m=6.0,
            canonical_divergence_radius_m=12.0,
            intended_candidate_pool=3,
            intended_bundle_size=2,
            exploit_bundle_size=1,
        ),
        author_reference_manifest=str(author_reference),
        bands=("1-10", "11-30", "31-60"),
        max_representatives_per_band=2,
        set_default_alias=True,
    )

    manifest = load_ghost_bundle_manifest(result.manifest_path)
    assert manifest["bundle_resolution_mode"] == "manual_rank_range_11_100"
    assert manifest["selected_training_family"] == "rank_011_100_bundle"
    assert [int(item["rank"]) for item in manifest["selected_trajectories"]] == [11, 12]
    assert read_json(tmp_path / "rank_bundle" / "ghost_bundle_manifest.json")["default_bundle_alias_of"] == str(
        result.manifest_path
    )


def test_route_aware_bundle_keeps_intended_selection_when_fast_unclassified_candidates_exist(tmp_path: Path) -> None:
    author_reference = _write_route_metadata(
        tmp_path,
        name="author_reference",
        rank=999,
        points=_straight_points(),
        record_time_ms=2500,
    )
    intended_paths = [
        _write_route_metadata(
            tmp_path,
            name="intended_rank_015",
            rank=15,
            points=_straight_points(z_offset=0.0),
            record_time_ms=2200,
        ),
        _write_route_metadata(
            tmp_path,
            name="intended_rank_022",
            rank=22,
            points=_straight_points(z_offset=1.0),
            record_time_ms=2300,
        ),
    ]
    ambiguous_path = _write_route_metadata(
        tmp_path,
        name="ambiguous_rank_012",
        rank=12,
        points=_straight_points(stop_x=30.0, z_offset=0.0),
        record_time_ms=1800,
    )

    result = build_ghost_bundle(
        map_uid="map",
        trajectory_metadata_paths=[*intended_paths, ambiguous_path],
        output_dir=tmp_path / "ambiguous_bundle",
        spacing_meters=0.5,
        ghost_config=GhostConfig(
            anchor_count=8,
            anchor_radius_m=6.0,
            canonical_divergence_radius_m=12.0,
            intended_candidate_pool=3,
            intended_bundle_size=2,
            exploit_bundle_size=1,
        ),
        author_reference_manifest=str(author_reference),
        bands=("1-10", "11-30", "31-60"),
        max_representatives_per_band=2,
    )
    manifest = load_ghost_bundle_manifest(result.manifest_path)

    assert manifest["selected_training_family"] == "intended_route"
    assert manifest["mixed_fallback"] is False
    assert manifest["bundle_resolution_mode"] == "intended_route"
    assert result.mixed_fallback_manifest_path is None


def test_rt_interface_prefers_intended_bundle_manifest_by_default(tmp_path: Path) -> None:
    ghost_root = tmp_path / "ghosts"
    map_root = ghost_root / "map"
    map_root.mkdir(parents=True, exist_ok=True)
    intended_manifest = map_root / "ghost_bundle_intended.json"
    write_json(intended_manifest, {"schema_version": "ghost_bundle_v1"})

    interface = TM20AIRtInterface.__new__(TM20AIRtInterface)
    interface.repo_root = tmp_path
    interface.config = SimpleNamespace(ghosts=GhostConfig(root=str(ghost_root), bundle_manifest=None))

    assert interface._ghost_bundle_manifest_for_map("map") == intended_manifest.resolve()

    intended_manifest.unlink()
    selected_override_manifest = map_root / "ghost_bundle_selected_override.json"
    write_json(selected_override_manifest, {"schema_version": "ghost_bundle_v1"})
    assert interface._ghost_bundle_manifest_for_map("map") == selected_override_manifest.resolve()

    explicit_manifest = map_root / "explicit_bundle.json"
    write_json(explicit_manifest, {"schema_version": "ghost_bundle_v1"})
    interface.config = SimpleNamespace(ghosts=GhostConfig(root=str(ghost_root), bundle_manifest=str(explicit_manifest)))

    assert interface._ghost_bundle_manifest_for_map("map") == explicit_manifest.resolve()
