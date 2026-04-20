from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tm20ai.bridge import TelemetryFrame
from tm20ai.config import GhostConfig, RewardConfig
from tm20ai.data.parquet_writer import read_json, write_json
from tm20ai.ghosts.dataset import build_ghost_bundle, extract_openplanet_export
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


def _frame(x: float, z: float = 0.0, *, finished: bool = False) -> TelemetryFrame:
    return TelemetryFrame(
        session_id="session",
        run_id="run",
        frame_id=int(x),
        timestamp_ns=0,
        map_uid="map",
        race_time_ms=int(x * 50),
        cp_count=0,
        cp_target=0,
        speed_kmh=50.0,
        gear=2,
        rpm=1000.0,
        pos_xyz=(x, 0.0, z),
        vel_xyz=(1.0, 0.0, 0.0),
        yaw_pitch_roll=(0.0, 0.0, 0.0),
        finished=finished,
        terminal_reason="finished" if finished else None,
    )


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
            "observation_npz_path": str(obs_npz),
        },
    )
    metadata = read_json(metadata_path)
    assert metadata["action_channel_valid"] is True
    assert metadata["offline_transition_count"] == 5

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
