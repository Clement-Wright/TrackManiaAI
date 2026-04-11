from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from .parquet_writer import read_json


@dataclass(slots=True, frozen=True)
class EpisodeRecord:
    episode_id: str
    steps_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    observation_npz_path: Path | None = None


class EpisodeDataset:
    """Read-only index over saved demo/eval episode artifacts."""

    def __init__(self, root: Path):
        self.root = root
        self._episodes = self._discover(root)

    @staticmethod
    def _discover(root: Path) -> list[EpisodeRecord]:
        if not root.exists():
            return []
        records: list[EpisodeRecord] = []
        for metadata_path in sorted(root.rglob("episodes/*.json")):
            metadata = read_json(metadata_path)
            steps_path = metadata_path.with_suffix(".parquet")
            if not steps_path.exists():
                continue
            sidecar_path = metadata_path.with_name(f"{metadata_path.stem}_observations.npz")
            records.append(
                EpisodeRecord(
                    episode_id=str(metadata.get("episode_id", metadata_path.stem)),
                    steps_path=steps_path,
                    metadata_path=metadata_path,
                    metadata=metadata,
                    observation_npz_path=sidecar_path if sidecar_path.exists() else None,
                )
            )
        return records

    @property
    def episodes(self) -> list[EpisodeRecord]:
        return list(self._episodes)

    def load_episode_steps(self, episode: EpisodeRecord) -> list[dict[str, Any]]:
        table = pq.read_table(episode.steps_path)
        return table.to_pylist()


@dataclass(slots=True, frozen=True)
class DemoDatasetSplit:
    train_episodes: list[EpisodeRecord]
    validation_episodes: list[EpisodeRecord]


def split_demo_dataset(
    root: Path,
    *,
    validation_fraction: float,
    seed: int,
) -> DemoDatasetSplit:
    dataset = EpisodeDataset(root)
    candidates = [episode for episode in dataset.episodes if episode.observation_npz_path is not None]
    if not candidates:
        return DemoDatasetSplit(train_episodes=[], validation_episodes=[])
    rng = np.random.default_rng(seed)
    order = np.arange(len(candidates))
    rng.shuffle(order)
    shuffled = [candidates[index] for index in order.tolist()]
    validation_count = max(1, int(round(len(shuffled) * validation_fraction))) if len(shuffled) > 1 else 0
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:] or shuffled
    if not train and validation:
        train = validation
        validation = []
    return DemoDatasetSplit(train_episodes=train, validation_episodes=validation)


class FullBehaviorCloningDataset(Dataset):
    def __init__(self, episodes: list[EpisodeRecord]) -> None:
        self._samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for episode in episodes:
            if episode.observation_npz_path is None:
                continue
            payload = np.load(episode.observation_npz_path)
            observations = payload["obs_uint8"]
            telemetry = payload["telemetry_float"]
            actions = payload["action"]
            for obs, telem, action in zip(observations, telemetry, actions, strict=True):
                self._samples.append(
                    (
                        np.asarray(obs, dtype=np.uint8),
                        np.asarray(telem, dtype=np.float32),
                        np.asarray(action, dtype=np.float32),
                    )
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._samples[index]


def seed_replay_from_demo_sidecars(replay, root: Path) -> int:  # noqa: ANN001
    dataset = EpisodeDataset(root)
    if getattr(replay, "mode", None) != "full":
        return 0
    seeded = 0
    for episode in dataset.episodes:
        if episode.observation_npz_path is None:
            continue
        payload = np.load(episode.observation_npz_path)
        observations = payload["obs_uint8"]
        telemetry = payload["telemetry_float"]
        actions = payload["action"]
        rows = pq.read_table(episode.steps_path).to_pylist()
        if len(observations) != len(rows):
            continue
        for index, row in enumerate(rows):
            next_index = min(index + 1, len(rows) - 1)
            replay.add(
                {
                    "obs_uint8": observations[index],
                    "telemetry_float": telemetry[index],
                    "action": actions[index],
                    "reward": float(row["reward"]),
                    "next_obs_uint8": observations[next_index],
                    "next_telemetry_float": telemetry[next_index],
                    "terminated": row.get("done_type") == "terminated",
                    "truncated": row.get("done_type") == "truncated",
                    "episode_id": episode.episode_id,
                    "map_uid": episode.metadata.get("map_uid"),
                    "step_idx": int(row.get("step_index", index)),
                }
            )
            seeded += 1
    return seeded
