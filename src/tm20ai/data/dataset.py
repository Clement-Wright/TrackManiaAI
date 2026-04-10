from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .parquet_writer import read_json


@dataclass(slots=True, frozen=True)
class EpisodeRecord:
    episode_id: str
    steps_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


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
            records.append(
                EpisodeRecord(
                    episode_id=str(metadata.get("episode_id", metadata_path.stem)),
                    steps_path=steps_path,
                    metadata_path=metadata_path,
                    metadata=metadata,
                )
            )
        return records

    @property
    def episodes(self) -> list[EpisodeRecord]:
        return list(self._episodes)

    def load_episode_steps(self, episode: EpisodeRecord) -> list[dict[str, Any]]:
        table = pq.read_table(episode.steps_path)
        return table.to_pylist()
