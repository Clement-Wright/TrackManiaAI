from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from ..action_space import ACTION_DIM, LEGACY_ACTION_DIM, clamp_action
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


@dataclass(slots=True, frozen=True)
class FullDemoDatasetValidation:
    root: Path
    episodes: list[EpisodeRecord]
    map_uid: str | None
    total_episode_count: int
    valid_episode_count: int
    sample_count: int
    total_nonzero_action_steps: int
    mean_abs_action: tuple[float, float]


def split_episode_records(
    episodes: list[EpisodeRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> DemoDatasetSplit:
    if not episodes:
        return DemoDatasetSplit(train_episodes=[], validation_episodes=[])
    rng = np.random.default_rng(seed)
    order = np.arange(len(episodes))
    rng.shuffle(order)
    shuffled = [episodes[index] for index in order.tolist()]
    validation_count = max(1, int(round(len(shuffled) * validation_fraction))) if len(shuffled) > 1 else 0
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:] or shuffled
    if not train and validation:
        train = validation
        validation = []
    return DemoDatasetSplit(train_episodes=train, validation_episodes=validation)


def split_demo_dataset(
    root: Path,
    *,
    validation_fraction: float,
    seed: int,
) -> DemoDatasetSplit:
    dataset = EpisodeDataset(root)
    candidates = [episode for episode in dataset.episodes if episode.observation_npz_path is not None]
    return split_episode_records(candidates, validation_fraction=validation_fraction, seed=seed)


def validate_full_demo_dataset(
    root: Path,
    *,
    allow_mixed_map_uids: bool = False,
) -> FullDemoDatasetValidation:
    dataset = EpisodeDataset(root)
    if not dataset.episodes:
        raise RuntimeError(f"No demo episodes were found under {root}.")

    valid_episodes: list[EpisodeRecord] = []
    invalid_messages: list[str] = []
    map_uids: set[str] = set()
    sample_count = 0
    total_nonzero_action_steps = 0
    total_abs_action = np.zeros(ACTION_DIM, dtype=np.float64)
    for episode in dataset.episodes:
        observation_mode = str(episode.metadata.get("observation_mode", "")).strip().lower()
        if observation_mode != "full":
            invalid_messages.append(
                f"{episode.metadata_path}: observation_mode must be 'full', got {observation_mode!r}."
            )
            continue
        if episode.observation_npz_path is None:
            invalid_messages.append(
                f"{episode.metadata_path}: FULL demo is missing the observation sidecar NPZ."
            )
            continue
        map_uid = episode.metadata.get("map_uid")
        if map_uid in (None, ""):
            invalid_messages.append(f"{episode.metadata_path}: map_uid is missing from episode metadata.")
            continue
        map_uids.add(str(map_uid))
        payload = np.load(episode.observation_npz_path)
        actions = np.asarray(payload["action"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] not in {ACTION_DIM, LEGACY_ACTION_DIM}:
            invalid_messages.append(
                f"{episode.observation_npz_path}: expected action sidecar shape (N, {ACTION_DIM}) or legacy (N, {LEGACY_ACTION_DIM}), got {actions.shape}."
            )
            continue
        if actions.shape[1] != ACTION_DIM:
            actions = np.stack([clamp_action(action) for action in actions], axis=0)
        valid_episodes.append(episode)
        sample_count += int(actions.shape[0])
        total_nonzero_action_steps += int(np.count_nonzero(np.any(np.abs(actions) > 1.0e-6, axis=1)))
        total_abs_action += np.abs(actions).sum(axis=0, dtype=np.float64)

    if invalid_messages:
        raise RuntimeError("FULL demo dataset validation failed:\n- " + "\n- ".join(invalid_messages))
    if not valid_episodes:
        raise RuntimeError(f"No valid FULL demo episodes were found under {root}.")
    if not allow_mixed_map_uids and len(map_uids) > 1:
        raise RuntimeError(
            "FULL demo dataset validation failed: demos span multiple map_uids: "
            + ", ".join(sorted(map_uids))
        )
    if total_nonzero_action_steps <= 0:
        raise RuntimeError("FULL demo dataset validation failed: demos contain no non-zero recorded actions.")

    map_uid = next(iter(sorted(map_uids)), None)
    return FullDemoDatasetValidation(
        root=root,
        episodes=valid_episodes,
        map_uid=map_uid,
        total_episode_count=len(dataset.episodes),
        valid_episode_count=len(valid_episodes),
        sample_count=sample_count,
        total_nonzero_action_steps=total_nonzero_action_steps,
        mean_abs_action=tuple((total_abs_action / max(1, sample_count)).astype(float).tolist()),
    )


class FullBehaviorCloningDataset(Dataset):
    def __init__(self, episodes: list[EpisodeRecord]) -> None:
        self._samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for episode in episodes:
            if episode.observation_npz_path is None:
                continue
            payload = np.load(episode.observation_npz_path)
            observations = payload["obs_uint8"]
            telemetry = payload["telemetry_float"]
            actions = np.asarray(payload["action"], dtype=np.float32)
            for obs, telem, action in zip(observations, telemetry, actions, strict=True):
                self._samples.append(
                    (
                        np.asarray(obs, dtype=np.uint8),
                        np.asarray(telem, dtype=np.float32),
                        clamp_action(action),
                    )
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._samples[index]


def seed_replay_from_demo_sidecars(replay, root: Path) -> int:  # noqa: ANN001
    validation = validate_full_demo_dataset(root)
    if getattr(replay, "mode", None) != "full":
        return 0
    seeded = 0
    for episode in validation.episodes:
        payload = np.load(episode.observation_npz_path)
        observations = payload["obs_uint8"]
        telemetry = payload["telemetry_float"]
        actions = np.asarray(payload["action"], dtype=np.float32)
        rows = pq.read_table(episode.steps_path).to_pylist()
        if len(observations) != len(rows):
            continue
        for index, row in enumerate(rows):
            next_index = min(index + 1, len(rows) - 1)
            replay.add(
                {
                    "obs_uint8": observations[index],
                    "telemetry_float": telemetry[index],
                    "action": clamp_action(actions[index]),
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
