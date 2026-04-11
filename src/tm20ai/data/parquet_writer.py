from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import TM20AIConfig


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_artifact_root(config: TM20AIConfig) -> Path:
    root = Path(config.artifacts.root)
    if root.is_absolute():
        return root
    return repo_root() / root


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_parquet_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "run"


@dataclass(slots=True, frozen=True)
class RunArtifactPaths:
    run_dir: Path
    episodes_dir: Path
    summary_json: Path
    episode_index_parquet: Path
    tensorboard_dir: Path


@dataclass(slots=True, frozen=True)
class EpisodeArtifactPaths:
    steps_parquet: Path
    metadata_json: Path
    frames_dir: Path | None
    video_path: Path | None
    observation_npz: Path | None


def build_run_artifact_paths(config: TM20AIConfig, *, mode: str, run_name: str) -> RunArtifactPaths:
    root = resolve_artifact_root(config)
    run_dir = ensure_directory(root / mode / run_name)
    episodes_dir = ensure_directory(run_dir / "episodes")
    return RunArtifactPaths(
        run_dir=run_dir,
        episodes_dir=episodes_dir,
        summary_json=run_dir / "summary.json",
        episode_index_parquet=run_dir / "episode_index.parquet",
        tensorboard_dir=ensure_directory(run_dir / "tensorboard"),
    )


def build_episode_artifact_paths(
    episodes_dir: Path,
    *,
    episode_id: str,
    record_video: bool,
    record_observation_sidecar: bool = False,
) -> EpisodeArtifactPaths:
    frames_dir = episodes_dir / f"{episode_id}_frames" if record_video else None
    if frames_dir is not None:
        ensure_directory(frames_dir)
    return EpisodeArtifactPaths(
        steps_parquet=episodes_dir / f"{episode_id}.parquet",
        metadata_json=episodes_dir / f"{episode_id}.json",
        frames_dir=frames_dir,
        video_path=None,
        observation_npz=episodes_dir / f"{episode_id}_observations.npz" if record_observation_sidecar else None,
    )
