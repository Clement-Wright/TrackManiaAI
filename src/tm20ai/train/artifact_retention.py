from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..data.parquet_writer import read_json


@dataclass(slots=True, frozen=True)
class ArtifactCleanupResult:
    kept_paths: tuple[str, ...]
    removed_paths: tuple[str, ...]


def _best_progress(summary: dict[str, Any]) -> float:
    best = 0.0
    for entry in summary.get("eval_history", []):
        payload = dict(entry.get("summary", {}))
        best = max(best, float(payload.get("mean_final_progress_index", 0.0) or 0.0))
    latest = dict(summary.get("latest_eval_summary") or {})
    best = max(best, float(latest.get("mean_final_progress_index", 0.0) or 0.0))
    return best


def discover_training_run_dirs(artifact_root: str | Path) -> list[Path]:
    train_root = Path(artifact_root).resolve() / "train"
    if not train_root.exists():
        return []
    return sorted(path for path in train_root.iterdir() if (path / "summary.json").exists())


def select_keeper_run_dirs(
    artifact_root: str | Path,
    *,
    keep_best_per_algorithm: int = 1,
    keep_latest_per_algorithm: int = 1,
    keep_run_names: Sequence[str] = (),
) -> list[Path]:
    run_dirs = discover_training_run_dirs(artifact_root)
    by_algorithm: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for run_dir in run_dirs:
        summary = read_json(run_dir / "summary.json")
        algorithm = str(summary.get("algorithm") or "unknown")
        by_algorithm.setdefault(algorithm, []).append((run_dir, summary))

    keepers: set[Path] = set()
    requested = set(keep_run_names)
    for entries in by_algorithm.values():
        entries.sort(key=lambda item: str(item[1].get("run_end_timestamp") or item[0].stat().st_mtime), reverse=True)
        keepers.update(run_dir for run_dir, _summary in entries[: max(0, int(keep_latest_per_algorithm))])
        best_sorted = sorted(entries, key=lambda item: _best_progress(item[1]), reverse=True)
        keepers.update(run_dir for run_dir, _summary in best_sorted[: max(0, int(keep_best_per_algorithm))])
        keepers.update(
            run_dir
            for run_dir, summary in entries
            if str(summary.get("run_name")) in requested or run_dir.name in requested
        )
    return sorted(keepers)


def referenced_eval_dirs(artifact_root: str | Path, keeper_run_dirs: Iterable[str | Path]) -> list[Path]:
    eval_root = Path(artifact_root).resolve() / "eval"
    if not eval_root.exists():
        return []
    prefixes = [f"{Path(run_dir).name}_step_" for run_dir in keeper_run_dirs]
    return sorted(
        path
        for path in eval_root.iterdir()
        if path.is_dir() and any(path.name.startswith(prefix) for prefix in prefixes)
    )


def cleanup_artifact_root(
    artifact_root: str | Path,
    *,
    keep_run_dirs: Sequence[str | Path],
    dry_run: bool = False,
) -> ArtifactCleanupResult:
    resolved_root = Path(artifact_root).resolve()
    keep_train = {Path(path).resolve() for path in keep_run_dirs}
    keep_eval = {path.resolve() for path in referenced_eval_dirs(resolved_root, keep_train)}
    keep_paths = keep_train | keep_eval
    removed_paths: list[str] = []

    for subdir_name in ("train", "eval"):
        subdir = resolved_root / subdir_name
        if not subdir.exists():
            continue
        for child in sorted(subdir.iterdir()):
            resolved_child = child.resolve()
            if resolved_child in keep_paths:
                continue
            if not child.is_dir():
                continue
            removed_paths.append(str(resolved_child))
            if not dry_run:
                shutil.rmtree(resolved_child)

    benchmarks_root = resolved_root / "benchmarks"
    if benchmarks_root.exists():
        benchmark_files = sorted(benchmarks_root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for stale_file in benchmark_files[2:]:
            removed_paths.append(str(stale_file.resolve()))
            if not dry_run:
                stale_file.unlink()

    return ArtifactCleanupResult(
        kept_paths=tuple(str(path) for path in sorted(keep_paths)),
        removed_paths=tuple(removed_paths),
    )
