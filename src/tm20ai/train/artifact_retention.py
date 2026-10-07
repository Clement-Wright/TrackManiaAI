from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..data.parquet_writer import read_json


@dataclass(slots=True, frozen=True)
class ArtifactCleanupResult:
    kept_paths: tuple[str, ...]
    removed_paths: tuple[str, ...]
    kept_bytes: int = 0
    removed_bytes: int = 0
    artifact_root_bytes_before: int = 0
    artifact_root_bytes_after: int = 0


@dataclass(slots=True, frozen=True)
class StoragePreflightResult:
    artifact_root: str
    free_bytes: int
    min_free_bytes: int
    artifact_root_bytes: int
    max_artifact_bytes: int | None
    ok: bool
    reasons: tuple[str, ...]


def bytes_from_gb(value: float | int | None) -> int | None:
    if value is None:
        return None
    return int(float(value) * 1024 * 1024 * 1024)


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    value_float = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while abs(value_float) >= 1024.0 and unit_index < len(units) - 1:
        value_float /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value_float)} {units[unit_index]}"
    return f"{value_float:.2f} {units[unit_index]}"


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        try:
            return int(root.stat().st_size)
        except OSError:
            return 0
    total = 0
    for child in root.rglob("*"):
        try:
            if child.is_file():
                total += int(child.stat().st_size)
        except OSError:
            continue
    return total


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _assert_safe_cleanup_root(artifact_root: str | Path, *, repo_root: str | Path | None = None) -> Path:
    resolved_root = Path(artifact_root).resolve()
    resolved_repo_root = _repo_root() if repo_root is None else Path(repo_root).resolve()
    protected_roots = (
        resolved_repo_root,
        resolved_repo_root / "src",
        resolved_repo_root / "scripts",
        resolved_repo_root / "tests",
        resolved_repo_root / "configs",
        resolved_repo_root / "docs",
        resolved_repo_root / "results",
        resolved_repo_root / "data",
        resolved_repo_root / "OpenplanetPlugin",
    )
    for protected in protected_roots:
        if resolved_root == protected:
            raise ValueError(f"Refusing to clean protected path: {resolved_root}")
        if protected != resolved_repo_root and _is_relative_to(resolved_root, protected):
            raise ValueError(f"Refusing to clean inside protected path {protected}: {resolved_root}")
    return resolved_root


def storage_preflight_report(
    artifact_root: str | Path,
    *,
    min_free_gb: float = 150.0,
    max_artifact_gb: float | None = 150.0,
    disk_usage_fn: Callable[[str | Path], Any] = shutil.disk_usage,
) -> StoragePreflightResult:
    resolved_root = Path(artifact_root).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    min_free_bytes = bytes_from_gb(min_free_gb) or 0
    max_artifact_bytes = bytes_from_gb(max_artifact_gb)
    disk_usage = disk_usage_fn(resolved_root)
    if hasattr(disk_usage, "free"):
        free_bytes = int(disk_usage.free)
    else:
        free_bytes = int(disk_usage[2])
    artifact_root_bytes = directory_size_bytes(resolved_root)
    reasons: list[str] = []
    if free_bytes < min_free_bytes:
        reasons.append(
            f"free_space_below_minimum:{format_bytes(free_bytes)}<{format_bytes(min_free_bytes)}"
        )
    if max_artifact_bytes is not None and artifact_root_bytes > max_artifact_bytes:
        reasons.append(
            f"artifact_root_above_quota:{format_bytes(artifact_root_bytes)}>{format_bytes(max_artifact_bytes)}"
        )
    return StoragePreflightResult(
        artifact_root=str(resolved_root),
        free_bytes=free_bytes,
        min_free_bytes=min_free_bytes,
        artifact_root_bytes=artifact_root_bytes,
        max_artifact_bytes=max_artifact_bytes,
        ok=not reasons,
        reasons=tuple(reasons),
    )


def enforce_storage_preflight(
    artifact_root: str | Path,
    *,
    min_free_gb: float = 150.0,
    max_artifact_gb: float | None = 150.0,
    disk_usage_fn: Callable[[str | Path], Any] = shutil.disk_usage,
) -> StoragePreflightResult:
    report = storage_preflight_report(
        artifact_root,
        min_free_gb=min_free_gb,
        max_artifact_gb=max_artifact_gb,
        disk_usage_fn=disk_usage_fn,
    )
    if not report.ok:
        raise RuntimeError(
            "Storage preflight failed for "
            f"{report.artifact_root}: {', '.join(report.reasons)}. "
            "Clean artifacts or override the storage thresholds explicitly."
        )
    return report


def _best_progress(summary: dict[str, Any]) -> float:
    exact_final_eval = dict(summary.get("exact_final_eval_summary") or {})
    if bool(summary.get("exact_final_eval_complete", False)) and exact_final_eval:
        return float(exact_final_eval.get("mean_final_progress_index", 0.0) or 0.0)
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
    run_names = [Path(run_dir).name for run_dir in keeper_run_dirs]
    return sorted(
        path
        for path in eval_root.iterdir()
        if path.is_dir()
        and any(path.name == run_name or path.name.startswith(f"{run_name}_") for run_name in run_names)
    )


def cleanup_artifact_root(
    artifact_root: str | Path,
    *,
    keep_run_dirs: Sequence[str | Path],
    dry_run: bool = False,
) -> ArtifactCleanupResult:
    resolved_root = _assert_safe_cleanup_root(artifact_root)
    keep_train = {Path(path).resolve() for path in keep_run_dirs}
    keep_eval = {path.resolve() for path in referenced_eval_dirs(resolved_root, keep_train)}
    keep_paths = keep_train | keep_eval
    removed_paths: list[str] = []
    artifact_root_bytes_before = directory_size_bytes(resolved_root)
    removed_bytes = 0

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
            removed_bytes += directory_size_bytes(resolved_child)
            removed_paths.append(str(resolved_child))
            if not dry_run:
                shutil.rmtree(resolved_child)

    benchmarks_root = resolved_root / "benchmarks"
    if benchmarks_root.exists():
        benchmark_files = sorted(benchmarks_root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for stale_file in benchmark_files[2:]:
            removed_bytes += directory_size_bytes(stale_file)
            removed_paths.append(str(stale_file.resolve()))
            if not dry_run:
                stale_file.unlink()

    kept_bytes = sum(directory_size_bytes(path) for path in keep_paths)
    artifact_root_bytes_after = artifact_root_bytes_before if dry_run else directory_size_bytes(resolved_root)
    return ArtifactCleanupResult(
        kept_paths=tuple(str(path) for path in sorted(keep_paths)),
        removed_paths=tuple(removed_paths),
        kept_bytes=kept_bytes,
        removed_bytes=removed_bytes,
        artifact_root_bytes_before=artifact_root_bytes_before,
        artifact_root_bytes_after=artifact_root_bytes_after,
    )
