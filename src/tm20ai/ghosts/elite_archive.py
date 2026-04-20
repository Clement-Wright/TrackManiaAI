from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..config import EliteArchiveConfig
from ..data.parquet_writer import read_json, write_json


class EliteArchive:
    """Small manifest-backed archive for high-progress or novel self-generated rollouts."""

    def __init__(self, *, config: EliteArchiveConfig, run_name: str) -> None:
        self.config = config
        self.root = Path(config.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "elite_archive_manifest.json"
        self.run_name = run_name

    def _load(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return read_json(self.manifest_path)
        return {"schema_version": "elite_archive_v1", "entries": []}

    @staticmethod
    def _cluster_key(summary: Mapping[str, Any]) -> str:
        progress = float(summary.get("mean_final_progress_index") or 0.0)
        completion = float(summary.get("completion_rate") or 0.0)
        steer = float(summary.get("mean_abs_steer") or 0.0)
        return f"progress_{int(progress // 250):03d}_completion_{int(completion * 10):02d}_steer_{int(steer * 10):02d}"

    def maybe_promote(
        self,
        *,
        summary: Mapping[str, Any],
        run_dir: str | None,
        mode: str,
        checkpoint_path: str | None,
    ) -> dict[str, Any]:
        manifest = self._load()
        entries = list(manifest.get("entries", []))
        progress = float(summary.get("mean_final_progress_index") or 0.0)
        best_progress = max([float(entry.get("mean_final_progress_index") or 0.0) for entry in entries] or [0.0])
        cluster_key = self._cluster_key(summary)
        novel_cluster = cluster_key not in {str(entry.get("cluster_key")) for entry in entries}
        improved = progress >= best_progress + float(self.config.min_progress_improvement)
        promoted = bool(improved or novel_cluster)
        if promoted:
            entries.append(
                {
                    "run_name": self.run_name,
                    "mode": mode,
                    "run_dir": run_dir,
                    "checkpoint_path": checkpoint_path,
                    "mean_final_progress_index": progress,
                    "completion_rate": summary.get("completion_rate"),
                    "cluster_key": cluster_key,
                    "promotion_reason": "progress_improvement" if improved else "novel_cluster",
                }
            )
            entries.sort(key=lambda item: float(item.get("mean_final_progress_index") or 0.0), reverse=True)
            manifest["entries"] = entries[: int(self.config.max_entries)]
            write_json(self.manifest_path, manifest)
        return {
            "promoted": promoted,
            "reason": "progress_improvement" if improved else "novel_cluster" if novel_cluster else "not_elite",
            "manifest_path": str(self.manifest_path),
            "cluster_key": cluster_key,
        }
