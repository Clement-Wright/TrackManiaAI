from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.data.parquet_writer import read_json, write_json
from tm20ai.ghosts.dataset import extract_openplanet_export


def _metadata_from_top_manifest(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    by_stem: dict[str, dict] = {}
    for entry in manifest.get("entries", []):
        status = entry.get("fetch_status") or {}
        replay_path = status.get("path")
        if not replay_path:
            continue
        by_stem[Path(str(replay_path)).stem] = {
            "map_uid": manifest.get("map_uid"),
            "account_id": entry.get("account_id"),
            "rank": entry.get("rank"),
            "record_time_ms": entry.get("time_ms"),
            "replay_source": "nadeo_top100",
        }
    return by_stem


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Openplanet-exported replay trajectories. Raw .gbx parsing is intentionally delegated "
            "to Openplanet; pass the resulting .json/.jsonl/.parquet exports here."
        )
    )
    parser.add_argument("--top100-manifest", default=None)
    parser.add_argument("--export", action="append", default=[], help="Openplanet export file; may be repeated.")
    parser.add_argument("--exports-dir", default=None, help="Directory containing .json/.jsonl/.parquet exports.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--map-uid", default=None)
    parser.add_argument("--observation-npz", default=None, help="Optional FULL-observation sidecar for action-valid pretraining.")
    args = parser.parse_args()

    exports: list[Path] = [Path(item).resolve() for item in args.export]
    if args.exports_dir is not None:
        exports.extend(
            sorted(
                path.resolve()
                for suffix in ("*.json", "*.jsonl", "*.parquet")
                for path in Path(args.exports_dir).glob(suffix)
            )
        )
    if not exports:
        parser.error("Provide at least one --export or --exports-dir with Openplanet export files.")

    top_manifest = None if args.top100_manifest is None else Path(args.top100_manifest).resolve()
    top_metadata = {} if top_manifest is None else _metadata_from_top_manifest(top_manifest)
    output_dir = Path(args.output_dir).resolve()
    metadata_paths = []
    for export_path in exports:
        metadata = dict(top_metadata.get(export_path.stem, {}))
        if args.map_uid is not None:
            metadata["map_uid"] = args.map_uid
        if args.observation_npz is not None:
            metadata["observation_npz_path"] = str(Path(args.observation_npz).resolve())
        metadata["trajectory_id"] = export_path.stem
        metadata_path = extract_openplanet_export(export_path, output_dir=output_dir, replay_metadata=metadata)
        metadata_paths.append(str(metadata_path))
        print(f"[extract-ghost-trajectories] trajectory_metadata={metadata_path}", flush=True)

    extraction_manifest_path = output_dir / "extraction_manifest.json"
    write_json(
        extraction_manifest_path,
        {
            "schema_version": "ghost_extraction_manifest_v1",
            "top100_manifest": None if top_manifest is None else str(top_manifest),
            "trajectory_metadata_paths": metadata_paths,
        },
    )
    print(f"[extract-ghost-trajectories] manifest={extraction_manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
