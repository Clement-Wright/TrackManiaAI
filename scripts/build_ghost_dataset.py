from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.config import load_tm20ai_config
from tm20ai.data.parquet_writer import read_json
from tm20ai.ghosts.dataset import build_ghost_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster/select top-player ghost trajectories and write ghost_bundle_manifest.json.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq.yaml"))
    parser.add_argument("--map-uid", required=True)
    parser.add_argument("--extraction-manifest", default=None)
    parser.add_argument("--trajectory-metadata", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bands", default=None, help="Comma-separated rank bands, e.g. 1-10,11-30,31-60,61-100.")
    parser.add_argument("--max-representatives-per-band", type=int, default=None)
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    metadata_paths = [str(Path(path).resolve()) for path in args.trajectory_metadata]
    if args.extraction_manifest is not None:
        extraction_manifest = read_json(Path(args.extraction_manifest).resolve())
        metadata_paths.extend(str(Path(path).resolve()) for path in extraction_manifest.get("trajectory_metadata_paths", []))
    if not metadata_paths:
        parser.error("Provide --trajectory-metadata or --extraction-manifest.")

    bands = (
        tuple(part.strip() for part in args.bands.split(",") if part.strip())
        if args.bands is not None
        else config.ghosts.default_bands
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (ROOT / config.ghosts.root / args.map_uid).resolve()
    )
    result = build_ghost_bundle(
        map_uid=args.map_uid,
        trajectory_metadata_paths=metadata_paths,
        output_dir=output_dir,
        bands=bands,
        max_representatives_per_band=args.max_representatives_per_band or config.ghosts.max_representatives_per_band,
    )
    print(f"[build-ghost-dataset] manifest={result.manifest_path}", flush=True)
    print(
        "[build-ghost-dataset] "
        f"selected={result.selected_count}/{result.trajectory_count} "
        f"action_channel_valid={result.action_channel_valid} "
        f"offline_transitions={result.offline_transition_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
