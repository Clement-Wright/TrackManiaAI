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
from tm20ai.ghosts.dataset import build_reference_target_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stable reference-target ghost bundle from an already extracted top-100 dataset. "
            "Use this for map-specific mainline targets such as tmrl-test rank11_100."
        )
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq_top100.yaml"))
    parser.add_argument("--map-uid", required=True)
    parser.add_argument("--strategy-manifest", default=None)
    parser.add_argument("--extraction-manifest", default=None)
    parser.add_argument("--trajectory-metadata", action="append", default=[])
    parser.add_argument("--rank-min", type=int, default=None)
    parser.add_argument("--rank-max", type=int, default=None)
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--family-label", required=True)
    parser.add_argument("--bundle-resolution-mode", required=True)
    parser.add_argument("--strategy-status", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--author-reference-manifest",
        default=None,
        help="Optional official author/reference trajectory metadata used for canonical arc-length provenance.",
    )
    parser.add_argument("--set-default-alias", action="store_true")
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    metadata_paths = [str(Path(path).resolve()) for path in args.trajectory_metadata]

    strategy_manifest_path = (
        Path(args.strategy_manifest).resolve()
        if args.strategy_manifest is not None
        else (ROOT / config.ghosts.root / args.map_uid / "ghost_strategy_manifest.json").resolve()
    )
    if strategy_manifest_path.exists():
        strategy_manifest = read_json(strategy_manifest_path)
        metadata_paths.extend(
            str(Path(item["metadata_path"]).resolve())
            for item in strategy_manifest.get("all_trajectories", [])
            if item.get("metadata_path") is not None
        )
    elif args.extraction_manifest is not None:
        extraction_manifest = read_json(Path(args.extraction_manifest).resolve())
        metadata_paths.extend(
            str(Path(path).resolve())
            for path in extraction_manifest.get("trajectory_metadata_paths", [])
        )

    deduped_paths: list[str] = []
    seen: set[str] = set()
    for path in metadata_paths:
        resolved = str(Path(path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_paths.append(resolved)
    if not deduped_paths:
        parser.error("No trajectory metadata paths were found. Provide --strategy-manifest, --extraction-manifest, or --trajectory-metadata.")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (ROOT / config.ghosts.root / args.map_uid).resolve()
    )

    result = build_reference_target_bundle(
        map_uid=args.map_uid,
        trajectory_metadata_paths=deduped_paths,
        output_dir=output_dir,
        manifest_name=args.manifest_name,
        selected_training_family=args.family_label,
        bundle_resolution_mode=args.bundle_resolution_mode,
        strategy_classification_status=args.strategy_status,
        rank_min=args.rank_min,
        rank_max=args.rank_max,
        spacing_meters=config.reward.spacing_meters,
        ghost_config=config.ghosts,
        author_reference_manifest=args.author_reference_manifest or config.ghosts.author_reference_manifest,
        bands=config.ghosts.default_bands,
        max_representatives_per_band=config.ghosts.max_representatives_per_band,
        set_default_alias=args.set_default_alias,
    )
    print(f"[build-reference-target-bundle] manifest={result.manifest_path}", flush=True)
    print(f"[build-reference-target-bundle] selected={result.selected_count}/{result.trajectory_count}", flush=True)
    print(f"[build-reference-target-bundle] resolution={result.bundle_resolution_mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
