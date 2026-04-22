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
    parser = argparse.ArgumentParser(
        description=(
            "Build route-aware ghost bundle manifests. One bundle should represent one driving strategy: intended "
            "route when available, otherwise a selected ghost override, then an author/reference fallback, then a hard error."
        )
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq.yaml"))
    parser.add_argument("--map-uid", required=True)
    parser.add_argument("--extraction-manifest", default=None)
    parser.add_argument("--trajectory-metadata", action="append", default=[])
    parser.add_argument(
        "--canonical-reference-manifest",
        "--author-reference-manifest",
        dest="canonical_reference_manifest",
        default=None,
        help=(
            "Author-run reference manifest or normalized trajectory metadata used as the canonical intended route. "
            "When omitted, the builder falls back to data/reward/<map_uid>/trajectory_<spacing>.npz if present."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bands", default=None, help="Comma-separated rank bands, e.g. 1-10,11-30,31-60,61-100.")
    parser.add_argument("--max-representatives-per-band", type=int, default=None)
    parser.add_argument(
        "--selected-ghost-name",
        default=None,
        help="Optional selected-ghost override name. Resolution order is exact preserved basename, substring, then rank.",
    )
    parser.add_argument(
        "--selected-ghost-rank",
        type=int,
        default=None,
        help="Optional selected-ghost override rank fallback used when the name does not resolve uniquely.",
    )
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
    selected_ghost_selector = None
    if args.selected_ghost_name is not None or args.selected_ghost_rank is not None:
        selected_ghost_selector = {}
        if args.selected_ghost_name is not None:
            selected_ghost_selector["ghost_name_contains"] = args.selected_ghost_name
        if args.selected_ghost_rank is not None:
            selected_ghost_selector["rank"] = args.selected_ghost_rank
    try:
        result = build_ghost_bundle(
            map_uid=args.map_uid,
            trajectory_metadata_paths=metadata_paths,
            output_dir=output_dir,
            spacing_meters=config.reward.spacing_meters,
            ghost_config=config.ghosts,
            author_reference_manifest=(
                args.canonical_reference_manifest
                if args.canonical_reference_manifest is not None
                else config.ghosts.author_reference_manifest
            ),
            selected_ghost_selector=selected_ghost_selector,
            bands=bands,
            max_representatives_per_band=args.max_representatives_per_band or config.ghosts.max_representatives_per_band,
        )
    except RuntimeError:
        strategy_manifest_path = output_dir / "ghost_strategy_manifest.json"
        if strategy_manifest_path.exists():
            strategy_manifest = read_json(strategy_manifest_path)
            print(
                "[build-ghost-dataset] "
                f"resolution={strategy_manifest.get('bundle_resolution_mode')} "
                f"strategy_manifest={strategy_manifest_path}",
                flush=True,
            )
        raise
    print(f"[build-ghost-dataset] manifest={result.manifest_path}", flush=True)
    print(f"[build-ghost-dataset] resolution={result.bundle_resolution_mode}", flush=True)
    if result.strategy_manifest_path is not None:
        print(f"[build-ghost-dataset] strategy_manifest={result.strategy_manifest_path}", flush=True)
    if result.intended_manifest_path is not None:
        print(f"[build-ghost-dataset] intended_manifest={result.intended_manifest_path}", flush=True)
    if result.exploit_manifest_path is not None:
        print(f"[build-ghost-dataset] exploit_manifest={result.exploit_manifest_path}", flush=True)
    if result.selected_override_manifest_path is not None:
        print(f"[build-ghost-dataset] selected_override_manifest={result.selected_override_manifest_path}", flush=True)
    if result.author_fallback_manifest_path is not None:
        print(f"[build-ghost-dataset] author_fallback_manifest={result.author_fallback_manifest_path}", flush=True)
    if result.mixed_fallback_manifest_path is not None:
        print(f"[build-ghost-dataset] mixed_fallback_manifest={result.mixed_fallback_manifest_path}", flush=True)
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
