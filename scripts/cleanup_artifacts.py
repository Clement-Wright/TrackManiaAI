from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def log(message: str) -> None:
    print(f"[cleanup-artifacts] {message}", flush=True)


def main() -> int:
    from tm20ai.train.artifact_retention import (
        cleanup_artifact_root,
        format_bytes,
        select_keeper_run_dirs,
        storage_preflight_report,
    )

    parser = argparse.ArgumentParser(description="Prune old training/eval artifacts while keeping best/latest runs.")
    parser.add_argument("--artifact-root", default=str(ROOT / "artifacts"))
    parser.add_argument("--keep-best-per-algorithm", type=int, default=1)
    parser.add_argument("--keep-latest-per-algorithm", type=int, default=1)
    parser.add_argument("--keep-run-name", action="append", default=[])
    parser.add_argument("--min-free-gb", type=float, default=150.0)
    parser.add_argument("--max-artifact-gb", type=float, default=150.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    preflight = storage_preflight_report(
        artifact_root,
        min_free_gb=args.min_free_gb,
        max_artifact_gb=args.max_artifact_gb,
    )
    keepers = select_keeper_run_dirs(
        artifact_root,
        keep_best_per_algorithm=args.keep_best_per_algorithm,
        keep_latest_per_algorithm=args.keep_latest_per_algorithm,
        keep_run_names=args.keep_run_name,
    )
    result = cleanup_artifact_root(
        artifact_root,
        keep_run_dirs=keepers,
        dry_run=args.dry_run,
    )
    log(f"artifact_root={artifact_root}")
    log(
        json.dumps(
            {
                "preflight": {
                    "ok": preflight.ok,
                    "reasons": preflight.reasons,
                    "free_bytes": preflight.free_bytes,
                    "free_human": format_bytes(preflight.free_bytes),
                    "artifact_root_bytes": preflight.artifact_root_bytes,
                    "artifact_root_human": format_bytes(preflight.artifact_root_bytes),
                    "max_artifact_bytes": preflight.max_artifact_bytes,
                    "max_artifact_human": format_bytes(preflight.max_artifact_bytes),
                },
                "kept_paths": result.kept_paths,
                "removed_paths": result.removed_paths,
                "kept_bytes": result.kept_bytes,
                "kept_human": format_bytes(result.kept_bytes),
                "removed_bytes": result.removed_bytes,
                "removed_human": format_bytes(result.removed_bytes),
                "artifact_root_bytes_before": result.artifact_root_bytes_before,
                "artifact_root_before_human": format_bytes(result.artifact_root_bytes_before),
                "artifact_root_bytes_after": result.artifact_root_bytes_after,
                "artifact_root_after_human": format_bytes(result.artifact_root_bytes_after),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
