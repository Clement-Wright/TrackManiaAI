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
    from tm20ai.train.artifact_retention import cleanup_artifact_root, select_keeper_run_dirs

    parser = argparse.ArgumentParser(description="Prune old training/eval artifacts while keeping best/latest runs.")
    parser.add_argument("--artifact-root", default=str(ROOT / "artifacts"))
    parser.add_argument("--keep-best-per-algorithm", type=int, default=1)
    parser.add_argument("--keep-latest-per-algorithm", type=int, default=1)
    parser.add_argument("--keep-run-name", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
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
    log(json.dumps({"kept_paths": result.kept_paths, "removed_paths": result.removed_paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
