from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.config import load_tm20ai_config
from tm20ai.ghosts.nadeo import fetch_top100_ghost_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch top-100 Trackmania leaderboard ghost metadata and replay .gbx files.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq.yaml"))
    parser.add_argument("--map-uid", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--group-uid", default=None)
    parser.add_argument("--include-regional", action="store_true")
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (ROOT / config.ghosts.root / args.map_uid).resolve()
    )
    manifest_path = fetch_top100_ghost_manifest(
        map_uid=args.map_uid,
        output_dir=output_dir,
        leaderboard_length=args.length or config.ghosts.leaderboard_length,
        group_uid=args.group_uid or config.ghosts.group_uid,
        only_world=not args.include_regional,
    )
    print(f"[fetch-top100-ghosts] manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
