from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.config import load_tm20ai_config
from tm20ai.train.evaluator import resolve_policy_adapter, run_policy_episodes


def log(message: str) -> None:
    print(f"[evaluate] {message}", flush=True)


def parse_action(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Fixed action must have exactly 3 comma-separated values.")
    return np.asarray([float(part) for part in parts], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation episodes against the custom env.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--policy", choices=("zero", "fixed", "scripted", "checkpoint"), default="zero")
    parser.add_argument("--fixed-action", default="0.0,0.0,0.0")
    parser.add_argument("--script", default=None, help="Scripted policy in 'module:attr' or 'path.py:attr' form.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    episodes = config.eval.episodes if args.episodes is None else args.episodes
    seed_base = config.eval.seed_base if args.seed_base is None else args.seed_base
    record_video = config.eval.record_video or args.record_video
    fixed_action = parse_action(args.fixed_action)
    policy = resolve_policy_adapter(
        policy=args.policy,
        fixed_action=fixed_action,
        script=args.script,
        checkpoint=args.checkpoint,
    )
    result = run_policy_episodes(
        config_path=args.config,
        mode="eval",
        policy=policy,
        episodes=episodes,
        seed_base=seed_base,
        record_video=record_video,
        checkpoint_path=args.checkpoint,
    )
    log(f"summary={result['summary_path']}")
    log(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
