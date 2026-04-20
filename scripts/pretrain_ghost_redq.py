from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.algos.redq import REDQSACAgent
from tm20ai.capture import lidar_feature_dim
from tm20ai.config import load_tm20ai_config
from tm20ai.data.parquet_writer import ensure_directory, sha256_file, timestamp_tag, write_json
from tm20ai.ghosts.offline import seed_replay_from_ghost_bundle
from tm20ai.ghosts.pretrain import offline_pretrain_redq
from tm20ai.train.features import TELEMETRY_DIM
from tm20ai.train.replay import ReplayBuffer


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline REDQ initialization from action-valid ghost bundle transitions.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq.yaml"))
    parser.add_argument("--ghost-bundle", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    config_snapshot = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if config.train.algorithm != "redq":
        raise RuntimeError("pretrain_ghost_redq.py requires a REDQ config.")
    requested_device = "cuda" if config.train.cuda_training and torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if config.observation.mode == "full":
        observation_shape = (
            config.full_observation.frame_stack,
            config.full_observation.output_height,
            config.full_observation.output_width,
        )
        telemetry_dim = TELEMETRY_DIM
    else:
        observation_shape = (lidar_feature_dim(config.lidar_observation),)
        telemetry_dim = 0
    replay = ReplayBuffer.from_config(config)
    seed_metadata = seed_replay_from_ghost_bundle(
        replay,
        args.ghost_bundle,
        require_actions=config.offline_pretrain.require_actions,
    )
    pretrain_config = config.offline_pretrain
    if args.gradient_steps is not None:
        pretrain_config.gradient_steps = int(args.gradient_steps)
    if args.batch_size is not None:
        pretrain_config.batch_size = int(args.batch_size)
    agent = REDQSACAgent(
        sac_config=config.sac,
        redq_config=config.redq,
        observation_mode=config.observation.mode,
        device=device,
        observation_shape=observation_shape,
        telemetry_dim=telemetry_dim,
    )
    result = offline_pretrain_redq(agent=agent, replay=replay, config=pretrain_config, device=device)
    run_name = args.run_name or f"ghost_redq_pretrain_{timestamp_tag()}"
    output_dir = ensure_directory(
        Path(args.output_dir).resolve() if args.output_dir is not None else ROOT / "artifacts" / "pretrain" / run_name
    )
    checkpoint_path = output_dir / "ghost_redq_pretrain.pt"
    payload = {
        **agent.state_dict(),
        "checkpoint_kind": "ghost_redq_offline_pretrain",
        "config_snapshot": config_snapshot,
        "ghost_bundle_manifest_path": str(Path(args.ghost_bundle).resolve()),
        "offline_pretrain_strategy": pretrain_config.strategy,
        "offline_dataset_hash": seed_metadata.get("offline_dataset_hash"),
        "offline_transition_count": seed_metadata.get("seeded", 0),
        "pretrain_result": asdict(result),
        "env_step": 0,
        "learner_step": int(result.critic_updates),
        "actor_step": int(result.bc_updates + result.awac_updates),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        checkpoint_path.with_suffix(".json"),
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "ghost_bundle_manifest_path": str(Path(args.ghost_bundle).resolve()),
            "seed_metadata": seed_metadata,
            "pretrain_result": asdict(result),
        },
    )
    print(f"[pretrain-ghost-redq] checkpoint={checkpoint_path}", flush=True)
    print(f"[pretrain-ghost-redq] metrics={result.metrics}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
