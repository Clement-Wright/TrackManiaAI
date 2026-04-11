from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.algos.bc import BehaviorCloningTrainer
from tm20ai.config import load_tm20ai_config
from tm20ai.data.dataset import FullBehaviorCloningDataset, split_demo_dataset
from tm20ai.data.parquet_writer import build_run_artifact_paths, sha256_file, timestamp_tag, write_json
from tm20ai.models.full_actor_critic import FullObservationActor
from tm20ai.train.metrics import TensorBoardScalarLogger


def log(message: str) -> None:
    print(f"[pretrain-bc] {message}", flush=True)


def _collate(batch):
    observations, telemetry, actions = zip(*batch, strict=True)
    obs = torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in observations], dim=0) / 255.0
    telem = torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in telemetry], dim=0)
    action = torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in actions], dim=0)
    return obs, telem, action


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain a FULL-observation actor with behavior cloning demos.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_bc.yaml"))
    parser.add_argument("--demos-root", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    split = split_demo_dataset(
        Path(args.demos_root).resolve(),
        validation_fraction=config.bc.validation_fraction,
        seed=config.train.seed,
    )
    train_dataset = FullBehaviorCloningDataset(split.train_episodes)
    validation_dataset = FullBehaviorCloningDataset(split.validation_episodes)
    if len(train_dataset) == 0:
        log("ERROR: no FULL demo observation sidecars were found.")
        return 1

    batch_size = args.batch_size or config.train.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    validation_loader = (
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
        if len(validation_dataset) > 0
        else None
    )

    requested_device = "cuda" if config.train.cuda_training and torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    actor = FullObservationActor()
    trainer = BehaviorCloningTrainer(
        actor=actor,
        device=device,
        learning_rate=config.bc.learning_rate,
        weight_decay=config.bc.weight_decay,
    )

    run_name = args.run_name or f"full_bc_{timestamp_tag()}"
    run_paths = build_run_artifact_paths(config, mode="bc", run_name=run_name)
    writer = TensorBoardScalarLogger(run_paths.tensorboard_dir)
    checkpoint_path = run_paths.run_dir / "actor_checkpoint.pt"
    summary_path = run_paths.summary_json

    best_validation = None
    epoch_rows: list[dict[str, float | int | None]] = []
    try:
        for epoch in range(config.bc.epochs):
            train_losses: list[float] = []
            for observation, telemetry, action in train_loader:
                update = trainer.train_step(
                    observation.to(device),
                    telemetry.to(device),
                    action.to(device),
                )
                train_losses.append(update.loss)

            validation_loss = None
            if validation_loader is not None:
                validation_losses: list[float] = []
                for observation, telemetry, action in validation_loader:
                    validation_losses.append(
                        trainer.evaluate(
                            observation.to(device),
                            telemetry.to(device),
                            action.to(device),
                        )
                    )
                validation_loss = _mean(validation_losses)
                if best_validation is None or validation_loss < best_validation:
                    best_validation = validation_loss
            epoch_summary = {
                "epoch": epoch + 1,
                "train_loss": _mean(train_losses),
                "validation_loss": validation_loss,
            }
            epoch_rows.append(epoch_summary)
            writer.add_scalar("bc/train_loss", epoch_summary["train_loss"], step=epoch + 1)
            if validation_loss is not None:
                writer.add_scalar("bc/validation_loss", validation_loss, step=epoch + 1)

        torch.save(
            {
                "observation_mode": "full",
                "actor_state_dict": actor.cpu().state_dict(),
                "observation_shape": (4, 64, 64),
                "telemetry_dim": 14,
                "action_dim": 3,
                "config_snapshot": {
                    "config_path": str(Path(args.config).resolve()),
                    "config_sha256": sha256_file(Path(args.config).resolve()),
                },
                "epochs": config.bc.epochs,
            },
            checkpoint_path,
        )
        summary = {
            "run_name": run_name,
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(Path(args.config).resolve()),
            "demos_root": str(Path(args.demos_root).resolve()),
            "device": str(device),
            "epochs": config.bc.epochs,
            "train_episode_count": len(split.train_episodes),
            "validation_episode_count": len(split.validation_episodes),
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(validation_dataset),
            "best_validation_loss": best_validation,
            "checkpoint_path": str(checkpoint_path),
            "history": epoch_rows,
        }
        write_json(summary_path, summary)
        log(f"summary={summary_path}")
        log(json.dumps(summary, indent=2))
        return 0
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
