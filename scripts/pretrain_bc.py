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
from tm20ai.data.dataset import (
    FullBehaviorCloningDataset,
    split_episode_records,
    validate_full_demo_dataset,
)
from tm20ai.data.parquet_writer import build_run_artifact_paths, sha256_file, timestamp_tag, write_json
from tm20ai.models.full_actor_critic import FullObservationActor
from tm20ai.train.features import ACTION_DIM, TELEMETRY_DIM
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


def _save_actor_checkpoint(
    checkpoint_path: Path,
    *,
    actor: FullObservationActor,
    config_path: Path,
    demos_root: Path,
    map_uid: str | None,
    epoch: int,
    train_loss: float,
    validation_loss: float | None,
) -> None:
    actor_state_dict = {key: value.detach().cpu() for key, value in actor.state_dict().items()}
    torch.save(
        {
            "checkpoint_kind": "bc_actor",
            "observation_mode": "full",
            "actor_state_dict": actor_state_dict,
            "observation_shape": (4, 64, 64),
            "telemetry_dim": TELEMETRY_DIM,
            "action_dim": ACTION_DIM,
            "map_uid": map_uid,
            "demo_root": str(demos_root),
            "config_snapshot": {
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
            },
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        checkpoint_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain a FULL-observation actor with behavior cloning demos.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_bc.yaml"))
    parser.add_argument("--demos-root", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--allow-mixed-map-uids", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    demos_root = Path(args.demos_root).resolve()
    config = load_tm20ai_config(config_path)
    validation = validate_full_demo_dataset(
        demos_root,
        allow_mixed_map_uids=args.allow_mixed_map_uids,
    )
    split = split_episode_records(
        validation.episodes,
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
    final_checkpoint_path = run_paths.run_dir / "actor_checkpoint_final.pt"
    best_checkpoint_path = run_paths.run_dir / "actor_checkpoint_best.pt"
    summary_path = run_paths.summary_json

    best_validation = None
    best_selection_metric = None
    best_epoch = None
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
            epoch_summary = {
                "epoch": epoch + 1,
                "train_loss": _mean(train_losses),
                "validation_loss": validation_loss,
            }
            selection_metric = epoch_summary["train_loss"] if validation_loss is None else validation_loss
            if best_selection_metric is None or selection_metric < best_selection_metric:
                best_selection_metric = selection_metric
                best_validation = validation_loss
                best_epoch = epoch + 1
                _save_actor_checkpoint(
                    best_checkpoint_path,
                    actor=actor,
                    config_path=config_path,
                    demos_root=demos_root,
                    map_uid=validation.map_uid,
                    epoch=epoch + 1,
                    train_loss=epoch_summary["train_loss"],
                    validation_loss=validation_loss,
                )
            epoch_rows.append(epoch_summary)
            writer.add_scalar("bc/train_loss", epoch_summary["train_loss"], step=epoch + 1)
            if validation_loss is not None:
                writer.add_scalar("bc/validation_loss", validation_loss, step=epoch + 1)

        _save_actor_checkpoint(
            final_checkpoint_path,
            actor=actor,
            config_path=config_path,
            demos_root=demos_root,
            map_uid=validation.map_uid,
            epoch=config.bc.epochs,
            train_loss=float(epoch_rows[-1]["train_loss"]),
            validation_loss=None if epoch_rows[-1]["validation_loss"] is None else float(epoch_rows[-1]["validation_loss"]),
        )
        summary = {
            "run_name": run_name,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "demos_root": str(demos_root),
            "device": str(device),
            "observation_mode": "full",
            "map_uid": validation.map_uid,
            "epochs": config.bc.epochs,
            "total_episode_count": validation.total_episode_count,
            "valid_episode_count": validation.valid_episode_count,
            "train_episode_count": len(split.train_episodes),
            "validation_episode_count": len(split.validation_episodes),
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(validation_dataset),
            "dataset_sample_count": validation.sample_count,
            "best_validation_loss": best_validation,
            "best_epoch": best_epoch,
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "allow_mixed_map_uids": args.allow_mixed_map_uids,
            "history": epoch_rows,
        }
        write_json(summary_path, summary)
        log(f"best_checkpoint={best_checkpoint_path}")
        log(f"final_checkpoint={final_checkpoint_path}")
        log(f"summary={summary_path}")
        log(json.dumps(summary, indent=2))
        return 0
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
