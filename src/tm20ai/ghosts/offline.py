from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..action_space import clamp_action
from .dataset import load_ghost_bundle_manifest


def _transition_from_npz(payload: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    transition: dict[str, Any] = {
        "action": clamp_action(payload["action"][index]),
        "reward": float(payload.get("reward", np.zeros((len(payload["action"]),), dtype=np.float32))[index]),
        "terminated": bool(payload.get("terminated", np.zeros((len(payload["action"]),), dtype=np.bool_))[index]),
        "truncated": bool(payload.get("truncated", np.zeros((len(payload["action"]),), dtype=np.bool_))[index]),
        "step_idx": int(payload.get("step_idx", np.arange(len(payload["action"]), dtype=np.int32))[index]),
    }
    if "obs_uint8" in payload:
        transition.update(
            {
                "obs_uint8": payload["obs_uint8"][index],
                "next_obs_uint8": payload["next_obs_uint8"][index],
                "telemetry_float": payload["telemetry_float"][index],
                "next_telemetry_float": payload["next_telemetry_float"][index],
            }
        )
    else:
        transition.update(
            {
                "obs_float": payload["obs_float"][index],
                "next_obs_float": payload["next_obs_float"][index],
            }
        )
    if "episode_id" in payload:
        transition["episode_id"] = str(payload["episode_id"][index])
    if "map_uid" in payload:
        transition["map_uid"] = str(payload["map_uid"][index])
    return transition


def seed_replay_from_ghost_bundle(
    replay,  # noqa: ANN001
    manifest_path: str | Path,
    *,
    require_actions: bool = True,
) -> dict[str, Any]:
    manifest = load_ghost_bundle_manifest(manifest_path)
    if require_actions and not bool(manifest.get("action_channel_valid")):
        raise RuntimeError(
            f"Ghost bundle {manifest_path} does not have validated action channels; "
            "refusing to seed actor/critic training from guessed actions."
        )
    npz_path = manifest.get("offline_transition_npz_path")
    if not npz_path:
        if require_actions:
            raise RuntimeError(
                f"Ghost bundle {manifest_path} does not contain offline transitions. "
                "Run the Openplanet extractor with observation/action sidecars before actor pretraining."
            )
        return {
            "seeded": 0,
            "manifest_path": str(Path(manifest_path).resolve()),
            "action_channel_valid": bool(manifest.get("action_channel_valid")),
            "reason": "no_offline_transition_npz_path",
        }
    loaded = np.load(Path(str(npz_path)).resolve(), allow_pickle=True)
    payload = {key: loaded[key] for key in loaded.files}
    action_count = int(payload["action"].shape[0])
    seeded = 0
    add_method = getattr(replay, "add_offline", replay.add)
    for index in range(action_count):
        add_method(_transition_from_npz(payload, index))
        seeded += 1
    return {
        "seeded": seeded,
        "manifest_path": str(Path(manifest_path).resolve()),
        "offline_transition_npz_path": str(Path(str(npz_path)).resolve()),
        "offline_dataset_hash": manifest.get("offline_dataset_hash"),
        "action_channel_valid": bool(manifest.get("action_channel_valid")),
    }
