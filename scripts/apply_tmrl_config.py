import argparse
import json
import os
from pathlib import Path
from typing import Any


USER_PROFILE = Path(os.environ["USERPROFILE"])
REPO_ROOT = Path(__file__).resolve().parents[1]
TMRL_CONFIG_PATH = USER_PROFILE / "TmrlData" / "config" / "config.json"
TEMPLATE_PATHS = {
    "lidar": REPO_ROOT / "config" / "tmrl.lidar.json",
    "full": REPO_ROOT / "config" / "tmrl.full.json",
}
CAMERA_HINTS = {
    "lidar": "Use camera 3 until the cockpit view hides the car.",
    "full": "Use camera 1 so the default camera shows the car.",
}


def deep_merge(base: Any, overrides: Any) -> Any:
    if isinstance(base, dict) and isinstance(overrides, dict):
        merged = dict(base)
        for key, value in overrides.items():
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return overrides


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a repo-owned TMRL config template.")
    parser.add_argument("--env", choices=sorted(TEMPLATE_PATHS), required=True)
    args = parser.parse_args()

    if not TMRL_CONFIG_PATH.exists():
        raise SystemExit(
            f"TMRL config not found at {TMRL_CONFIG_PATH}. "
            "Run scripts\\bootstrap_phase1.ps1 first."
        )

    template_path = TEMPLATE_PATHS[args.env]
    current_config = load_json(TMRL_CONFIG_PATH)
    template_config = load_json(template_path)
    merged_config = deep_merge(current_config, template_config)

    TMRL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TMRL_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(merged_config, handle, indent=2)
        handle.write("\n")

    env_config = merged_config["ENV"]
    print(f"Applied {args.env} config to {TMRL_CONFIG_PATH}")
    print(f"Interface: {env_config['RTGYM_INTERFACE']}")
    print(f"Window: {env_config['WINDOW_WIDTH']}x{env_config['WINDOW_HEIGHT']}")
    print(CAMERA_HINTS[args.env])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
