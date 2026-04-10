from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.bridge import BridgeClient, BridgeConnectionConfig, assess_bridge_status


VALID_GATE_RACE_STATES = frozenset({"outside_race", "start_line", "running", "finished"})


def log(message: str) -> None:
    print(f"[check-environment] {message}", flush=True)


def load_base_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a mapping in {path}.")
    return payload


def get_installed_plugin_dir() -> Path:
    return Path(os.environ["USERPROFILE"]) / "OpenplanetNext" / "Plugins" / "TM20AIBridge"

def get_installed_plugin_archive() -> Path:
    return Path(os.environ["USERPROFILE"]) / "OpenplanetNext" / "Plugins" / "TM20AIBridge.op"

def require_plugin_directory_layout(base_path: Path, label: str) -> list[str]:
    failures: list[str] = []
    if not base_path.exists():
        failures.append(f"{label} path is missing: {base_path}")
        return failures
    if not base_path.is_dir():
        failures.append(f"{label} path is not a directory: {base_path}")
        return failures
    for relative_path in (
        "info.toml",
        "Main.as",
        "Protocol.as",
        "Telemetry.as",
        "Reset.as",
        "Recorder.as",
    ):
        candidate = base_path / relative_path
        if not candidate.exists():
            failures.append(f"{label} is missing {candidate.name}: {candidate}")
    return failures


def require_plugin_archive_layout(archive_path: Path, label: str) -> list[str]:
    failures: list[str] = []
    if not archive_path.exists():
        failures.append(f"{label} archive is missing: {archive_path}")
        return failures
    if not archive_path.is_file():
        failures.append(f"{label} archive is not a file: {archive_path}")
        return failures
    try:
        with zipfile.ZipFile(archive_path) as plugin_archive:
            entries = {name.rstrip("/") for name in plugin_archive.namelist()}
    except zipfile.BadZipFile as exc:
        failures.append(f"{label} archive is not a valid .op/.zip file: {archive_path} ({exc})")
        return failures

    for expected_entry in (
        "info.toml",
        "Main.as",
        "Protocol.as",
        "Telemetry.as",
        "Reset.as",
        "Recorder.as",
    ):
        if expected_entry not in entries:
            failures.append(f"{label} archive is missing {expected_entry} at zip root: {archive_path}")
    return failures


def require_plugin_layout(repo_plugin_path: Path, installed_plugin_dir: Path, installed_plugin_archive: Path) -> list[str]:
    failures = require_plugin_directory_layout(repo_plugin_path, "repo plugin")
    if installed_plugin_dir.exists():
        failures.extend(require_plugin_directory_layout(installed_plugin_dir, "installed plugin"))
        return failures
    if installed_plugin_archive.exists():
        failures.extend(require_plugin_archive_layout(installed_plugin_archive, "installed plugin"))
        return failures
    failures.append(
        "installed plugin is missing; expected either "
        f"{installed_plugin_dir} or {installed_plugin_archive}"
    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that the Phase 2 custom bridge is installed and healthy.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "base.yaml"),
        help="Path to the Phase 2 base config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        log(f"ERROR: config path does not exist: {config_path}")
        return 1

    config = load_base_config(config_path)
    bridge_config = BridgeConnectionConfig.from_mapping(config.get("bridge", {}))
    repo_plugin_path = ROOT / "openplanet" / "TM20AIBridge"
    installed_plugin_dir = get_installed_plugin_dir()
    installed_plugin_archive = get_installed_plugin_archive()

    layout_failures = require_plugin_layout(repo_plugin_path, installed_plugin_dir, installed_plugin_archive)
    if layout_failures:
        for failure in layout_failures:
            log(f"ERROR: {failure}")
        return 1

    try:
        with BridgeClient(bridge_config) as client:
            report = assess_bridge_status(
                client,
                frame_timeout=bridge_config.initial_frame_timeout,
                health_timeout=bridge_config.command_timeout,
                stale_after_seconds=bridge_config.stale_timeout,
            )
    except Exception as exc:  # noqa: BLE001 - tool should surface the bridge exception directly
        log(f"ERROR: failed to connect to the custom bridge: {exc}")
        return 1

    log(f"bridge_status={report.status}")
    if report.health is not None:
        log(
            "health="
            f"ok:{report.health.ok} "
            f"race_state:{report.health.race_state} "
            f"session_id:{report.health.session_id} "
            f"run_id:{report.health.run_id} "
            f"map_uid:{report.health.map_uid}"
        )
    if report.latest_frame is not None:
        log(
            "latest_frame="
            f"frame_id:{report.latest_frame.frame_id} "
            f"timestamp_ns:{report.latest_frame.timestamp_ns} "
            f"session_id:{report.latest_frame.session_id} "
            f"run_id:{report.latest_frame.run_id} "
            f"map_uid:{report.latest_frame.map_uid}"
        )

    for warning in report.warnings:
        log(f"WARNING: {warning}")
    for issue in report.issues:
        log(f"ERROR: {issue}")

    race_state = report.health.race_state if report.health is not None else None
    if race_state not in VALID_GATE_RACE_STATES:
        log(f"ERROR: bridge race_state {race_state!r} is not one of {sorted(VALID_GATE_RACE_STATES)}")
        return 1

    if not report.ok:
        log("ERROR: bridge did not pass the environment gate.")
        return 1

    log("Environment gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
