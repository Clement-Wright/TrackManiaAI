from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.bridge import (
    BridgeClient,
    BridgeConnectionConfig,
    assess_bridge_status,
    run_reset_validation,
    run_telemetry_soak,
)


def log(message: str) -> None:
    print(f"[check-bridge] {message}", flush=True)


def load_base_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a mapping in {path}.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run diagnostics against the custom dual-port Openplanet bridge.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "base.yaml"),
        help="Path to the Phase 2 base config.",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Telemetry soak duration in seconds.")
    parser.add_argument("--reset-count", type=int, default=3, help="Number of soft resets to validate.")
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=None,
        help="Per-reset timeout in seconds. Defaults to the bridge reset_timeout from the config.",
    )
    parser.add_argument(
        "--reset-sleep",
        type=float,
        default=None,
        help="Sleep between reset attempts in seconds. Defaults to runtime.sleep_time_at_reset from the config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        log(f"ERROR: config path does not exist: {config_path}")
        return 1

    config = load_base_config(config_path)
    bridge_config = BridgeConnectionConfig.from_mapping(config.get("bridge", {}))
    per_reset_timeout = bridge_config.reset_timeout if args.reset_timeout is None else args.reset_timeout
    runtime_config = config.get("runtime", {}) if isinstance(config.get("runtime", {}), dict) else {}
    reset_sleep = (
        float(runtime_config.get("sleep_time_at_reset", runtime_config.get("reset_sleep", 0.0)))
        if args.reset_sleep is None
        else args.reset_sleep
    )

    try:
        with BridgeClient(bridge_config) as client:
            initial_report = assess_bridge_status(
                client,
                frame_timeout=bridge_config.initial_frame_timeout,
                health_timeout=bridge_config.command_timeout,
                stale_after_seconds=bridge_config.stale_timeout,
            )
            log(f"initial_status={initial_report.status}")
            for warning in initial_report.warnings:
                log(f"WARNING: {warning}")
            for issue in initial_report.issues:
                log(f"ERROR: {issue}")
            if not initial_report.ok:
                log("ERROR: bridge failed the initial health gate.")
                return 1

            soak = run_telemetry_soak(
                client,
                duration_seconds=args.duration,
                stale_after_seconds=bridge_config.stale_timeout,
            )
            log(
                "telemetry_soak="
                f"ok:{soak.ok} "
                f"frames_seen:{soak.frames_seen} "
                f"first_frame_id:{soak.first_frame_id} "
                f"last_frame_id:{soak.last_frame_id} "
                f"disconnects_seen:{soak.disconnects_seen} "
                f"stale_events:{soak.stale_events}"
            )
            for issue in soak.issues:
                log(f"ERROR: {issue}")

            reset_result = run_reset_validation(
                client,
                reset_count=args.reset_count,
                per_reset_timeout_seconds=per_reset_timeout,
                sleep_between_resets_seconds=reset_sleep,
            )
            log(
                "reset_validation="
                f"ok:{reset_result.ok} "
                f"requested:{reset_result.requested} "
                f"succeeded:{reset_result.succeeded}"
            )
            for failure in reset_result.failures:
                log(f"ERROR: {failure}")

            final_report = assess_bridge_status(
                client,
                frame_timeout=bridge_config.command_timeout,
                health_timeout=bridge_config.command_timeout,
                stale_after_seconds=bridge_config.stale_timeout,
            )
            log(f"final_status={final_report.status}")
            for warning in final_report.warnings:
                log(f"WARNING: {warning}")
            for issue in final_report.issues:
                log(f"ERROR: {issue}")
    except Exception as exc:  # noqa: BLE001 - tool should surface the bridge exception directly
        log(f"ERROR: bridge diagnostics failed unexpectedly: {exc}")
        return 1

    if not soak.ok or not reset_result.ok or not final_report.ok:
        log("ERROR: bridge diagnostics failed.")
        return 1

    log("Bridge diagnostics passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
