from __future__ import annotations

import time
from dataclasses import dataclass, field

from .client import BridgeClient
from .messages import BridgeHealth, TelemetryFrame


@dataclass(slots=True)
class BridgeStatusReport:
    status: str
    health: BridgeHealth | None
    latest_frame: TelemetryFrame | None
    stale: bool
    telemetry_disconnects: int
    telemetry_connections: int
    last_telemetry_error: str | None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "healthy" and not self.issues


@dataclass(slots=True)
class TelemetrySoakResult:
    duration_seconds: float
    frames_seen: int
    first_frame_id: int | None
    last_frame_id: int | None
    session_id: str | None
    run_id: str | None
    stale_events: int
    disconnects_seen: int
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.frames_seen > 0 and self.stale_events == 0 and not self.issues


@dataclass(slots=True)
class ResetValidationResult:
    requested: int
    succeeded: int
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.requested == self.succeeded and not self.failures


def assess_bridge_status(
    client: BridgeClient,
    *,
    frame_timeout: float | None = None,
    health_timeout: float | None = None,
    stale_after_seconds: float | None = None,
) -> BridgeStatusReport:
    issues: list[str] = []
    warnings: list[str] = []
    health: BridgeHealth | None = None
    latest_frame = client.get_latest_frame()

    try:
        health = client.health(timeout=health_timeout)
    except Exception as exc:  # noqa: BLE001 - diagnostics should preserve bridge error details
        issues.append(f"Bridge health request failed: {exc}")

    if latest_frame is None:
        try:
            latest_frame = client.wait_for_frame(
                timeout=client.config.initial_frame_timeout if frame_timeout is None else frame_timeout
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should preserve bridge error details
            issues.append(f"Telemetry frame unavailable: {exc}")

    stale_threshold = client.config.stale_timeout if stale_after_seconds is None else stale_after_seconds
    stale = latest_frame is None or client.is_stale(stale_threshold)
    if stale:
        issues.append(f"Telemetry is stale (>{stale_threshold:.3f}s).")

    if latest_frame is not None:
        if not latest_frame.session_id:
            issues.append("Latest telemetry frame has an empty session_id.")
        if not latest_frame.run_id:
            issues.append("Latest telemetry frame has an empty run_id.")
        if latest_frame.finished and latest_frame.terminal_reason != "finished":
            issues.append("Finished telemetry frame must carry terminal_reason='finished'.")

    if health is not None and latest_frame is not None:
        if health.session_id is not None and health.session_id != latest_frame.session_id:
            issues.append(
                f"Bridge health session_id {health.session_id!r} does not match latest telemetry {latest_frame.session_id!r}."
            )
        if health.run_id is not None and health.run_id != latest_frame.run_id:
            issues.append(f"Bridge health run_id {health.run_id!r} does not match latest telemetry {latest_frame.run_id!r}.")
        if health.map_uid is not None and health.map_uid != latest_frame.map_uid:
            issues.append(f"Bridge health map_uid {health.map_uid!r} does not match latest telemetry {latest_frame.map_uid!r}.")
        # Health is sampled over the command socket while telemetry can advance on the stream
        # between reads, so allow a small bounded drift instead of exact equality.
        if health.last_frame_id is not None:
            frame_delta = abs(health.last_frame_id - latest_frame.frame_id)
            if frame_delta > 5:
                warnings.append(
                    "Bridge health last_frame_id drifted too far from latest telemetry: "
                    f"{health.last_frame_id} vs {latest_frame.frame_id} (delta={frame_delta})."
                )
        if health.last_timestamp_ns is not None:
            timestamp_delta_ns = abs(health.last_timestamp_ns - latest_frame.timestamp_ns)
            max_timestamp_delta_ns = int(max(stale_threshold, 1.0) * 1_000_000_000)
            if timestamp_delta_ns > max_timestamp_delta_ns:
                warnings.append(
                    "Bridge health last_timestamp_ns drifted too far from latest telemetry: "
                    f"{health.last_timestamp_ns} vs {latest_frame.timestamp_ns} "
                    f"(delta_ns={timestamp_delta_ns})."
                )
        if not latest_frame.map_uid and (health.race_state or "") != "outside_race":
            issues.append("Latest telemetry frame is missing map_uid while the bridge is not in outside_race.")

    if health is not None and not health.ok:
        warnings.append(f"Bridge health reported ok=false: {health.message or 'no message provided'}")

    if health is None or latest_frame is None:
        status = "disconnected"
    elif stale:
        status = "degraded" if client.telemetry_connections > 0 else "disconnected"
    elif issues or warnings or (health is not None and not health.ok):
        status = "degraded"
    else:
        status = "healthy"

    return BridgeStatusReport(
        status=status,
        health=health,
        latest_frame=latest_frame,
        stale=stale,
        telemetry_disconnects=client.telemetry_disconnects,
        telemetry_connections=client.telemetry_connections,
        last_telemetry_error=client.last_telemetry_error,
        issues=issues,
        warnings=warnings,
    )


def run_telemetry_soak(
    client: BridgeClient,
    *,
    duration_seconds: float,
    stale_after_seconds: float,
) -> TelemetrySoakResult:
    initial = client.wait_for_frame(timeout=max(1.0, min(10.0, duration_seconds)))
    start_disconnects = client.telemetry_disconnects
    issues: list[str] = []
    frames_seen = 0
    stale_events = 0
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    session_id: str | None = initial.session_id
    run_id: str | None = initial.run_id
    previous_frame: TelemetryFrame | None = None

    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        frames = client.pop_received_frames()
        if not frames:
            if client.is_stale(stale_after_seconds):
                stale_events += 1
                issues.append(f"Telemetry became stale for longer than {stale_after_seconds:.3f}s.")
                break
            time.sleep(0.05)
            continue

        for frame in frames:
            if first_frame_id is None:
                first_frame_id = frame.frame_id
            last_frame_id = frame.frame_id
            run_id = frame.run_id
            frames_seen += 1

            if not frame.session_id:
                issues.append("Telemetry frame had an empty session_id.")
            if not frame.run_id:
                issues.append("Telemetry frame had an empty run_id.")
            if frame.finished and frame.terminal_reason != "finished":
                issues.append("Finished telemetry frame did not expose terminal_reason='finished'.")

            if previous_frame is not None:
                if frame.frame_id <= previous_frame.frame_id:
                    issues.append(
                        f"frame_id stopped being monotonic: {frame.frame_id} after {previous_frame.frame_id}."
                    )
                if frame.timestamp_ns <= previous_frame.timestamp_ns:
                    issues.append(
                        "timestamp_ns stopped being monotonic: "
                        f"{frame.timestamp_ns} after {previous_frame.timestamp_ns}."
                    )
                if frame.session_id != previous_frame.session_id:
                    issues.append(
                        f"session_id changed during telemetry soak: {previous_frame.session_id} -> {frame.session_id}."
                    )
                if frame.map_uid != previous_frame.map_uid and frame.run_id == previous_frame.run_id:
                    issues.append(
                        f"map_uid changed from {previous_frame.map_uid!r} to {frame.map_uid!r} without a new run_id."
                    )
            previous_frame = frame

        time.sleep(0.01)

    disconnects_seen = client.telemetry_disconnects - start_disconnects
    return TelemetrySoakResult(
        duration_seconds=duration_seconds,
        frames_seen=frames_seen,
        first_frame_id=first_frame_id,
        last_frame_id=last_frame_id,
        session_id=session_id,
        run_id=run_id,
        stale_events=stale_events,
        disconnects_seen=disconnects_seen,
        issues=issues,
    )


def run_reset_validation(
    client: BridgeClient,
    *,
    reset_count: int,
    per_reset_timeout_seconds: float,
    sleep_between_resets_seconds: float = 0.0,
) -> ResetValidationResult:
    latest = client.wait_for_frame(timeout=client.config.initial_frame_timeout)
    expected_session = latest.session_id
    expected_map_uid = latest.map_uid
    succeeded = 0
    failures: list[str] = []

    for index in range(reset_count):
        before = client.get_latest_frame()
        if before is None:
            failures.append(f"Reset {index + 1}: no latest frame was available before reset.")
            break

        previous_run_id = before.run_id
        response = client.reset_to_start(timeout=per_reset_timeout_seconds)
        if not response.success:
            failures.append(f"Reset {index + 1}: bridge reported failure: {response.message}")
            continue

        target_run_id = response.payload.get("run_id")
        target_frame_id = response.payload.get("frame_id")
        target_timestamp_ns = response.payload.get("timestamp_ns")
        target_map_uid = response.payload.get("map_uid")
        target_race_state = response.payload.get("race_state")
        if target_race_state != "start_line":
            failures.append(f"Reset {index + 1}: reset response race_state was {target_race_state!r}, expected 'start_line'.")
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue
        if target_map_uid != expected_map_uid:
            failures.append(
                f"Reset {index + 1}: reset response map_uid was {target_map_uid!r}, expected {expected_map_uid!r}."
            )
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue
        deadline = time.monotonic() + per_reset_timeout_seconds
        acknowledged_frame: TelemetryFrame | None = None
        while time.monotonic() < deadline:
            candidate = client.wait_for_frame(after_frame_id=before.frame_id, timeout=0.5)
            if candidate.run_id == previous_run_id:
                continue
            if isinstance(target_run_id, str) and candidate.run_id != target_run_id:
                continue
            acknowledged_frame = candidate
            break

        if acknowledged_frame is None:
            failures.append(f"Reset {index + 1}: timed out waiting for a new run_id.")
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue

        if acknowledged_frame.session_id != expected_session:
            failures.append(
                f"Reset {index + 1}: session_id changed from {expected_session} to {acknowledged_frame.session_id}."
            )
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue
        if expected_map_uid and acknowledged_frame.map_uid != expected_map_uid:
            failures.append(
                f"Reset {index + 1}: map_uid changed from {expected_map_uid} to {acknowledged_frame.map_uid}."
            )
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue
        if acknowledged_frame.run_id == previous_run_id:
            failures.append(f"Reset {index + 1}: run_id did not change after reset.")
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue
        if target_run_id != acknowledged_frame.run_id:
            failures.append(
                f"Reset {index + 1}: response run_id {target_run_id!r} did not match telemetry {acknowledged_frame.run_id!r}."
            )
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue
        if target_frame_id != acknowledged_frame.frame_id:
            failures.append(
                f"Reset {index + 1}: response frame_id {target_frame_id!r} did not match telemetry {acknowledged_frame.frame_id!r}."
            )
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue
        if target_timestamp_ns != acknowledged_frame.timestamp_ns:
            failures.append(
                "Reset "
                f"{index + 1}: response timestamp_ns {target_timestamp_ns!r} did not match telemetry {acknowledged_frame.timestamp_ns!r}."
            )
            if sleep_between_resets_seconds > 0.0:
                time.sleep(sleep_between_resets_seconds)
            continue

        succeeded += 1
        if sleep_between_resets_seconds > 0.0:
            time.sleep(sleep_between_resets_seconds)

    return ResetValidationResult(requested=reset_count, succeeded=succeeded, failures=failures)
