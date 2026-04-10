from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

VALID_RACE_STATES = frozenset({"outside_race", "start_line", "running", "finished"})
VALID_TERMINAL_REASONS = frozenset({"finished", "outside_active_race", "map_changed"})
VALID_COMMANDS = frozenset({"health", "race_state", "reset_to_start", "set_recording_mode"})


class BridgeProtocolError(RuntimeError):
    """Raised when the custom bridge payload does not match the contract."""


def _require_key(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise BridgeProtocolError(f"Missing required field: {key}")
    return payload[key]


def _ensure_only_keys(payload: Mapping[str, Any], allowed_keys: set[str], context: str) -> None:
    unexpected = sorted(set(payload) - allowed_keys)
    if unexpected:
        raise BridgeProtocolError(f"Unexpected field(s) in {context}: {', '.join(unexpected)}")


def _parse_str(payload: Mapping[str, Any], key: str, *, allow_none: bool = False) -> str | None:
    value = _require_key(payload, key)
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise BridgeProtocolError(f"Field {key} must be a string.")
    return value


def _parse_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = _require_key(payload, key)
    if not isinstance(value, bool):
        raise BridgeProtocolError(f"Field {key} must be a bool.")
    return value


def _parse_int(payload: Mapping[str, Any], key: str) -> int:
    value = _require_key(payload, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgeProtocolError(f"Field {key} must be an int.")
    return value


def _parse_float(payload: Mapping[str, Any], key: str) -> float:
    value = _require_key(payload, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeProtocolError(f"Field {key} must be a number.")
    return float(value)


def _parse_vec3(payload: Mapping[str, Any], key: str) -> tuple[float, float, float] | None:
    value = _require_key(payload, key)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise BridgeProtocolError(f"Field {key} must be null or a 3-element list.")
    coords: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise BridgeProtocolError(f"Field {key} must contain only numbers.")
        coords.append(float(item))
    return (coords[0], coords[1], coords[2])


def _parse_payload_dict(raw: Any, field_name: str = "payload") -> dict[str, JsonValue]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise BridgeProtocolError(f"Field {field_name} must be an object.")
    return dict(raw)


def _parse_race_state(payload: Mapping[str, Any], key: str, *, allow_none: bool = False) -> str | None:
    value = _parse_str(payload, key, allow_none=allow_none)
    if value is None:
        return None
    if value not in VALID_RACE_STATES:
        raise BridgeProtocolError(
            f"Field {key} must be one of {sorted(VALID_RACE_STATES)}, got {value!r}."
        )
    return value


def _parse_terminal_reason(payload: Mapping[str, Any], key: str) -> str | None:
    value = _parse_str(payload, key, allow_none=True)
    if value is None:
        return None
    if value not in VALID_TERMINAL_REASONS:
        raise BridgeProtocolError(
            f"Field {key} must be one of {sorted(VALID_TERMINAL_REASONS)} or null, got {value!r}."
        )
    return value


def validate_command_request_payload(command: str, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    payload_dict = dict(payload)
    if command == "health":
        _ensure_only_keys(payload_dict, set(), "health payload")
        return payload_dict
    if command == "race_state":
        _ensure_only_keys(payload_dict, set(), "race_state payload")
        return payload_dict
    if command == "set_recording_mode":
        _ensure_only_keys(payload_dict, {"enabled"}, "set_recording_mode payload")
        if not isinstance(payload_dict.get("enabled"), bool):
            raise BridgeProtocolError("set_recording_mode payload must contain a boolean enabled field.")
        return payload_dict
    if command == "reset_to_start":
        _ensure_only_keys(payload_dict, {"timeout_ms"}, "reset_to_start payload")
        timeout_ms = payload_dict.get("timeout_ms")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise BridgeProtocolError("reset_to_start payload must contain a positive integer timeout_ms field.")
        return payload_dict
    raise BridgeProtocolError(f"Unsupported command: {command}")


def validate_command_response_payload(command: str, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    payload_dict = dict(payload)
    if command == "health":
        BridgeHealth.from_mapping(payload_dict)
        return payload_dict

    if command == "race_state":
        _ensure_only_keys(payload_dict, {"race_state", "session_id", "run_id", "map_uid"}, "race_state response")
        _parse_race_state(payload_dict, "race_state")
        _parse_str(payload_dict, "session_id")
        _parse_str(payload_dict, "run_id")
        _parse_str(payload_dict, "map_uid")
        return payload_dict

    if command == "set_recording_mode":
        _ensure_only_keys(
            payload_dict,
            {"recording_mode", "session_id", "run_id", "map_uid"},
            "set_recording_mode response",
        )
        _parse_bool(payload_dict, "recording_mode")
        _parse_str(payload_dict, "session_id")
        _parse_str(payload_dict, "run_id")
        _parse_str(payload_dict, "map_uid")
        return payload_dict

    if command == "reset_to_start":
        _ensure_only_keys(
            payload_dict,
            {"run_id", "frame_id", "timestamp_ns", "map_uid", "race_state"},
            "reset_to_start response",
        )
        _parse_str(payload_dict, "run_id")
        _parse_int(payload_dict, "frame_id")
        _parse_int(payload_dict, "timestamp_ns")
        _parse_str(payload_dict, "map_uid")
        race_state = _parse_race_state(payload_dict, "race_state")
        if race_state != "start_line":
            raise BridgeProtocolError(
                f"reset_to_start response must acknowledge race_state='start_line', got {race_state!r}."
            )
        return payload_dict

    raise BridgeProtocolError(f"Unsupported command: {command}")


@dataclass(slots=True)
class TelemetryFrame:
    session_id: str
    run_id: str
    frame_id: int
    timestamp_ns: int
    map_uid: str
    race_time_ms: int
    cp_count: int
    cp_target: int
    speed_kmh: float
    gear: int
    rpm: float
    pos_xyz: tuple[float, float, float] | None
    vel_xyz: tuple[float, float, float] | None
    yaw_pitch_roll: tuple[float, float, float] | None
    finished: bool
    terminal_reason: str | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TelemetryFrame":
        expected_keys = {
            "session_id",
            "run_id",
            "frame_id",
            "timestamp_ns",
            "map_uid",
            "race_time_ms",
            "cp_count",
            "cp_target",
            "speed_kmh",
            "gear",
            "rpm",
            "pos_xyz",
            "vel_xyz",
            "yaw_pitch_roll",
            "finished",
            "terminal_reason",
        }
        _ensure_only_keys(payload, expected_keys, "telemetry frame")
        frame = cls(
            session_id=_parse_str(payload, "session_id") or "",
            run_id=_parse_str(payload, "run_id") or "",
            frame_id=_parse_int(payload, "frame_id"),
            timestamp_ns=_parse_int(payload, "timestamp_ns"),
            map_uid=_parse_str(payload, "map_uid") or "",
            race_time_ms=_parse_int(payload, "race_time_ms"),
            cp_count=_parse_int(payload, "cp_count"),
            cp_target=_parse_int(payload, "cp_target"),
            speed_kmh=_parse_float(payload, "speed_kmh"),
            gear=_parse_int(payload, "gear"),
            rpm=_parse_float(payload, "rpm"),
            pos_xyz=_parse_vec3(payload, "pos_xyz"),
            vel_xyz=_parse_vec3(payload, "vel_xyz"),
            yaw_pitch_roll=_parse_vec3(payload, "yaw_pitch_roll"),
            finished=_parse_bool(payload, "finished"),
            terminal_reason=_parse_terminal_reason(payload, "terminal_reason"),
        )
        if frame.frame_id < 0:
            raise BridgeProtocolError("frame_id must be non-negative.")
        if frame.timestamp_ns < 0:
            raise BridgeProtocolError("timestamp_ns must be non-negative.")
        if frame.race_time_ms < 0:
            raise BridgeProtocolError("race_time_ms must be non-negative.")
        if frame.cp_count < 0 or frame.cp_target < 0:
            raise BridgeProtocolError("cp_count and cp_target must be non-negative.")
        if frame.finished and frame.terminal_reason not in {None, "finished"}:
            raise BridgeProtocolError(
                "finished telemetry frames may only use terminal_reason null or 'finished'."
            )
        return frame

    @classmethod
    def from_json_line(cls, line: str) -> "TelemetryFrame":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise BridgeProtocolError("Telemetry frame must be a JSON object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "map_uid": self.map_uid,
            "race_time_ms": self.race_time_ms,
            "cp_count": self.cp_count,
            "cp_target": self.cp_target,
            "speed_kmh": self.speed_kmh,
            "gear": self.gear,
            "rpm": self.rpm,
            "pos_xyz": list(self.pos_xyz) if self.pos_xyz is not None else None,
            "vel_xyz": list(self.vel_xyz) if self.vel_xyz is not None else None,
            "yaw_pitch_roll": list(self.yaw_pitch_roll) if self.yaw_pitch_roll is not None else None,
            "finished": self.finished,
            "terminal_reason": self.terminal_reason,
        }


@dataclass(slots=True)
class BridgeHealth:
    ok: bool
    heartbeat_ns: int
    session_id: str | None = None
    run_id: str | None = None
    map_uid: str | None = None
    race_state: str | None = None
    recording_mode: bool = False
    telemetry_clients: int = 0
    command_clients: int = 0
    plugin_version: str | None = None
    last_frame_id: int | None = None
    last_timestamp_ns: int | None = None
    message: str | None = None
    extra: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BridgeHealth":
        known_keys = {
            "ok",
            "heartbeat_ns",
            "session_id",
            "run_id",
            "map_uid",
            "race_state",
            "recording_mode",
            "telemetry_clients",
            "command_clients",
            "plugin_version",
            "last_frame_id",
            "last_timestamp_ns",
            "message",
        }
        extra = {key: value for key, value in payload.items() if key not in known_keys}
        return cls(
            ok=_parse_bool(payload, "ok"),
            heartbeat_ns=_parse_int(payload, "heartbeat_ns"),
            session_id=_parse_str(payload, "session_id", allow_none=True),
            run_id=_parse_str(payload, "run_id", allow_none=True),
            map_uid=_parse_str(payload, "map_uid", allow_none=True),
            race_state=_parse_race_state(payload, "race_state", allow_none=True),
            recording_mode=bool(payload.get("recording_mode", False)),
            telemetry_clients=int(payload.get("telemetry_clients", 0)),
            command_clients=int(payload.get("command_clients", 0)),
            plugin_version=_parse_str(payload, "plugin_version", allow_none=True)
            if "plugin_version" in payload
            else None,
            last_frame_id=int(payload["last_frame_id"]) if payload.get("last_frame_id") is not None else None,
            last_timestamp_ns=int(payload["last_timestamp_ns"]) if payload.get("last_timestamp_ns") is not None else None,
            message=_parse_str(payload, "message", allow_none=True) if "message" in payload else None,
            extra=extra,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "ok": self.ok,
            "heartbeat_ns": self.heartbeat_ns,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "map_uid": self.map_uid,
            "race_state": self.race_state,
            "recording_mode": self.recording_mode,
            "telemetry_clients": self.telemetry_clients,
            "command_clients": self.command_clients,
            "plugin_version": self.plugin_version,
            "last_frame_id": self.last_frame_id,
            "last_timestamp_ns": self.last_timestamp_ns,
            "message": self.message,
        }
        data.update(self.extra)
        return data


@dataclass(slots=True)
class CommandRequest:
    request_id: str
    command: str
    payload: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def new(cls, command: str, payload: Mapping[str, JsonValue] | None = None) -> "CommandRequest":
        payload_dict = validate_command_request_payload(command, dict(payload or {}))
        return cls(request_id=uuid.uuid4().hex, command=command, payload=payload_dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CommandRequest":
        _ensure_only_keys(payload, {"request_id", "command", "payload"}, "command request")
        command = _parse_str(payload, "command") or ""
        if command not in VALID_COMMANDS:
            raise BridgeProtocolError(f"Unsupported command: {command}")
        payload_dict = validate_command_request_payload(command, _parse_payload_dict(payload.get("payload")))
        return cls(
            request_id=_parse_str(payload, "request_id") or "",
            command=command,
            payload=payload_dict,
        )

    @classmethod
    def from_json_line(cls, line: str) -> "CommandRequest":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise BridgeProtocolError("Command request must be a JSON object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "request_id": self.request_id,
            "command": self.command,
            "payload": self.payload,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":")) + "\n"


@dataclass(slots=True)
class CommandResponse:
    request_id: str
    success: bool
    message: str
    payload: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CommandResponse":
        _ensure_only_keys(payload, {"request_id", "success", "message", "payload"}, "command response")
        return cls(
            request_id=_parse_str(payload, "request_id") or "",
            success=_parse_bool(payload, "success"),
            message=_parse_str(payload, "message") or "",
            payload=_parse_payload_dict(payload.get("payload")),
        )

    @classmethod
    def from_json_line(cls, line: str) -> "CommandResponse":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise BridgeProtocolError("Command response must be a JSON object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "message": self.message,
            "payload": self.payload,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":")) + "\n"
