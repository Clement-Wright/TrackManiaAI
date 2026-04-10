from __future__ import annotations

import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from .messages import (
    BridgeHealth,
    BridgeProtocolError,
    CommandRequest,
    CommandResponse,
    JsonValue,
    TelemetryFrame,
    validate_command_response_payload,
)


@dataclass(slots=True)
class BridgeConnectionConfig:
    host: str = "127.0.0.1"
    telemetry_port: int = 9100
    command_port: int = 9101
    connect_timeout: float = 5.0
    command_timeout: float = 5.0
    initial_frame_timeout: float = 10.0
    reconnect_delay: float = 1.0
    stale_timeout: float = 0.25
    reset_timeout: float = 5.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BridgeConnectionConfig":
        return cls(
            host=str(payload.get("host", "127.0.0.1")),
            telemetry_port=int(payload.get("telemetry_port", 9100)),
            command_port=int(payload.get("command_port", 9101)),
            connect_timeout=float(payload.get("connect_timeout", 5.0)),
            command_timeout=float(payload.get("command_timeout", 5.0)),
            initial_frame_timeout=float(payload.get("initial_frame_timeout", 10.0)),
            reconnect_delay=float(payload.get("reconnect_delay", 1.0)),
            stale_timeout=float(payload.get("stale_timeout", 0.25)),
            reset_timeout=float(payload.get("reset_timeout", 5.0)),
        )


class BridgeClient:
    """Persistent client for the custom dual-port Openplanet bridge."""

    def __init__(self, config: BridgeConnectionConfig):
        self.config = config
        self._stop_event = threading.Event()
        self._frame_event = threading.Event()
        self._lock = threading.Lock()
        self._command_lock = threading.Lock()

        self._telemetry_thread: threading.Thread | None = None
        self._latest_frame: TelemetryFrame | None = None
        self._latest_receive_monotonic: float | None = None
        self._received_frames: deque[TelemetryFrame] = deque(maxlen=4096)
        self._telemetry_disconnects = 0
        self._telemetry_connections = 0
        self._last_telemetry_error: str | None = None

        self._command_socket: socket.socket | None = None
        self._command_reader = None
        self._command_writer = None

    def start(self) -> None:
        if self._telemetry_thread is not None:
            return
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, name="tm20ai-telemetry", daemon=True)
        self._telemetry_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self._close_command_stream()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=2.0)
            self._telemetry_thread = None

    def __enter__(self) -> "BridgeClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def telemetry_disconnects(self) -> int:
        with self._lock:
            return self._telemetry_disconnects

    @property
    def telemetry_connections(self) -> int:
        with self._lock:
            return self._telemetry_connections

    @property
    def last_telemetry_error(self) -> str | None:
        with self._lock:
            return self._last_telemetry_error

    def get_latest_frame(self) -> TelemetryFrame | None:
        with self._lock:
            return self._latest_frame

    def pop_received_frames(self) -> list[TelemetryFrame]:
        with self._lock:
            frames = list(self._received_frames)
            self._received_frames.clear()
        return frames

    def get_received_frames_snapshot(self) -> list[TelemetryFrame]:
        with self._lock:
            return list(self._received_frames)

    def wait_for_frame(self, *, after_frame_id: int | None = None, timeout: float | None = None) -> TelemetryFrame:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                latest = self._latest_frame
            if latest is not None and (after_frame_id is None or latest.frame_id > after_frame_id):
                return latest

            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0.0:
                raise TimeoutError("Timed out waiting for a telemetry frame.")
            self._frame_event.wait(timeout=0.25 if remaining is None else min(0.25, remaining))
            self._frame_event.clear()

    def is_stale(self, stale_after: float | None = None) -> bool:
        threshold = stale_after if stale_after is not None else self.config.stale_timeout
        with self._lock:
            latest_receive = self._latest_receive_monotonic
        if latest_receive is None:
            return True
        return (time.monotonic() - latest_receive) > threshold

    def health(self, *, timeout: float | None = None) -> BridgeHealth:
        response = self.request("health", timeout=timeout)
        if not response.success:
            raise RuntimeError(f"Bridge health request failed: {response.message}")
        validate_command_response_payload("health", response.payload)
        return BridgeHealth.from_mapping(response.payload)

    def race_state(self, *, timeout: float | None = None) -> CommandResponse:
        response = self.request("race_state", timeout=timeout)
        if not response.success:
            raise RuntimeError(f"Bridge race_state request failed: {response.message}")
        validate_command_response_payload("race_state", response.payload)
        return response

    def set_recording_mode(self, enabled: bool, *, timeout: float | None = None) -> CommandResponse:
        response = self.request("set_recording_mode", {"enabled": enabled}, timeout=timeout)
        if response.success:
            validate_command_response_payload("set_recording_mode", response.payload)
        return response

    def reset_to_start(self, *, timeout: float | None = None) -> CommandResponse:
        effective_timeout = self.config.reset_timeout if timeout is None else timeout
        response = self.request(
            "reset_to_start",
            {"timeout_ms": int(round(effective_timeout * 1000.0))},
            timeout=max(self.config.command_timeout, effective_timeout + 1.0),
        )
        if response.success:
            validate_command_response_payload("reset_to_start", response.payload)
        return response

    def request(
        self,
        command: str,
        payload: Mapping[str, JsonValue] | None = None,
        *,
        timeout: float | None = None,
    ) -> CommandResponse:
        request = CommandRequest.new(command, payload)
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._command_lock:
            for attempt in range(2):
                try:
                    self._ensure_command_stream()
                    assert self._command_writer is not None
                    assert self._command_reader is not None
                    self._command_writer.write(request.to_json_line())
                    self._command_writer.flush()

                    remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                    if remaining is not None and remaining <= 0.0:
                        raise TimeoutError(f"Timed out waiting for command response to {command}.")
                    if self._command_socket is not None:
                        self._command_socket.settimeout(self.config.command_timeout if remaining is None else remaining)

                    line = self._command_reader.readline()
                    if not line:
                        raise ConnectionError("Command bridge closed the socket.")
                    response = CommandResponse.from_json_line(line)
                    if response.request_id != request.request_id:
                        raise BridgeProtocolError(
                            f"Command response request_id mismatch: expected {request.request_id}, got {response.request_id}"
                        )
                    return response
                except (OSError, TimeoutError, ValueError, BridgeProtocolError) as exc:
                    self._close_command_stream()
                    if attempt == 1:
                        raise RuntimeError(f"Command request {command!r} failed: {exc}") from exc

        raise RuntimeError(f"Command request {command!r} failed unexpectedly.")

    def _telemetry_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with socket.create_connection(
                    (self.config.host, self.config.telemetry_port),
                    timeout=self.config.connect_timeout,
                ) as telemetry_socket:
                    telemetry_socket.settimeout(1.0)
                    with self._lock:
                        self._telemetry_connections += 1
                        self._last_telemetry_error = None

                    buffer = bytearray()
                    while not self._stop_event.is_set():
                        chunk = telemetry_socket.recv(4096)
                        if not chunk:
                            raise ConnectionError("Telemetry bridge closed the socket.")
                        buffer.extend(chunk)

                        while True:
                            separator = buffer.find(b"\n")
                            if separator < 0:
                                break
                            raw_line = bytes(buffer[:separator]).strip()
                            del buffer[: separator + 1]
                            if not raw_line:
                                continue
                            frame = TelemetryFrame.from_json_line(raw_line.decode("utf-8"))
                            with self._lock:
                                self._latest_frame = frame
                                self._latest_receive_monotonic = time.monotonic()
                                self._received_frames.append(frame)
                            self._frame_event.set()
            except Exception as exc:  # noqa: BLE001 - reconnect loop wants the original error string
                with self._lock:
                    self._telemetry_disconnects += 1
                    self._last_telemetry_error = str(exc)
                if self._stop_event.wait(self.config.reconnect_delay):
                    return

    def _ensure_command_stream(self) -> None:
        if self._command_socket is not None and self._command_reader is not None and self._command_writer is not None:
            return
        command_socket = socket.create_connection(
            (self.config.host, self.config.command_port),
            timeout=self.config.connect_timeout,
        )
        command_socket.settimeout(self.config.command_timeout)
        self._command_socket = command_socket
        self._command_reader = command_socket.makefile("r", encoding="utf-8", newline="\n")
        self._command_writer = command_socket.makefile("w", encoding="utf-8", newline="\n")

    def _close_command_stream(self) -> None:
        for handle_name in ("_command_reader", "_command_writer"):
            handle = getattr(self, handle_name)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, handle_name, None)

        if self._command_socket is not None:
            try:
                self._command_socket.close()
            except OSError:
                pass
            self._command_socket = None
