from __future__ import annotations

import json
import msvcrt
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ClassVar

from ..config import TM20AIConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_live_env_lock_path(config: TM20AIConfig) -> Path:
    artifacts_root = Path(config.artifacts.root)
    if not artifacts_root.is_absolute():
        artifacts_root = _repo_root() / artifacts_root
    return artifacts_root / "live_env.lock"


@dataclass(slots=True)
class LiveEnvLock:
    path: Path

    _process_lock: ClassVar[threading.Lock] = threading.Lock()
    _held_paths: ClassVar[set[Path]] = set()

    _handle: BinaryIO | None = None
    _acquired: bool = False

    def acquire(self) -> None:
        normalized = self.path.resolve()
        with self._process_lock:
            if normalized in self._held_paths:
                raise RuntimeError(
                    f"Live env lock is already held in this process: {normalized}. "
                    "Only one Trackmania live env client is allowed at a time."
                )

        normalized.parent.mkdir(parents=True, exist_ok=True)
        handle = normalized.open("a+b")
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                f"Could not acquire live env lock at {normalized}. "
                "Another live env client is already running."
            ) from exc

        payload = {"pid": os.getpid(), "path": str(normalized)}
        handle.seek(0)
        handle.truncate(0)
        handle.write(json.dumps(payload, indent=2).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())

        with self._process_lock:
            self._held_paths.add(normalized)
        self.path = normalized
        self._handle = handle
        self._acquired = True

    def release(self) -> None:
        if not self._acquired or self._handle is None:
            return
        try:
            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            try:
                self._handle.close()
            finally:
                with self._process_lock:
                    self._held_paths.discard(self.path.resolve())
                self._handle = None
                self._acquired = False

    def __enter__(self) -> "LiveEnvLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
