from __future__ import annotations

import pytest

from tm20ai.env.live_lock import LiveEnvLock


def test_live_env_lock_enforces_single_owner_and_releases_cleanly(tmp_path) -> None:
    lock_path = tmp_path / "artifacts" / "live_env.lock"
    first = LiveEnvLock(lock_path)
    second = LiveEnvLock(lock_path)

    first.acquire()
    with pytest.raises(RuntimeError):
        second.acquire()

    first.release()
    second.acquire()
    second.release()
