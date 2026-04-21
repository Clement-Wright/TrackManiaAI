from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..data.parquet_writer import sha256_file, write_json


CORE_BASE_URL = "https://prod.trackmania.core.nadeo.online"
LIVE_BASE_URL = "https://live-services.trackmania.nadeo.live"
AUTH_BASE_URL = "https://prod.trackmania.core.nadeo.online/v2/authentication/token/basic"


@dataclass(frozen=True, slots=True)
class NadeoCredentials:
    dedicated_login: str
    dedicated_password: str
    user_agent: str
    core_token: str | None = None
    live_token: str | None = None

    @classmethod
    def from_env(cls) -> "NadeoCredentials":
        login = _read_environment_value("TM20AI_NADEO_DEDI_LOGIN")
        password = _read_environment_value("TM20AI_NADEO_DEDI_PASSWORD")
        user_agent = _read_environment_value("TM20AI_NADEO_USER_AGENT")
        if not login or not password or not user_agent:
            raise RuntimeError(
                "Nadeo ghost ingestion requires environment variables "
                "TM20AI_NADEO_DEDI_LOGIN, TM20AI_NADEO_DEDI_PASSWORD, and TM20AI_NADEO_USER_AGENT."
            )
        return cls(
            dedicated_login=login,
            dedicated_password=password,
            user_agent=user_agent,
            core_token=_read_environment_value("TM20AI_NADEO_CORE_TOKEN"),
            live_token=_read_environment_value("TM20AI_NADEO_LIVE_TOKEN"),
        )


def _read_environment_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows guard
        return None
    for hive, subkey in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                registry_value, _value_type = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if registry_value:
            return str(registry_value)
    try:
        user_index = 0
        while True:
            sid = winreg.EnumKey(winreg.HKEY_USERS, user_index)
            user_index += 1
            try:
                with winreg.OpenKey(winreg.HKEY_USERS, rf"{sid}\Environment") as key:
                    registry_value, _value_type = winreg.QueryValueEx(key, name)
            except OSError:
                continue
            if registry_value:
                return str(registry_value)
    except OSError:
        pass
    return None


class NadeoServicesClient:
    def __init__(
        self,
        credentials: NadeoCredentials,
        *,
        core_base_url: str = CORE_BASE_URL,
        live_base_url: str = LIVE_BASE_URL,
        auth_url: str = AUTH_BASE_URL,
    ) -> None:
        self.credentials = credentials
        self.core_base_url = core_base_url.rstrip("/")
        self.live_base_url = live_base_url.rstrip("/")
        self.auth_url = auth_url
        self._core_token = credentials.core_token
        self._live_token = credentials.live_token

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        token: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "User-Agent": self.credentials.user_agent,
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        if token is not None:
            headers["Authorization"] = f"nadeo_v1 t={token}"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:  # noqa: S310 - user-requested API client
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Nadeo request failed {exc.code} for {url}: {body}") from exc

    def _auth_token(self, audience: str) -> str:
        basic = base64.b64encode(
            f"{self.credentials.dedicated_login}:{self.credentials.dedicated_password}".encode("utf-8")
        ).decode("ascii")
        request = urllib.request.Request(
            self.auth_url,
            data=json.dumps({"audience": audience}).encode("utf-8"),
            headers={
                "User-Agent": self.credentials.user_agent,
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Basic {basic}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:  # noqa: S310 - user-requested API client
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Nadeo authentication failed {exc.code} for audience {audience}: {body}") from exc
        token = payload.get("accessToken") or payload.get("token")
        if not token:
            raise RuntimeError(f"Nadeo authentication response for {audience} did not include an access token.")
        return str(token)

    @property
    def core_token(self) -> str:
        if self._core_token is None:
            self._core_token = self._auth_token("NadeoServices")
        return self._core_token

    @property
    def live_token(self) -> str:
        if self._live_token is None:
            self._live_token = self._auth_token("NadeoLiveServices")
        return self._live_token

    def resolve_map_uid(self, map_uid: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"mapUidList": map_uid})
        payload = self._request_json(
            f"{self.core_base_url}/maps/by-uid/?{query}",
            token=self.core_token,
        )
        if isinstance(payload, list) and payload:
            return dict(payload[0])
        if isinstance(payload, Mapping):
            maps = payload.get("mapList") or payload.get("maps") or payload.get("data")
            if isinstance(maps, list) and maps:
                return dict(maps[0])
        raise RuntimeError(f"Could not resolve map UID {map_uid!r} from Nadeo response.")

    def leaderboard_top(
        self,
        *,
        map_uid: str,
        group_uid: str,
        length: int,
        only_world: bool = True,
    ) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode(
            {
                "mapUid": map_uid,
                "groupUid": group_uid,
                "offset": 0,
                "length": int(length),
                "onlyWorld": str(bool(only_world)).lower(),
            }
        )
        payload = self._request_json(
            f"{self.live_base_url}/api/token/leaderboard/group/{group_uid}/map/{map_uid}/top?{query}",
            token=self.live_token,
        )
        rows = payload.get("tops") or payload.get("top") or payload.get("scores") if isinstance(payload, Mapping) else payload
        if isinstance(rows, list) and rows and isinstance(rows[0], Mapping) and isinstance(rows[0].get("top"), list):
            rows = rows[0]["top"]
        if isinstance(rows, Mapping):
            rows = rows.get("records") or rows.get("scores") or rows.get("tops") or rows.get("top")
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected leaderboard response shape for map UID {map_uid!r}.")
        return [dict(row) for row in rows[:length]]

    def map_records_by_accounts(self, *, map_id: str, account_ids: list[str]) -> list[dict[str, Any]]:
        if not account_ids:
            return []
        query = urllib.parse.urlencode({"accountIdList": ",".join(account_ids)})
        payload = self._request_json(
            f"{self.core_base_url}/v2/mapRecords/by-account/?mapId={urllib.parse.quote(map_id)}&{query}",
            token=self.core_token,
        )
        records = payload.get("mapRecordList") or payload.get("records") if isinstance(payload, Mapping) else payload
        if not isinstance(records, list):
            raise RuntimeError(f"Unexpected map-record response shape for map ID {map_id!r}.")
        return [dict(record) for record in records]

    def download_replay(self, url: str, destination: str | Path) -> dict[str, Any]:
        destination_path = Path(destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": self.credentials.user_agent}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=60.0) as response:  # noqa: S310 - user-requested download
                destination_path.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "status": exc.code,
                "error": body,
                "path": str(destination_path),
            }
        return {
            "ok": True,
            "status": 200,
            "path": str(destination_path),
            "sha256": sha256_file(destination_path),
            "bytes": destination_path.stat().st_size,
        }


def fetch_top100_ghost_manifest(
    *,
    map_uid: str,
    output_dir: str | Path,
    leaderboard_length: int = 100,
    group_uid: str = "Personal_Best",
    only_world: bool = True,
    client: NadeoServicesClient | None = None,
) -> Path:
    client = client or NadeoServicesClient(NadeoCredentials.from_env())
    output_root = Path(output_dir).resolve()
    replay_dir = output_root / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    map_info = client.resolve_map_uid(map_uid)
    map_id = str(map_info.get("mapId") or map_info.get("uid") or map_uid)
    leaderboard_rows = client.leaderboard_top(
        map_uid=map_uid,
        group_uid=group_uid,
        length=leaderboard_length,
        only_world=only_world,
    )
    account_ids = [
        str(row.get("accountId") or row.get("account_id"))
        for row in leaderboard_rows
        if row.get("accountId") or row.get("account_id")
    ]
    record_rows = client.map_records_by_accounts(map_id=map_id, account_ids=account_ids)
    records_by_account = {
        str(row.get("accountId") or row.get("account_id")): row
        for row in record_rows
        if row.get("accountId") or row.get("account_id")
    }
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(leaderboard_rows, start=1):
        account_id = str(row.get("accountId") or row.get("account_id") or "")
        record = dict(records_by_account.get(account_id, {}))
        replay_url = record.get("url") or record.get("replayUrl") or record.get("replay_url")
        replay_result = {"ok": False, "status": None, "path": None, "error": "missing_replay_url"}
        if replay_url:
            replay_result = client.download_replay(
                str(replay_url),
                replay_dir / f"rank_{index:03d}_{account_id or 'unknown'}.gbx",
            )
        entries.append(
            {
                "rank": int(row.get("rank") or index),
                "account_id": account_id,
                "score": row.get("score"),
                "time_ms": row.get("time") or row.get("score"),
                "leaderboard_row": row,
                "record_metadata": record,
                "replay_url": replay_url,
                "fetch_status": replay_result,
            }
        )
    manifest_path = output_root / "top100_ghost_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "nadeo_top100_ghost_manifest_v1",
            "map_uid": map_uid,
            "map_id": map_id,
            "map_info": map_info,
            "group_uid": group_uid,
            "only_world": bool(only_world),
            "leaderboard_length": int(leaderboard_length),
            "entries": entries,
        },
    )
    return manifest_path
