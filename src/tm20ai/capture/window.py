from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class WindowGeometry:
    hwnd: int
    title: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_region(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(slots=True, frozen=True)
class MonitorGeometry:
    handle: int
    device_name: str
    left: int
    top: int
    right: int
    bottom: int
    is_primary: bool

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(slots=True, frozen=True)
class TrackmaniaWindowLocator:
    title_substring: str = "Trackmania"

    def locate_tm_window(self) -> tuple[int, WindowGeometry]:
        hwnd = find_window(self.title_substring)
        return hwnd, get_client_geometry(hwnd)


def _load_win32_modules():
    try:
        import win32api
        import win32con
        import win32gui
        import win32process
    except ImportError as exc:  # pragma: no cover - Windows dependency guard
        raise RuntimeError("pywin32 is required to locate the Trackmania window.") from exc
    return win32api, win32con, win32gui, win32process


def _window_matches(title_fragment: str, *, title: str, process_path: str | None) -> bool:
    normalized_title = title.strip().lower()
    process_name = Path(process_path).name.lower() if process_path else ""
    if process_name.startswith(title_fragment):
        return True
    if normalized_title == title_fragment:
        return True
    if normalized_title.startswith(f"{title_fragment} "):
        return True
    return False


def _get_process_path(win32api, win32con, win32process, hwnd: int) -> str | None:
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
        try:
            return win32process.GetModuleFileNameEx(handle, 0)
        finally:
            win32api.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        return None


def find_window(title_substring: str = "Trackmania") -> int:
    win32api, win32con, win32gui, win32process = _load_win32_modules()
    title_fragment = title_substring.lower()
    hwnds: list[int] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        process_path = _get_process_path(win32api, win32con, win32process, hwnd)
        if _window_matches(title_fragment, title=title, process_path=process_path):
            hwnds.append(hwnd)
        return True

    win32gui.EnumWindows(callback, 0)
    if not hwnds:
        raise RuntimeError(f"Could not find a visible window containing {title_substring!r}.")
    return hwnds[0]


def get_client_geometry(hwnd: int) -> WindowGeometry:
    _, _win32con, win32gui, _ = _load_win32_modules()
    title = win32gui.GetWindowText(hwnd)
    client_rect = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
    right, bottom = win32gui.ClientToScreen(hwnd, (client_rect[2], client_rect[3]))
    return WindowGeometry(hwnd=hwnd, title=title, left=left, top=top, right=right, bottom=bottom)


def get_window_monitor_geometry(hwnd: int) -> MonitorGeometry:
    win32api, win32con, _win32gui, _win32process = _load_win32_modules()
    monitor_handle = int(win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST))
    info: dict[str, Any] = win32api.GetMonitorInfo(monitor_handle)
    left, top, right, bottom = info["Monitor"]
    flags = int(info.get("Flags", 0))
    return MonitorGeometry(
        handle=monitor_handle,
        device_name=str(info.get("Device", "")),
        left=int(left),
        top=int(top),
        right=int(right),
        bottom=int(bottom),
        is_primary=bool(flags & 1),
    )
