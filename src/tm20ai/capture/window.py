from __future__ import annotations

from dataclasses import dataclass


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


def _load_win32_modules():
    try:
        import win32gui
    except ImportError as exc:  # pragma: no cover - Windows dependency guard
        raise RuntimeError("pywin32 is required to locate the Trackmania window.") from exc
    return win32gui


def find_window(title_substring: str = "Trackmania") -> int:
    win32gui = _load_win32_modules()
    title_fragment = title_substring.lower()
    hwnds: list[int] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if title_fragment in title.lower():
            hwnds.append(hwnd)
        return True

    win32gui.EnumWindows(callback, 0)
    if not hwnds:
        raise RuntimeError(f"Could not find a visible window containing {title_substring!r}.")
    return hwnds[0]


def get_client_geometry(hwnd: int) -> WindowGeometry:
    win32gui = _load_win32_modules()
    title = win32gui.GetWindowText(hwnd)
    client_rect = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
    right, bottom = win32gui.ClientToScreen(hwnd, (client_rect[2], client_rect[3]))
    return WindowGeometry(hwnd=hwnd, title=title, left=left, top=top, right=right, bottom=bottom)
