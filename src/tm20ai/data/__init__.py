"""Data artifact helpers for evaluation, demos, and datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["DemoRecorder", "EpisodeDataset", "EpisodeRecord"]

if TYPE_CHECKING:
    from .dataset import EpisodeDataset, EpisodeRecord
    from .demo_recorder import DemoRecorder


def __getattr__(name: str) -> Any:
    if name in {"EpisodeDataset", "EpisodeRecord"}:
        from .dataset import EpisodeDataset, EpisodeRecord

        return {"EpisodeDataset": EpisodeDataset, "EpisodeRecord": EpisodeRecord}[name]
    if name == "DemoRecorder":
        from .demo_recorder import DemoRecorder

        return DemoRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
