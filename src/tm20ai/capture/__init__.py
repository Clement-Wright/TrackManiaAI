"""Window capture helpers for Trackmania 2020."""

from .dxcam_capture import DXCamCapture
from .preprocess import FrameStackPreprocessor

__all__ = ["DXCamCapture", "FrameStackPreprocessor"]
