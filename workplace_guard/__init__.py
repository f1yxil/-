"""Computer-vision components for workplace monitoring."""

from .camera import Camera, CameraError
from .detection import Detection, HOGPersonDetector, PersonDetector

__all__ = [
    "Camera",
    "CameraError",
    "Detection",
    "HOGPersonDetector",
    "PersonDetector",
]
