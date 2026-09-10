"""Web-camera access isolated from the application loop."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or read."""


class Camera:
    """Own an OpenCV video capture and provide validated frames."""

    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        capture: Any | None = None,
    ) -> None:
        self.index = index
        self._capture = capture if capture is not None else cv2.VideoCapture(index)

        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(f"Не удалось открыть камеру с индексом {index}")

    def read(self) -> np.ndarray:
        """Return the next frame or raise a descriptive error."""
        success, frame = self._capture.read()
        if not success or frame is None:
            raise CameraError("Не удалось получить кадр с камеры")
        return frame

    def release(self) -> None:
        """Release the underlying operating-system camera resource."""
        self._capture.release()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
