"""Person-detector contracts and the dependency-free OpenCV MVP backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """A detected person represented by an ``(x, y, width, height)`` box."""

    box: tuple[int, int, int, int]
    confidence: float
    label: str = "person"


class PersonDetector(Protocol):
    """Interface to be implemented by HOG, YOLO, or another backend."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return every person found in a BGR image."""
        ...


class HOGPersonDetector:
    """Detect pedestrians with OpenCV's pre-trained default HOG descriptor."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold должен быть от 0 до 1")
        self.confidence_threshold = confidence_threshold
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run HOG detection and discard results below the threshold."""
        boxes, weights = self._hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections: list[Detection] = []
        for box, weight in zip(boxes, weights):
            confidence = float(weight)
            if confidence >= self.confidence_threshold:
                detections.append(
                    Detection(tuple(int(value) for value in box), confidence)
                )
        return detections
