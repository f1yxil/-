from unittest.mock import Mock

import numpy as np
import pytest

from workplace_guard.detection import Detection, HOGPersonDetector


def test_detection_is_immutable_value_object() -> None:
    assert Detection((1, 2, 3, 4), 0.9).label == "person"


def test_hog_detector_filters_low_confidence_results() -> None:
    detector = HOGPersonDetector(confidence_threshold=0.5)
    detector._hog = Mock()
    detector._hog.detectMultiScale.return_value = (
        np.array([[1, 2, 30, 40], [5, 6, 70, 80]]),
        np.array([0.2, 0.8]),
    )

    assert detector.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == [
        Detection((5, 6, 70, 80), 0.8)
    ]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_hog_detector_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        HOGPersonDetector(confidence_threshold=threshold)
