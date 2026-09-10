from unittest.mock import Mock

import numpy as np
import pytest

from workplace_guard.camera import Camera, CameraError


def test_camera_reads_frame_and_releases_capture() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)

    with Camera(capture=capture) as camera:
        assert camera.read() is frame

    capture.release.assert_called_once()


def test_camera_raises_when_capture_cannot_be_opened() -> None:
    capture = Mock()
    capture.isOpened.return_value = False

    with pytest.raises(CameraError, match="Не удалось открыть камеру"):
        Camera(index=3, capture=capture)

    capture.release.assert_called_once()


def test_camera_raises_when_frame_cannot_be_read() -> None:
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (False, None)

    camera = Camera(capture=capture)
    with pytest.raises(CameraError, match="Не удалось получить кадр"):
        camera.read()
