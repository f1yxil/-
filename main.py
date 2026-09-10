"""Command-line entry point for the workplace-monitoring MVP."""

from __future__ import annotations

import argparse
import time

import cv2

from workplace_guard import Camera, CameraError, HOGPersonDetector, PersonDetector

WINDOW_TITLE = "Workplace Guard — q для выхода"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обнаружение людей в видеопотоке с веб-камеры"
    )
    parser.add_argument("--camera-index", type=int, default=0, help="индекс камеры")
    parser.add_argument("--width", type=int, default=None, help="ширина кадра")
    parser.add_argument("--height", type=int, default=None, help="высота кадра")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="минимальная уверенность детектора (0–1)",
    )
    return parser.parse_args()


def draw_detections(frame, detector: PersonDetector) -> int:
    """Draw detector output on a frame and return the number of people."""
    detections = detector.detect(frame)
    for detection in detections:
        x, y, width, height = detection.box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 0), 2)
        cv2.putText(
            frame,
            f"person {detection.confidence:.2f}",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
        )
    return len(detections)


def run(args: argparse.Namespace) -> None:
    detector = HOGPersonDetector(args.confidence)
    with Camera(args.camera_index, args.width, args.height) as camera:
        previous_time = time.perf_counter()
        while True:
            frame = camera.read()
            people_count = draw_detections(frame, detector)

            current_time = time.perf_counter()
            fps = 1 / max(current_time - previous_time, 1e-9)
            previous_time = current_time
            cv2.putText(
                frame,
                f"People: {people_count} | FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (CameraError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
