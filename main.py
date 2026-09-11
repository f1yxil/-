"""Запуск MVP: камера → детектор людей → окно OpenCV."""

from __future__ import annotations

import multiprocessing

# PyInstaller must divert camera worker processes before importing OpenCV or CLI.
if __name__ == "__main__":
    multiprocessing.freeze_support()

import argparse
from collections.abc import Sequence
import logging
import math
import sys
import time

import cv2
import numpy as np

from workplace_guard import (
    Camera, CameraError, DetectorError, HOGPersonDetector, PersonDetector, YOLOPersonDetector,
)
from workplace_guard.diagnostics import diagnose_cameras
from workplace_guard.config import load_config
from workplace_guard.pose_validator import DEFAULT_POSE_MODEL, PoseValidator
from workplace_guard.tracking import PersonTracker, TrackedDetection, TrackingConfig
from workplace_guard.version import __version__
from workplace_guard.yolo_detector import DEFAULT_MODEL

WINDOW_TITLE = "Workplace Guard | q / Esc - exit"


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("значение должно быть неотрицательным")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return number


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("значение должно быть конечным числом")
    return number


def positive_finite_float(value: str) -> float:
    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return number


def confidence_value(value: str) -> float:
    number = finite_float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("confidence должен быть от 0 до 1")
    return number


def image_size(value: str) -> int:
    number = positive_int(value)
    if number % 32:
        raise argparse.ArgumentTypeError("размер входа YOLO должен быть кратен 32")
    return number


def positive_probability(value: str) -> float:
    number = confidence_value(value)
    if number == 0:
        raise argparse.ArgumentTypeError("значение должно быть больше 0")
    return number


def non_negative_float(value: str) -> float:
    number = finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("значение должно быть неотрицательным")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обнаружение людей в видеопотоке с веб-камеры"
    )
    parser.add_argument("--version", action="version", version=f"Workplace Guard {__version__}")
    parser.add_argument("--config", help="путь к JSON с настройками; CLI имеет приоритет")
    parser.add_argument("--self-test", action="store_true", help="проверить обе модели на CPU без камеры")
    parser.add_argument(
        "--detector", choices=("yolo", "hog"), default="yolo",
        help="детектор: yolo (по умолчанию) или резервный hog",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="путь к весам YOLO (.pt)")
    parser.add_argument(
        "--confidence", type=confidence_value, default=0.3,
        help="минимальный confidence YOLO от 0 до 1 (0.3)",
    )
    parser.add_argument(
        "--imgsz", type=image_size, default=320,
        help="размер обработки YOLO, кратный 32 (320); 640 повышает детализацию",
    )
    parser.add_argument(
        "--device", default="cpu", help="устройство YOLO: cpu или, например, 0 для GPU",
    )
    parser.add_argument(
        "--camera-index", type=non_negative_int, default=0, help="индекс камеры (0)"
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "dshow", "msmf"),
        default="auto",
        help="backend камеры: auto на Windows пробует DSHOW, затем MSMF",
    )
    parser.add_argument(
        "--open-timeout",
        type=positive_finite_float,
        default=5.0,
        help="таймаут открытия и первого кадра для каждого backend, секунды (5)",
    )
    parser.add_argument(
        "--read-timeout",
        type=positive_finite_float,
        default=3.0,
        help="таймаут получения очередного кадра, секунды (3)",
    )
    parser.add_argument(
        "--diagnose-cameras",
        action="store_true",
        help="проверить камеры без окна видео и детектора людей",
    )
    parser.add_argument(
        "--max-camera-index",
        type=non_negative_int,
        default=4,
        help="последний индекс диагностики включительно: проверяются 0..N (4)",
    )
    parser.add_argument(
        "--width", type=positive_int, default=640, help="желаемая ширина кадра (640)"
    )
    parser.add_argument(
        "--height", type=positive_int, default=480, help="желаемая высота кадра (480)"
    )
    parser.add_argument(
        "--hog-threshold",
        type=finite_float,
        default=0.5,
        help="порог оценки SVM для HOG (0.5); не вероятность, может быть больше 1",
    )
    parser.add_argument("--confirm-frames", type=positive_int, default=4,
                        help="последовательных кадров до подтверждения человека (4)")
    parser.add_argument("--lost-timeout", type=positive_finite_float, default=0.6,
                        help="сколько секунд сохранять человека при пропуске обнаружения (0.6)")
    parser.add_argument("--match-iou", type=positive_probability, default=0.3,
                        help="минимальное пересечение рамок для отслеживания (0.3)")
    parser.add_argument("--pose-threshold", type=confidence_value, default=0.7,
                        help="YOLO confidence ниже этого значения требует keypoints (0.7)")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL, help="веса YOLO pose (.pt)")
    parser.add_argument("--pose-imgsz", type=image_size, default=256, help="размер входа pose (256)")
    parser.add_argument("--pose-keypoint-confidence", type=positive_probability, default=0.4,
                        help="минимальная уверенность каждого keypoint (0.4)")
    parser.add_argument("--pose-recheck-frames", type=positive_int, default=8,
                        help="интервал повторной проверки pose в кадрах (8)")
    parser.add_argument("--pose-cache-seconds", type=non_negative_float, default=0.6,
                        help="предельный возраст результата pose в секундах (0.6)")
    parser.add_argument("--pose-max-per-frame", type=positive_int, default=2,
                        help="максимум проверок pose за кадр (2)")
    parser.add_argument("--debug-detections", action=argparse.BooleanOptionalAction, default=False,
                        help="показать кандидатов, состояния, confidence и причины отклонения")
    # Help/version must work even with a damaged user configuration.
    actual_argv = list(argv) if argv is not None else sys.argv[1:]
    if not any(option in actual_argv for option in ("--help", "-h", "--version")):
        bootstrap = argparse.ArgumentParser(add_help=False)
        bootstrap.add_argument("--config")
        preliminary, _ = bootstrap.parse_known_args(actual_argv)
        try:
            parser.set_defaults(**load_config(preliminary.config))
        except ValueError as error:
            parser.error(str(error))
    return parser.parse_args(argv)


def create_detector(args: argparse.Namespace) -> PersonDetector:
    if args.detector == "hog":
        logging.getLogger(__name__).info("Выбран HOG: score — оценка SVM, не confidence")
        return HOGPersonDetector(args.hog_threshold)
    return YOLOPersonDetector(
        model_path=args.model, confidence_threshold=args.confidence,
        imgsz=args.imgsz, device=args.device,
    )


def create_tracker(args: argparse.Namespace) -> PersonTracker:
    pose = None
    if args.detector == "yolo":
        pose = PoseValidator(model_path=args.pose_model, imgsz=args.pose_imgsz,
                             device=args.device, keypoint_confidence=args.pose_keypoint_confidence)
    config = TrackingConfig(
        confirm_frames=args.confirm_frames, lost_timeout=args.lost_timeout,
        match_iou=args.match_iou, pose_threshold=args.pose_threshold if pose else 0,
        pose_recheck_frames=args.pose_recheck_frames, pose_cache_seconds=args.pose_cache_seconds,
        pose_max_per_frame=args.pose_max_per_frame,
    )
    logging.getLogger(__name__).info(
        "Подтверждение: %d кадров; удержание %.2f с; pose для confidence < %.2f",
        config.confirm_frames, config.lost_timeout, config.pose_threshold,
    )
    return PersonTracker(config, pose_validator=pose)


def draw_detections(
    frame: np.ndarray, tracks: list[TrackedDetection], *, score_label: str = "score",
    debug: bool = False,
) -> int:
    """Count confirmed tracks only; explain rejected candidates in debug mode."""
    for track in tracks:
        if track.state != "confirmed" and not debug:
            continue
        detection = track.detection
        color = {"confirmed": (0, 200, 0), "candidate": (0, 165, 255), "lost": (128, 128, 128)}[track.state]
        x, y, width, height = detection.box
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        label = f"person | {score_label}: {detection.score:.2f}"
        if debug:
            label += f" | #{track.track_id} {track.state}"
        elif not track.visible:
            label += " | held"
        text_y = min(frame.shape[0] - 10, max(y - 8, 52))
        cv2.putText(
            frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
        if debug:
            cv2.putText(frame, track.reason, (x, min(text_y + 18, frame.shape[0] - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)
    return sum(track.state == "confirmed" for track in tracks)


def run(args: argparse.Namespace) -> None:
    try:
        detector = create_detector(args)
        tracker = create_tracker(args)
        with Camera(
            args.camera_index,
            args.width,
            args.height,
            backend=args.camera_backend,
            open_timeout=args.open_timeout,
            read_timeout=args.read_timeout,
        ) as camera:
            previous_time = time.perf_counter()
            while True:
                frame = camera.read()
                tracks = tracker.update(frame, detector.detect(frame))
                people_count = draw_detections(
                    frame, tracks, score_label="confidence" if args.detector == "yolo" else "score",
                    debug=args.debug_detections,
                )

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
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                try:
                    visible = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    # Некоторые GUI-бэкенды (например, Qt) после крестика
                    # сообщают об уничтоженном окне исключением.
                    break
                if visible < 1:
                    break
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Например, HighGUI недоступен: сохранить исходную ошибку запуска.
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.self_test:
            from workplace_guard.self_test import run_self_test
            run_self_test(args.model, args.pose_model)
            return 0
        if args.diagnose_cameras:
            results = diagnose_cameras(
                max_index=args.max_camera_index,
                width=args.width,
                height=args.height,
                backend=args.camera_backend,
                open_timeout=args.open_timeout,
                read_timeout=args.read_timeout,
            )
            return 0 if any(result.opened for result in results) else 1
        else:
            run(args)
    except KeyboardInterrupt:
        print("Работа завершена.")
    except (CameraError, DetectorError, ValueError, cv2.error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
