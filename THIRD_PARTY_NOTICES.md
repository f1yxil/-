# Уведомления о сторонних компонентах

Workplace Guard распространяется вместе со сторонними библиотеками и их
лицензионными файлами в каталоге сборки. Этот файл помогает найти основные
уведомления; юридические тексты из установленных пакетов следует сохранять
вместе с релизом.

- **Ultralytics** — GNU AGPL-3.0. Используется для YOLO11n и YOLO11n-pose.
  Исходный код и условия: <https://github.com/ultralytics/ultralytics>.
- **PyTorch** — BSD 3-Clause. Используется как вычислительный runtime.
  Условия: <https://github.com/pytorch/pytorch/blob/main/LICENSE>.
- **Torchvision** — BSD 3-Clause. Используется зависимостями Ultralytics.
  Условия: <https://github.com/pytorch/vision/blob/main/LICENSE>.
- **NumPy** — BSD 3-Clause. Используется для представления кадров и массивов.
  Условия: <https://github.com/numpy/numpy/blob/main/LICENSE.txt>.
- **OpenCV-Python** — Apache License 2.0 и уведомления компонентов сборки.
  Условия: <https://github.com/opencv/opencv/blob/4.x/LICENSE>.
- **PyInstaller** — GPLv2 с исключением для распространяемых приложений.
  Условия: <https://github.com/pyinstaller/pyinstaller/blob/main/COPYING.txt>.

Полный список версий фиксируется в `build/models-manifest.json`, который
создаётся скриптом сборки и включается в onedir-дистрибутив. Перед публичным
релизом проверьте актуальные LICENSE/NOTICE файлы всех зависимостей и условия
распространения выбранных весов моделей.
