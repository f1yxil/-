# -*- mode: python ; coding: utf-8 -*-
"""Windows x64 onedir distribution with models and read-only defaults."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, copy_metadata


root = Path(SPECPATH)
required_assets = [
    (root / "models" / "yolo11n.pt", "models"),
    (root / "models" / "yolo11n-pose.pt", "models"),
    (root / "config" / "defaults.json", "config"),
]
for source, _ in required_assets:
    if not source.is_file():
        raise FileNotFoundError(
            f"Required offline asset is missing: {source}. Use scripts/build.ps1."
        )

datas = [(str(source), destination) for source, destination in required_assets]
for filename in ("README.md", "LICENSE", "LICENSE.txt", "THIRD_PARTY_NOTICES.md"):
    path = root / filename
    if path.is_file():
        datas.append((str(path), "."))
manifest = root / "build" / "models-manifest.json"
if manifest.is_file():
    datas.append((str(manifest), "."))

# Ultralytics resolves model classes dynamically and needs packaged YAML files.
# The maintained PyInstaller hooks collect OpenCV, NumPy, Torch/Torchvision DLLs
# and their transitive dependencies; do not remove Torch DLLs to shrink a build.
ultralytics_datas, binaries, hiddenimports = collect_all("ultralytics")
datas += ultralytics_datas
for distribution in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python"):
    datas += copy_metadata(distribution)
datas += collect_data_files("torch", includes=["**/LICENSE*", "**/licenses/**"])
hiddenimports += ["torch", "torchvision", "torchvision.ops", "cv2", "numpy"]
# Newer CPU torchvision wheels expose the operator extension as _C_stable.pyd;
# collect it explicitly because generic hooks may only look for torchvision._C.
binaries += collect_dynamic_libs("torchvision")
for filename in ("_C_stable.pyd", "image_stable.pyd"):
    extension = root / ".venv" / "Lib" / "site-packages" / "torchvision" / filename
    if extension.is_file():
        binaries.append((str(extension), "torchvision"))

icon = root / "assets" / "workplace_guard.ico"
analysis = Analysis(
    [str(root / "main.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(root / "scripts" / "runtime_hook.py")],
    excludes=["pytest", "IPython", "jupyter", "notebook", "tensorboard", "tensorflow"],
    noarchive=False,
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="WorkplaceGuard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    version=str(root / "scripts" / "windows_version_info.txt"),
    icon=str(icon) if icon.is_file() else None,
    contents_directory="_internal",
)
distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="WorkplaceGuard",
)
