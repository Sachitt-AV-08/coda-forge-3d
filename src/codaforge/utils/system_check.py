from __future__ import annotations

import logging
import shutil

logger = logging.getLogger(__name__)


def check_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def check_colmap() -> bool:
    return shutil.which("colmap") is not None or shutil.which("colmap.exe") is not None


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None or shutil.which("ffmpeg.exe") is not None


def check_mediapipe() -> bool:
    try:
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


def check_xatlas() -> bool:
    try:
        import xatlas  # noqa: F401

        return True
    except ImportError:
        return False


def check_opencv() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def run_system_check(verbose: bool = True) -> dict[str, bool]:
    checks = {
        "torch": check_torch(),
        "colmap": check_colmap(),
        "ffmpeg": check_ffmpeg(),
        "mediapipe": check_mediapipe(),
        "xatlas": check_xatlas(),
        "opencv": check_opencv(),
    }
    if verbose:
        for name, ok in checks.items():
            status = "OK" if ok else "MISSING"
            logger.info("%s: %s", name, status)
    return checks
