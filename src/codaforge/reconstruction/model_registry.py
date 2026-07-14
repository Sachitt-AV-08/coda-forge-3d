from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_model_cache: dict[str, bool] = {}


def _check_module(name: str, package: str | None = None) -> bool:
    if name in _model_cache:
        return _model_cache[name]
    try:
        if package:
            __import__(package)
        else:
            __import__(name)
        _model_cache[name] = True
        return True
    except ImportError:
        _model_cache[name] = False
        return False


def check_torch() -> bool:
    return _check_module("torch")


def check_mediapipe() -> bool:
    return _check_module("mediapipe")


def check_rembg() -> bool:
    return _check_module("rembg")


def check_xatlas() -> bool:
    return _check_module("xatlas")


def check_pymcubes() -> bool:
    return _check_module("PyMCubes", "PyMCubes")


def list_available() -> dict[str, bool]:
    return {
        "torch": check_torch(),
        "mediapipe": check_mediapipe(),
        "rembg": check_rembg(),
        "xatlas": check_xatlas(),
        "pymcubes": check_pymcubes(),
    }
