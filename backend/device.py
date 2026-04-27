"""
Device + memory detection for the local backend.

Principle: do not hardcode platform. Detect what's available at startup,
report capability flags via /health, let the frontend pick a sensible
default model from the registry.
"""
from __future__ import annotations

import platform
from typing import Literal

import torch

DeviceName = Literal["cuda", "mps", "cpu"]


def detect_device() -> DeviceName:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def total_memory_gb(device: DeviceName) -> float | None:
    """Approximate total memory available to the model.

    For CUDA, returns dedicated VRAM. For MPS and CPU, returns system memory
    (since unified memory is shared). None if we can't tell.
    """
    if device == "cuda":
        try:
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            return None
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        return None


def default_dtype(device: DeviceName) -> torch.dtype:
    """Half precision on accelerators, full on CPU."""
    return torch.float16 if device != "cpu" else torch.float32
