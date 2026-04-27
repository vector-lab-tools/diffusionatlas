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
    """Conservative default when model id is unknown."""
    return torch.float16 if device == "cuda" else torch.float32


def dtype_for_model(model_id: str, device: DeviceName) -> torch.dtype:
    """
    Per-model precision policy.

    On CUDA fp16 works for all current diffusion models, so use it.

    On MPS:
      - SD 1.x at fp16 has a well-known NaN-in-VAE bug that produces black
        images. fp32 throughout fits at 512×512 in unified memory and is
        the only stable configuration. SD 2.x is similarly old and
        treated the same way.
      - SDXL, SD3, and FLUX are large enough that fp32 won't fit, but they
        are stable at bfloat16 — wider exponent range than fp16 so the VAE
        doesn't blow up, and roughly half the memory of fp32. PyTorch 2.4+
        on Apple Silicon supports bf16 across the relevant ops.

    On CPU, fp32 is the only realistic option.
    """
    if device == "cuda":
        return torch.float16
    if device == "cpu":
        return torch.float32

    # MPS path: lower-case match on the model id to pick a family.
    mid = model_id.lower()

    # Older Stable Diffusion 1.x / 2.x: fp32 on MPS.
    if "stable-diffusion-v1-" in mid or mid.endswith("/sd-v1"):
        return torch.float32
    if "stable-diffusion-2" in mid or "stable-diffusion-v2-" in mid:
        return torch.float32

    # SDXL, SD3, FLUX, and anything else heavy: bfloat16.
    if (
        "xl" in mid
        or "stable-diffusion-3" in mid
        or "sd3" in mid
        or "flux" in mid
        or "kandinsky" in mid
        or "playground" in mid
    ):
        return torch.bfloat16

    # Fallback: fp32 is safe but slow.
    return torch.float32
