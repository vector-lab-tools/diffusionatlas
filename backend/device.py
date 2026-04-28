"""
Device + memory detection for the local backend.

Principle: do not hardcode platform. Detect what's available at startup,
report capability flags via /health, let the frontend pick a sensible
default model from the registry.
"""
from __future__ import annotations

import os
import platform
from typing import Literal

import torch


# Enable mixed-precision VAE for SD 1.x / 2.x on MPS: U-Net and text
# encoder run at fp16 (or bf16), VAE runs at fp32. This dodges the
# fp16-VAE NaN bug that originally pushed us to full fp32 while halving
# the resident weight footprint (~5-7 GB → ~3 GB peak). Off by default
# until verified on reference seeds; enable per-session via
# `MIXED_PRECISION_VAE=1 uvicorn main:app …` and watch for black images.
MIXED_PRECISION_VAE = os.environ.get("MIXED_PRECISION_VAE", "0") == "1"

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
        treated the same way. (Memory pressure on 24 GB boxes is handled
        upstream via PYTORCH_MPS_HIGH_WATERMARK_RATIO + aggressive cache
        clearing on model swap, not by trading off correctness for RAM.)
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

    # Older Stable Diffusion 1.x / 2.x:
    #   - fp32 by default (the safe option, what we shipped with)
    #   - fp16 when MIXED_PRECISION_VAE is on — combined with an explicit
    #     pipe.vae.to(fp32) cast in session.load() this dodges the
    #     fp16-VAE NaN bug while halving resident weight memory.
    if "stable-diffusion-v1-" in mid or mid.endswith("/sd-v1"):
        return torch.float16 if MIXED_PRECISION_VAE else torch.float32
    if "stable-diffusion-2" in mid or "stable-diffusion-v2-" in mid:
        return torch.float16 if MIXED_PRECISION_VAE else torch.float32

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


# ---------------------------------------------------------------------
# Memory fit check
# ---------------------------------------------------------------------

# Approximate resident-memory footprint per model family, in GB at the
# dtype the policy above selects. These are loaded-weight estimates, not
# peak-during-inference; the activations on top are usually under 2 GB
# at 512×512 / 1024×1024 with attention slicing on. Numbers are
# deliberately conservative: better to refuse a load than crash the OS.
MODEL_FOOTPRINTS_GB: dict[str, float] = {
    "sd1.5": 4.0,    # fp32: ~4 GB resident
    "sd2": 5.0,      # fp32: ~5 GB
    "sdxl": 7.0,     # bf16: ~7 GB
    "sd3": 10.0,     # bf16: ~10 GB
    "flux-schnell": 12.0,
    "flux-dev": 24.0,
    "default": 8.0,  # unknown model: assume mid-range
}


def estimated_footprint_gb(model_id: str) -> float:
    mid = model_id.lower()
    if "flux" in mid and ("dev" in mid or "1-dev" in mid):
        return MODEL_FOOTPRINTS_GB["flux-dev"]
    if "flux" in mid:
        return MODEL_FOOTPRINTS_GB["flux-schnell"]
    if "stable-diffusion-3" in mid or "sd3" in mid:
        return MODEL_FOOTPRINTS_GB["sd3"]
    if "xl" in mid:
        return MODEL_FOOTPRINTS_GB["sdxl"]
    if "stable-diffusion-2" in mid or "stable-diffusion-v2-" in mid:
        return MODEL_FOOTPRINTS_GB["sd2"]
    if "stable-diffusion-v1-" in mid or "/sd-v1" in mid:
        return MODEL_FOOTPRINTS_GB["sd1.5"]
    return MODEL_FOOTPRINTS_GB["default"]


# Reserved headroom for OS + browser + Claude Code + the dev server, on
# unified-memory boxes. On 24 GB this leaves ~18 GB nominally available
# for the model — and the watermark ratio (0.7) further caps allocations
# to ~17 GB so the kernel never enters heavy swap. CUDA devices have
# dedicated VRAM so the headroom is much smaller.
HEADROOM_GB_BY_DEVICE: dict[DeviceName, float] = {
    "mps": 6.0,
    "cuda": 1.0,
    "cpu": 4.0,
}


def fit_check(model_id: str, device: DeviceName) -> tuple[bool, float, float, float]:
    """
    Return `(fits, footprint_gb, available_gb, headroom_gb)` for the
    given model + device. Caller decides whether to refuse or warn.
    `available_gb` is total memory minus headroom.
    """
    footprint = estimated_footprint_gb(model_id)
    total = total_memory_gb(device) or 0.0
    headroom = HEADROOM_GB_BY_DEVICE.get(device, 4.0)
    available = max(0.0, total - headroom)
    fits = footprint <= available
    return fits, footprint, available, headroom
