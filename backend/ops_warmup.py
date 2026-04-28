"""
POST /warmup — load and warm a diffusion pipeline.

The first inference on Apple Silicon (MPS) takes ~80 s/step while the
backend compiles MPS kernels, allocates buffers, and hits the VAE for
the first time. After that the same pipeline drops to ~1-2 s/step.

This endpoint is the cheap public path to that warmup: load the model
if not already loaded, run a single denoising step at the model's
native resolution, decode through the VAE once. The image is thrown
away — only the side effects (warm caches) matter.

Frontend triggers this when the user switches to the Local backend or
on first load so the first real Sweep / Neighbourhood / Bench cell
isn't a five-minute black box.
"""
from __future__ import annotations

import time
from typing import Any

import torch
from fastapi import HTTPException
from pydantic import BaseModel


class WarmupRequest(BaseModel):
    modelId: str = "runwayml/stable-diffusion-v1-5"
    overrideMemoryCheck: bool = False


def run(req: WarmupRequest, session_state) -> dict[str, Any]:
    from session import ModelTooLargeError
    started = time.time()

    # 1. Load (download + dtype + scheduler swap). No-op if already loaded.
    try:
        load_started = time.time()
        already_loaded = (
            session_state.current_model_id == req.modelId
            and session_state.pipeline is not None
        )
        session_state.load(req.modelId, force=req.overrideMemoryCheck)
        load_ms = int((time.time() - load_started) * 1000)
    except ModelTooLargeError as e:
        raise HTTPException(status_code=413, detail={
            "code": "model_too_large",
            "message": str(e),
            "modelId": e.model_id,
            "footprintGb": e.footprint_gb,
            "availableGb": e.available_gb,
            "headroomGb": e.headroom_gb,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {req.modelId}: {e}")

    # 2. Run one denoising step at 256×256, regardless of the model's
    #    native resolution. Activation memory scales with H×W, so SDXL
    #    at native 1024×1024 would peak at ~16× the activation
    #    footprint of an SD 1.5 warmup at 512×512 — the warmup itself
    #    becomes a memory-pressure event. 256×256 still touches every
    #    kernel and forces VAE compilation, but at a fraction of peak.
    #    The "first real call feels snappy" property is preserved
    #    because the kernels you needed are now cached.
    pipe = session_state.pipeline
    width = 256
    height = 256

    inference_started = time.time()
    try:
        generator = torch.Generator(device=session_state.device).manual_seed(0)
        # 1 inference step = scheduler emits a single timestep, U-Net
        # runs once, VAE decodes once. Touches everything that's slow on
        # the first call without spending real wall-clock on denoising.
        pipe(
            prompt="warmup",
            num_inference_steps=1,
            guidance_scale=0.0,
            width=width,
            height=height,
            generator=generator,
        )
    except Exception as e:
        # Warmup failure is informational, not fatal — the next real
        # call will still try to run.
        raise HTTPException(status_code=500, detail=f"Warmup forward pass failed: {e}")
    inference_ms = int((time.time() - inference_started) * 1000)

    nw, nh = session_state.native_dims()
    total_ms = int((time.time() - started) * 1000)
    return {
        "ok": True,
        "modelId": req.modelId,
        "alreadyLoaded": already_loaded,
        "loadMs": load_ms,
        "inferenceMs": inference_ms,
        "totalMs": total_ms,
        "device": session_state.device,
        "dtype": str(session_state.dtype).replace("torch.", ""),
        "nativeWidth": nw,
        "nativeHeight": nh,
    }
