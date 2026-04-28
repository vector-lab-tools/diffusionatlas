"""
POST /generate — single image, blocking.

Returns { images: [data:image/png;base64,...], meta: {...} }, the same shape
the frontend's /api/diffuse expects from any provider.
"""
from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any

import torch
from fastapi import HTTPException
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    modelId: str = "runwayml/stable-diffusion-v1-5"
    prompt: str
    negativePrompt: str | None = None
    seed: int = 0
    steps: int = Field(20, ge=1, le=200)
    cfg: float = Field(7.5, ge=0.0, le=30.0)
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    overrideMemoryCheck: bool = False


def run(req: GenerateRequest, session_state) -> dict[str, Any]:
    from session import ModelTooLargeError
    try:
        session_state.load(req.modelId, force=req.overrideMemoryCheck)
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

    pipe = session_state.pipeline
    # Drop allocator-cached buffers from any previous forward pass — a
    # warmup at 256² or a previous /generate at 512² can leave several
    # GB of cached attention and VAE buffers behind on MPS, which then
    # collide with the next call's allocations and cause OOM.
    session_state.reset_activation_cache()

    # Refuse obviously-impossible resolution / model combos with a 422
    # rather than letting the U-Net OOM mid-forward. Attention QK.T at
    # double the model's native resolution is ~4× the activations of a
    # native-resolution call (16384² vs 4096² spatial), which on SD 1.5
    # fp32 + a 12 GB MPS cap is the OOM you've been seeing.
    nw, nh = session_state.native_dims()
    if nw and nh and (req.width > nw * 1.5 or req.height > nh * 1.5):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Resolution {req.width}×{req.height} is too large for "
                f"{req.modelId} (native {nw}×{nh}). Attention activations "
                f"at fp32 would exceed the MPS memory cap. Use the model's "
                f"native resolution, switch Width/Height in the form, or "
                f"pick an SDXL/FLUX checkpoint that supports {req.width}×."
            ),
        )

    generator = torch.Generator(device=session_state.device).manual_seed(req.seed)

    # Clamp steps to the scheduler's training-timestep ceiling, matching the
    # defence in ops_trajectory.py. Past num_train_timesteps - 1 the
    # alphas_cumprod table is out of bounds.
    scheduler_max = getattr(pipe.scheduler.config, "num_train_timesteps", 1000)
    safe_steps = max(1, min(int(req.steps), int(scheduler_max) - 1))

    # Normalise negative_prompt: empty string can confuse some diffusers
    # pipelines into producing a [0, 40] embedding tensor that then
    # fails a batched matmul mid-forward. None is the well-defined
    # "no negative prompt" value across all SD/SDXL/SD3/FLUX paths.
    neg = req.negativePrompt
    if neg is not None and neg.strip() == "":
        neg = None
    # Same for the prompt itself — defensive guard, should never fire
    # because the request schema requires it, but if a future caller
    # passes whitespace we'd rather refuse early than crash inside the
    # text encoder.
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=422, detail="Empty prompt — give the model something to denoise toward.")

    started = time.time()
    try:
        out = pipe(
            prompt=req.prompt,
            negative_prompt=neg,
            num_inference_steps=safe_steps,
            guidance_scale=req.cfg,
            width=req.width,
            height=req.height,
            generator=generator,
        )
    except torch.cuda.OutOfMemoryError as e:
        raise HTTPException(status_code=507, detail=f"CUDA out of memory: {e}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise HTTPException(status_code=507, detail=f"Out of memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = int((time.time() - started) * 1000)
    image = out.images[0]

    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "images": [f"data:image/png;base64,{b64}"],
        "meta": {
            "providerId": "local",
            "modelId": req.modelId,
            "seed": req.seed,
            "steps": req.steps,
            "cfg": req.cfg,
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "responseTimeMs": elapsed_ms,
        },
    }
