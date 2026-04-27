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


def run(req: GenerateRequest, session_state) -> dict[str, Any]:
    try:
        session_state.load(req.modelId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {req.modelId}: {e}")

    pipe = session_state.pipeline
    generator = torch.Generator(device=session_state.device).manual_seed(req.seed)

    # Clamp steps to the scheduler's training-timestep ceiling, matching the
    # defence in ops_trajectory.py. Past num_train_timesteps - 1 the
    # alphas_cumprod table is out of bounds.
    scheduler_max = getattr(pipe.scheduler.config, "num_train_timesteps", 1000)
    safe_steps = max(1, min(int(req.steps), int(scheduler_max) - 1))

    started = time.time()
    try:
        out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negativePrompt,
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
