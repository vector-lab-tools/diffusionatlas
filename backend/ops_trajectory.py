"""
POST /trajectory — streams an NDJSON sequence of per-step latents.

Each line is one JSON object:
  { "event": "start", "meta": {...} }
  { "event": "step", "step": int, "totalSteps": int, "shape": [..],
    "latentB64": "<float32 bytes, base64>" }
  { "event": "done", "imageDataUrl": "...", "responseTimeMs": int }
  { "event": "error", "message": str }

The diffusers callback fires synchronously inside pipe() so we run the
pipeline in a worker thread and drain a queue from the response generator
to keep the bytes flowing while denoising progresses.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from io import BytesIO
from queue import Queue
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class TrajectoryRequest(BaseModel):
    modelId: str = "runwayml/stable-diffusion-v1-5"
    prompt: str
    negativePrompt: str | None = None
    seed: int = 0
    steps: int = Field(20, ge=1, le=100)
    cfg: float = Field(7.5, ge=0.0, le=30.0)
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)


_DONE = object()


def _encode_latent(t: torch.Tensor) -> str:
    arr = t.detach().to(dtype=torch.float32, device="cpu").numpy().astype(np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def stream(req: TrajectoryRequest, session_state) -> StreamingResponse:
    try:
        session_state.load(req.modelId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {req.modelId}: {e}")

    pipe = session_state.pipeline
    device = session_state.device

    q: Queue = Queue()

    def step_callback(pipe_, step: int, timestep, kwargs):
        latent = kwargs.get("latents")
        if latent is None:
            return kwargs
        try:
            q.put({
                "event": "step",
                "step": int(step) + 1,
                "totalSteps": int(req.steps),
                "shape": list(latent.shape),
                "latentB64": _encode_latent(latent),
            })
        except Exception as e:
            q.put({"event": "error", "message": f"Encoding step {step} failed: {e}"})
        return kwargs

    def worker() -> None:
        try:
            generator = torch.Generator(device=device).manual_seed(int(req.seed))
            started = time.time()
            out = pipe(
                prompt=req.prompt,
                negative_prompt=req.negativePrompt,
                num_inference_steps=int(req.steps),
                guidance_scale=float(req.cfg),
                width=int(req.width),
                height=int(req.height),
                generator=generator,
                callback_on_step_end=step_callback,
            )
            image = out.images[0]
            buf = BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            elapsed = int((time.time() - started) * 1000)
            q.put({
                "event": "done",
                "imageDataUrl": f"data:image/png;base64,{b64}",
                "responseTimeMs": elapsed,
            })
        except torch.cuda.OutOfMemoryError as e:
            q.put({"event": "error", "message": f"CUDA out of memory: {e}"})
        except Exception as e:
            q.put({"event": "error", "message": str(e)})
        finally:
            q.put(_DONE)

    def line_iter():
        meta: dict[str, Any] = {
            "providerId": "local",
            "modelId": req.modelId,
            "seed": req.seed,
            "steps": req.steps,
            "cfg": req.cfg,
            "width": req.width,
            "height": req.height,
        }
        yield json.dumps({"event": "start", "meta": meta}) + "\n"

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is _DONE:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(line_iter(), media_type="application/x-ndjson")
