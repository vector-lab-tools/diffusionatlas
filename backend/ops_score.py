"""
POST /score — CLIP-based image–text similarity for the Compositional Bench.

Loads `openai/clip-vit-base-patch32` lazily on first call (small, ~150 MB).
Accepts parallel arrays of images (data URLs) and texts; returns one
cosine-similarity score per pair, in the conventional CLIP scale (typically
0.18–0.35 for "image matches text"). Caller chooses the threshold.

Single-tenant; the CLIP model lives alongside the diffusers pipeline in
session memory.
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import torch
from fastapi import HTTPException
from PIL import Image
from pydantic import BaseModel

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class ScoreRequest(BaseModel):
    images: list[str]
    prompts: list[str]


class _ClipState:
    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.device: str | None = None

    def ensure(self, device: str) -> None:
        if self.model is not None and self.device == device:
            return
        from transformers import CLIPModel, CLIPProcessor

        # CLIP is small enough that fp32 is safe everywhere and avoids the
        # occasional fp16 normalisation quirk on MPS.
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch.float32)
        model = model.to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.model = model
        self.processor = processor
        self.device = device


_state = _ClipState()


def _decode_data_url(data_url: str) -> Image.Image:
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def run(req: ScoreRequest, session_state) -> dict[str, Any]:
    if not req.images:
        raise HTTPException(status_code=400, detail="images is empty")
    if len(req.images) != len(req.prompts):
        raise HTTPException(
            status_code=400,
            detail=f"images ({len(req.images)}) and prompts ({len(req.prompts)}) must have equal length",
        )

    device = session_state.device

    try:
        _state.ensure(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load CLIP: {e}")

    try:
        images = [_decode_data_url(u) for u in req.images]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode an image: {e}")

    inputs = _state.processor(
        text=req.prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = _state.model(**inputs)

    image_emb = out.image_embeds
    text_emb = out.text_embeds
    img_n = image_emb / image_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    txt_n = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Per-pair cosine similarity: image i vs text i.
    scores = (img_n * txt_n).sum(dim=-1).tolist()

    return {"scores": scores, "modelId": CLIP_MODEL_ID}
