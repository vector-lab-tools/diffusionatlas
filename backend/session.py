"""
Session state for the local backend: holds the currently-loaded diffusers
pipeline, lazily loads on first request, swaps cleanly on model change.

Single global instance — local backend is single-tenant by design.
"""
from __future__ import annotations

import gc
from typing import Any

import torch

from device import (
    detect_device,
    default_dtype,
    is_apple_silicon,
    total_memory_gb,
)


class SessionState:
    def __init__(self) -> None:
        self.device = detect_device()
        self.dtype = default_dtype(self.device)
        self.pipeline: Any = None
        self.current_model_id: str | None = None

    def load(self, model_id: str) -> None:
        if self.current_model_id == model_id and self.pipeline is not None:
            return

        # Unload previous pipeline before loading the next so we don't hold two.
        self.pipeline = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Imported lazily so `import session` doesn't pull diffusers on cold start.
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=self.dtype)
        pipe = pipe.to(self.device)

        # On Apple Silicon (and any constrained accelerator) these reduce peak
        # memory at modest cost in throughput. Safe to enable broadly.
        if is_apple_silicon() or self.device == "cuda":
            try:
                pipe.enable_attention_slicing()
            except AttributeError:
                pass
            try:
                pipe.enable_vae_tiling()
            except AttributeError:
                pass

        self.pipeline = pipe
        self.current_model_id = model_id

    def health(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "torchVersion": torch.__version__,
            "appleSilicon": is_apple_silicon(),
            "totalMemoryGb": total_memory_gb(self.device),
            "currentModelId": self.current_model_id,
            "ready": True,
        }


# Module-level singleton.
session_state = SessionState()
