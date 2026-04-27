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
    dtype_for_model,
    is_apple_silicon,
    total_memory_gb,
)


class SessionState:
    def __init__(self) -> None:
        self.device = detect_device()
        # Default until first model load; updated to the per-model dtype
        # whenever load() runs.
        self.dtype = default_dtype(self.device)
        self.pipeline: Any = None
        self.current_model_id: str | None = None

    def load(self, model_id: str) -> None:
        if self.current_model_id == model_id and self.pipeline is not None:
            return

        # Pick the right precision for this model+device combination.
        self.dtype = dtype_for_model(model_id, self.device)

        # Unload previous pipeline before loading the next so we don't hold two.
        self.pipeline = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Imported lazily so `import session` doesn't pull diffusers on cold start.
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=self.dtype)
        pipe = pipe.to(self.device)

        # Force DPMSolverMultistepScheduler ("DPM++ 2M") in Karras-sigmas
        # mode. Two reasons:
        #   1. SD 1.5/2.x ship with PNDMScheduler, which has a warmup-phase
        #      off-by-one: at certain num_inference_steps it computes a
        #      prk_timestep equal to num_train_timesteps (1000), indexes
        #      alphas_cumprod[1000], and crashes with `index 1001 is out of
        #      bounds for dimension 0 with size 1000`.
        #   2. DPM++ 2M is the modern "fast and good" choice — it converges
        #      at ~12 steps for SD 1.5 (vs ~20-25 for Euler, ~50 for DDIM),
        #      so the default trajectory feels snappy without going mushy.
        # Karras sigmas (`use_karras_sigmas=True`) shape the noise schedule
        # for cleaner detail at low step counts. `algorithm_type="dpmsolver++"`
        # enables the second-order solver. Both are safe across SD 1.x /
        # 2.x / SDXL / SD 3.
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
        except Exception:
            # If the model's scheduler config isn't compatible (some FLUX /
            # SD3 transformer pipelines), leave the original in place.
            # Trajectory clamping in ops_trajectory.py is the second line of
            # defence.
            pass

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

    def native_dims(self) -> tuple[int | None, int | None]:
        """
        Derive the model's native pixel resolution from the loaded pipeline.

        Source of truth: `unet.config.sample_size` (or `transformer.config`
        for DiT-style models like SD 3 / FLUX) gives the *latent-grid* edge
        length; multiplying by the VAE scale factor yields pixels. This is
        what diffusers itself uses to default `height` / `width` when the
        caller doesn't pass them, so it is the authoritative per-checkpoint
        answer rather than convention.
        """
        pipe = self.pipeline
        if pipe is None:
            return None, None

        sample = None
        for attr in ("unet", "transformer"):
            sub = getattr(pipe, attr, None)
            if sub is not None and hasattr(sub, "config"):
                sample = getattr(sub.config, "sample_size", None)
                if sample is not None:
                    break
        if sample is None:
            return None, None
        if isinstance(sample, (list, tuple)) and sample:
            sample = sample[0]

        scale = getattr(pipe, "vae_scale_factor", None)
        if scale is None and hasattr(pipe, "vae") and hasattr(pipe.vae, "config"):
            block = getattr(pipe.vae.config, "block_out_channels", None)
            if block:
                scale = 2 ** (len(block) - 1)
        if scale is None:
            scale = 8

        try:
            edge = int(sample) * int(scale)
        except (TypeError, ValueError):
            return None, None
        return edge, edge

    def health(self) -> dict[str, Any]:
        nw, nh = self.native_dims()
        return {
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "torchVersion": torch.__version__,
            "appleSilicon": is_apple_silicon(),
            "totalMemoryGb": total_memory_gb(self.device),
            "currentModelId": self.current_model_id,
            "nativeWidth": nw,
            "nativeHeight": nh,
            "ready": True,
        }


# Module-level singleton.
session_state = SessionState()
