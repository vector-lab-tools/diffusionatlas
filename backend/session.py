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
    estimated_footprint_gb,
    fit_check,
    is_apple_silicon,
    total_memory_gb,
    MIXED_PRECISION_VAE,
)


class ModelTooLargeError(Exception):
    """Raised when a model wouldn't fit in available memory and `force`
    is not set. Callers map this to HTTP 413 Payload Too Large."""

    def __init__(self, model_id: str, footprint_gb: float, available_gb: float, headroom_gb: float):
        self.model_id = model_id
        self.footprint_gb = footprint_gb
        self.available_gb = available_gb
        self.headroom_gb = headroom_gb
        super().__init__(
            f"Model {model_id} estimated at {footprint_gb:.1f} GB, "
            f"only {available_gb:.1f} GB available "
            f"(after {headroom_gb:.1f} GB OS/browser headroom). "
            f"Pass overrideMemoryCheck=true to load anyway, or pick a smaller model."
        )


class SessionState:
    def __init__(self) -> None:
        self.device = detect_device()
        # Default until first model load; updated to the per-model dtype
        # whenever load() runs.
        self.dtype = default_dtype(self.device)
        self.pipeline: Any = None
        self.current_model_id: str | None = None

    def load(self, model_id: str, force: bool = False) -> None:
        if self.current_model_id == model_id and self.pipeline is not None:
            return

        # Refuse loads that would push the OS into encrypted swap unless
        # the caller has explicitly opted in. This is the dev-mode
        # warning made teeth: a 24 GB MacBook should not silently try
        # to load FLUX-dev at fp32. The frontend can prompt and re-call
        # with force=True if the user wants to risk it.
        if not force:
            fits, footprint, available, headroom = fit_check(model_id, self.device)
            if not fits:
                raise ModelTooLargeError(model_id, footprint, available, headroom)

        # Pick the right precision for this model+device combination.
        self.dtype = dtype_for_model(model_id, self.device)

        # Aggressive cleanup before loading the next pipeline. On 24 GB
        # MacBooks SD 1.5 at fp32 leaves enough resident weight that
        # holding two simultaneously will push the kernel into encrypted
        # swap. `del` + gc + empty_cache on every supported device drops
        # the previous pipeline before we ask diffusers for the next.
        if self.pipeline is not None:
            try:
                del self.pipeline
            except Exception:
                pass
        self.pipeline = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            try:
                torch.mps.empty_cache()
                # synchronize forces the command queue to drain before
                # we measure or allocate again — without it, deferred
                # frees on the GPU side mean empty_cache underreports
                # how much was actually returned.
                torch.mps.synchronize()
            except Exception:
                # Older torches don't expose torch.mps.empty_cache —
                # gc.collect alone is the next best thing on MPS.
                pass

        # Imported lazily so `import session` doesn't pull diffusers on cold start.
        from diffusers import DiffusionPipeline, EulerDiscreteScheduler

        # `low_cpu_mem_usage=False` disables accelerate's meta-tensor
        # lazy-loading optimisation. With it on (the default in recent
        # diffusers), the subsequent `.to(self.device)` errors with
        # `Cannot copy out of meta tensor; no data!` because the weights
        # are placeholders until first forward. Loading directly into
        # RAM is fine on a 24 GB box for SD 1.5/SDXL — peak load memory
        # is ~5-7 GB which the watermark cap leaves room for.
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )
        pipe = pipe.to(self.device)

        # Mixed-precision VAE: on SD 1.x/2.x with MIXED_PRECISION_VAE on,
        # the U-Net and text encoder are loaded at fp16 (set above by
        # dtype_for_model). The VAE is the part with the historic NaN
        # underflow bug, so cast it back to fp32 individually. Net
        # effect: ~3 GB peak instead of ~5-7 GB, no black images.
        if (
            MIXED_PRECISION_VAE
            and self.device == "mps"
            and self.dtype == torch.float16
            and hasattr(pipe, "vae")
        ):
            try:
                pipe.vae.to(torch.float32)
            except Exception:
                # Cast failures are non-fatal — the worst case is the
                # VAE stays at fp16 and we get black images, which is
                # the bug we already know how to spot.
                pass

        # Force EulerDiscreteScheduler. Three reasons:
        #
        #   1. SD 1.5/2.x ship with PNDMScheduler, which has a warmup-phase
        #      off-by-one: at certain num_inference_steps it computes a
        #      prk_timestep equal to num_train_timesteps (1000), indexes
        #      alphas_cumprod[1000], and crashes with
        #      `index 1001 is out of bounds for dimension 0 with size 1000`.
        #
        #   2. DPMSolverMultistepScheduler ("DPM++ 2M") is faster-converging
        #      on CUDA but on **MPS** it has known numerical instabilities:
        #      the 2nd-order velocity prediction can push a few latent
        #      values past fp32's representable range, producing NaN that
        #      the VAE then decodes to all-black pixels. Different
        #      CFG/seed/step combinations hit it unpredictably (we've seen
        #      it at CFG 2.5 and 7.5 on the same prompt; CFG 4 and 12
        #      escape). Disabling Karras sigmas didn't fix it.
        #
        #   3. EulerDiscreteScheduler is 1st-order, doesn't accumulate
        #      velocity error, and is rock-solid across the whole CFG
        #      range on MPS. It converges slightly slower than DPM++ at
        #      the same step count (~20-step Euler ≈ ~12-step DPM++ in
        #      visual fidelity), but "always renders" beats "sometimes
        #      faster" for a research instrument. Bump steps to 20 if
        #      the output looks soft.
        try:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        except Exception:
            # If the model's scheduler config isn't compatible (some FLUX /
            # SD3 transformer pipelines), leave the original in place.
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

    def reset_activation_cache(self) -> None:
        """
        Drop any allocator-cached buffers from previous forward passes,
        without unloading the pipeline. Called before every inference
        so a warmup at 256×256 doesn't leave 5 GB of cached attention
        and VAE buffers sitting around when the next call wants to
        allocate at 512×512 or higher and immediately OOMs.

        Also re-instantiates the scheduler from its config — this is
        the fix for the `IndexError: index 21 is out of bounds for
        dimension 0 with size 21` we hit on EulerDiscreteScheduler:
        the scheduler holds `step_index` as instance state and doesn't
        always reset it cleanly when set_timesteps is called twice in
        a row from the pipeline. A fresh scheduler per call is cheap
        and stateless-by-construction.
        """
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            try:
                torch.mps.empty_cache()
                torch.mps.synchronize()
            except Exception:
                pass
        # Stateless reset for the scheduler.
        if self.pipeline is not None:
            try:
                from diffusers import EulerDiscreteScheduler
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            except Exception:
                # If something goes wrong, fall through — the existing
                # scheduler may still work; the worst case is the
                # next call hits the same IndexError again.
                pass

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
