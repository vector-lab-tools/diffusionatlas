"use client";

import { OperationStub } from "@/components/shared/OperationStub";

export function DenoiseTrajectory() {
  return (
    <OperationStub
      title="Denoise Trajectory"
      description="Trace the iterative denoising path through latent space. Per-step latents are reduced via UMAP/PCA to a 3D curve, with frame-stamped image previews along the trajectory."
      backendNote="Local FastAPI required (per-step latents are unavailable from hosted providers)."
    />
  );
}
