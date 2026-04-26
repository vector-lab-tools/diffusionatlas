"use client";

import { OperationStub } from "@/components/shared/OperationStub";

export function LatentNeighbourhood() {
  return (
    <OperationStub
      title="Latent Neighbourhood"
      description="Sample nearby points around an anchor seed or initial latent. Reveals the local manifold structure: smoothness, basins, and where small perturbations produce categorical jumps."
      backendNote="Hosted (seed jitter) or Local (true latent-space perturbation at chosen σ)."
    />
  );
}
