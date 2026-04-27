/**
 * Sensible starting-point model id per provider. Naming conventions diverge
 * (Replicate: owner/model · Fal: fal-ai/<model> · local: HF id), so anytime
 * the provider changes the model id should follow.
 */

import type { ProviderId } from "./types";

export const PROVIDER_DEFAULT_MODEL: Record<ProviderId, string> = {
  replicate: "black-forest-labs/flux-schnell",
  fal: "fal-ai/flux/schnell",
  together: "black-forest-labs/FLUX.1-schnell-Free",
  stability: "stable-diffusion-xl-1024-v1-0",
  local: "runwayml/stable-diffusion-v1-5",
};

export const HOSTED_PROVIDERS: ProviderId[] = ["replicate", "fal", "together", "stability"];
export const ALL_PROVIDERS: ProviderId[] = [...HOSTED_PROVIDERS, "local"];

/** Pretty label for UI. */
export function providerLabel(id: ProviderId): string {
  switch (id) {
    case "replicate": return "Replicate";
    case "fal": return "Fal.ai";
    case "together": return "Together";
    case "stability": return "Stability AI";
    case "local": return "Local FastAPI";
  }
}
