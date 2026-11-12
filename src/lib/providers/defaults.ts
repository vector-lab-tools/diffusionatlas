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
  local: "stable-diffusion-v1-5/stable-diffusion-v1-5",
};

// Suggestions only: the Model field stays free text so any id from the
// provider's catalogue works when these go stale. Checked 2026-11-12.
export const PROVIDER_SUGGESTED_MODELS: Record<ProviderId, string[]> = {
  replicate: [
    "black-forest-labs/flux-schnell",
    "black-forest-labs/flux-dev",
    "black-forest-labs/flux-2-dev",
    "stability-ai/stable-diffusion-3.5-medium",
    "bytedance/sdxl-lightning-4step",
  ],
  fal: ["fal-ai/flux/schnell", "fal-ai/flux/dev", "fal-ai/flux-2", "fal-ai/fast-sdxl"],
  together: ["black-forest-labs/FLUX.1-schnell-Free"],
  stability: ["stable-diffusion-xl-1024-v1-0"],
  local: [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3.5-medium",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
  ],
};

// Ids that moved; saved settings are rewritten on load.
export const RENAMED_MODELS: Record<string, string> = {
  "runwayml/stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
};

// Only providers registered in dispatcher.ts; together and stability are typed but not wired.
export const HOSTED_PROVIDERS: ProviderId[] = ["replicate", "fal"];
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
