/**
 * Fal.ai hosted provider.
 *
 * Direct fetch against fal.run (sync endpoint) — no SDK dependency.
 * Fal's rate limits are typically an order of magnitude more permissive
 * than Replicate's free tier, which makes it the better default for
 * sweep- and bench-heavy work.
 *
 * Different Fal models accept different input shapes:
 *   - flux/schnell:    { prompt, image_size, num_inference_steps (1-12) }
 *   - flux/dev:        { prompt, image_size, num_inference_steps, guidance_scale }
 *   - flux-pro / flux/pro/v1.1, flux/pro/new, flux-2-*: similar to dev
 *   - older SD-shaped models: width/height/cfg/steps explicit
 */

import {
  type DiffusionProvider,
  type DiffusionResult,
  type ProviderCallOptions,
  AuthError,
  PaymentRequiredError,
  RateLimitError,
} from "./types";

const FAL_BASE = "https://fal.run";

interface FalImage {
  url: string;
  width?: number;
  height?: number;
  content_type?: string;
}

interface FalResponse {
  images: FalImage[];
  seed?: number;
  has_nsfw_concepts?: boolean[];
  prompt?: string;
}

function imageSize(width: number, height: number): string | { width: number; height: number } {
  // Try to match a Fal preset; otherwise pass explicit pixels.
  if (width === height) return "square_hd";
  if (width === 1024 && height === 768) return "landscape_4_3";
  if (width === 768 && height === 1024) return "portrait_4_3";
  return { width, height };
}

function buildInput(modelId: string, req: { prompt: string; negativePrompt?: string; seed: number; steps: number; cfg: number; width: number; height: number }): Record<string, unknown> {
  const id = modelId.toLowerCase();
  const isSchnell = id.includes("schnell");

  if (id.startsWith("fal-ai/flux") || id.includes("/flux")) {
    const steps = isSchnell ? Math.min(Math.max(req.steps, 1), 12) : req.steps;
    return {
      prompt: req.prompt,
      image_size: imageSize(req.width, req.height),
      num_inference_steps: steps,
      seed: req.seed,
      enable_safety_checker: true,
      ...(isSchnell ? {} : { guidance_scale: req.cfg }),
    };
  }

  // Generic SDXL-shaped fallback.
  return {
    prompt: req.prompt,
    negative_prompt: req.negativePrompt,
    seed: req.seed,
    num_inference_steps: req.steps,
    guidance_scale: req.cfg,
    image_size: imageSize(req.width, req.height),
  };
}

async function fetchAsBlob(url: string): Promise<Blob> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.blob();
}

export const falProvider: DiffusionProvider = {
  id: "fal",
  backend: "hosted",
  capabilities: { perStepLatents: false, cfgSweep: true, batch: false },

  async generate(req, opts?: ProviderCallOptions): Promise<DiffusionResult> {
    const apiKey = opts?.apiKey;
    if (!apiKey) throw new AuthError("fal");

    const startedAt = Date.now();
    let res: Response;
    try {
      res = await fetch(`${FAL_BASE}/${req.modelId}`, {
        method: "POST",
        headers: {
          Authorization: `Key ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(buildInput(req.modelId, req)),
      });
    } catch (err) {
      throw new Error(`Fal request failed: ${err instanceof Error ? err.message : String(err)}`);
    }

    if (res.status === 401 || res.status === 403) throw new AuthError("fal");
    if (res.status === 402) {
      throw new PaymentRequiredError(
        "fal",
        "Fal account is out of credit or requires a paid plan for this model.",
        "https://fal.ai/dashboard/billing",
      );
    }
    if (res.status === 429) {
      const retryAfter = parseInt(res.headers.get("retry-after") ?? "30", 10);
      throw new RateLimitError(Number.isFinite(retryAfter) ? retryAfter : 30);
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Fal ${res.status}: ${text.slice(0, 400)}`);
    }

    const data: FalResponse = await res.json();
    if (!data.images || data.images.length === 0) {
      throw new Error("Fal returned no images");
    }
    const blobs = await Promise.all(data.images.map((img) => fetchAsBlob(img.url)));

    return {
      images: blobs,
      meta: {
        providerId: "fal",
        modelId: req.modelId,
        seed: data.seed ?? req.seed,
        steps: req.steps,
        cfg: req.cfg,
        generatedAt: new Date().toISOString(),
        responseTimeMs: Date.now() - startedAt,
      },
    };
  },
};
