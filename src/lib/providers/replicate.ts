/**
 * Replicate hosted provider.
 *
 * Replicate exposes models via prediction API: POST /predictions, poll until
 * status === "succeeded", then download output URLs as image blobs.
 *
 * Per-step latents are NOT available, so trajectory() is unimplemented.
 */

import Replicate from "replicate";
import {
  type DiffusionProvider,
  type DiffusionRequest,
  type DiffusionResult,
  AuthError,
  PaymentRequiredError,
  RateLimitError,
} from "./types";

type ReplicateInput = Record<string, unknown>;

/**
 * Replicate models expose different input schemas:
 * - flux-2-* takes { prompt, resolution, aspect_ratio, output_format, ... }
 * - flux-1-* / schnell / dev takes { prompt, aspect_ratio, num_inference_steps, ... }
 * - SDXL-shaped models (bytedance/sdxl-lightning, older forks) take
 *   { prompt, num_inference_steps, guidance_scale, width, height, ... }
 *
 * We detect the family from modelId and build the matching shape.
 */
function aspectRatio(w: number, h: number): string {
  if (w === h) return "1:1";
  if (w > h) {
    if (Math.abs(w / h - 16 / 9) < 0.05) return "16:9";
    if (Math.abs(w / h - 4 / 3) < 0.05) return "4:3";
    if (Math.abs(w / h - 3 / 2) < 0.05) return "3:2";
  } else {
    if (Math.abs(h / w - 16 / 9) < 0.05) return "9:16";
    if (Math.abs(h / w - 4 / 3) < 0.05) return "3:4";
    if (Math.abs(h / w - 3 / 2) < 0.05) return "2:3";
  }
  return "1:1";
}

function resolutionMP(w: number, h: number): string {
  const mp = (w * h) / 1_000_000;
  if (mp < 0.75) return "0.5 MP";
  if (mp < 1.5) return "1 MP";
  return "2 MP";
}

function buildInput(req: DiffusionRequest): ReplicateInput {
  const id = req.modelId.toLowerCase();
  const ratio = aspectRatio(req.width, req.height);

  if (id.includes("flux-2")) {
    return {
      prompt: req.prompt,
      resolution: resolutionMP(req.width, req.height),
      aspect_ratio: ratio,
      output_format: "png",
      output_quality: 90,
      safety_tolerance: 2,
    };
  }

  if (id.startsWith("black-forest-labs/flux")) {
    // flux-1 family (schnell, dev, pro)
    return {
      prompt: req.prompt,
      aspect_ratio: ratio,
      num_inference_steps: req.steps,
      output_format: "png",
      output_quality: 90,
      ...(id.includes("schnell") ? {} : { guidance: req.cfg }),
      ...(req.seed !== undefined ? { seed: req.seed } : {}),
    };
  }

  // SDXL-shaped fallback
  return {
    prompt: req.prompt,
    negative_prompt: req.negativePrompt,
    seed: req.seed,
    num_inference_steps: req.steps,
    guidance_scale: req.cfg,
    width: req.width,
    height: req.height,
    scheduler: req.scheduler,
  };
}

async function fetchAsBlob(url: string): Promise<Blob> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.blob();
}

export const replicateProvider: DiffusionProvider = {
  id: "replicate",
  backend: "hosted",
  capabilities: { perStepLatents: false, cfgSweep: true, batch: false },

  async generate(req, apiKey) {
    if (!apiKey) throw new AuthError("replicate");

    const client = new Replicate({ auth: apiKey, useFileOutput: false });
    const startedAt = Date.now();

    let outputs: unknown;
    try {
      outputs = await client.run(req.modelId as `${string}/${string}`, {
        input: buildInput(req),
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      if (/401|unauthor/i.test(message)) throw new AuthError("replicate");
      if (/402|insufficient credit|payment required/i.test(message)) {
        throw new PaymentRequiredError(
          "replicate",
          "Insufficient Replicate credit. Purchase credit and try again.",
          "https://replicate.com/account/billing#billing",
        );
      }
      if (/429|rate.?limit/i.test(message)) {
        const m = message.match(/retry.?after[^\d]*(\d+)/i);
        const retryAfter = m ? parseInt(m[1], 10) : 30;
        throw new RateLimitError(retryAfter);
      }
      throw err;
    }

    // Replicate output is typically string[] of URLs, or a single URL string.
    const urls: string[] = Array.isArray(outputs)
      ? (outputs as string[])
      : typeof outputs === "string"
        ? [outputs]
        : [];

    if (urls.length === 0) {
      throw new Error("Replicate returned no image outputs");
    }

    const images = await Promise.all(urls.map(fetchAsBlob));

    const result: DiffusionResult = {
      images,
      meta: {
        providerId: "replicate",
        modelId: req.modelId,
        seed: req.seed,
        steps: req.steps,
        cfg: req.cfg,
        generatedAt: new Date().toISOString(),
        responseTimeMs: Date.now() - startedAt,
      },
    };
    return result;
  },
};
