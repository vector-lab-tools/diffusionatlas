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
  RateLimitError,
} from "./types";

interface ReplicateInput {
  prompt: string;
  negative_prompt?: string;
  seed?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  width?: number;
  height?: number;
  scheduler?: string;
}

function buildInput(req: DiffusionRequest): ReplicateInput {
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

    const client = new Replicate({ auth: apiKey });
    const startedAt = Date.now();

    let outputs: unknown;
    try {
      outputs = await client.run(req.modelId as `${string}/${string}`, {
        input: buildInput(req),
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      if (/401|unauthor/i.test(message)) throw new AuthError("replicate");
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
