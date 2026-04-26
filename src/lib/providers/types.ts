/**
 * Provider abstraction for hosted (Replicate, Fal, Together, Stability)
 * and local (FastAPI) diffusion backends.
 */

export type Backend = "hosted" | "local";

export interface DiffusionRequest {
  modelId: string;
  prompt: string;
  negativePrompt?: string;
  seed: number;
  steps: number;
  cfg: number;
  width: number;
  height: number;
  scheduler?: string;
}

export interface DiffusionResultMeta {
  providerId: string;
  modelId: string;
  seed: number;
  steps: number;
  cfg: number;
  generatedAt: string;
  responseTimeMs: number;
}

export interface DiffusionResult {
  images: Blob[];
  latents?: Float32Array[];
  meta: DiffusionResultMeta;
}

export interface StepFrame {
  step: number;
  totalSteps: number;
  latent: Float32Array;
  previewPng?: Blob;
}

export interface DiffusionProvider {
  id: "replicate" | "fal" | "together" | "stability" | "local";
  backend: Backend;
  capabilities: {
    perStepLatents: boolean;
    cfgSweep: boolean;
    batch: boolean;
  };
  generate(req: DiffusionRequest, apiKey?: string): Promise<DiffusionResult>;
  trajectory?(
    req: DiffusionRequest,
    onStep: (frame: StepFrame) => void,
    apiKey?: string,
  ): Promise<DiffusionResult>;
}

export class CapabilityError extends Error {
  constructor(public capability: string, public providerId: string) {
    super(`Provider ${providerId} does not support ${capability}`);
  }
}

export class RateLimitError extends Error {
  constructor(public retryAfterSeconds: number) {
    super(`Rate limited; retry after ${retryAfterSeconds}s`);
  }
}

export class AuthError extends Error {
  constructor(public providerId: string) {
    super(`Missing or invalid API key for ${providerId}`);
  }
}
