/**
 * Local FastAPI provider.
 *
 * Talks to the diffusers-backed FastAPI service at `localBaseUrl`
 * (default http://localhost:8000). The service must be running:
 * see backend/README.md.
 *
 * Per-step latents will be exposed via the trajectory() method in v0.2.1.
 */

import {
  type DiffusionProvider,
  type DiffusionResult,
  type DiffusionResultMeta,
} from "./types";

const DEFAULT_BASE_URL = "http://localhost:8000";

interface LocalGenerateResponse {
  images: string[]; // data URLs
  meta: DiffusionResultMeta;
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? "image/png";
  const bytes = Buffer.from(b64, "base64");
  return new Blob([bytes], { type: mime });
}

export const localProvider: DiffusionProvider = {
  id: "local",
  backend: "local",
  // perStepLatents flips to true once /trajectory ships in v0.2.1.
  capabilities: { perStepLatents: false, cfgSweep: true, batch: false },

  async generate(req, opts): Promise<DiffusionResult> {
    const baseUrl = opts?.localBaseUrl ?? DEFAULT_BASE_URL;

    let res: Response;
    try {
      res = await fetch(`${baseUrl}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });
    } catch (err) {
      throw new Error(
        `Cannot reach local backend at ${baseUrl}. Is uvicorn running? (${err instanceof Error ? err.message : String(err)})`,
      );
    }

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Local backend ${res.status}: ${text.slice(0, 400)}`);
    }

    const data: LocalGenerateResponse = await res.json();
    const images = data.images.map(dataUrlToBlob);

    return { images, meta: data.meta };
  },
};
