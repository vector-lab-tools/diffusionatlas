/**
 * /api/diffuse — server-side dispatcher for image generation.
 *
 * Receives { providerId, request } from the client. The API key is read
 * from .env.local (preferred) or from the X-Diffusion-API-Key header (so
 * users can paste a key in Settings without writing to disk).
 *
 * Returns a multipart-ish JSON shape: images as base64 data URLs plus meta.
 * Blob caching happens client-side after this response.
 */

import { NextResponse } from "next/server";
import { getProvider } from "@/lib/providers/dispatcher";
import {
  type DiffusionRequest,
  type ProviderId,
  AuthError,
  PaymentRequiredError,
  RateLimitError,
  CapabilityError,
} from "@/lib/providers/types";

const ENV_KEY_BY_PROVIDER: Record<ProviderId, string | null> = {
  replicate: "REPLICATE_API_TOKEN",
  fal: "FAL_KEY",
  together: "TOGETHER_API_KEY",
  stability: "STABILITY_API_KEY",
  local: null,
};

interface DiffuseRequestBody {
  providerId: ProviderId;
  request: DiffusionRequest;
  /** Required for providerId === "local"; ignored otherwise. */
  localBaseUrl?: string;
}

async function blobToDataURL(blob: Blob): Promise<string> {
  const buf = Buffer.from(await blob.arrayBuffer());
  const mime = blob.type || "image/png";
  return `data:${mime};base64,${buf.toString("base64")}`;
}

export async function POST(req: Request) {
  let body: DiffuseRequestBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { providerId, request, localBaseUrl } = body;
  if (!providerId || !request) {
    return NextResponse.json({ error: "Missing providerId or request" }, { status: 400 });
  }

  let provider;
  try {
    provider = getProvider(providerId);
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Unknown provider" },
      { status: 400 },
    );
  }

  const envKeyName = ENV_KEY_BY_PROVIDER[providerId];
  const headerKey = req.headers.get("x-diffusion-api-key") ?? undefined;
  const apiKey = (envKeyName ? process.env[envKeyName] : undefined) ?? headerKey;

  try {
    const result = await provider.generate(request, { apiKey, localBaseUrl });
    const images = await Promise.all(result.images.map(blobToDataURL));
    return NextResponse.json({ images, meta: result.meta });
  } catch (err: unknown) {
    if (err instanceof AuthError) {
      return NextResponse.json(
        { error: "auth", providerId, message: err.message },
        { status: 401 },
      );
    }
    if (err instanceof PaymentRequiredError) {
      return NextResponse.json(
        { error: "payment_required", providerId, message: err.message, billingUrl: err.billingUrl },
        { status: 402 },
      );
    }
    if (err instanceof RateLimitError) {
      return NextResponse.json(
        { error: "rate_limit", retryAfterSeconds: err.retryAfterSeconds, message: err.message },
        { status: 429 },
      );
    }
    if (err instanceof CapabilityError) {
      return NextResponse.json(
        { error: "capability", capability: err.capability, providerId: err.providerId, message: err.message },
        { status: 422 },
      );
    }
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: "unknown", message }, { status: 500 });
  }
}
