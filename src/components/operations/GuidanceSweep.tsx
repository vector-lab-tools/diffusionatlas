"use client";

import { useState } from "react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { DiffusionRequest, DiffusionResultMeta } from "@/lib/providers/types";

interface DiffuseResponse {
  images: string[]; // data URLs
  meta: DiffusionResultMeta;
}

interface GenError {
  error: string;
  message?: string;
  retryAfterSeconds?: number;
  billingUrl?: string;
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? "image/png";
  const bytes = atob(b64);
  const buf = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
  return new Blob([buf], { type: mime });
}

export function GuidanceSweep() {
  const { settings } = useSettings();
  const { set: cacheImage } = useImageBlobCache();

  const [prompt, setPrompt] = useState("a red cube on a blue cube, photorealistic");
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(settings.defaults.steps);
  const [cfg, setCfg] = useState(settings.defaults.cfg);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<DiffuseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorLink, setErrorLink] = useState<{ href: string; label: string } | null>(null);

  async function generate() {
    setRunning(true);
    setError(null);
    setErrorLink(null);
    setResult(null);

    const request: DiffusionRequest = {
      modelId: settings.modelId,
      prompt,
      seed,
      steps,
      cfg,
      width: settings.defaults.width,
      height: settings.defaults.height,
      scheduler: settings.defaults.scheduler,
    };

    const apiKey = settings.apiKeys[settings.providerId];
    try {
      const res = await fetch("/api/diffuse", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { "X-Diffusion-API-Key": apiKey } : {}),
        },
        body: JSON.stringify({ providerId: settings.providerId, request }),
      });

      if (!res.ok) {
        const err: GenError = await res.json().catch(() => ({ error: "unknown" }));
        if (err.error === "auth") {
          setError("Missing or invalid API key. Open Settings and paste a Replicate token.");
        } else if (err.error === "payment_required") {
          setError(err.message ?? "Insufficient credit on the provider account.");
          if (err.billingUrl) setErrorLink({ href: err.billingUrl, label: "Add credit on Replicate" });
        } else if (err.error === "rate_limit") {
          setError(`Rate limited. Retry after ${err.retryAfterSeconds ?? 30}s.`);
        } else {
          setError(err.message ?? `Generation failed (${res.status}).`);
        }
        return;
      }

      const data: DiffuseResponse = await res.json();
      setResult(data);

      const runId = `${data.meta.providerId}::${data.meta.modelId}::${seed}::${steps}::${cfg}::${Date.now()}`;
      await Promise.all(
        data.images.map((url, i) => cacheImage(`${runId}::${i}`, dataUrlToBlob(url))),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="card-editorial p-6 max-w-3xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">Guidance Sweep</h2>
      <p className="font-body text-body-sm text-foreground mb-4">
        Generate the same prompt and seed across CFG values. v0.1.3 ships single-image generation; the parallel sweep
        across multiple CFGs follows next.
      </p>

      <div className="grid grid-cols-1 gap-3 mb-4">
        <label className="block">
          <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">Prompt</span>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="input-editorial mt-1"
          />
        </label>
        <div className="grid grid-cols-3 gap-3">
          <label className="block">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">Seed</span>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">Steps</span>
            <input
              type="number"
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value, 10) || 1)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">CFG</span>
            <input
              type="number"
              step="0.5"
              value={cfg}
              onChange={(e) => setCfg(parseFloat(e.target.value) || 0)}
              className="input-editorial mt-1"
            />
          </label>
        </div>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={() => void generate()}
          disabled={running}
          className={running ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running ? "Generating…" : "Generate"}
        </button>
        <span className="font-sans text-caption text-muted-foreground">
          {settings.backend === "local" ? "Local" : "Hosted"} · {settings.providerId} · {settings.modelId}
        </span>
      </div>

      {error && (
        <div className="border border-burgundy/40 bg-burgundy/5 text-burgundy p-3 mb-4 font-sans text-body-sm rounded-sm">
          {error}
          {errorLink && (
            <>
              {" "}
              <a
                href={errorLink.href}
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-2 hover:text-burgundy-900"
              >
                {errorLink.label} →
              </a>
            </>
          )}
        </div>
      )}

      {result && (
        <div className="border-t border-parchment pt-4">
          <div className="grid grid-cols-1 gap-3">
            {result.images.map((src, i) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={i} src={src} alt={`Generation ${i}`} className="w-full max-w-xl rounded-sm shadow-editorial" />
            ))}
          </div>
          <dl className="mt-3 grid grid-cols-[120px_1fr] gap-y-1 font-sans text-caption text-muted-foreground">
            <dt>Provider</dt><dd className="text-foreground">{result.meta.providerId}</dd>
            <dt>Model</dt><dd className="text-foreground">{result.meta.modelId}</dd>
            <dt>Seed</dt><dd className="text-foreground">{result.meta.seed}</dd>
            <dt>Steps</dt><dd className="text-foreground">{result.meta.steps}</dd>
            <dt>CFG</dt><dd className="text-foreground">{result.meta.cfg}</dd>
            <dt>Time</dt><dd className="text-foreground">{(result.meta.responseTimeMs / 1000).toFixed(1)}s</dd>
          </dl>
        </div>
      )}
    </div>
  );
}
