"use client";

import { useState } from "react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { DiffusionRequest, DiffusionResultMeta } from "@/lib/providers/types";
import { saveRun } from "@/lib/cache/runs";
import type { Run, RunSampleRef } from "@/types/run";

interface DiffuseResponse {
  images: string[];
  meta: DiffusionResultMeta;
}

interface GenError {
  error: string;
  message?: string;
  retryAfterSeconds?: number;
  billingUrl?: string;
}

interface SweepRow {
  cfg: number;
  status: "pending" | "running" | "ok" | "error";
  imageDataUrl?: string;
  meta?: DiffusionResultMeta;
  errorMessage?: string;
}

const DEFAULT_CFG_SET = "1, 2.5, 4, 7.5, 12";

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? "image/png";
  const bytes = atob(b64);
  const buf = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
  return new Blob([buf], { type: mime });
}

function parseCfgList(raw: string): number[] {
  return raw
    .split(/[,\s]+/)
    .map((s) => parseFloat(s))
    .filter((n) => Number.isFinite(n) && n >= 0);
}

function modelIgnoresCfg(modelId: string): boolean {
  const id = modelId.toLowerCase();
  return id.includes("schnell") || id.includes("flux-2");
}

export function GuidanceSweep() {
  const { settings } = useSettings();
  const { set: cacheImage } = useImageBlobCache();

  const [prompt, setPrompt] = useState("a red cube on a blue cube, photorealistic");
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(settings.defaults.steps);
  const [cfgList, setCfgList] = useState(DEFAULT_CFG_SET);
  const [rows, setRows] = useState<SweepRow[]>([]);
  const [running, setRunning] = useState(false);
  const [topError, setTopError] = useState<string | null>(null);
  const [errorLink, setErrorLink] = useState<{ href: string; label: string } | null>(null);

  const cfgs = parseCfgList(cfgList);
  const cfgWarning = modelIgnoresCfg(settings.modelId);

  async function callOnce(cfg: number): Promise<{ ok: boolean; data?: DiffuseResponse; err?: GenError; status?: number }> {
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
        body: JSON.stringify({
          providerId: settings.providerId,
          request,
          localBaseUrl: settings.backend === "local" ? settings.localBaseUrl : undefined,
        }),
      });
      if (!res.ok) {
        const err: GenError = await res.json().catch(() => ({ error: "unknown" }));
        return { ok: false, err, status: res.status };
      }
      const data: DiffuseResponse = await res.json();
      return { ok: true, data };
    } catch (err) {
      return { ok: false, err: { error: "network", message: err instanceof Error ? err.message : String(err) } };
    }
  }

  function setRowAt(idx: number, patch: Partial<SweepRow>) {
    setRows((prev) => prev.map((r, j) => (j === idx ? { ...r, ...patch } : r)));
  }

  async function runOne(cfg: number, idx: number, imageKeyRef: { key?: string; meta?: DiffusionResultMeta }): Promise<{ abortAll?: boolean }> {
    const MAX_ATTEMPTS = 4;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const r = await callOnce(cfg);
      if (r.ok && r.data) {
        const dataUrl = r.data.images[0];
        const imageKey = `sweep::${r.data.meta.providerId}::${r.data.meta.modelId}::${seed}::${steps}::${cfg}::${Date.now()}`;
        await cacheImage(imageKey, dataUrlToBlob(dataUrl));
        imageKeyRef.key = imageKey;
        imageKeyRef.meta = r.data.meta;
        setRowAt(idx, { status: "ok", imageDataUrl: dataUrl, meta: r.data.meta });
        return {};
      }

      const err = r.err ?? { error: "unknown" };
      if (err.error === "rate_limit" && attempt < MAX_ATTEMPTS) {
        const wait = (err.retryAfterSeconds ?? 5) + 1;
        for (let s = wait; s > 0; s--) {
          setRowAt(idx, { status: "running", errorMessage: `Rate limited; retrying in ${s}s` });
          await new Promise((res) => setTimeout(res, 1000));
        }
        setRowAt(idx, { status: "running", errorMessage: undefined });
        continue;
      }

      if (err.error === "auth" || err.error === "payment_required") {
        if (err.error === "auth") {
          setTopError("Missing or invalid API key. Open Settings and paste a token.");
        } else {
          setTopError(err.message ?? "Insufficient credit on the provider account.");
          if (err.billingUrl) setErrorLink({ href: err.billingUrl, label: "Add credit on Replicate" });
        }
        setRowAt(idx, { status: "error", errorMessage: err.message ?? "Aborted" });
        return { abortAll: true };
      }

      setRowAt(idx, { status: "error", errorMessage: err.message ?? `Failed (${r.status ?? "?"})` });
      return {};
    }
    setRowAt(idx, { status: "error", errorMessage: "Rate limited; gave up after retries" });
    return {};
  }

  async function runSweep() {
    if (cfgs.length === 0) {
      setTopError("Enter at least one CFG value.");
      return;
    }
    setRunning(true);
    setTopError(null);
    setErrorLink(null);
    const initial: SweepRow[] = cfgs.map((cfg) => ({ cfg, status: "pending" }));
    setRows(initial);

    let aborted = false;
    const samples: RunSampleRef[] = [];
    let providerIdSeen: string | undefined;
    let modelIdSeen: string | undefined;
    for (let i = 0; i < cfgs.length; i++) {
      if (aborted) {
        setRows((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Skipped" } : r)));
        break;
      }
      setRowAt(i, { status: "running" });
      const ref: { key?: string; meta?: DiffusionResultMeta } = {};
      const { abortAll } = await runOne(cfgs[i], i, ref);
      if (ref.key && ref.meta) {
        samples.push({ imageKey: ref.key, variable: cfgs[i], responseTimeMs: ref.meta.responseTimeMs });
        providerIdSeen = ref.meta.providerId;
        modelIdSeen = ref.meta.modelId;
      }
      if (abortAll) aborted = true;
      if (i < cfgs.length - 1) await new Promise((res) => setTimeout(res, 1500));
    }

    if (samples.length > 0 && providerIdSeen && modelIdSeen) {
      const run: Run = {
        id: `sweep::${Date.now()}`,
        kind: "sweep",
        createdAt: new Date().toISOString(),
        providerId: providerIdSeen,
        modelId: modelIdSeen,
        prompt,
        seed,
        steps,
        cfg: cfgs[0],
        width: settings.defaults.width,
        height: settings.defaults.height,
        samples,
        extra: { cfgList: cfgs },
      };
      await saveRun(run);
    }

    setRunning(false);
  }

  return (
    <div className="card-editorial p-6 max-w-4xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">Guidance Sweep</h2>
      <p className="font-body text-body-sm text-foreground mb-4">
        Same prompt and seed across a range of CFG values. Reveals the controllability surface and where mode collapse begins.
      </p>

      {cfgWarning && (
        <div className="border border-gold/40 bg-gold/10 text-foreground p-3 mb-4 font-sans text-caption rounded-sm">
          The current model (<strong>{settings.modelId}</strong>) does not use a guidance scale.
          A sweep will produce nearly identical images. For meaningful results switch to <code>black-forest-labs/flux-dev</code> or another CFG-bearing model in Settings.
        </div>
      )}

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
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">CFG list</span>
            <input
              type="text"
              value={cfgList}
              onChange={(e) => setCfgList(e.target.value)}
              placeholder="1, 2.5, 4, 7.5, 12"
              className="input-editorial mt-1"
            />
          </label>
        </div>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={() => void runSweep()}
          disabled={running}
          className={running ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running ? "Running sweep…" : `Run sweep (${cfgs.length} ${cfgs.length === 1 ? "image" : "images"})`}
        </button>
        <span className="font-sans text-caption text-muted-foreground">
          {settings.backend === "local" ? "Local" : "Hosted"} · {settings.providerId} · {settings.modelId}
        </span>
      </div>

      {topError && (
        <div className="border border-burgundy/40 bg-burgundy/5 text-burgundy p-3 mb-4 font-sans text-body-sm rounded-sm">
          {topError}
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

      {rows.length > 0 && (
        <div className="border-t border-parchment pt-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {rows.map((row, i) => (
              <div key={i} className="flex flex-col">
                <div className="aspect-square bg-cream/50 border border-parchment rounded-sm overflow-hidden flex items-center justify-center">
                  {row.status === "ok" && row.imageDataUrl && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={row.imageDataUrl} alt={`CFG ${row.cfg}`} className="w-full h-full object-cover" />
                  )}
                  {row.status === "running" && (
                    <span className="font-sans text-caption text-muted-foreground">Generating…</span>
                  )}
                  {row.status === "pending" && (
                    <span className="font-sans text-caption text-muted-foreground">Queued</span>
                  )}
                  {row.status === "error" && (
                    <span className="font-sans text-caption text-burgundy text-center px-2">
                      {row.errorMessage ?? "Failed"}
                    </span>
                  )}
                </div>
                <div className="mt-1 font-sans text-caption text-muted-foreground text-center">
                  CFG <span className="text-foreground font-medium">{row.cfg}</span>
                  {row.meta && <> · {(row.meta.responseTimeMs / 1000).toFixed(1)}s</>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
