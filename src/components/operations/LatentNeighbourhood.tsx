"use client";

import { useState } from "react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { DiffusionRequest, DiffusionResultMeta, ProviderId } from "@/lib/providers/types";
import { saveRun } from "@/lib/cache/runs";
import type { Run, RunSampleRef } from "@/types/run";
import { ALL_PROVIDERS, PROVIDER_DEFAULT_MODEL, providerLabel } from "@/lib/providers/defaults";
import { DeepDive } from "@/components/shared/DeepDive";
import { Table } from "@/components/shared/Table";
import { ExportButtons } from "@/components/shared/ExportButtons";
import { downloadCsv } from "@/lib/export/csv";
import { downloadPdf } from "@/lib/export/pdf";
import { downloadJson } from "@/lib/export/json";
import { lookup as lookupTerm, termsFor } from "@/lib/docs/glossary";

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

interface NeighbourRow {
  seed: number;
  status: "pending" | "running" | "ok" | "error";
  imageDataUrl?: string;
  meta?: DiffusionResultMeta;
  errorMessage?: string;
}

interface ProviderConfig {
  providerId: ProviderId;
  modelId: string;
  apiKey?: string;
  localBaseUrl?: string;
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? "image/png";
  const bytes = atob(b64);
  const buf = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
  return new Blob([buf], { type: mime });
}

/**
 * Deterministic seed jitter around an anchor: produces k integer offsets in
 * [-radius, +radius] excluding 0, evenly spread. Anchor itself is included as
 * the first sample so the user can see what they're sampling around.
 */
function buildSeedSet(anchor: number, k: number, radius: number): number[] {
  const out = [anchor];
  const offsets: number[] = [];
  let state = (anchor >>> 0) ^ 0x9e3779b9;
  while (offsets.length < k - 1) {
    state = Math.imul(state ^ (state >>> 15), 0x85ebca6b);
    state = Math.imul(state ^ (state >>> 13), 0xc2b2ae35);
    state ^= state >>> 16;
    const norm = (state >>> 0) / 0xffffffff;
    const offset = Math.round((norm * 2 - 1) * radius);
    if (offset !== 0 && !offsets.includes(offset)) offsets.push(offset);
  }
  for (const o of offsets) out.push(anchor + o);
  return out;
}

export function LatentNeighbourhood() {
  const { settings } = useSettings();
  const { set: cacheImage } = useImageBlobCache();

  const [prompt, setPrompt] = useState("a red cube on a blue cube, photorealistic");
  const [anchor, setAnchor] = useState(42);
  const [k, setK] = useState(6);
  const [radius, setRadius] = useState(1000);
  const [steps, setSteps] = useState(settings.defaults.steps);
  const [cfg] = useState(settings.defaults.cfg);
  const [rows, setRows] = useState<NeighbourRow[]>([]);
  const [rowsB, setRowsB] = useState<NeighbourRow[]>([]);
  const [running, setRunning] = useState(false);
  const [topError, setTopError] = useState<string | null>(null);
  const [errorLink, setErrorLink] = useState<{ href: string; label: string } | null>(null);

  // Cross-backend comparison
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [compareProviderId, setCompareProviderId] = useState<ProviderId>("local");
  const [compareModelId, setCompareModelId] = useState<string>(PROVIDER_DEFAULT_MODEL.local);

  const primaryConfig: ProviderConfig = {
    providerId: settings.providerId,
    modelId: settings.modelId,
    apiKey: settings.apiKeys[settings.providerId],
    localBaseUrl: settings.backend === "local" ? settings.localBaseUrl : undefined,
  };
  const compareConfig: ProviderConfig = {
    providerId: compareProviderId,
    modelId: compareModelId,
    apiKey: settings.apiKeys[compareProviderId],
    localBaseUrl: compareProviderId === "local" ? settings.localBaseUrl : undefined,
  };

  async function callOnce(seed: number, cfg_: ProviderConfig): Promise<{ ok: boolean; data?: DiffuseResponse; err?: GenError; status?: number }> {
    const request: DiffusionRequest = {
      modelId: cfg_.modelId,
      prompt,
      seed,
      steps,
      cfg,
      width: settings.defaults.width,
      height: settings.defaults.height,
      scheduler: settings.defaults.scheduler,
    };
    try {
      const res = await fetch("/api/diffuse", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(cfg_.apiKey ? { "X-Diffusion-API-Key": cfg_.apiKey } : {}),
        },
        body: JSON.stringify({ providerId: cfg_.providerId, request, localBaseUrl: cfg_.localBaseUrl }),
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

  function makeSetRowAt(setter: React.Dispatch<React.SetStateAction<NeighbourRow[]>>) {
    return (idx: number, patch: Partial<NeighbourRow>) =>
      setter((prev) => prev.map((r, j) => (j === idx ? { ...r, ...patch } : r)));
  }

  async function runOne(
    seed: number,
    idx: number,
    cfg_: ProviderConfig,
    setRowAt: (idx: number, patch: Partial<NeighbourRow>) => void,
    keyPrefix: string,
    imageKeyRef: { key?: string; meta?: DiffusionResultMeta },
  ): Promise<{ abortAll?: boolean }> {
    const MAX_ATTEMPTS = 4;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const r = await callOnce(seed, cfg_);
      if (r.ok && r.data) {
        const dataUrl = r.data.images[0];
        const imageKey = `${keyPrefix}::${r.data.meta.providerId}::${r.data.meta.modelId}::${seed}::${steps}::${cfg}::${Date.now()}`;
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
          setTopError(`Missing or invalid API key for ${cfg_.providerId}. Open Settings.`);
        } else {
          setTopError(`${providerLabel(cfg_.providerId)}: ${err.message ?? "insufficient credit"}`);
          if (err.billingUrl) setErrorLink({ href: err.billingUrl, label: `Top up ${providerLabel(cfg_.providerId)}` });
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

  async function runLane(
    seeds: number[],
    cfg_: ProviderConfig,
    keyPrefix: string,
    setter: React.Dispatch<React.SetStateAction<NeighbourRow[]>>,
  ): Promise<{ samples: RunSampleRef[]; providerIdSeen?: string; modelIdSeen?: string }> {
    setter(seeds.map((seed) => ({ seed, status: "pending" })));
    const setRowAt = makeSetRowAt(setter);
    const samples: RunSampleRef[] = [];
    let providerIdSeen: string | undefined;
    let modelIdSeen: string | undefined;
    let aborted = false;
    for (let i = 0; i < seeds.length; i++) {
      if (aborted) {
        setter((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Skipped" } : r)));
        break;
      }
      setRowAt(i, { status: "running" });
      const ref: { key?: string; meta?: DiffusionResultMeta } = {};
      const { abortAll } = await runOne(seeds[i], i, cfg_, setRowAt, keyPrefix, ref);
      if (ref.key && ref.meta) {
        samples.push({ imageKey: ref.key, variable: seeds[i], responseTimeMs: ref.meta.responseTimeMs });
        providerIdSeen = ref.meta.providerId;
        modelIdSeen = ref.meta.modelId;
      }
      if (abortAll) aborted = true;
      if (i < seeds.length - 1) await new Promise((res) => setTimeout(res, 1500));
    }
    return { samples, providerIdSeen, modelIdSeen };
  }

  async function persistRun(
    samples: RunSampleRef[],
    providerIdSeen: string | undefined,
    modelIdSeen: string | undefined,
    seeds: number[],
    extra: Record<string, unknown>,
  ) {
    if (samples.length === 0 || !providerIdSeen || !modelIdSeen) return;
    const run: Run = {
      id: `nbr::${providerIdSeen}::${Date.now()}`,
      kind: "neighbourhood",
      createdAt: new Date().toISOString(),
      providerId: providerIdSeen,
      modelId: modelIdSeen,
      prompt,
      seed: anchor,
      steps,
      cfg,
      width: settings.defaults.width,
      height: settings.defaults.height,
      samples,
      extra: { k, radius, anchor, seedList: seeds, ...extra },
    };
    await saveRun(run);
  }

  async function runNeighbourhood() {
    if (k < 1) {
      setTopError("k must be ≥ 1.");
      return;
    }
    setRunning(true);
    setTopError(null);
    setErrorLink(null);
    if (!compareEnabled) setRowsB([]);

    const seeds = buildSeedSet(anchor, k, radius);

    if (compareEnabled) {
      const [a, b] = await Promise.all([
        runLane(seeds, primaryConfig, "nbr", setRows),
        runLane(seeds, compareConfig, "nbr-cmp", setRowsB),
      ]);
      await persistRun(a.samples, a.providerIdSeen, a.modelIdSeen, seeds, { lane: "primary" });
      await persistRun(b.samples, b.providerIdSeen, b.modelIdSeen, seeds, { lane: "compare" });
    } else {
      const a = await runLane(seeds, primaryConfig, "nbr", setRows);
      await persistRun(a.samples, a.providerIdSeen, a.modelIdSeen, seeds, {});
    }
    setRunning(false);
  }

  return (
    <div className="card-editorial p-6 max-w-5xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">Latent Neighbourhood</h2>
      <p className="font-body text-body-sm text-foreground mb-1">
        Sample nearby points by jittering the seed around an anchor. Reveals the local manifold structure: smoothness, basins, and where small perturbations produce categorical jumps in the output.
      </p>
      <p className="font-sans text-caption italic text-muted-foreground mb-4">
        Read the grid as: nearby seeds that look similar mean a smooth basin; sudden categorical jumps (a cat becomes a dog) mean a basin boundary. Small radius probes local geometry; large radius probes how connected the manifold is. With Compare on, two providers' neighbourhoods around the same anchor reveal where their geometries agree on what "near" means.
      </p>

      <div className="border border-cream/80 bg-cream/30 text-foreground p-3 mb-4 font-sans text-caption rounded-sm">
        Hosted mode samples by changing the seed (each seed maps to a different starting point in latent space). True latent-space perturbation at a chosen σ requires a `/neighbourhood` endpoint on the local backend, queued.
      </div>

      <div className="grid grid-cols-1 gap-3 mb-4">
        <label className="block" title={lookupTerm("Prompt")}>
          <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Prompt</span>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="input-editorial mt-1"
          />
        </label>
        <div className="grid grid-cols-4 gap-3">
          <label className="block" title={lookupTerm("Anchor seed")}>
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Anchor seed</span>
            <input
              type="number"
              value={anchor}
              onChange={(e) => setAnchor(parseInt(e.target.value, 10) || 0)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block" title={lookupTerm("k samples")}>
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">k samples</span>
            <input
              type="number"
              min={1}
              max={24}
              value={k}
              onChange={(e) => setK(parseInt(e.target.value, 10) || 1)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block" title={lookupTerm("Radius")}>
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Radius</span>
            <input
              type="number"
              min={1}
              value={radius}
              onChange={(e) => setRadius(parseInt(e.target.value, 10) || 1)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block" title={lookupTerm("Steps")}>
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Steps</span>
            <input
              type="number"
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value, 10) || 1)}
              className="input-editorial mt-1"
            />
          </label>
        </div>
      </div>

      {/* Compare-with: cross-backend agreement */}
      <div className="border border-parchment rounded-sm p-3 mb-4 bg-cream/20">
        <label className="flex items-center gap-2 font-sans text-body-sm">
          <input
            type="checkbox"
            checked={compareEnabled}
            onChange={(e) => setCompareEnabled(e.target.checked)}
          />
          <span>Compare with a second provider (sample the same anchor in two manifolds)</span>
        </label>
        {compareEnabled && (
          <div className="grid grid-cols-2 gap-3 mt-3">
            <label className="block">
              <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">Compare provider</span>
              <select
                value={compareProviderId}
                onChange={(e) => {
                  const p = e.target.value as ProviderId;
                  setCompareProviderId(p);
                  setCompareModelId(PROVIDER_DEFAULT_MODEL[p]);
                }}
                className="input-editorial mt-1"
              >
                {ALL_PROVIDERS.filter((p) => p !== settings.providerId).map((p) => (
                  <option key={p} value={p}>{providerLabel(p)}</option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">Compare model</span>
              <input
                type="text"
                value={compareModelId}
                onChange={(e) => setCompareModelId(e.target.value)}
                className="input-editorial mt-1"
              />
            </label>
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <button
          onClick={() => void runNeighbourhood()}
          disabled={running}
          className={running ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running
            ? "Sampling…"
            : compareEnabled
              ? `Sample × 2 (${k * 2} images)`
              : `Sample neighbourhood (${k} ${k === 1 ? "image" : "images"})`}
        </button>
        <span className="font-sans text-caption text-muted-foreground">
          {providerLabel(settings.providerId)} · {settings.modelId}
          {compareEnabled && <> ↔ {providerLabel(compareProviderId)} · {compareModelId}</>}
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
        <NeighbourLane
          label={`${providerLabel(settings.providerId)} · ${settings.modelId}`}
          rows={rows}
        />
      )}
      {compareEnabled && rowsB.length > 0 && (
        <NeighbourLane
          label={`${providerLabel(compareProviderId)} · ${compareModelId}`}
          rows={rowsB}
        />
      )}

      {(rows.length > 0 || rowsB.length > 0) && (
        <NeighbourDeepDive
          prompt={prompt}
          anchor={anchor}
          k={k}
          radius={radius}
          steps={steps}
          primaryLabel={`${providerLabel(settings.providerId)} · ${settings.modelId}`}
          rowsA={rows}
          compareEnabled={compareEnabled}
          compareLabel={`${providerLabel(compareProviderId)} · ${compareModelId}`}
          rowsB={rowsB}
        />
      )}
    </div>
  );
}

interface NeighbourDeepDiveProps {
  prompt: string;
  anchor: number;
  k: number;
  radius: number;
  steps: number;
  primaryLabel: string;
  rowsA: NeighbourRow[];
  compareEnabled: boolean;
  compareLabel: string;
  rowsB: NeighbourRow[];
}

function NeighbourDeepDive({ prompt, anchor, k, radius, steps, primaryLabel, rowsA, compareEnabled, compareLabel, rowsB }: NeighbourDeepDiveProps) {
  function laneRows(rows: NeighbourRow[]) {
    return rows.map((r, i) => [
      r.seed,
      i === 0 ? "anchor" : "neighbour",
      r.status,
      r.meta?.providerId ?? "—",
      r.meta?.modelId ?? "—",
      r.meta ? (r.meta.responseTimeMs / 1000).toFixed(2) + "s" : "—",
      r.errorMessage ?? "",
    ]);
  }
  const headers = ["seed", "role", "status", "provider", "model", "time", "error"];
  const tableRows: Array<Array<string | number>> = [
    ...laneRows(rowsA).map((r) => ["primary", ...r]),
    ...(compareEnabled ? laneRows(rowsB).map((r) => ["compare", ...r]) : []),
  ];

  const stamp = `nbr-${anchor}-${Date.now()}`;
  function exportCsv() { downloadCsv(`${stamp}.csv`, ["lane", ...headers], tableRows); }
  function exportJson() {
    downloadJson(`${stamp}.json`, {
      operation: "latent-neighbourhood",
      prompt, anchor, k, radius, steps,
      primary: { label: primaryLabel, rows: rowsA },
      compare: compareEnabled ? { label: compareLabel, rows: rowsB } : null,
    });
  }
  function exportPdf() {
    const images = [
      ...rowsA.filter((r) => r.imageDataUrl).map((r, i) => ({
        dataUrl: r.imageDataUrl as string,
        caption: `primary · seed ${r.seed}${i === 0 ? " (anchor)" : ""}`,
      })),
      ...(compareEnabled ? rowsB.filter((r) => r.imageDataUrl).map((r, i) => ({
        dataUrl: r.imageDataUrl as string,
        caption: `compare · seed ${r.seed}${i === 0 ? " (anchor)" : ""}`,
      })) : []),
    ];
    downloadPdf(`${stamp}.pdf`, {
      meta: {
        title: "Latent Neighbourhood",
        subtitle: compareEnabled ? `${primaryLabel}  ↔  ${compareLabel}` : primaryLabel,
        fields: [
          { label: "Prompt", value: prompt },
          { label: "Anchor seed", value: anchor },
          { label: "k samples", value: k },
          { label: "Radius", value: radius },
          { label: "Steps", value: steps },
        ],
      },
      images,
      appendix: [
        {
          title: "Per-seed results",
          caption: "Each row in the neighbourhood: seed, role (anchor / neighbour), provider, model, response time. With Compare on, both lanes are interleaved.",
          table: { headers: ["lane", ...headers], rows: tableRows },
        },
      ],
      glossary: termsFor(["Prompt", "Anchor seed", "k samples", "Radius", "Steps", "lane", "seed", "role", "status", "provider", "model", "time", "error"]),
    });
  }

  return (
    <DeepDive actions={<ExportButtons onCsv={exportCsv} onPdf={exportPdf} onJson={exportJson} />}>
      <Table headers={["lane", ...headers]} rows={tableRows} numericColumns={[1, 5]} />
    </DeepDive>
  );
}

interface NeighbourLaneProps {
  label: string;
  rows: NeighbourRow[];
}

function NeighbourLane({ label, rows }: NeighbourLaneProps) {
  return (
    <div className="border-t border-parchment pt-4 mt-4">
      <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
        {label}
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {rows.map((row, i) => (
          <div key={i} className="flex flex-col">
            <div className="aspect-square bg-cream/50 border border-parchment rounded-sm overflow-hidden flex items-center justify-center">
              {row.status === "ok" && row.imageDataUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={row.imageDataUrl} alt={`seed ${row.seed}`} className="w-full h-full object-cover" />
              )}
              {row.status === "running" && (
                <span className="font-sans text-caption text-muted-foreground text-center px-2">
                  {row.errorMessage ?? "Generating…"}
                </span>
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
              seed <span className="text-foreground font-medium">{row.seed}</span>
              {i === 0 && <span className="text-burgundy"> · anchor</span>}
              {row.meta && <> · {(row.meta.responseTimeMs / 1000).toFixed(1)}s</>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
