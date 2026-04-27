"use client";

import { useEffect, useMemo, useState } from "react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { DiffusionRequest, DiffusionResultMeta, ProviderId } from "@/lib/providers/types";
import { saveRun } from "@/lib/cache/runs";
import type { Run, RunSampleRef } from "@/types/run";
import { ahash, normalisedDrift, type Hash } from "@/lib/geometry/perceptual_hash";
import { ALL_PROVIDERS, PROVIDER_DEFAULT_MODEL, providerLabel } from "@/lib/providers/defaults";
import { DeepDive } from "@/components/shared/DeepDive";
import { Table } from "@/components/shared/Table";
import { ExportButtons } from "@/components/shared/ExportButtons";
import { CameraRoll } from "@/components/shared/CameraRoll";
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

interface SweepRow {
  cfg: number;
  status: "pending" | "running" | "ok" | "error";
  imageDataUrl?: string;
  meta?: DiffusionResultMeta;
  errorMessage?: string;
  hash?: Hash;
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

function DriftCurve({ cfgs, drift }: { cfgs: number[]; drift: Array<number | null> }) {
  const W = 480, H = 100, P = 22;
  const xs = cfgs;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const xScale = (x: number) => P + ((x - minX) / (maxX - minX || 1)) * (W - 2 * P);
  const yScale = (y: number) => H - P - y * (H - 2 * P); // 0..1 → bottom..top

  const points = cfgs
    .map((cfg, i) => (drift[i] !== null ? `${xScale(cfg)},${yScale(drift[i] as number)}` : null))
    .filter((p): p is string => p !== null);
  const path = points.length >= 2 ? `M ${points.join(" L ")}` : "";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-2xl">
      {/* Axes */}
      <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="hsl(var(--parchment-dark))" strokeWidth={1} />
      <line x1={P} y1={P} x2={P} y2={H - P} stroke="hsl(var(--parchment-dark))" strokeWidth={1} />
      <text x={P} y={H - 4} fontSize={9} fill="hsl(var(--muted-foreground))" fontFamily="sans-serif">
        CFG {minX}
      </text>
      <text x={W - P - 16} y={H - 4} fontSize={9} fill="hsl(var(--muted-foreground))" fontFamily="sans-serif">
        {maxX}
      </text>
      <text x={2} y={P + 4} fontSize={9} fill="hsl(var(--muted-foreground))" fontFamily="sans-serif">1</text>
      <text x={2} y={H - P} fontSize={9} fill="hsl(var(--muted-foreground))" fontFamily="sans-serif">0</text>
      {/* Path */}
      {path && <path d={path} fill="none" stroke="hsl(var(--burgundy))" strokeWidth={1.5} />}
      {/* Points */}
      {cfgs.map((cfg, i) =>
        drift[i] !== null ? (
          <g key={i}>
            <circle cx={xScale(cfg)} cy={yScale(drift[i] as number)} r={3} fill="hsl(var(--burgundy))" />
            <text
              x={xScale(cfg)}
              y={yScale(drift[i] as number) - 6}
              fontSize={9}
              textAnchor="middle"
              fill="hsl(var(--foreground))"
              fontFamily="sans-serif"
            >
              {(drift[i] as number).toFixed(2)}
            </text>
          </g>
        ) : null,
      )}
    </svg>
  );
}

interface ProviderConfig {
  providerId: ProviderId;
  modelId: string;
  apiKey?: string;
  localBaseUrl?: string;
}

export function GuidanceSweep() {
  const { settings } = useSettings();
  const { set: cacheImage } = useImageBlobCache();

  const [prompt, setPrompt] = useState("a red cube on a blue cube, photorealistic");
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(settings.defaults.steps);
  const [cfgList, setCfgList] = useState(DEFAULT_CFG_SET);
  const [rows, setRows] = useState<SweepRow[]>([]);
  const [rowsB, setRowsB] = useState<SweepRow[]>([]);
  const [running, setRunning] = useState(false);
  const [topError, setTopError] = useState<string | null>(null);
  const [errorLink, setErrorLink] = useState<{ href: string; label: string } | null>(null);

  // Cross-backend agreement: optionally run a second sweep against a different
  // provider in parallel and show the two grids side by side.
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [compareProviderId, setCompareProviderId] = useState<ProviderId>("local");
  const [compareModelId, setCompareModelId] = useState<string>(PROVIDER_DEFAULT_MODEL.local);

  const cfgs = parseCfgList(cfgList);
  const cfgWarning = modelIgnoresCfg(settings.modelId);

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

  // Hash any new ok rows in either lane so drift can be plotted.
  useEffect(() => {
    let cancelled = false;
    function hashLane(lane: SweepRow[], setter: React.Dispatch<React.SetStateAction<SweepRow[]>>) {
      void (async () => {
        for (let i = 0; i < lane.length; i++) {
          const row = lane[i];
          if (row.status === "ok" && row.imageDataUrl && !row.hash) {
            try {
              const h = await ahash(row.imageDataUrl);
              if (cancelled) return;
              setter((prev) => prev.map((r, j) => (j === i ? { ...r, hash: h } : r)));
            } catch {
              /* ignore */
            }
          }
        }
      })();
    }
    hashLane(rows, setRows);
    hashLane(rowsB, setRowsB);
    return () => {
      cancelled = true;
    };
  }, [rows, rowsB]);

  function laneDrift(lane: SweepRow[]): Array<number | null> | null {
    const ok = lane.filter((r) => r.status === "ok" && r.hash);
    if (ok.length < 2) return null;
    const baseline = ok.reduce((best, r) => (Math.abs(r.cfg - 7.5) < Math.abs(best.cfg - 7.5) ? r : best));
    return lane.map((r) => (r.status === "ok" && r.hash ? normalisedDrift(baseline.hash!, r.hash) : null));
  }

  const drift = useMemo(() => laneDrift(rows), [rows]);
  const driftB = useMemo(() => (compareEnabled ? laneDrift(rowsB) : null), [rowsB, compareEnabled]);

  async function callOnce(cfg: number, cfg_: ProviderConfig): Promise<{ ok: boolean; data?: DiffuseResponse; err?: GenError; status?: number }> {
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

  function makeSetRowAt(setter: React.Dispatch<React.SetStateAction<SweepRow[]>>) {
    return (idx: number, patch: Partial<SweepRow>) =>
      setter((prev) => prev.map((r, j) => (j === idx ? { ...r, ...patch } : r)));
  }

  async function runOne(
    cfg: number,
    idx: number,
    cfg_: ProviderConfig,
    setRowAt: (idx: number, patch: Partial<SweepRow>) => void,
    keyPrefix: string,
    imageKeyRef: { key?: string; meta?: DiffusionResultMeta },
  ): Promise<{ abortAll?: boolean }> {
    const MAX_ATTEMPTS = 4;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const r = await callOnce(cfg, cfg_);
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
    cfg_: ProviderConfig,
    keyPrefix: string,
    setter: React.Dispatch<React.SetStateAction<SweepRow[]>>,
  ): Promise<{ samples: RunSampleRef[]; providerIdSeen?: string; modelIdSeen?: string }> {
    const initial: SweepRow[] = cfgs.map((cfg) => ({ cfg, status: "pending" }));
    setter(initial);
    const setRowAt = makeSetRowAt(setter);
    const samples: RunSampleRef[] = [];
    let providerIdSeen: string | undefined;
    let modelIdSeen: string | undefined;
    let aborted = false;
    for (let i = 0; i < cfgs.length; i++) {
      if (aborted) {
        setter((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Skipped" } : r)));
        break;
      }
      setRowAt(i, { status: "running" });
      const ref: { key?: string; meta?: DiffusionResultMeta } = {};
      const { abortAll } = await runOne(cfgs[i], i, cfg_, setRowAt, keyPrefix, ref);
      if (ref.key && ref.meta) {
        samples.push({ imageKey: ref.key, variable: cfgs[i], responseTimeMs: ref.meta.responseTimeMs });
        providerIdSeen = ref.meta.providerId;
        modelIdSeen = ref.meta.modelId;
      }
      if (abortAll) aborted = true;
      if (i < cfgs.length - 1) await new Promise((res) => setTimeout(res, 1500));
    }
    return { samples, providerIdSeen, modelIdSeen };
  }

  async function persistRun(
    samples: RunSampleRef[],
    providerIdSeen: string | undefined,
    modelIdSeen: string | undefined,
    extra: Record<string, unknown>,
  ) {
    if (samples.length === 0 || !providerIdSeen || !modelIdSeen) return;
    const run: Run = {
      id: `sweep::${providerIdSeen}::${Date.now()}`,
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
      extra,
    };
    await saveRun(run);
  }

  async function runSweep() {
    if (cfgs.length === 0) {
      setTopError("Enter at least one CFG value.");
      return;
    }
    setRunning(true);
    setTopError(null);
    setErrorLink(null);
    if (!compareEnabled) setRowsB([]);

    if (compareEnabled) {
      // Run both lanes in parallel — independent providers, independent rate limits.
      const [a, b] = await Promise.all([
        runLane(primaryConfig, "sweep", setRows),
        runLane(compareConfig, "sweep-cmp", setRowsB),
      ]);
      await persistRun(a.samples, a.providerIdSeen, a.modelIdSeen, { cfgList: cfgs, lane: "primary" });
      await persistRun(b.samples, b.providerIdSeen, b.modelIdSeen, { cfgList: cfgs, lane: "compare" });
    } else {
      const a = await runLane(primaryConfig, "sweep", setRows);
      await persistRun(a.samples, a.providerIdSeen, a.modelIdSeen, { cfgList: cfgs });
    }

    setRunning(false);
  }

  return (
    <div className="card-editorial p-6 max-w-4xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">Guidance Sweep</h2>
      <p className="font-body text-body-sm text-foreground mb-1">
        Same prompt and seed across a range of CFG values. Reveals the controllability surface and where mode collapse begins.
      </p>
      <p className="font-sans text-caption italic text-muted-foreground mb-4">
        Read the drift curve as: low CFG → image drifts off prompt; high CFG → oversaturated mode collapse. The valley around CFG ≈ 7.5 is where guidance is most stable. With Compare on, two providers' valleys side by side show whether that "stable zone" is structural to the regime or specific to one model.
      </p>

      {cfgWarning && (
        <div className="border border-gold/40 bg-gold/10 text-foreground p-3 mb-4 font-sans text-caption rounded-sm">
          The current model (<strong>{settings.modelId}</strong>) does not use a guidance scale.
          A sweep will produce nearly identical images. For meaningful results switch to <code>black-forest-labs/flux-dev</code> or another CFG-bearing model in Settings.
        </div>
      )}

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
        <div className="grid grid-cols-3 gap-3">
          <label className="block" title={lookupTerm("Seed")}>
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Seed</span>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
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
          <label className="block" title={lookupTerm("CFG list")}>
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">CFG list</span>
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

      {/* Compare-with: cross-backend agreement */}
      <div className="border border-parchment rounded-sm p-3 mb-4 bg-cream/20">
        <label className="flex items-center gap-2 font-sans text-body-sm">
          <input
            type="checkbox"
            checked={compareEnabled}
            onChange={(e) => setCompareEnabled(e.target.checked)}
          />
          <span>Compare with a second provider (run two sweeps in parallel)</span>
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
          onClick={() => void runSweep()}
          disabled={running}
          className={running ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running
            ? "Running sweep…"
            : compareEnabled
              ? `Run sweep × 2 (${cfgs.length * 2} images)`
              : `Run sweep (${cfgs.length} ${cfgs.length === 1 ? "image" : "images"})`}
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

      {/* Lanes: primary, plus optional comparison */}
      {rows.length > 0 && (
        <SweepLane
          label={`${providerLabel(settings.providerId)} · ${settings.modelId}`}
          rows={rows}
          drift={drift}
        />
      )}
      {compareEnabled && rowsB.length > 0 && (
        <SweepLane
          label={`${providerLabel(compareProviderId)} · ${compareModelId}`}
          rows={rowsB}
          drift={driftB}
        />
      )}

      {(rows.length > 0 || rowsB.length > 0) && (
        <SweepDeepDive
          prompt={prompt}
          seed={seed}
          steps={steps}
          primaryLabel={`${providerLabel(settings.providerId)} · ${settings.modelId}`}
          rowsA={rows}
          driftA={drift}
          compareEnabled={compareEnabled}
          compareLabel={`${providerLabel(compareProviderId)} · ${compareModelId}`}
          rowsB={rowsB}
          driftB={driftB}
        />
      )}
    </div>
  );
}

interface SweepDeepDiveProps {
  prompt: string;
  seed: number;
  steps: number;
  primaryLabel: string;
  rowsA: SweepRow[];
  driftA: Array<number | null> | null;
  compareEnabled: boolean;
  compareLabel: string;
  rowsB: SweepRow[];
  driftB: Array<number | null> | null;
}

function SweepDeepDive({ prompt, seed, steps, primaryLabel, rowsA, driftA, compareEnabled, compareLabel, rowsB, driftB }: SweepDeepDiveProps) {
  function laneRows(rows: SweepRow[], drift: Array<number | null> | null) {
    return rows.map((r, i) => [
      r.cfg,
      r.status,
      r.meta?.providerId ?? "—",
      r.meta?.modelId ?? "—",
      r.meta?.seed ?? "—",
      r.meta ? (r.meta.responseTimeMs / 1000).toFixed(2) + "s" : "—",
      drift?.[i] != null ? (drift[i] as number).toFixed(3) : "—",
      r.errorMessage ?? "",
    ]);
  }

  const headers = ["CFG", "status", "provider", "model", "seed", "time", "drift", "error"];

  const csvRows: Array<Array<string | number>> = [];
  for (const row of laneRows(rowsA, driftA)) csvRows.push(["primary", ...row]);
  if (compareEnabled) for (const row of laneRows(rowsB, driftB)) csvRows.push(["compare", ...row]);

  function exportCsv() {
    downloadCsv(`guidance-sweep-${seed}-${Date.now()}.csv`, ["lane", ...headers], csvRows);
  }

  function exportJson() {
    downloadJson(`guidance-sweep-${seed}-${Date.now()}.json`, {
      operation: "guidance-sweep",
      prompt, seed, steps,
      primary: { label: primaryLabel, rows: rowsA, drift: driftA },
      compare: compareEnabled ? { label: compareLabel, rows: rowsB, drift: driftB } : null,
    });
  }

  function exportPdf() {
    const appendixRows: Array<Array<string | number>> = [];
    for (const row of laneRows(rowsA, driftA)) appendixRows.push(["primary", ...row]);
    if (compareEnabled) for (const row of laneRows(rowsB, driftB)) appendixRows.push(["compare", ...row]);
    const images = [
      ...rowsA.filter((r) => r.imageDataUrl).map((r) => ({
        dataUrl: r.imageDataUrl as string,
        caption: `primary · CFG ${r.cfg}`,
      })),
      ...(compareEnabled ? rowsB.filter((r) => r.imageDataUrl).map((r) => ({
        dataUrl: r.imageDataUrl as string,
        caption: `compare · CFG ${r.cfg}`,
      })) : []),
    ];
    downloadPdf(`guidance-sweep-${seed}-${Date.now()}.pdf`, {
      meta: {
        title: "Guidance Sweep",
        subtitle: compareEnabled ? `${primaryLabel}  ↔  ${compareLabel}` : primaryLabel,
        fields: [
          { label: "Prompt", value: prompt },
          { label: "Seed", value: seed },
          { label: "Steps", value: steps },
          { label: "Lanes", value: compareEnabled ? "2 (cross-backend)" : "1" },
        ],
      },
      images,
      appendix: [
        {
          title: "Per-cell results",
          caption: "Full sweep output: status, provider, model, response time, and perceptual-hash drift from the CFG ≈ 7.5 baseline. With Compare on, both lanes are interleaved.",
          table: { headers: ["lane", ...headers], rows: appendixRows },
        },
      ],
      glossary: termsFor(["Prompt", "Seed", "Steps", "CFG list", "lane", "CFG", "status", "provider", "model", "time", "drift", "error"]),
    });
  }

  function rowDetails(r: SweepRow, lane: "primary" | "compare", driftVal: number | null | undefined): Array<{ label: string; value: string | number }> {
    const out: Array<{ label: string; value: string | number }> = [
      { label: "Lane", value: lane },
      { label: "CFG", value: r.cfg },
      { label: "Prompt", value: prompt },
      { label: "Seed", value: seed },
      { label: "Steps", value: steps },
    ];
    if (r.meta) {
      out.push({ label: "Provider", value: r.meta.providerId });
      out.push({ label: "Model", value: r.meta.modelId });
      out.push({ label: "Time", value: `${(r.meta.responseTimeMs / 1000).toFixed(1)}s` });
    }
    if (driftVal != null) out.push({ label: "Drift", value: driftVal.toFixed(3) });
    return out;
  }

  const cameraEntries = [
    ...rowsA.map((r, i) => r.imageDataUrl ? {
      src: r.imageDataUrl as string,
      caption: `CFG ${r.cfg}`,
      subcaption: `primary · ${r.meta?.responseTimeMs != null ? `${(r.meta.responseTimeMs / 1000).toFixed(1)}s` : ""}`,
      details: rowDetails(r, "primary", driftA?.[i] ?? null),
    } : null).filter((e): e is NonNullable<typeof e> => e !== null),
    ...(compareEnabled ? rowsB.map((r, i) => r.imageDataUrl ? {
      src: r.imageDataUrl as string,
      caption: `CFG ${r.cfg}`,
      subcaption: `compare · ${r.meta?.responseTimeMs != null ? `${(r.meta.responseTimeMs / 1000).toFixed(1)}s` : ""}`,
      details: rowDetails(r, "compare", driftB?.[i] ?? null),
    } : null).filter((e): e is NonNullable<typeof e> => e !== null) : []),
  ];

  return (
    <DeepDive actions={<ExportButtons onCsv={exportCsv} onPdf={exportPdf} onJson={exportJson} />}>
      <div className="space-y-6">
        <CameraRoll entries={cameraEntries} />
        <Table
          headers={["lane", ...headers]}
          rows={[
            ...laneRows(rowsA, driftA).map((r) => ["primary", ...r]),
            ...(compareEnabled ? laneRows(rowsB, driftB).map((r) => ["compare", ...r]) : []),
          ]}
          numericColumns={[1, 6, 7]}
        />
      </div>
    </DeepDive>
  );
}

interface SweepLaneProps {
  label: string;
  rows: SweepRow[];
  drift: Array<number | null> | null;
}

function SweepLane({ label, rows, drift }: SweepLaneProps) {
  return (
    <div className="border-t border-parchment pt-4 mt-4">
      <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
        {label}
      </h3>
      {drift && drift.some((d) => d !== null) && (
        <div className="mb-3">
          <p className="font-sans text-caption text-muted-foreground mb-1">
            Drift from CFG ≈ 7.5 baseline (perceptual hash distance)
          </p>
          <DriftCurve cfgs={rows.map((r) => r.cfg)} drift={drift} />
        </div>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {rows.map((row, i) => (
          <div key={i} className="flex flex-col">
            <div className="aspect-square bg-cream/50 border border-parchment rounded-sm overflow-hidden flex items-center justify-center">
              {row.status === "ok" && row.imageDataUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={row.imageDataUrl} alt={`CFG ${row.cfg}`} className="w-full h-full object-cover" />
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
              CFG <span className="text-foreground font-medium">{row.cfg}</span>
              {row.meta && <> · {(row.meta.responseTimeMs / 1000).toFixed(1)}s</>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
