"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Square } from "lucide-react";
import { useSettings, effectiveSteps } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { DiffusionRequest, DiffusionResultMeta, ProviderId } from "@/lib/providers/types";
import { saveRun } from "@/lib/cache/runs";
import type { Run, RunSampleRef } from "@/types/run";
import { ahash, normalisedDrift, type Hash } from "@/lib/geometry/perceptual_hash";
import { ALL_PROVIDERS, PROVIDER_DEFAULT_MODEL, providerLabel } from "@/lib/providers/defaults";
import { DeepDive } from "@/components/shared/DeepDive";
import { Table } from "@/components/shared/Table";
import { ExportButtons } from "@/components/shared/ExportButtons";
import { CameraRoll, FrameModal, type CameraRollEntry } from "@/components/shared/CameraRoll";
import { ContactSheetFrame, SprocketRow } from "@/components/shared/ContactSheetFrame";
import { cfgCaption } from "@/components/shared/CfgSelect";
import { downloadCsv } from "@/lib/export/csv";
import { downloadPdf } from "@/lib/export/pdf";
import { downloadJson } from "@/lib/export/json";
import { lookup as lookupTerm, termsFor } from "@/lib/docs/glossary";
import { PromptChips, STARTER_PRESETS } from "@/components/shared/PromptChips";
import { RandomSeedButton, nextSeed, type SeedMode } from "@/components/shared/RandomSeedButton";
import { WARMUP_LABEL, WARMUP_TOOLTIP, isWarmupMessage, shortenBackendError } from "@/lib/local/warmup";
import { useBackendHealth } from "@/context/BackendHealthContext";

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

/**
 * One complete sweep run, persisted as a "layer". The temp/locked
 * model from DenoiseTrajectory: every completed sweep auto-saves as
 * unlocked; running again drops existing unlocked layer(s) and pushes
 * a new one. Click the padlock to keep a layer across future runs.
 */
interface SweepLayer {
  id: string;
  label: string;
  colour: string;
  locked: boolean;
  visible: boolean;
  prompt: string;
  seed: number;
  steps: number;
  cfgList: number[];
  rows: SweepRow[];
  providerId: ProviderId;
  modelId: string;
  /** Distinguishes the two sub-rows when compare-with is on. */
  lane: "primary" | "compare";
  createdAt: number;
}

// Drops the previous 2.5 slot — that value sits in the empirically-
// fragile low-CFG range where SD 1.5 fp32 on MPS hits NaN-in-VAE on
// many seed/prompt combinations. 4 still anchors the "soft" end of
// the surface, 18 shows where amplification crosses into oversaturation.
const DEFAULT_CFG_SET = "1, 4, 7.5, 12, 18";

const LAYER_COLOURS = ["#7c2d36", "#c9a227", "#2e5d8a", "#3b7d4f", "#8a3b6e", "#5e5e5e"];
const SWEEP_PERSIST_KEY = "sweep.layers.v1";

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

  // Y-axis gridline values (drift is normalised to [0, 1]).
  const yGrid = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-2xl">
      {/* Horizontal gridlines + Y-axis tick labels */}
      {yGrid.map((g) => (
        <g key={`yg-${g}`}>
          <line
            x1={P}
            y1={yScale(g)}
            x2={W - P}
            y2={yScale(g)}
            stroke="hsl(var(--parchment))"
            strokeWidth={0.5}
            strokeDasharray={g === 0 || g === 1 ? "none" : "2,2"}
          />
          <text
            x={P - 4}
            y={yScale(g) + 3}
            fontSize={8}
            fill="hsl(var(--muted-foreground))"
            fontFamily="sans-serif"
            textAnchor="end"
          >
            {g.toFixed(g === 0 || g === 1 ? 0 : 2)}
          </text>
        </g>
      ))}
      {/* Vertical gridlines at each CFG sample */}
      {cfgs.map((cfg, i) => (
        <g key={`xg-${i}`}>
          <line
            x1={xScale(cfg)}
            y1={P}
            x2={xScale(cfg)}
            y2={H - P}
            stroke="hsl(var(--parchment))"
            strokeWidth={0.5}
            strokeDasharray="2,2"
          />
          <text
            x={xScale(cfg)}
            y={H - P + 10}
            fontSize={8}
            fill="hsl(var(--muted-foreground))"
            fontFamily="sans-serif"
            textAnchor="middle"
          >
            {cfg}
          </text>
        </g>
      ))}
      {/* Solid axes on top of the grid */}
      <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="hsl(var(--parchment-dark))" strokeWidth={1} />
      <line x1={P} y1={P} x2={P} y2={H - P} stroke="hsl(var(--parchment-dark))" strokeWidth={1} />
      {/* Axis legends */}
      <text x={W / 2} y={H - 2} fontSize={8} fill="hsl(var(--muted-foreground))" fontFamily="sans-serif" textAnchor="middle">
        CFG
      </text>
      <text
        x={4}
        y={H / 2}
        fontSize={8}
        fill="hsl(var(--muted-foreground))"
        fontFamily="sans-serif"
        transform={`rotate(-90 4 ${H / 2})`}
        textAnchor="middle"
      >
        drift
      </text>
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
  const { report: healthReport } = useBackendHealth();

  const [prompt, setPrompt] = useState("a red cube on a blue cube, photorealistic");
  const [seed, setSeed] = useState(42);
  const [seedMode, setSeedMode] = useState<SeedMode>("off");
  const [seedSpinning, setSeedSpinning] = useState(false);

  // Sweep abort: ref the per-CFG loop checks between cells, plus a
  // single AbortController bound to the current /api/diffuse fetch so
  // Stop terminates the in-flight call as well.
  const sweepAbortRef = useRef(false);
  const fetchAbortRef = useRef<AbortController | null>(null);
  // Ref so callOnce / runOne can read the freshly-rolled seed without
  // waiting for React to re-render their closures.
  const seedRef = useRef(seed);
  useEffect(() => { seedRef.current = seed; }, [seed]);
  // 20 default with EulerDiscreteScheduler (the backend pins it because
  // DPM++ produces NaN on MPS at certain CFGs). 20 Euler ≈ 12 DPM++ in
  // visual fidelity — necessary for a CFG sweep where you want crisp
  // images at each value to read drift cleanly.
  const [steps, setSteps] = useState(Math.max(20, settings.defaults.steps));
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
    // For the local lane, override the global settings.defaults
    // resolution with the loaded model's native size (from /health).
    // settings.defaults is preserved for hosted lanes since flux /
    // SDXL on hosted providers often expect 1024 anyway.
    const isLocal = cfg_.providerId === "local";
    const w = isLocal
      ? (healthReport?.nativeWidth ?? 512)
      : settings.defaults.width;
    const h = isLocal
      ? (healthReport?.nativeHeight ?? 512)
      : settings.defaults.height;
    const request: DiffusionRequest = {
      modelId: cfg_.modelId,
      prompt,
      seed: seedRef.current,
      steps: effectiveSteps(steps, settings),
      cfg,
      width: w,
      height: h,
      scheduler: settings.defaults.scheduler,
    };
    const controller = new AbortController();
    fetchAbortRef.current = controller;
    try {
      const res = await fetch("/api/diffuse", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(cfg_.apiKey ? { "X-Diffusion-API-Key": cfg_.apiKey } : {}),
        },
        body: JSON.stringify({ providerId: cfg_.providerId, request, localBaseUrl: cfg_.localBaseUrl }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const err: GenError = await res.json().catch(() => ({ error: "unknown" }));
        return { ok: false, err, status: res.status };
      }
      const data: DiffuseResponse = await res.json();
      return { ok: true, data };
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return { ok: false, err: { error: "aborted", message: "Stopped by user" } };
      }
      return { ok: false, err: { error: "network", message: err instanceof Error ? err.message : String(err) } };
    } finally {
      if (fetchAbortRef.current === controller) fetchAbortRef.current = null;
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
        const imageKey = `${keyPrefix}::${r.data.meta.providerId}::${r.data.meta.modelId}::${seedRef.current}::${steps}::${cfg}::${Date.now()}`;
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
      // User-initiated stop terminates the per-CFG loop on the next
      // iteration; the in-flight fetch is already aborted in stopSweep.
      if (sweepAbortRef.current) {
        setter((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Stopped" } : r)));
        break;
      }
      if (aborted) {
        setter((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Skipped" } : r)));
        break;
      }
      // First local cell on a cold backend can take 1–2 min for MPS
      // warmup (no streaming for /generate) — surface that so the user
      // doesn't think it's stuck. Subsequent cells in the same lane are
      // typically 1–3s on the warmed pipeline.
      const isLocalLane = cfg_.providerId === "local";
      setRowAt(i, {
        status: "running",
        errorMessage: isLocalLane && i === 0 ? WARMUP_LABEL : undefined,
      });
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
      seed: seedRef.current,
      steps,
      cfg: cfgs[0],
      width: settings.defaults.width,
      height: settings.defaults.height,
      samples,
      extra,
    };
    await saveRun(run);
  }

  function stopSweep() {
    sweepAbortRef.current = true;
    fetchAbortRef.current?.abort();
  }

  async function runSweep() {
    if (cfgs.length === 0) {
      setTopError("Enter at least one CFG value.");
      return;
    }
    sweepAbortRef.current = false;
    // Apply shuffle / increment before the sweep starts, so a held seed
    // walks predictably across runs. Update both state (for the input)
    // and ref (so the in-flight closures read the fresh value).
    if (seedMode !== "off") {
      const fresh = nextSeed(seedMode, seedRef.current);
      seedRef.current = fresh;
      setSeed(fresh);
      setSeedSpinning(true);
      await new Promise((r) => setTimeout(r, 450));
      setSeedSpinning(false);
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
        <PromptChips active={prompt} presets={STARTER_PRESETS} onPick={setPrompt} />
        <div className="grid grid-cols-3 gap-3">
          <label className="block" title={lookupTerm("Seed")}>
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Seed</span>
            <div className="flex items-stretch gap-1 mt-1">
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
                className="input-editorial flex-1 min-w-0"
              />
              <RandomSeedButton
                onPick={setSeed}
                mode={seedMode}
                onModeChange={setSeedMode}
                spinning={seedSpinning}
              />
            </div>
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
        <p className="font-sans text-caption italic text-muted-foreground mt-2">
          <span className="not-italic font-medium">CFG (classifier-free guidance)</span> amplifies the prompt's pull on each denoising step: output = <span className="font-mono not-italic">unconditional + CFG × (conditional − unconditional)</span>. <span className="font-mono not-italic">0</span> = prompt off; <span className="font-mono not-italic">1</span> = prompt on, no extra amplification; <span className="font-mono not-italic">7.5</span> = balanced default; <span className="font-mono not-italic">12+</span> = oversaturated / mode collapse. The sweep generates the same prompt + seed at each value so you can see where the controllability surface bends. The drift curve measures perceptual distance from the CFG-7.5 baseline.
        </p>
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
            <label className="block" title={lookupTerm("Compare provider")}>
              <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Compare provider</span>
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
            <label className="block" title={lookupTerm("Compare model")}>
              <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Compare model</span>
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
        {running && (
          <button
            onClick={stopSweep}
            className="px-3 py-2 border border-burgundy bg-burgundy text-cream rounded-sm hover:bg-burgundy-900 flex items-center gap-1.5 font-sans text-body-sm"
            title="Abort the running sweep. Cells already completed are kept; remaining cells are marked Stopped."
          >
            <Square size={12} fill="currentColor" /> Stop
          </button>
        )}
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
          prompt={prompt}
          seed={seed}
          modelId={settings.modelId}
          providerId={settings.providerId}
        />
      )}
      {compareEnabled && rowsB.length > 0 && (
        <SweepLane
          label={`${providerLabel(compareProviderId)} · ${compareModelId}`}
          rows={rowsB}
          drift={driftB}
          prompt={prompt}
          seed={seed}
          modelId={compareModelId}
          providerId={compareProviderId}
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
      driftCurves: [
        ...(driftA && driftA.some((d) => d !== null)
          ? [{ label: primaryLabel, domain: rowsA.map((r) => r.cfg), values: driftA }]
          : []),
        ...(compareEnabled && driftB && driftB.some((d) => d !== null)
          ? [{ label: compareLabel, domain: rowsB.map((r) => r.cfg), values: driftB }]
          : []),
      ],
      glossary: termsFor(["Prompt", "Seed", "Steps", "CFG list", "lane", "CFG", "status", "provider", "model", "time", "drift", "error"]),
      notes: [
        {
          title: "Reading the CFG axis (and what 'error' cells mean)",
          body:
            "CFG sweeps on SD 1.5 fp32 on Apple Silicon (MPS) sometimes produce all-black cells at specific CFG values — these are NaN failures in the U-Net's attention path, not bugs in the prompt or seed. The pattern reflects how classifier-free guidance arithmetic interacts with fp32 numerical stability:\n\n" +
            "CFG 1: doesn't actually do CFG arithmetic — the pipeline takes a single prompt-conditioned forward pass with no amplification. No (conditional − unconditional) subtraction, nothing to over-amplify. The safe path.\n\n" +
            "CFG 4 and above: amplification is large enough that intermediate activations cluster well away from fp32's underflow regions. The numbers stay in well-conditioned parts of the network and the trajectory completes cleanly.\n\n" +
            "CFG 7.5: same safe regime as 4, with a larger amplitude. This is the conventional default for a reason — it's both visually balanced and numerically robust.\n\n" +
            "CFG 12 and above: largest amplification, most stable in this sense. Risk shifts from underflow to oversaturation / mode collapse, which is a perceptual problem, not a numerical one.\n\n" +
            "CFG 2.5 (and the 2-3 range generally): small but non-zero amplification of (conditional − unconditional). Apple's MPS fp32 implementation has subtly different reduction order from CUDA's fp32, particularly inside attention's softmax denominator. That tiny numerical difference compounds across denoising steps; at some intermediate step a single value occasionally overflows. Once one position is NaN, attention spreads it across the whole tensor on the next layer, the VAE decodes NaN to all-zero pixels, and the cell renders black.\n\n" +
            "Workarounds: try a different seed (the failure is path-dependent — the specific noise + CFG combination that overflows is rare); try CFG 2 or 3 instead of 2.5; switch SD 1.5 to bfloat16 (same exponent range as fp32, half the memory, no overflow). The Diffusion Atlas backend has an opt-in MIXED_PRECISION_VAE flag that uses a different numerical path and often clears these cases.",
        },
      ],
    });
  }

  function rowDetails(r: SweepRow, lane: "primary" | "compare", driftVal: number | null | undefined): Array<{ label: string; value: string | number }> {
    const out: Array<{ label: string; value: string | number }> = [
      { label: "Lane", value: lane },
      { label: "CFG", value: `${r.cfg} (${cfgCaption(r.cfg)})` },
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
        {/* Drift curve(s) — moved here from the contact sheet because
            it's analytical detail, not visual gestalt. The contact
            sheet stays a pure visual surface; this is where you read
            the numbers. */}
        {((driftA && driftA.some((d) => d !== null)) ||
          (compareEnabled && driftB && driftB.some((d) => d !== null))) && (
          <div>
            <h4 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
              Drift curve
            </h4>
            <p className="font-sans text-caption text-muted-foreground italic mb-2">
              Perceptual hash distance from the cell nearest CFG ≈ 7.5 (the
              conventional baseline). 0 = visually identical to the baseline,
              1 = maximally different. The curve traces the controllability
              surface — where it bends sharply, the model's behaviour at that
              CFG is qualitatively different from the default.
            </p>
            {driftA && driftA.some((d) => d !== null) && (
              <div className="mb-3">
                <p className="font-sans text-caption text-muted-foreground mb-1">
                  {primaryLabel}
                </p>
                <DriftCurve cfgs={rowsA.map((r) => r.cfg)} drift={driftA} />
              </div>
            )}
            {compareEnabled && driftB && driftB.some((d) => d !== null) && (
              <div>
                <p className="font-sans text-caption text-muted-foreground mb-1">
                  {compareLabel}
                </p>
                <DriftCurve cfgs={rowsB.map((r) => r.cfg)} drift={driftB} />
              </div>
            )}
          </div>
        )}
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
  prompt: string;
  seed: number;
  modelId: string;
  providerId: ProviderId;
}

function SweepLane({ label, rows, drift, prompt, seed, modelId, providerId }: SweepLaneProps) {
  // Click-to-inspect: opening a frame surfaces the same RGB-overlay
  // histogram + image-stats panel used in DenoiseTrajectory's camera
  // roll. Keeps the modal logic inside the lane so each lane has its
  // own open state and prev/next walks within that lane.
  const okRows = useMemo(() => rows.filter((r) => r.imageDataUrl), [rows]);
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  const openEntry = openIdx != null ? entryFor(okRows[openIdx]) : null;

  function entryFor(row: SweepRow): CameraRollEntry {
    const details: Array<{ label: string; value: string | number }> = [
      { label: "Prompt", value: prompt },
      { label: "CFG", value: `${row.cfg} (${cfgCaption(row.cfg)})` },
      { label: "Seed", value: seed },
      { label: "Provider", value: providerId },
      { label: "Model", value: modelId },
    ];
    if (row.meta?.responseTimeMs != null) {
      details.push({ label: "Time", value: `${(row.meta.responseTimeMs / 1000).toFixed(1)}s` });
    }
    return {
      src: row.imageDataUrl as string,
      caption: `CFG ${row.cfg}`,
      subcaption: `${providerLabel(providerId)} · seed ${seed}`,
      details,
    };
  }

  // Contact-sheet styling: dark sprocketed strip with frames in a row,
  // white edge-print metadata band below with CFG / drift / time per
  // frame and a tiny clickable RGB histogram. Same metaphor as
  // DenoiseTrajectory's FilmStrip, just keyed on CFG instead of step.
  const FRAME_PX = 150;
  const COL_GAP_PX = 12;
  const stripMinWidth = rows.length * FRAME_PX + Math.max(0, rows.length - 1) * COL_GAP_PX;

  return (
    <div className="rounded-sm border border-parchment overflow-hidden mt-4">
      <div className="bg-card px-3 py-1.5 flex items-center justify-between border-b border-parchment">
        <span
          className="font-mono text-[9px] uppercase tracking-[0.18em]"
          style={{ color: "#a36b3a" }}
        >
          DIFFUSION ATLAS · CFG SWEEP
        </span>
        <span className="font-sans text-caption text-muted-foreground">{label}</span>
      </div>

      <div className="overflow-x-auto">
        <div style={{ minWidth: `${stripMinWidth + 24}px` }}>
          <div className="bg-[#111] py-2">
            <SprocketRow frameCount={rows.length} />
            <div className="px-3 py-2">
              <div className="flex items-start" style={{ gap: `${COL_GAP_PX}px` }}>
                {rows.map((row, i) => (
                  <div
                    key={i}
                    className="flex-shrink-0 bg-black border-2 border-[#0a0a0a] rounded-sm overflow-hidden"
                    style={{ width: `${FRAME_PX}px`, height: `${FRAME_PX}px` }}
                  >
                    {row.status === "ok" && row.imageDataUrl && (
                      <button
                        type="button"
                        onClick={() => {
                          const j = okRows.findIndex((r) => r === row);
                          if (j >= 0) setOpenIdx(j);
                        }}
                        className="w-full h-full block hover:opacity-90 transition-opacity"
                        title="Click for full image + RGB histogram + stats"
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={row.imageDataUrl} alt={`CFG ${row.cfg}`} className="w-full h-full object-cover" />
                      </button>
                    )}
                    {row.status === "running" && (
                      <div className="w-full h-full flex items-center justify-center px-2">
                        <span
                          className={`text-[10px] text-[#999] font-mono uppercase tracking-wider text-center leading-tight ${isWarmupMessage(row.errorMessage) ? "cursor-help underline decoration-dotted decoration-[#666]/60 underline-offset-2" : ""}`}
                          title={isWarmupMessage(row.errorMessage) ? WARMUP_TOOLTIP : undefined}
                        >
                          {row.errorMessage ?? "generating…"}
                        </span>
                      </div>
                    )}
                    {row.status === "pending" && (
                      <div className="w-full h-full flex items-center justify-center text-[10px] text-[#666] font-mono uppercase tracking-wider">
                        queued
                      </div>
                    )}
                    {row.status === "error" && (() => {
                      const { short, full } = shortenBackendError(row.errorMessage);
                      return (
                        <div className="w-full h-full flex flex-col items-center justify-center px-2 gap-1" title={full}>
                          <span className="text-[10px] text-burgundy font-mono uppercase tracking-wider text-center leading-tight">
                            {short}
                          </span>
                          <span className="text-[8px] text-[#888] font-sans italic text-center">
                            hover for details
                          </span>
                        </div>
                      );
                    })()}
                  </div>
                ))}
              </div>
            </div>
            <SprocketRow frameCount={rows.length} />
          </div>

          <div className="bg-card px-3 py-2 border-t border-parchment">
            <div className="flex items-start" style={{ gap: `${COL_GAP_PX}px` }}>
              {rows.map((row, i) => {
                const driftValue = drift?.[i];
                const scalars: Array<{ key: string; value: string }> = [];
                if (driftValue != null) scalars.push({ key: "drift", value: driftValue.toFixed(2) });
                if (row.meta?.responseTimeMs != null) scalars.push({ key: "time", value: `${(row.meta.responseTimeMs / 1000).toFixed(1)}s` });
                scalars.push({ key: "seed", value: String(seed) });
                return (
                  <ContactSheetFrame
                    key={i}
                    width={FRAME_PX}
                    primary={`CFG ${row.cfg}`}
                    secondary={row.status === "ok" ? "ok" : row.status}
                    caption={cfgCaption(row.cfg)}
                    scalars={scalars}
                    preview={row.imageDataUrl ?? null}
                    onOpen={() => {
                      const j = okRows.findIndex((r) => r === row);
                      if (j >= 0) setOpenIdx(j);
                    }}
                  />
                );
              })}
            </div>
          </div>
        </div>
      </div>
      {openIdx != null && openEntry && (
        <FrameModal
          entry={openEntry}
          index={openIdx}
          total={okRows.length}
          onClose={() => setOpenIdx(null)}
          onPrev={openIdx > 0 ? () => setOpenIdx(openIdx - 1) : undefined}
          onNext={openIdx < okRows.length - 1 ? () => setOpenIdx(openIdx + 1) : undefined}
        />
      )}
    </div>
  );
}
