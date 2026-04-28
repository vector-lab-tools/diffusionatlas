"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSettings, effectiveSteps } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { DiffusionRequest, DiffusionResultMeta, ProviderId } from "@/lib/providers/types";
import {
  CATEGORIES,
  type BenchTask,
  type TaskCategoryId,
} from "@/lib/bench/tasks";
import { BENCH_PACKS, DEFAULT_PACK_ID, packById } from "@/lib/bench/packs";
import { Trash2, Plus, Square } from "lucide-react";
import { saveRun } from "@/lib/cache/runs";
import type { Run, RunSampleRef } from "@/types/run";
import { ALL_PROVIDERS, PROVIDER_DEFAULT_MODEL, providerLabel } from "@/lib/providers/defaults";
import { DeepDive } from "@/components/shared/DeepDive";
import { Table } from "@/components/shared/Table";
import { ExportButtons } from "@/components/shared/ExportButtons";
import { CameraRoll } from "@/components/shared/CameraRoll";
import { RandomSeedButton, nextSeed, type SeedMode } from "@/components/shared/RandomSeedButton";
import { WARMUP_LABEL, WARMUP_TOOLTIP, isWarmupMessage } from "@/lib/local/warmup";
import { useBackendHealth } from "@/context/BackendHealthContext";
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

type RowStatus = "pending" | "running" | "ok" | "error";
type Verdict = "pass" | "fail" | null;

interface BenchRow {
  task: BenchTask;
  status: RowStatus;
  imageDataUrl?: string;
  meta?: DiffusionResultMeta;
  errorMessage?: string;
  verdict: Verdict;
  clipScore?: number;
}

interface ProviderConfig {
  providerId: ProviderId;
  modelId: string;
  apiKey?: string;
  localBaseUrl?: string;
}

interface LaneScores {
  byCat: Record<TaskCategoryId, { pass: number; fail: number; pending: number }>;
  totalScored: number;
  totalPass: number;
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? "image/png";
  const bytes = atob(b64);
  const buf = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
  return new Blob([buf], { type: mime });
}

function freshRows(tasks: BenchTask[]): BenchRow[] {
  return tasks.map((task) => ({ task, status: "pending" as RowStatus, verdict: null }));
}

const PACK_STORAGE_KEY = "diffusion-atlas:bench-pack-overrides";
type PackOverrides = Record<string, BenchTask[]>;

function loadOverrides(): PackOverrides {
  try {
    const raw = localStorage.getItem(PACK_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as PackOverrides) : {};
  } catch {
    return {};
  }
}

function saveOverrides(overrides: PackOverrides) {
  try { localStorage.setItem(PACK_STORAGE_KEY, JSON.stringify(overrides)); } catch {}
}

/**
 * Plain-English read for a CLIP cosine score so the "0.262" number is
 * interpretable without the user looking up the literature. Bands are
 * approximate — CLIP cosine has no calibrated scale, but the ranges
 * here track community convention for ViT-B/32 on natural images.
 */
function scoreVerdictLabel(score: number): string {
  if (score < 0.18) return "weak match — image probably off-prompt";
  if (score < 0.22) return "marginal — borderline match";
  if (score < 0.27) return "plausible match";
  if (score < 0.32) return "strong match";
  return "very strong match";
}

function computeScores(rows: BenchRow[]): LaneScores {
  const byCat: Record<TaskCategoryId, { pass: number; fail: number; pending: number }> = {
    "single-object": { pass: 0, fail: 0, pending: 0 },
    "two-objects": { pass: 0, fail: 0, pending: 0 },
    counting: { pass: 0, fail: 0, pending: 0 },
    "colour-binding": { pass: 0, fail: 0, pending: 0 },
  };
  for (const row of rows) {
    const slot = byCat[row.task.category];
    if (row.verdict === "pass") slot.pass++;
    else if (row.verdict === "fail") slot.fail++;
    else slot.pending++;
  }
  const totalScored = rows.filter((r) => r.verdict !== null).length;
  const totalPass = rows.filter((r) => r.verdict === "pass").length;
  return { byCat, totalScored, totalPass };
}

export function CompositionalBench() {
  const { settings, setSettingsOpen } = useSettings();
  const { set: cacheImage } = useImageBlobCache();
  const { report: healthReport } = useBackendHealth();

  const [seed, setSeed] = useState(42);
  const [seedMode, setSeedMode] = useState<SeedMode>("off");
  const [seedSpinning, setSeedSpinning] = useState(false);
  const seedRef = useRef(seed);
  useEffect(() => { seedRef.current = seed; }, [seed]);
  // Bench needs higher fidelity than the global 4-step default (which
  // is tuned for FLUX schnell / fast smoke tests). 12 with DPM++ 2M
  // Karras gives recognisable images on SD 1.5/SDXL — necessary for
  // CLIP scoring to discriminate compositional success vs. mush.
  const [steps, setSteps] = useState(Math.max(12, settings.defaults.steps));
  // CFG held constant at 7.5 (the convention) so per-task CLIP scores
  // are comparable. Default may be 0 in settings (FLUX-schnell-tuned),
  // which produces noise on SD 1.5/SDXL — so we floor it here.
  const [cfg, setCfg] = useState(settings.defaults.cfg > 0 ? settings.defaults.cfg : 7.5);

  // Pack selection + editable overrides per pack.
  const [activePackId, setActivePackId] = useState<string>(DEFAULT_PACK_ID);
  const [packOverrides, setPackOverrides] = useState<PackOverrides>({});

  // Hydrate overrides once.
  useEffect(() => {
    setPackOverrides(loadOverrides());
  }, []);

  const activeTasks: BenchTask[] = useMemo(() => {
    if (packOverrides[activePackId]) return packOverrides[activePackId];
    return packById(activePackId)?.tasks ?? [];
  }, [activePackId, packOverrides]);

  const [rows, setRows] = useState<BenchRow[]>(() => freshRows(packById(DEFAULT_PACK_ID)?.tasks ?? []));
  const [rowsB, setRowsB] = useState<BenchRow[]>(() => freshRows(packById(DEFAULT_PACK_ID)?.tasks ?? []));

  // When the active pack (or its overrides) changes, reset both lanes to
  // pending rows for the new pack so the grid mirrors the editable list.
  useEffect(() => {
    setRows(freshRows(activeTasks));
    setRowsB(freshRows(activeTasks));
  }, [activeTasks]);
  const [running, setRunning] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [clipThreshold, setClipThreshold] = useState(0.25);
  const [topError, setTopError] = useState<string | null>(null);
  const [errorLink, setErrorLink] = useState<{ href: string; label: string } | null>(null);
  const [errorAction, setErrorAction] = useState<{ label: string; onClick: () => void } | null>(null);

  // Cross-backend comparison state
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [compareProviderId, setCompareProviderId] = useState<ProviderId>("local");
  const [compareModelId, setCompareModelId] = useState<string>(PROVIDER_DEFAULT_MODEL.local);

  // Bench abort: a single ref that the lane loop checks between tasks
  // and that the in-flight fetch's signal is bound to.  stopBench()
  // sets it and aborts; lanes break out of their per-task loop on the
  // next iteration.
  const benchAbortRef = useRef(false);
  const fetchAbortRef = useRef<AbortController | null>(null);

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

  async function callOnce(prompt: string, cfg_: ProviderConfig): Promise<{ ok: boolean; data?: DiffuseResponse; err?: GenError; status?: number }> {
    // Local lane: prefer the loaded model's native size from /health.
    // If health hasn't reported yet, fall back to a safe 512×512 (not
    // settings.defaults.width which may still be 1024 in localStorage)
    // — SD 1.5 / SDXL Sweep at 1024² fp32 OOMs on a 24 GB box, and
    // the backend's pre-flight check refuses with 422 anyway.
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
      // Clear the ref so subsequent calls don't carry a dead controller.
      if (fetchAbortRef.current === controller) fetchAbortRef.current = null;
    }
  }

  function makeSetRowAt(setter: React.Dispatch<React.SetStateAction<BenchRow[]>>) {
    return (idx: number, patch: Partial<BenchRow>) =>
      setter((prev) => prev.map((r, j) => (j === idx ? { ...r, ...patch } : r)));
  }

  async function runOne(
    idx: number,
    task: BenchTask,
    cfg_: ProviderConfig,
    setRowAt: (idx: number, patch: Partial<BenchRow>) => void,
    keyPrefix: string,
    imageKeyRef: { key?: string; meta?: DiffusionResultMeta },
  ): Promise<{ abortAll?: boolean }> {
    const MAX_ATTEMPTS = 4;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const r = await callOnce(task.prompt, cfg_);
      if (r.ok && r.data) {
        const dataUrl = r.data.images[0];
        const imageKey = `${keyPrefix}::${r.data.meta.providerId}::${r.data.meta.modelId}::${task.id}::${seedRef.current}::${Date.now()}`;
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
    setter: React.Dispatch<React.SetStateAction<BenchRow[]>>,
  ): Promise<{ samples: RunSampleRef[]; providerIdSeen?: string; modelIdSeen?: string }> {
    const tasks = activeTasks;
    setter(freshRows(tasks));
    const setRowAt = makeSetRowAt(setter);
    const samples: RunSampleRef[] = [];
    let providerIdSeen: string | undefined;
    let modelIdSeen: string | undefined;
    let aborted = false;
    for (let i = 0; i < tasks.length; i++) {
      // Honour user-initiated stop between tasks.
      if (benchAbortRef.current) {
        setter((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Stopped" } : r)));
        break;
      }
      if (aborted) {
        setter((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Skipped" } : r)));
        break;
      }
      // First local task on a cold backend warms the MPS pipeline
      // (~1-2 min); subsequent tasks reuse the warmed scheduler.
      const isLocalLane = cfg_.providerId === "local";
      setRowAt(i, {
        status: "running",
        errorMessage: isLocalLane && i === 0 ? WARMUP_LABEL : undefined,
      });
      const ref: { key?: string; meta?: DiffusionResultMeta } = {};
      const { abortAll } = await runOne(i, tasks[i], cfg_, setRowAt, keyPrefix, ref);
      if (ref.key && ref.meta) {
        samples.push({ imageKey: ref.key, variable: tasks[i].id, responseTimeMs: ref.meta.responseTimeMs });
        providerIdSeen = ref.meta.providerId;
        modelIdSeen = ref.meta.modelId;
      }
      if (abortAll) aborted = true;
      if (i < tasks.length - 1) await new Promise((res) => setTimeout(res, 1500));
    }
    return { samples, providerIdSeen, modelIdSeen };
  }

  async function persistRun(samples: RunSampleRef[], providerIdSeen?: string, modelIdSeen?: string, lane?: string) {
    if (samples.length === 0 || !providerIdSeen || !modelIdSeen) return;
    const run: Run = {
      id: `bench::${providerIdSeen}::${Date.now()}`,
      kind: "bench",
      createdAt: new Date().toISOString(),
      providerId: providerIdSeen,
      modelId: modelIdSeen,
      prompt: "(GenEval-lite pack)",
      seed: seedRef.current,
      steps,
      cfg: settings.defaults.cfg,
      width: settings.defaults.width,
      height: settings.defaults.height,
      samples,
      extra: { taskCount: activeTasks.length, categoryCount: CATEGORIES.length, packId: activePackId, ...(lane ? { lane } : {}) },
    };
    await saveRun(run);
  }

  function stopBench() {
    benchAbortRef.current = true;
    fetchAbortRef.current?.abort();
  }

  async function runBench() {
    benchAbortRef.current = false;
    // Apply shuffle / increment so a benchmark sweep across seeds is
    // one click away. Updates state + ref so closures and persisted
    // run records see the fresh value.
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

    if (compareEnabled) {
      const [a, b] = await Promise.all([
        runLane(primaryConfig, "bench", setRows),
        runLane(compareConfig, "bench-cmp", setRowsB),
      ]);
      await persistRun(a.samples, a.providerIdSeen, a.modelIdSeen, "primary");
      await persistRun(b.samples, b.providerIdSeen, b.modelIdSeen, "compare");
    } else {
      setRowsB(freshRows(activeTasks));
      const a = await runLane(primaryConfig, "bench", setRows);
      await persistRun(a.samples, a.providerIdSeen, a.modelIdSeen);
    }

    setRunning(false);
  }

  function setVerdict(setter: React.Dispatch<React.SetStateAction<BenchRow[]>>, idx: number, verdict: Verdict) {
    setter((prev) => prev.map((r, j) => (j === idx ? { ...r, verdict } : r)));
  }

  async function scoreLane(lane: BenchRow[], setter: React.Dispatch<React.SetStateAction<BenchRow[]>>): Promise<string | null> {
    const scorable = lane
      .map((r, i) => ({ r, i }))
      .filter(({ r }) => r.status === "ok" && r.imageDataUrl);
    if (scorable.length === 0) return null;

    try {
      const res = await fetch(`${settings.localBaseUrl}/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          images: scorable.map(({ r }) => r.imageDataUrl),
          prompts: scorable.map(({ r }) => r.task.prompt),
        }),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        if (res.status === 404) {
          return `Local backend at ${settings.localBaseUrl} returned 404 for /score. Your uvicorn process is likely running a pre-v0.2.3 build. Stop it and restart: cd backend && uvicorn main:app --reload --port 8000`;
        }
        return `Local backend ${res.status}: ${text.slice(0, 400)}`;
      }
      const data: { scores: number[]; modelId: string } = await res.json();
      setter((prev) =>
        prev.map((r, i) => {
          const sIdx = scorable.findIndex((s) => s.i === i);
          if (sIdx < 0) return r;
          const score = data.scores[sIdx];
          return { ...r, clipScore: score, verdict: score >= clipThreshold ? "pass" : "fail" };
        }),
      );
      return null;
    } catch (err) {
      return `Cannot reach local backend at ${settings.localBaseUrl}: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  async function autoScore() {
    const anyImages =
      rows.some((r) => r.status === "ok" && r.imageDataUrl) ||
      (compareEnabled && rowsB.some((r) => r.status === "ok" && r.imageDataUrl));
    if (!anyImages) {
      setTopError("Run the bench first; auto-score needs generated images.");
      return;
    }
    if (!settings.apiKeys.local && settings.backend !== "local") {
      // The /score endpoint is on the local backend regardless of the active hosted provider.
      // We don't strictly need backend === "local"; we just need the URL to be reachable.
    }

    setScoring(true);
    setTopError(null);
    setErrorAction(null);
    const results = await Promise.all([
      scoreLane(rows, setRows),
      compareEnabled ? scoreLane(rowsB, setRowsB) : Promise.resolve(null),
    ]);
    const errs = results.filter((e): e is string => !!e);
    if (errs.length > 0) setTopError(errs[0]);
    setScoring(false);
  }

  const scoresA = useMemo(() => computeScores(rows), [rows]);
  const scoresB = useMemo(() => computeScores(rowsB), [rowsB]);

  return (
    <div className="card-editorial p-6 max-w-5xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">Compositional Bench</h2>
      <p className="font-body text-body-sm text-foreground mb-1">
        GenEval-style scored tasks: single object, two objects, counting, colour binding. Generate the pack at a fixed seed; score with CLIP or mark pass/fail manually.
      </p>
      <p className="font-sans text-caption italic text-muted-foreground mb-4">
        Read the score panel as: where a category drops below the others, that's where the model fails compositionality (binding, counting, placement). With Compare on, two providers' panels side by side make "what does a smaller open model lose, by category?" empirically visible.
      </p>

      <div className="border border-cream/80 bg-cream/30 text-foreground p-3 mb-4 font-sans text-caption rounded-sm">
        Mark pass/fail manually, or run <strong>Auto-score (CLIP)</strong> against the local backend to set verdicts from CLIP image-text similarity. The threshold is the cosine cutoff (0.25 is reasonable for `clip-vit-base-patch32`); you can override any verdict afterwards.
      </div>

      {/* Pack picker */}
      <div className="mb-4">
        <div className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
          Task pack
        </div>
        <div className="flex flex-wrap gap-1.5 mb-2">
          {BENCH_PACKS.map((pack) => {
            const isActive = pack.id === activePackId;
            return (
              <button
                key={pack.id}
                onClick={() => setActivePackId(pack.id)}
                title={pack.description}
                className={`px-3 py-1 font-sans text-caption rounded-sm border transition-colors ${
                  isActive
                    ? "bg-burgundy text-primary-foreground border-burgundy"
                    : "bg-cream/40 text-foreground border-parchment-dark hover:bg-cream/70"
                }`}
              >
                {pack.label}
              </button>
            );
          })}
        </div>
        <p className="font-sans text-caption italic text-muted-foreground">
          {packById(activePackId)?.description}
        </p>
      </div>

      {/* Editable task list */}
      <BenchPackEditor
        tasks={activeTasks}
        packId={activePackId}
        onChange={(next) => {
          const overrides = { ...packOverrides, [activePackId]: next };
          setPackOverrides(overrides);
          saveOverrides(overrides);
        }}
        onReset={() => {
          const next = { ...packOverrides };
          delete next[activePackId];
          setPackOverrides(next);
          saveOverrides(next);
        }}
        isOverridden={!!packOverrides[activePackId]}
      />

      <div className="grid grid-cols-4 gap-3 mb-4">
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
        <label className="block" title={lookupTerm("CFG")}>
          <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">CFG</span>
          <input
            type="number"
            step="0.5"
            min={0}
            max={30}
            list="cfg-presets-bench"
            value={cfg}
            onChange={(e) => setCfg(parseFloat(e.target.value) || 0)}
            className="input-editorial mt-1"
          />
          <datalist id="cfg-presets-bench">
            <option value="0" label="prompt off" />
            <option value="1" label="no amplification" />
            <option value="4" label="soft" />
            <option value="7.5" label="balanced default" />
            <option value="12" label="aggressive" />
          </datalist>
        </label>
        <div className="block" title={lookupTerm("Pack size")}>
          <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Pack size</span>
          <div className="font-sans text-body-sm mt-2">{activeTasks.length} tasks · {CATEGORIES.length} categories</div>
        </div>
      </div>
      <p className="font-sans text-caption italic text-muted-foreground -mt-2 mb-4">
        <span className="not-italic font-medium">CFG (classifier-free guidance)</span> amplifies the prompt's pull on each step: <span className="font-mono not-italic">unconditional + CFG × (conditional − unconditional)</span>. Held at <span className="font-mono not-italic">{cfg}</span> across every task so per-task scoring is comparable. <span className="font-mono not-italic">0</span> = prompt off; <span className="font-mono not-italic">7.5</span> = balanced default; <span className="font-mono not-italic">12+</span> = oversaturated.
      </p>

      {/* Compare-with: cross-backend */}
      <div className="border border-parchment rounded-sm p-3 mb-4 bg-cream/20">
        <label className="flex items-center gap-2 font-sans text-body-sm">
          <input
            type="checkbox"
            checked={compareEnabled}
            onChange={(e) => setCompareEnabled(e.target.checked)}
          />
          <span>Compare with a second provider (run the same pack against both, score side by side)</span>
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
          onClick={() => void runBench()}
          disabled={running}
          className={running ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running
            ? "Running bench…"
            : compareEnabled
              ? `Run bench × 2 (${activeTasks.length * 2} images)`
              : `Run bench (${activeTasks.length} images)`}
        </button>
        {running && (
          <button
            onClick={stopBench}
            className="px-3 py-2 border border-burgundy bg-burgundy text-cream rounded-sm hover:bg-burgundy-900 flex items-center gap-1.5 font-sans text-body-sm"
            title="Abort the running bench. Tasks already completed are kept; remaining tasks are marked Stopped."
          >
            <Square size={12} fill="currentColor" /> Stop
          </button>
        )}
        <button
          onClick={() => void autoScore()}
          disabled={running || scoring}
          className={running || scoring ? "btn-editorial-secondary opacity-50" : "btn-editorial-secondary"}
          title="Score every generated image against its prompt with CLIP (local backend required)"
        >
          {scoring ? "Scoring…" : "Auto-score (CLIP)"}
        </button>
        <label className="flex items-center gap-2 font-sans text-caption text-muted-foreground" title={lookupTerm("Threshold")}>
          <span className="cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Threshold</span>
          <input
            type="number"
            step="0.01"
            min={0}
            max={1}
            value={clipThreshold}
            onChange={(e) => setClipThreshold(parseFloat(e.target.value) || 0)}
            className="input-editorial w-20 py-1"
          />
        </label>
        <span className="font-sans text-caption text-muted-foreground">
          {providerLabel(settings.providerId)} · {settings.modelId}
          {compareEnabled && <> ↔ {providerLabel(compareProviderId)} · {compareModelId}</>}
        </span>
      </div>

      {topError && (
        <div className="border border-burgundy/40 bg-burgundy/5 text-burgundy p-3 mb-4 font-sans text-body-sm rounded-sm">
          {topError}
          {errorAction && (
            <>
              {" "}
              <button
                onClick={errorAction.onClick}
                className="underline underline-offset-2 hover:text-burgundy-900 font-medium"
              >
                {errorAction.label} →
              </button>
            </>
          )}
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

      <BenchLane
        label={`${providerLabel(settings.providerId)} · ${settings.modelId}`}
        rows={rows}
        scores={scoresA}
        clipThreshold={clipThreshold}
        onMark={(idx, v) => setVerdict(setRows, idx, v)}
      />
      {compareEnabled && (
        <BenchLane
          label={`${providerLabel(compareProviderId)} · ${compareModelId}`}
          rows={rowsB}
          scores={scoresB}
          clipThreshold={clipThreshold}
          onMark={(idx, v) => setVerdict(setRowsB, idx, v)}
        />
      )}

      <BenchDeepDive
        seed={seed}
        steps={steps}
        clipThreshold={clipThreshold}
        primaryLabel={`${providerLabel(settings.providerId)} · ${settings.modelId}`}
        rowsA={rows}
        scoresA={scoresA}
        compareEnabled={compareEnabled}
        compareLabel={`${providerLabel(compareProviderId)} · ${compareModelId}`}
        rowsB={rowsB}
        scoresB={scoresB}
        activeTasks={activeTasks}
        packId={activePackId}
        packLabel={packById(activePackId)?.label ?? activePackId}
        packIsOverridden={!!packOverrides[activePackId]}
      />
    </div>
  );
}

interface BenchPackEditorProps {
  tasks: BenchTask[];
  packId: string;
  onChange: (next: BenchTask[]) => void;
  onReset: () => void;
  isOverridden: boolean;
}

function BenchPackEditor({ tasks, packId, onChange, onReset, isOverridden }: BenchPackEditorProps) {
  const [open, setOpen] = useState(false);

  function update(idx: number, patch: Partial<BenchTask>) {
    onChange(tasks.map((t, i) => (i === idx ? { ...t, ...patch } : t)));
  }
  function remove(idx: number) {
    onChange(tasks.filter((_, i) => i !== idx));
  }
  function add() {
    const newId = `${packId}-custom-${Date.now()}`;
    onChange([...tasks, { id: newId, category: "single-object", prompt: "a photograph of …", criterion: "Describe what should be visible." }]);
  }

  return (
    <div className="border border-parchment rounded-sm bg-cream/20 mb-4">
      <div className="flex items-center justify-between px-3 py-2">
        <button
          onClick={() => setOpen((v) => !v)}
          className="font-sans text-caption uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
        >
          {open ? "▼ Hide tasks" : `▶ Edit tasks (${tasks.length})`}
          {isOverridden && <span className="ml-2 text-burgundy">· edited</span>}
        </button>
        {isOverridden && (
          <button onClick={onReset} className="font-sans text-caption text-burgundy hover:text-burgundy-900 underline underline-offset-2">
            Reset to default
          </button>
        )}
      </div>
      {open && (
        <div className="px-3 pb-3 space-y-2">
          <div className="grid grid-cols-[110px_1fr_1fr_30px] gap-2 font-sans text-caption uppercase tracking-wider text-muted-foreground border-b border-parchment pb-1">
            <span>Category</span>
            <span>Prompt</span>
            <span>Criterion</span>
            <span></span>
          </div>
          {tasks.map((task, i) => (
            <div key={task.id} className="grid grid-cols-[110px_1fr_1fr_30px] gap-2 items-center">
              <select
                value={task.category}
                onChange={(e) => update(i, { category: e.target.value as TaskCategoryId })}
                className="input-editorial py-1 text-caption"
              >
                {CATEGORIES.map((c) => (
                  <option key={c.id} value={c.id}>{c.label}</option>
                ))}
              </select>
              <input
                type="text"
                value={task.prompt}
                onChange={(e) => update(i, { prompt: e.target.value })}
                className="input-editorial py-1 text-caption"
              />
              <input
                type="text"
                value={task.criterion}
                onChange={(e) => update(i, { criterion: e.target.value })}
                className="input-editorial py-1 text-caption"
              />
              <button
                onClick={() => remove(i)}
                className="btn-editorial-ghost p-1"
                title="Delete this task"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
          <button onClick={add} className="btn-editorial-secondary px-3 py-1 text-caption flex items-center gap-1 mt-2">
            <Plus size={12} /> Add task
          </button>
        </div>
      )}
    </div>
  );
}

interface BenchLaneProps {
  label: string;
  rows: BenchRow[];
  scores: LaneScores;
  clipThreshold: number;
  onMark: (idx: number, v: Verdict) => void;
}

function BenchLane({ label, rows, scores, clipThreshold, onMark }: BenchLaneProps) {
  const overallPct = scores.totalScored === 0 ? null : (scores.totalPass / scores.totalScored) * 100;
  return (
    <div className="border-t border-parchment pt-4 mt-4">
      <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">{label}</h3>

      {scores.totalScored > 0 && (
        <div className="mb-4">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {CATEGORIES.map((cat) => {
              const s = scores.byCat[cat.id];
              const scored = s.pass + s.fail;
              const pct = scored === 0 ? null : (s.pass / scored) * 100;
              return (
                <div key={cat.id} className="border border-parchment p-2 rounded-sm">
                  <div className="font-sans text-caption text-muted-foreground">{cat.label}</div>
                  <div className="font-display text-display-md font-bold text-burgundy">
                    {pct === null ? "—" : `${pct.toFixed(0)}%`}
                  </div>
                  <div className="font-sans text-caption text-muted-foreground">
                    {s.pass}/{scored} scored
                  </div>
                </div>
              );
            })}
            <div className="border border-burgundy bg-burgundy/5 p-2 rounded-sm">
              <div className="font-sans text-caption text-muted-foreground">Overall</div>
              <div className="font-display text-display-md font-bold text-burgundy">
                {overallPct === null ? "—" : `${overallPct.toFixed(0)}%`}
              </div>
              <div className="font-sans text-caption text-muted-foreground">
                {scores.totalPass}/{scores.totalScored} scored
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {rows.map((row, i) => (
          <div key={row.task.id} className="border border-parchment rounded-sm overflow-hidden flex flex-col">
            <div className="aspect-square bg-cream/50 flex items-center justify-center">
              {row.status === "ok" && row.imageDataUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={row.imageDataUrl} alt={row.task.prompt} className="w-full h-full object-cover" />
              )}
              {row.status === "running" && (
                <span
                  className={`font-sans text-caption text-muted-foreground text-center px-2 ${isWarmupMessage(row.errorMessage) ? "cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4" : ""}`}
                  title={isWarmupMessage(row.errorMessage) ? WARMUP_TOOLTIP : undefined}
                >
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
            <div className="p-2 flex-1 flex flex-col gap-2">
              <div className="font-sans text-caption uppercase tracking-wider text-muted-foreground">
                {CATEGORIES.find((c) => c.id === row.task.category)?.label}
              </div>
              <div className="font-body text-body-sm text-foreground line-clamp-2">{row.task.prompt}</div>
              <div className="font-sans text-caption text-muted-foreground italic line-clamp-2">{row.task.criterion}</div>
              {row.clipScore !== undefined && (
                <div
                  className="font-sans text-caption text-muted-foreground cursor-help"
                  title={
                    "CLIP cosine similarity between the generated image and the source prompt, " +
                    "scored by openai/clip-vit-base-patch32. The number is the cosine of the angle " +
                    "between the two CLIP embeddings (1.0 = identical direction, 0.0 = orthogonal). " +
                    "For matching pairs, scores typically land in [0.18, 0.35]. The threshold is a " +
                    "methodological choice, not a fact: 0.20 = permissive, 0.25 = standard, 0.30 = strict. " +
                    "It does not directly measure object count, attribute binding, or spatial relations — " +
                    "use it as a coarse 'plausible match' signal and override with manual pass/fail when " +
                    "compositional fidelity is actually what you're testing."
                  }
                >
                  CLIP score:{" "}
                  <span className={row.clipScore >= clipThreshold ? "text-burgundy font-medium" : "text-foreground"}>
                    {row.clipScore.toFixed(3)}
                  </span>
                  <span className="ml-1">
                    ({row.clipScore >= clipThreshold ? "passes" : "below"} threshold {clipThreshold.toFixed(2)})
                  </span>
                  <span className="ml-1 italic">
                    · {scoreVerdictLabel(row.clipScore)}
                  </span>
                </div>
              )}
              <div className="flex gap-2 mt-auto">
                <button
                  onClick={() => onMark(i, row.verdict === "pass" ? null : "pass")}
                  disabled={row.status !== "ok"}
                  className={
                    row.verdict === "pass"
                      ? "btn-editorial-primary px-3 py-1 text-caption"
                      : "btn-editorial-secondary px-3 py-1 text-caption"
                  }
                >
                  Pass
                </button>
                <button
                  onClick={() => onMark(i, row.verdict === "fail" ? null : "fail")}
                  disabled={row.status !== "ok"}
                  className={
                    row.verdict === "fail"
                      ? "btn-editorial-primary px-3 py-1 text-caption"
                      : "btn-editorial-secondary px-3 py-1 text-caption"
                  }
                >
                  Fail
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

interface BenchDeepDiveProps {
  seed: number;
  steps: number;
  clipThreshold: number;
  primaryLabel: string;
  rowsA: BenchRow[];
  scoresA: LaneScores;
  compareEnabled: boolean;
  compareLabel: string;
  rowsB: BenchRow[];
  scoresB: LaneScores;
  activeTasks: BenchTask[];
  packId: string;
  packLabel: string;
  packIsOverridden: boolean;
}

function BenchDeepDive({ seed, steps, clipThreshold, primaryLabel, rowsA, scoresA, compareEnabled, compareLabel, rowsB, scoresB, activeTasks, packId, packLabel, packIsOverridden }: BenchDeepDiveProps) {
  function laneTaskRows(rows: BenchRow[]) {
    return rows.map((r) => [
      r.task.id,
      r.task.category,
      r.task.prompt,
      r.status,
      r.verdict ?? "—",
      r.clipScore !== undefined ? r.clipScore.toFixed(3) : "—",
      r.meta?.providerId ?? "—",
      r.meta?.modelId ?? "—",
      r.meta ? (r.meta.responseTimeMs / 1000).toFixed(2) + "s" : "—",
    ]);
  }
  const taskHeaders = ["task", "category", "prompt", "status", "verdict", "CLIP", "provider", "model", "time"];

  const taskTableRows: Array<Array<string | number>> = [
    ...laneTaskRows(rowsA).map((r) => ["primary", ...r]),
    ...(compareEnabled ? laneTaskRows(rowsB).map((r) => ["compare", ...r]) : []),
  ];

  function categoryRows(scores: LaneScores) {
    return CATEGORIES.map((cat) => {
      const s = scores.byCat[cat.id];
      const scored = s.pass + s.fail;
      const pct = scored === 0 ? "—" : `${((s.pass / scored) * 100).toFixed(0)}%`;
      return [cat.label, s.pass, s.fail, s.pending, pct];
    });
  }
  const categoryHeaders = ["category", "pass", "fail", "pending", "accuracy"];

  const stamp = `bench-${seed}-${Date.now()}`;
  function exportCsv() {
    downloadCsv(`${stamp}.csv`, ["lane", ...taskHeaders], taskTableRows);
  }
  function exportJson() {
    downloadJson(`${stamp}.json`, {
      operation: "compositional-bench",
      seed, steps, clipThreshold,
      pack: { id: packId, label: packLabel, isOverridden: packIsOverridden, tasks: activeTasks },
      primary: { label: primaryLabel, rows: rowsA, scores: scoresA },
      compare: compareEnabled ? { label: compareLabel, rows: rowsB, scores: scoresB } : null,
    });
  }
  function exportPdf() {
    const images = [
      ...rowsA.filter((r) => r.imageDataUrl).map((r) => ({
        dataUrl: r.imageDataUrl as string,
        caption: `primary · ${r.task.id} · ${r.verdict ?? "—"}`,
      })),
      ...(compareEnabled ? rowsB.filter((r) => r.imageDataUrl).map((r) => ({
        dataUrl: r.imageDataUrl as string,
        caption: `compare · ${r.task.id} · ${r.verdict ?? "—"}`,
      })) : []),
    ];
    const summaryHeaders = ["lane", ...categoryHeaders];
    const summaryRows: Array<Array<string | number>> = [
      ...categoryRows(scoresA).map((r) => ["primary", ...r]),
      ...(compareEnabled ? categoryRows(scoresB).map((r) => ["compare", ...r]) : []),
    ];
    downloadPdf(`${stamp}.pdf`, {
      meta: {
        title: "Compositional Bench",
        subtitle: compareEnabled ? `${primaryLabel}  ↔  ${compareLabel}` : primaryLabel,
        fields: [
          { label: "Seed", value: seed },
          { label: "Steps", value: steps },
          { label: "CLIP threshold", value: clipThreshold.toFixed(2) },
          { label: "Tasks", value: activeTasks.length },
          { label: "Pack", value: packId },
        ],
      },
      table: { headers: summaryHeaders, rows: summaryRows },
      images,
      appendix: [
        {
          title: `Task pack · ${packLabel}${packIsOverridden ? " (edited)" : ""}`,
          caption: `Definition of every task in the pack at run time: ${activeTasks.length} entries.`,
          table: { headers: packHeaders, rows: packTableRows },
        },
        {
          title: "Per-task results",
          caption: "Every task with its category, prompt, status, verdict, CLIP score (where available), provider, model, and response time. With Compare on, both lanes are interleaved.",
          table: { headers: ["lane", ...taskHeaders], rows: taskTableRows },
        },
      ],
      glossary: termsFor(["Seed", "Steps", "Threshold", "lane", "task", "category", "verdict", "CLIP", "status", "provider", "model", "time", "pass", "fail", "pending", "accuracy"]),
    });
  }

  // Pack definition table — shown even before running, so the user can see
  // exactly what's in the pack they're about to run.
  const packTableRows: Array<Array<string | number>> = activeTasks.map((t) => [t.id, t.category, t.prompt, t.criterion]);
  const packHeaders = ["task", "category", "prompt", "criterion"];

  function benchDetails(r: BenchRow, lane: "primary" | "compare"): Array<{ label: string; value: string | number }> {
    const out: Array<{ label: string; value: string | number }> = [
      { label: "Lane", value: lane },
      { label: "Task", value: r.task.id },
      { label: "Category", value: r.task.category },
      { label: "Prompt", value: r.task.prompt },
      { label: "Criterion", value: r.task.criterion },
      { label: "Seed", value: seed },
      { label: "Steps", value: steps },
      { label: "Verdict", value: r.verdict ?? "—" },
    ];
    if (r.clipScore !== undefined) out.push({ label: "CLIP", value: r.clipScore.toFixed(3) });
    if (r.meta) {
      out.push({ label: "Provider", value: r.meta.providerId });
      out.push({ label: "Model", value: r.meta.modelId });
      out.push({ label: "Time", value: `${(r.meta.responseTimeMs / 1000).toFixed(1)}s` });
    }
    return out;
  }

  const cameraEntries = [
    ...rowsA.filter((r) => r.imageDataUrl).map((r) => ({
      src: r.imageDataUrl as string,
      caption: r.task.id,
      subcaption: `primary · ${r.verdict ?? "—"}${r.clipScore !== undefined ? ` · ${r.clipScore.toFixed(2)}` : ""}`,
      details: benchDetails(r, "primary"),
    })),
    ...(compareEnabled ? rowsB.filter((r) => r.imageDataUrl).map((r) => ({
      src: r.imageDataUrl as string,
      caption: r.task.id,
      subcaption: `compare · ${r.verdict ?? "—"}${r.clipScore !== undefined ? ` · ${r.clipScore.toFixed(2)}` : ""}`,
      details: benchDetails(r, "compare"),
    })) : []),
  ];

  return (
    <DeepDive actions={<ExportButtons onCsv={exportCsv} onPdf={exportPdf} onJson={exportJson} />}>
      <div className="space-y-6">
        {cameraEntries.length > 0 && <CameraRoll entries={cameraEntries} />}
        <div>
          <h4 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
            Task pack · {packLabel}
            {packIsOverridden && <span className="text-burgundy ml-2">(edited)</span>}
            <span className="text-foreground ml-2">— {activeTasks.length} tasks</span>
          </h4>
          <Table headers={packHeaders} rows={packTableRows} />
        </div>
        {!rowsA.every((r) => r.status === "pending") && (
          <>
            <div>
              <h4 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
                Per-category accuracy
              </h4>
              <div className={compareEnabled ? "grid grid-cols-1 lg:grid-cols-2 gap-4" : ""}>
                <div>
                  <p className="font-sans text-caption text-foreground mb-1">{primaryLabel}</p>
                  <Table headers={categoryHeaders} rows={categoryRows(scoresA)} numericColumns={[1, 2, 3, 4]} />
                </div>
                {compareEnabled && (
                  <div>
                    <p className="font-sans text-caption text-foreground mb-1">{compareLabel}</p>
                    <Table headers={categoryHeaders} rows={categoryRows(scoresB)} numericColumns={[1, 2, 3, 4]} />
                  </div>
                )}
              </div>
            </div>
            <div>
              <h4 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
                Per-task results
              </h4>
              <Table headers={["lane", ...taskHeaders]} rows={taskTableRows} numericColumns={[6, 9]} />
            </div>
          </>
        )}
      </div>
    </DeepDive>
  );
}
