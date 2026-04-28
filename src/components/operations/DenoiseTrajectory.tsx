"use client";

import { useEffect, useRef, useState } from "react";
import { UMAP } from "umap-js";
import { X, Plus, Lock, Unlock, Square, Search, Play } from "lucide-react";
import { useSettings, effectiveSteps } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import { useBackendHealth } from "@/context/BackendHealthContext";
import { saveRun } from "@/lib/cache/runs";
import { pca3D, type Point3 } from "@/lib/geometry/pca";
import { TrajectoryThree } from "@/components/viz/TrajectoryThree";
import type { Run, RunSampleRef } from "@/types/run";
import { DeepDive } from "@/components/shared/DeepDive";
import { Table } from "@/components/shared/Table";
import { ExportButtons } from "@/components/shared/ExportButtons";
import { CameraRoll, FrameModal, type CameraRollEntry } from "@/components/shared/CameraRoll";
import { downloadCsv } from "@/lib/export/csv";
import { downloadPdf } from "@/lib/export/pdf";
import { downloadJson } from "@/lib/export/json";
import { computeImageStats } from "@/lib/image/stats";
import { downloadSvg, downloadPngFromSvg, escXml } from "@/lib/export/svg";
import { Download } from "lucide-react";
import { lookup as lookupTerm, termsFor } from "@/lib/docs/glossary";
import { PromptChips, STARTER_PRESETS } from "@/components/shared/PromptChips";
import { RandomSeedButton, nextSeed, type SeedMode } from "@/components/shared/RandomSeedButton";
import { CfgSelect, cfgCaption } from "@/components/shared/CfgSelect";
import { useLayerStack, type BaseLayer, TEMP_COLOUR } from "@/lib/operations/useLayerStack";
import { LayerStackPanel } from "@/components/shared/LayerStackPanel";

// IDB key for the persistent layers stack. Bump if the TrajectoryLayer
// shape changes incompatibly so old snapshots don't crash the loader.
const TRAJ_PERSIST_KEY = "trajectory.savedLayers.v1";

type ProjectionKind = "pca" | "umap" | "film";

type StepStat = Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">;

interface TrajectoryLayer extends BaseLayer {
  /**
   * Inherits id / label / colour / visible / locked / createdAt
   * from BaseLayer (see useLayerStack). The locked semantics:
   *   • `false` = a temporary layer that will be replaced by the next run.
   *   • `true`  = the user has locked this layer in place; subsequent runs
   *               leave it alone and add their own (initially temporary)
   *               layer alongside it.
   */
  prompt: string;
  seed: number;
  steps: number;
  cfg: number;
  width: number;
  height: number;
  modelId: string;
  latents: Float32Array[];
  previews: Array<string | null>;
  stepStats: StepStat[];
  finalImage: string | null;
  responseTimeMs: number | null;
  points: Point3[]; // re-projected client-side per current projection
}

const LAYER_COLOURS = ["#7c2d36", "#c9a227", "#2e5d8a", "#3b7d4f", "#8a3b6e", "#5e5e5e"];

/**
 * Canonical diffusion-model resolutions. Multiples of 64, covering the
 * sizes the major SD/SDXL/FLUX/SD3 checkpoints were actually trained at.
 * Free-form pixel input is unhelpful — typing 517 just produces a broken
 * image. The annotations remind the user which models match each size.
 */
// Bare numeric labels keep the dropdown narrow enough to sit on the same
// row as Seed / Steps / CFG / Preview every. The hint text (which model
// each size matches) lives on each option's `title` attribute and on the
// "native" badge in the field label.
const DIFFUSION_SIZES: Array<{ value: number; label: string; hint?: string }> = [
  { value: 256, label: "256", hint: "small / preview" },
  { value: 384, label: "384" },
  { value: 512, label: "512", hint: "SD 1.5 / 2.0" },
  { value: 640, label: "640" },
  { value: 768, label: "768", hint: "SD 2.x" },
  { value: 896, label: "896", hint: "SDXL aspect" },
  { value: 1024, label: "1024", hint: "SDXL / FLUX / SD 3" },
  { value: 1152, label: "1152", hint: "SDXL aspect" },
  { value: 1280, label: "1280" },
  { value: 1536, label: "1536" },
  { value: 2048, label: "2048", hint: "large" },
];

/**
 * Builds the dropdown option list with the loaded model's native size
 * marked with a ★. If the native size is not in the canonical list
 * (uncommon, but possible — e.g. a fine-tune trained at 720) it is
 * inserted in the right sort position so the user can still pick it.
 */
function dimensionOptions(native: number | null): Array<{ value: number; label: string; hint?: string }> {
  const merged = [...DIFFUSION_SIZES];
  if (native && !merged.some((s) => s.value === native)) {
    merged.push({ value: native, label: String(native), hint: "model native" });
    merged.sort((a, b) => a.value - b.value);
  }
  return merged.map((s) =>
    native && s.value === native
      ? { value: s.value, label: `★ ${s.label}`, hint: s.hint ? `${s.hint} · native` : "native" }
      : s,
  );
}

function nextColour(i: number): string {
  return LAYER_COLOURS[i % LAYER_COLOURS.length];
}

interface StartEvent { event: "start"; meta: Record<string, unknown> }
interface ModelLoadingEvent { event: "model_loading"; modelId: string; alreadyLoaded: boolean; message: string }
interface ReadyEvent { event: "ready"; device: string }
interface StepEvent {
  event: "step";
  step: number;
  totalSteps: number;
  shape: number[];
  latentB64: string;
  previewDataUrl?: string;
  timestep?: number | null;
  sigma?: number | null;
  latentMean?: number;
  latentStd?: number;
  latentMin?: number;
  latentMax?: number;
  latentNorm?: number;
}
interface DoneEvent { event: "done"; imageDataUrl: string; responseTimeMs: number }
interface ErrorEvent { event: "error"; message: string }

type StreamEvent = StartEvent | ModelLoadingEvent | ReadyEvent | StepEvent | DoneEvent | ErrorEvent;

function decodeLatent(b64: string): Float32Array {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? "image/png";
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Blob([buf], { type: mime });
}

export function DenoiseTrajectory() {
  const { settings, setSettingsOpen } = useSettings();
  const { set: cacheImage } = useImageBlobCache();
  const { report: healthReport } = useBackendHealth();
  const nativeWidth = healthReport?.nativeWidth ?? null;
  const nativeHeight = healthReport?.nativeHeight ?? null;
  // Track whether the user has manually picked a size so we don't clobber
  // their choice every time the model changes (or every poll).
  const [userSetSize, setUserSetSize] = useState(false);

  const [prompt, setPrompt] = useState("a cat sitting on a wooden chair, photorealistic");
  const [seed, setSeed] = useState(42);
  // Seed mode: "off" leaves the seed alone; "shuffle" rolls a random seed
  // before each run; "increment" bumps the seed by +1 before each run.
  // The dice icon animates differently per mode so the user can see at a
  // glance which kind of roll happened.
  const [seedMode, setSeedMode] = useState<SeedMode>("off");
  const [seedSpinning, setSeedSpinning] = useState(false);

  // AbortController for in-flight runs. Stored in a ref so the Stop
  // button can reach the live controller without re-renders.
  const abortRef = useRef<AbortController | null>(null);

  // Batch run: kick off N trajectories in sequence. Combined with the
  // shuffle / +1 seed mode this walks a range of seeds; each
  // intermediate run is auto-locked so all N layers persist for
  // comparison rather than overwriting each other. 1 = current
  // single-run behaviour. Stop aborts the whole batch via batchAbortRef.
  const [batchSize, setBatchSize] = useState(1);
  const [batchProgress, setBatchProgress] = useState<{ done: number; total: number } | null>(null);
  const batchAbortRef = useRef(false);
  // 20-step default with EulerDiscreteScheduler (the backend pins this
  // because DPM++ 2M produces NaN/black images at specific CFG × seed
  // combinations on MPS, regardless of Karras sigmas). Euler is slower-
  // converging than DPM++ at the same step count — 20 Euler ≈ 12 DPM++
  // in fidelity. Drop to 12-15 for fast smoke tests, bump to 30+ for
  // research-grade output.
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(7.5);

  const [previewEvery, setPreviewEvery] = useState(1);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [stepInspectorOpen, setStepInspectorOpen] = useState(false);

  // Film is the default — most intuitive read of a trajectory for a researcher
  // unfamiliar with PCA/UMAP. PCA and UMAP remain one click away.
  // (Set in the projection state initialisation below.)
  const [projection, setProjection] = useState<ProjectionKind>("film");
  // PCA/UMAP thumbnail density. 1 = every step, higher = thinned. Lets the
  // user clear the swarm of mid-trajectory thumbnails so the curve itself
  // is legible. Defaults to a sensible "every other" so dense runs aren't
  // overwhelming on first render.
  const [thumbStride, setThumbStride] = useState(4);
  // SD 1.5 was trained at 512×512 — running it at 1024 produces black images.
  // SDXL and FLUX want 1024. Default to 512 as a safe pre-handshake guess;
  // a useEffect below snaps to the loaded model's native size as soon as
  // /health reports it (and resets when the model changes), unless the user
  // has manually picked a size.
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);

  // Snap width/height to the model's native resolution as soon as the
  // backend reports it. Reset on model change so a fresh model gets its
  // own native default. Honours `userSetSize` so manual choices stick.
  useEffect(() => {
    if (!nativeWidth || !nativeHeight) return;
    if (userSetSize) return;
    setWidth(nativeWidth);
    setHeight(nativeHeight);
  }, [nativeWidth, nativeHeight, userSetSize]);

  // Reset the manual-override flag when the loaded model changes — a new
  // checkpoint should snap to its own native default once /health updates.
  useEffect(() => {
    setUserSetSize(false);
  }, [healthReport?.currentModelId]);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [latents, setLatents] = useState<Float32Array[]>([]);
  const [previews, setPreviews] = useState<Array<string | null>>([]);
  const [stepStats, setStepStats] = useState<Array<Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">>>([]);
  const [points, setPoints] = useState<Point3[]>([]);
  const [finalImage, setFinalImage] = useState<string | null>(null);
  const [responseTimeMs, setResponseTimeMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Layers stack: temp/locked, palette colours on lock, persisted to
  // IDB. Generic primitive shared with Sweep / Neighbourhood / Bench
  // — see `useLayerStack`. The validate function rejects malformed
  // snapshots from older schema versions.
  const layerStack = useLayerStack<TrajectoryLayer>(TRAJ_PERSIST_KEY, (stored) => {
    if (!Array.isArray(stored)) return null;
    return stored.filter((l): l is TrajectoryLayer =>
      typeof l === "object" && l !== null && Array.isArray((l as TrajectoryLayer).latents)
    );
  });
  const savedLayers = layerStack.layers;
  // Compatibility shim: existing run() code uses functional setState
  // directly. The hook exposes the same setLayers fn.
  const setSavedLayers = layerStack.setLayers;

  const isLocal = settings.backend === "local";

  // Reproject when the user toggles PCA/UMAP. UMAP needs ≥4 samples; below
  // that we silently fall back to PCA so the curve stays sensible.
  useEffect(() => {
    if (latents.length < 2) return;
    if (projection === "pca" || latents.length < 4) {
      setPoints(pca3D(latents));
      return;
    }
    try {
      const data = latents.map((l) => Array.from(l));
      const umap = new UMAP({
        nComponents: 3,
        nNeighbors: Math.min(latents.length - 1, 5),
        minDist: 0.1,
      });
      const projected = umap.fit(data) as number[][];
      setPoints(projected.map((p) => [p[0], p[1], p[2]]));
    } catch {
      setPoints(pca3D(latents));
    }
  }, [projection, latents]);

  async function run(opts: { lockOnComplete?: boolean } = {}) {
    if (!isLocal) {
      setError("Denoise Trajectory requires the Local FastAPI backend (per-step latents are not exposed by hosted providers). Open Settings and switch Backend to Local.");
      return;
    }

    // Drop any existing unlocked (temporary) layer — a fresh run
    // replaces the previous temp result. Locked layers are untouched.
    setSavedLayers((prev) => prev.filter((l) => l.locked));

    // Apply the seed mode (shuffle / increment / off) and let the dice
    // animate briefly so the user sees the change before the request
    // fires. The freshly-derived value is used directly in the request
    // body — `seed` from the closure is stale until React re-renders.
    let activeSeed = seed;
    if (seedMode !== "off") {
      activeSeed = nextSeed(seedMode, seed);
      setSeed(activeSeed);
      setSeedSpinning(true);
      await new Promise((r) => setTimeout(r, 450));
      setSeedSpinning(false);
    }

    setRunning(true);
    setError(null);
    setProgress(null);
    setStatusMsg("Connecting to local backend…");
    setLatents([]);
    setPreviews([]);
    setStepStats([]);
    setPoints([]);
    setFinalImage(null);
    setResponseTimeMs(null);

    // Fresh AbortController for this run. Stop button calls .abort().
    const controller = new AbortController();
    abortRef.current = controller;

    let res: Response;
    try {
      res = await fetch(`${settings.localBaseUrl}/trajectory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modelId: settings.modelId,
          prompt,
          seed: activeSeed,
          steps: effectiveSteps(steps, settings),
          cfg,
          width,
          height,
          previewEvery,
        }),
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setStatusMsg("Stopped by user before the backend replied.");
        setRunning(false);
        abortRef.current = null;
        return;
      }
      setError(`Cannot reach local backend at ${settings.localBaseUrl}. Is uvicorn running? (${err instanceof Error ? err.message : String(err)})`);
      setRunning(false);
      abortRef.current = null;
      return;
    }

    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      setError(`Local backend ${res.status}: ${text.slice(0, 400)}`);
      setRunning(false);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const collected: Float32Array[] = [];
    const collectedPreviews: Array<string | null> = [];
    type StepStat = Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">;
    const collectedStats: StepStat[] = [];
    let finalUrl: string | null = null;
    let respMs: number | null = null;
    let streamErr: string | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          let event: StreamEvent;
          try { event = JSON.parse(line) as StreamEvent; } catch { continue; }
          if (event.event === "model_loading") {
            setStatusMsg(event.message);
          } else if (event.event === "ready") {
            setStatusMsg(`Model ready on ${event.device}. Generating…`);
          } else if (event.event === "step") {
            collected.push(decodeLatent(event.latentB64));
            collectedPreviews.push(event.previewDataUrl ?? null);
            // Track the latest step as the selected one during streaming so
            // the slider follows the live preview in real time.
            setSelectedStep(event.step - 1);
            collectedStats.push({
              step: event.step,
              totalSteps: event.totalSteps,
              timestep: event.timestep ?? null,
              sigma: event.sigma ?? null,
              latentMean: event.latentMean,
              latentStd: event.latentStd,
              latentMin: event.latentMin,
              latentMax: event.latentMax,
              latentNorm: event.latentNorm,
            });
            setProgress({ done: event.step, total: event.totalSteps });
            setStatusMsg(null);
            setLatents([...collected]);
            setPreviews([...collectedPreviews]);
            setStepStats([...collectedStats]);
          } else if (event.event === "done") {
            finalUrl = event.imageDataUrl;
            respMs = event.responseTimeMs;
          } else if (event.event === "error") {
            streamErr = event.message;
          }
        }
      }
    } catch (err) {
      // AbortError from reader.read() when the user clicks Stop is
      // expected — the controller-aborted branch below handles cleanup.
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        streamErr = err instanceof Error ? err.message : String(err);
      }
    }

    // User-initiated stop: keep the partial trajectory visible (it can
    // still be saved as a layer if it has ≥2 steps) but skip the
    // auto-save block that expects a final image.
    if (controller.signal.aborted) {
      setStatusMsg(`Stopped by user at step ${collected.length}/${steps}. Partial trajectory kept.`);
      setRunning(false);
      abortRef.current = null;
      // If we already have ≥2 steps, snapshot them as a temp layer so
      // the user can still inspect what was captured before the stop.
      if (collected.length >= 2) {
        const newLayer: TrajectoryLayer = {
          id: `layer-${Date.now()}`,
          label: `${prompt.slice(0, 32)}${prompt.length > 32 ? "…" : ""} · seed ${activeSeed} · stopped @${collected.length}`,
          colour: "#1a1a1a",
          visible: true,
          locked: false,
          createdAt: Date.now(),
          prompt, seed: activeSeed, steps, cfg, width, height,
          modelId: settings.modelId,
          latents: collected.slice(),
          previews: collectedPreviews.slice(),
          stepStats: collectedStats.slice(),
          finalImage: null,
          responseTimeMs: null,
          points: pca3D(collected),
        };
        setSavedLayers((prev) => [newLayer, ...prev.filter((l) => l.locked)]);
      }
      return;
    }

    if (streamErr) {
      setError(streamErr);
      setRunning(false);
      abortRef.current = null;
      return;
    }

    // Initial projection (PCA — UMAP toggle re-runs via useEffect).
    if (collected.length >= 2) {
      const projected = pca3D(collected);
      setPoints(projected);
    }
    setResponseTimeMs(respMs);

    if (finalUrl) {
      setFinalImage(finalUrl);
      const imageKey = `traj::local::${settings.modelId}::${activeSeed}::${steps}::${cfg}::${Date.now()}`;
      // IDB writes can fail (connection closing during HMR, version-change
      // race, quota exceeded) — log and continue so the auto-save-as-layer
      // below still runs. The final image is also kept in the temp layer's
      // `finalImage` field, so it's not lost on a cache miss.
      try {
        await cacheImage(imageKey, dataUrlToBlob(finalUrl));

        const samples: RunSampleRef[] = [{ imageKey, variable: "final", responseTimeMs: respMs ?? undefined }];
        const run: Run = {
          id: `traj::${Date.now()}`,
          kind: "single",
          createdAt: new Date().toISOString(),
          providerId: "local",
          modelId: settings.modelId,
          prompt,
          seed: activeSeed,
          steps,
          cfg,
          width: settings.defaults.width,
          height: settings.defaults.height,
          samples,
          extra: { trajectory: true, stepCount: collected.length },
        };
        await saveRun(run);
      } catch (err) {
        console.warn("Trajectory cache write failed (continuing):", err);
      }
    }

    // Auto-save the just-finished run as a layer. By default temporary
    // (unlocked, neutral colour) — replaced by the next single run.
    // When `opts.lockOnComplete` is true (batch run, intermediate
    // iteration), save as locked with a palette colour so it persists
    // alongside its siblings.
    if (collected.length >= 2 && finalUrl) {
      const lockNow = opts.lockOnComplete ?? false;
      setSavedLayers((prev) => {
        const lockedCount = prev.filter((l) => l.locked).length;
        const newLayer: TrajectoryLayer = {
          id: `layer-${Date.now()}`,
          label: `${prompt.slice(0, 32)}${prompt.length > 32 ? "…" : ""} · seed ${activeSeed}`,
          colour: lockNow ? nextColour(lockedCount) : "#1a1a1a",
          visible: true,
          locked: lockNow,
          createdAt: Date.now(),
          prompt,
          seed: activeSeed,
          steps,
          cfg,
          width,
          height,
          modelId: settings.modelId,
          latents: collected.slice(),
          previews: collectedPreviews.slice(),
          stepStats: collectedStats.slice(),
          finalImage: finalUrl,
          responseTimeMs: respMs,
          points: pca3D(collected),
        };
        return [newLayer, ...prev.filter((l) => l.locked)];
      });
    }

    setRunning(false);
    abortRef.current = null;
  }

  function stopRun() {
    // Set the batch flag first so a multi-iteration loop breaks
    // between iterations, then abort the current fetch.
    batchAbortRef.current = true;
    abortRef.current?.abort();
  }

  /**
   * Run N trajectories in sequence. Each iteration except the last is
   * auto-locked so all N persist as their own layers (otherwise the
   * temp-replace rule would leave only the last). The seed mode
   * (shuffle / increment / off) fires once per iteration inside `run()`,
   * so combining batchSize=10 with mode="increment" walks 10
   * neighbouring seeds and keeps all 10 trajectories side by side for
   * comparison.
   */
  async function runBatch() {
    const n = Math.max(1, Math.min(50, batchSize | 0));
    if (n <= 1) {
      // Single-run path — auto-save as temp like before.
      void run();
      return;
    }
    batchAbortRef.current = false;
    setBatchProgress({ done: 0, total: n });
    for (let i = 0; i < n; i++) {
      if (batchAbortRef.current) break;
      setBatchProgress({ done: i + 1, total: n });
      // Lock all but the final iteration so every result stays in the
      // layers list. The last one stays temp so the existing
      // click-padlock-to-keep affordance still applies.
      await run({ lockOnComplete: i < n - 1 });
      if (batchAbortRef.current) break;
    }
    setBatchProgress(null);
    batchAbortRef.current = false;
  }

  // All layer mutations now flow through the shared hook so they
  // stay in sync with persistence + the temp/locked invariants.
  const updateLayer = layerStack.updateLayer;
  const deleteLayer = layerStack.deleteLayer;
  const lockLayer = layerStack.lockLayer;
  const unlockLayer = layerStack.unlockLayer;

  // The in-flight run is rendered from transient state so the user sees
  // partial latents/previews as they stream. Once `running` flips to
  // false the auto-save in run() has already snapshotted the result into
  // savedLayers as `locked: false`, so we drop the activeLayer at that
  // point to avoid duplicating it on screen.
  const activeLayer: TrajectoryLayer | null =
    running && latents.length > 0
      ? {
          id: "active",
          label: "active run",
          colour: "#1a1a1a",
          visible: true,
          locked: false,
          createdAt: Date.now(),
          prompt, seed, steps, cfg, width, height,
          modelId: settings.modelId,
          latents,
          previews,
          stepStats,
          finalImage,
          responseTimeMs,
          points,
        }
      : null;

  // Newest-first: while a run is streaming the in-flight activeLayer
  // sits at the top; once the run completes it has already been
  // snapshotted into savedLayers (as the unlocked / temporary entry at
  // the head), so we just render savedLayers in their existing order.
  const visibleLayers = [
    ...(activeLayer ? [activeLayer] : []),
    ...savedLayers.filter((l) => l.visible),
  ];

  return (
    <div className="card-editorial p-6 max-w-5xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">Denoise Trajectory</h2>
      <p className="font-body text-body-sm text-foreground mb-1">
        Trace the iterative denoising path the model walks across its learned manifold. The local backend streams each step's <em>latent</em> — a tensor in motion within the model's vector space — and the client projects them to 3D via PCA (or UMAP).
      </p>
      <p className="font-sans text-caption italic text-muted-foreground mb-4">
        Read the curve as: gold marker is pure noise (a sample from the vector space's prior); burgundy is the final image (the manifold's destination for this prompt + seed). Bends in the curve are where the denoising made a "decision" about composition. Thumbnails along the path show the image emerging from noise — the gap between an early-step blur and the final image is where most of the generative work happens. Unlike LLM trajectories, which have to be reconstructed from token logits, this trajectory is directly observed: every step is a tensor you can encode, decode, and compare.
      </p>

      {!isLocal && (
        <div className="border border-burgundy/40 bg-burgundy/5 text-burgundy p-3 mb-4 font-sans text-body-sm rounded-sm">
          Local FastAPI backend required. Hosted providers return only the final image; the per-step latents that constitute the trajectory are not part of their response.{" "}
          <button
            onClick={() => setSettingsOpen(true)}
            className="underline underline-offset-2 hover:text-burgundy-900 font-medium"
          >
            Open Settings →
          </button>{" "}
          and switch Backend to <strong>Local FastAPI</strong>.
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
        <div
          className="grid gap-2"
          style={{
            // CFG holds a decimal (7.5) + spinner ≥ 96px. "Preview every"
            // wraps onto two lines under ~88px, so 100px keeps the label
            // on one line.
            gridTemplateColumns:
              "minmax(220px, 1.4fr) 72px 96px 100px minmax(108px, 1fr) minmax(108px, 1fr)",
          }}
        >
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
              min={2}
              max={100}
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value, 10) || 1)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block" title={lookupTerm("CFG")}>
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">CFG</span>
            <CfgSelect value={cfg} onChange={setCfg} />
          </label>
          <label className="block" title="Decode a thumbnail every N steps. 1 = capture every step (richer Deep Dive + step scrubber, ~0.5s extra per step). 0 disables previews entirely. Higher values make the run faster at the cost of less data in the inspector.">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4 whitespace-nowrap">Preview every</span>
            <input
              type="number"
              min={0}
              max={50}
              value={previewEvery}
              onChange={(e) => setPreviewEvery(parseInt(e.target.value, 10) || 0)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block" title="Image width in pixels. Snaps to the loaded model's native resolution (derived from unet/transformer sample_size × VAE scale factor) as soon as the backend reports it. Off-native sizes are kept available for SDXL aspect-ratio bucketing and experimentation, but produce black or distorted output on most checkpoints.">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4 whitespace-nowrap">
              W{nativeWidth ? <span className="ml-1 normal-case tracking-normal text-burgundy">· {nativeWidth}★</span> : null}
            </span>
            <select
              value={width}
              onChange={(e) => { setWidth(parseInt(e.target.value, 10)); setUserSetSize(true); }}
              className="input-editorial mt-1"
            >
              {dimensionOptions(nativeWidth).map((s) => (
                <option key={s.value} value={s.value} title={s.hint ?? undefined}>{s.label}</option>
              ))}
            </select>
          </label>
          <label className="block" title="Image height in pixels. Match Width and the model's training resolution unless you are exploring SDXL aspect-ratio buckets.">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4 whitespace-nowrap">
              H{nativeHeight ? <span className="ml-1 normal-case tracking-normal text-burgundy">· {nativeHeight}★</span> : null}
            </span>
            <select
              value={height}
              onChange={(e) => { setHeight(parseInt(e.target.value, 10)); setUserSetSize(true); }}
              className="input-editorial mt-1"
            >
              {dimensionOptions(nativeHeight).map((s) => (
                <option key={s.value} value={s.value} title={s.hint ?? undefined}>{s.label}</option>
              ))}
            </select>
          </label>
        </div>
        <p className="font-sans text-caption italic text-muted-foreground mt-2">
          <span className="not-italic font-medium">CFG (classifier-free guidance):</span> how much the prompt's pull is <em>amplified</em> at each denoising step. The U-Net runs twice (with prompt + without) and outputs <span className="font-mono not-italic">unconditional + CFG × (conditional − unconditional)</span>. <span className="font-mono not-italic">0</span> = ignore the prompt entirely; <span className="font-mono not-italic">1</span> = use the prompt with no extra push; <span className="font-mono not-italic">7.5</span> = balanced default; <span className="font-mono not-italic">12+</span> = oversaturated / mode collapse. Different CFGs produce different trajectories — the Guidance Sweep operation is built around this lever.
        </p>
      </div>

      <LayerStackPanel
        layers={savedLayers}
        operationName="Trajectory"
        onRename={(id, label) => updateLayer(id, { label })}
        onToggleVisible={(id, visible) => updateLayer(id, { visible })}
        onLock={lockLayer}
        onUnlock={unlockLayer}
        onDelete={deleteLayer}
        onReset={layerStack.reset}
        renderMetadata={(l) =>
          `${l.latents.length} steps · seed ${l.seed} · ${l.modelId.split("/").pop()}`
        }
      />

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <button
          onClick={() => void runBatch()}
          disabled={running || !isLocal}
          className={running || !isLocal ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running
            ? batchProgress
              ? `Batch ${batchProgress.done}/${batchProgress.total} · step ${progress?.done ?? 0}/${progress?.total ?? steps}…`
              : statusMsg && progress === null
                ? statusMsg
                : `Streaming step ${progress?.done ?? 0}/${progress?.total ?? steps}…`
            : batchSize > 1
              ? `Run trajectory × ${batchSize}`
              : `Run trajectory (${steps} steps)`}
        </button>
        {/* Batch stepper. Pairs naturally with shuffle / +1: × 10 with
            +1 mode walks ten neighbouring seeds and locks all but the
            last so they all stay in the layers list. */}
        <label
          className="flex items-center gap-1.5 font-sans text-caption text-muted-foreground"
          title={
            seedMode === "off" && batchSize > 1
              ? "Batch run: N trajectories in sequence. Pair with shuffle (random seed each) or +1 (incrementing seed) to walk a range, otherwise all N runs use the same seed."
              : "Batch run: N trajectories in sequence. Each intermediate run is auto-locked so all N persist as layers; the last stays temporary."
          }
        >
          <span className="uppercase tracking-wider cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">
            ×
          </span>
          <input
            type="number"
            min={1}
            max={50}
            value={batchSize}
            onChange={(e) => setBatchSize(Math.max(1, Math.min(50, parseInt(e.target.value, 10) || 1)))}
            disabled={running}
            className="input-editorial w-16 px-2 py-1 text-body-sm"
          />
        </label>
        {running && (
          <button
            onClick={stopRun}
            className="px-3 py-2 border border-burgundy bg-burgundy text-cream rounded-sm hover:bg-burgundy-900 flex items-center gap-1.5 font-sans text-body-sm"
            title={batchProgress
              ? `Abort the batch run. Iterations completed so far stay as locked layers; the in-flight one is kept as a partial temp if it has ≥ 2 steps.`
              : "Abort the in-flight run. The partial trajectory captured so far will be kept as a temporary layer if it has at least 2 steps."}
          >
            <Square size={12} fill="currentColor" /> Stop
          </button>
        )}
      </div>

      {error && (
        <div className="border border-burgundy/40 bg-burgundy/5 text-burgundy p-3 mb-4 font-sans text-body-sm rounded-sm">
          {error}
        </div>
      )}

      {(running || latents.length > 0) && (
        <div className="border-t border-parchment pt-4">
          <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
            <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground">
              Trajectory · {latents.length} step{latents.length !== 1 ? "s" : ""} captured
            </h3>
            <div className="flex items-center gap-2">
              <span className="font-sans text-caption text-muted-foreground">Projection</span>
              <button
                onClick={() => setProjection("pca")}
                className={projection === "pca" ? "btn-editorial-primary px-3 py-1 text-caption" : "btn-editorial-secondary px-3 py-1 text-caption"}
                title="Linear projection onto the three directions of greatest variance. Preserves global distances; deterministic; fast."
              >
                PCA
              </button>
              <button
                onClick={() => setProjection("umap")}
                className={projection === "umap" ? "btn-editorial-primary px-3 py-1 text-caption" : "btn-editorial-secondary px-3 py-1 text-caption"}
                title={latents.length < 4 ? "UMAP needs ≥4 steps; falls back to PCA" : "Non-linear projection that preserves local neighbourhood structure. Stochastic; slower; can reveal basin/cluster shape PCA flattens."}
              >
                UMAP
              </button>
              <button
                onClick={() => setProjection("film")}
                className={projection === "film" ? "btn-editorial-primary px-3 py-1 text-caption" : "btn-editorial-secondary px-3 py-1 text-caption"}
                title="35mm film-strip view: scroll through every captured step in order, with per-step scalars underneath each frame."
              >
                Film
              </button>
              {responseTimeMs !== null && (
                <span className="font-sans text-caption text-muted-foreground ml-2">
                  {(responseTimeMs / 1000).toFixed(1)}s total
                </span>
              )}
            </div>
          </div>
          <p className="font-sans text-caption italic text-muted-foreground mb-2">
            PCA and UMAP show different curves because they answer different questions. PCA is a linear projection onto the three directions of greatest variance, so it preserves <em>global</em> distances — long stretches of the trajectory keep their relative scale. UMAP is a non-linear method that preserves <em>local</em> neighbourhood structure at the cost of distorting global distances — it can pull apart steps that PCA bunches together when they sit on different parts of a curved surface. Same trajectory, two views: PCA reads it as a path, UMAP reads it as a topology. Disagreement between them is itself a finding about the manifold's local curvature.
          </p>

          {/* Step inspector — opens a modal with the slider + thumbnail
              + per-step scalars. Replaces the always-visible scrubber
              that was crowding the projection view. */}
          {latents.length > 1 && (
            <div className="mb-3 flex items-center gap-3">
              <button
                onClick={() => setStepInspectorOpen(true)}
                className="btn-editorial-secondary px-3 py-1 text-caption flex items-center gap-1.5"
                title="Open the step inspector to scrub through every captured latent with its preview thumbnail and full scalar stats."
              >
                <Search size={12} /> Inspect steps
              </button>
              <span className="font-sans text-caption text-muted-foreground italic">
                {latents.length} of {steps} step{steps !== 1 ? "s" : ""} captured · click to scrub
              </span>
            </div>
          )}
          {stepInspectorOpen && latents.length > 1 && (
            <StepScrubberModal
              captured={latents.length}
              requestedSteps={steps}
              selected={selectedStep ?? latents.length - 1}
              onChange={setSelectedStep}
              onClose={() => setStepInspectorOpen(false)}
              previews={previews}
              stepStats={stepStats}
              cfg={cfg}
            />
          )}
          {projection === "film" && visibleLayers.length > 0 ? (
            <div className="space-y-3">
              {visibleLayers.map((l) => (
                <div key={l.id}>
                  {visibleLayers.length > 1 && (
                    <div className="font-sans text-[8px] text-muted-foreground mb-1 flex items-center gap-1.5">
                      <span className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: l.colour }} />
                      {l.label}
                    </div>
                  )}
                  <FilmStrip
                    previews={l.previews}
                    latents={l.latents}
                    stepStats={l.stepStats}
                    cfg={l.cfg}
                    prompt={l.prompt}
                    modelId={l.modelId}
                    seed={l.seed}
                  />
                </div>
              ))}
            </div>
          ) : visibleLayers.some((l) => l.points.length >= 2) ? (
            <div className="space-y-2">
              <div className="flex items-center gap-3 px-1">
                <label className="font-sans text-caption text-muted-foreground flex items-center gap-2 flex-1">
                  <span className="uppercase tracking-wider whitespace-nowrap" title="Show a preview thumbnail every Nth step. Higher values thin the swarm so the curve is easier to read; the final-step thumbnail is always shown.">
                    Thumbnails every
                  </span>
                  <input
                    type="range"
                    min={1}
                    max={Math.max(1, Math.min(20, latents.length || 1))}
                    value={thumbStride}
                    onChange={(e) => setThumbStride(parseInt(e.target.value, 10) || 1)}
                    className="flex-1 max-w-xs"
                  />
                  <span className="font-mono text-foreground w-10 text-right">{thumbStride}</span>
                  <span className="italic">{thumbStride === 1 ? "every step" : `every ${thumbStride}th step`}</span>
                </label>
              </div>
              <TrajectoryThree
                previewStride={thumbStride}
                layers={visibleLayers
                  .filter((l) => l.points.length >= 2)
                  .map((l) => ({
                    id: l.id,
                    label: l.label,
                    colour: l.colour,
                    points: l.points,
                    previews: l.previews,
                  }))}
              />
            </div>
          ) : (
            <div className="border border-parchment bg-cream/30 p-8 text-center font-sans text-body-sm text-muted-foreground rounded-sm">
              {running ? (
                <>
                  {statusMsg ?? `Streaming latents · ${latents.length} step${latents.length === 1 ? "" : "s"} captured of ${steps}`}
                  <div className="mt-2 h-1 bg-parchment rounded overflow-hidden max-w-md mx-auto">
                    <div
                      className="h-full bg-burgundy transition-all duration-300"
                      style={{ width: `${Math.min(100, ((progress?.done ?? 0) / Math.max(1, progress?.total ?? steps)) * 100)}%` }}
                    />
                  </div>
                  {previews.some((p) => p) && (
                    <div className="mt-4 flex gap-2 justify-center flex-wrap">
                      {previews.map((url, i) =>
                        url ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <div key={i} className="flex flex-col items-center">
                            <img src={url} alt={`step ${i + 1}`} className="w-16 h-16 object-cover rounded-sm border border-parchment" />
                            <span className="font-sans text-[10px] text-muted-foreground mt-0.5">step {i + 1}</span>
                          </div>
                        ) : null,
                      )}
                    </div>
                  )}
                </>
              ) : (
                "Need at least 2 steps to render a trajectory."
              )}
            </div>
          )}
        </div>
      )}

      {visibleLayers.length > 0 && (
        <TrajectoryDeepDive layers={visibleLayers} />
      )}
    </div>
  );
}

interface TrajectoryDeepDiveProps {
  layers: TrajectoryLayer[];
}

function l2(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
}

interface RichStepRow {
  step: number;
  norm: number;
  stepSize: number | null;
  cosToFinal: number;
  cosToStart: number;
  timestep: number | null | undefined;
  sigma: number | null | undefined;
  latentMean: number | undefined;
  latentStd: number | undefined;
  latentMin: number | undefined;
  latentMax: number | undefined;
  preview: string | null;
}

function buildRichRows(layer: TrajectoryLayer): RichStepRow[] {
  const final = layer.latents[layer.latents.length - 1];
  return layer.latents.map((l, i) => {
    const norm = Math.sqrt(l.reduce((a, x) => a + x * x, 0));
    const stepSize = i > 0 ? l2(l, layer.latents[i - 1]) : 0;
    const cosToFinal = cosine(l, final);
    const cosToStart = cosine(l, layer.latents[0]);
    const stat = layer.stepStats[i] ?? {};
    return {
      step: i + 1,
      norm,
      stepSize: i > 0 ? stepSize : null,
      cosToFinal,
      cosToStart,
      timestep: stat.timestep ?? null,
      sigma: stat.sigma ?? null,
      latentMean: stat.latentMean,
      latentStd: stat.latentStd,
      latentMin: stat.latentMin,
      latentMax: stat.latentMax,
      preview: layer.previews[i] ?? null,
    };
  });
}

function rowsToTableArray(rows: RichStepRow[]): Array<Array<string | number>> {
  return rows.map((r) => [
    r.step,
    r.timestep != null ? r.timestep.toFixed(1) : "—",
    r.sigma != null ? r.sigma.toFixed(3) : "—",
    r.norm.toFixed(3),
    r.stepSize != null ? r.stepSize.toFixed(3) : "—",
    r.cosToFinal.toFixed(3),
    r.cosToStart.toFixed(3),
    r.latentMean != null ? r.latentMean.toFixed(3) : "—",
    r.latentStd != null ? r.latentStd.toFixed(3) : "—",
    r.latentMin != null ? r.latentMin.toFixed(3) : "—",
    r.latentMax != null ? r.latentMax.toFixed(3) : "—",
    r.preview ? "yes" : "—",
  ]);
}

const TRAJ_HEADERS = ["step", "timestep", "sigma", "‖z‖", "Δ to prev", "cos→final", "cos→start", "mean", "std", "min", "max", "preview"];

function buildCameraEntries(layer: TrajectoryLayer): Array<{
  src: string;
  caption?: string;
  subcaption?: string;
  details?: Array<{ label: string; value: string | number }>;
}> {
  const out: ReturnType<typeof buildCameraEntries> = [];
  layer.previews.forEach((url, i) => {
    if (!url) return;
    const stat = layer.stepStats[i] ?? {};
    const details: Array<{ label: string; value: string | number }> = [
      { label: "Layer", value: layer.label },
      { label: "Step", value: i + 1 },
      { label: "CFG", value: `${layer.cfg} (${cfgCaption(layer.cfg)})` },
      { label: "Seed", value: layer.seed },
    ];
    if (stat.timestep != null) details.push({ label: "Timestep", value: stat.timestep.toFixed(2) });
    if (stat.sigma != null) details.push({ label: "Sigma", value: stat.sigma.toFixed(4) });
    if (stat.latentNorm != null) details.push({ label: "‖z‖", value: stat.latentNorm.toFixed(4) });
    if (stat.latentMean != null) details.push({ label: "Mean", value: stat.latentMean.toFixed(4) });
    if (stat.latentStd != null) details.push({ label: "Std", value: stat.latentStd.toFixed(4) });
    if (stat.latentMin != null) details.push({ label: "Min", value: stat.latentMin.toFixed(4) });
    if (stat.latentMax != null) details.push({ label: "Max", value: stat.latentMax.toFixed(4) });
    out.push({
      src: url,
      caption: `step ${i + 1}`,
      subcaption: `${layer.label} · click for stats`,
      details,
    });
  });
  if (layer.finalImage) {
    out.push({
      src: layer.finalImage,
      caption: "final",
      subcaption: `${layer.label} · ${layer.steps} steps`,
      details: [
        { label: "Layer", value: layer.label },
        { label: "Prompt", value: layer.prompt },
        { label: "Seed", value: layer.seed },
        { label: "Steps", value: layer.steps },
        { label: "CFG", value: `${layer.cfg} (${cfgCaption(layer.cfg)})` },
        { label: "Model", value: layer.modelId },
        ...(layer.responseTimeMs !== null ? [{ label: "Time", value: `${(layer.responseTimeMs / 1000).toFixed(1)}s` }] : []),
      ],
    });
  }
  return out;
}

function TrajectoryDeepDive({ layers }: TrajectoryDeepDiveProps) {
  // Aggregate everything for cross-layer exports.
  const stamp = `traj-${Date.now()}`;

  function exportCsv() {
    const headers = ["layer", ...TRAJ_HEADERS];
    const rows: Array<Array<string | number>> = [];
    for (const layer of layers) {
      for (const r of rowsToTableArray(buildRichRows(layer))) {
        rows.push([layer.label, ...r]);
      }
    }
    downloadCsv(`${stamp}.csv`, headers, rows);
  }

  function exportJson() {
    downloadJson(`${stamp}.json`, {
      operation: "denoise-trajectory",
      layers: layers.map((layer) => ({
        id: layer.id,
        label: layer.label,
        modelId: layer.modelId,
        prompt: layer.prompt,
        seed: layer.seed,
        steps: layer.steps,
        cfg: layer.cfg,
        latentDim: layer.latents[0]?.length ?? 0,
        stepCount: layer.latents.length,
        responseTimeMs: layer.responseTimeMs,
        perStep: rowsToTableArray(buildRichRows(layer)),
      })),
    });
  }

  async function exportPdf() {
    // Compute image stats for every previewed step in every layer. The
    // computeImageStats helper is cached per dataUrl so re-opening the PDF
    // export after browsing the camera roll is essentially instant.
    const imageStatsByLayer: Array<Map<number, Awaited<ReturnType<typeof computeImageStats>>>> = [];
    for (const layer of layers) {
      const map = new Map<number, Awaited<ReturnType<typeof computeImageStats>>>();
      const tasks: Array<Promise<void>> = [];
      layer.previews.forEach((url, i) => {
        if (!url) return;
        tasks.push(
          computeImageStats(url).then((s) => {
            map.set(i, s);
          }).catch(() => undefined),
        );
      });
      await Promise.all(tasks);
      imageStatsByLayer.push(map);
    }

    const IMG_HEADERS = [
      "step",
      "dims",
      "R", "G", "B",
      "hue°",
      "bright",
      "contrast",
      "satur.",
      "entropy",
      "edges",
      "centre (x,y)",
      "HF energy",
      "PNG bytes",
    ];

    // One PdfGroup per layer — newest at the top, matching the on-screen
    // order (the caller already passes layers newest-first). Each group is
    // self-contained: layer header, its own camera roll, then its
    // latent-geometry and image-stats tables, all on its own page(s).
    const groups = layers.map((layer, li) => {
      const images: Array<{ dataUrl: string; caption?: string }> = [];
      layer.previews.forEach((p, i) => {
        if (p) images.push({ dataUrl: p, caption: `step ${i + 1}` });
      });
      if (layer.finalImage) images.push({ dataUrl: layer.finalImage, caption: "final" });

      const tables: Array<{ title: string; caption: string; table: { headers: string[]; rows: Array<Array<string | number>> } }> = [];
      tables.push({
        title: "Per-step latent geometry",
        caption: `Prompt: ${layer.prompt} · seed ${layer.seed} · ${layer.steps} steps · CFG ${layer.cfg} · model ${layer.modelId}`,
        table: { headers: TRAJ_HEADERS, rows: rowsToTableArray(buildRichRows(layer)) },
      });

      const statsMap = imageStatsByLayer[li];
      const imgRows: Array<Array<string | number>> = [];
      layer.previews.forEach((url, i) => {
        if (!url) return;
        const s = statsMap.get(i);
        if (!s) return;
        imgRows.push([
          i + 1,
          `${s.width}×${s.height}`,
          s.meanR.toFixed(0),
          s.meanG.toFixed(0),
          s.meanB.toFixed(0),
          s.meanHue.toFixed(0),
          s.meanLuma.toFixed(1),
          s.stdLuma.toFixed(1),
          s.saturation.toFixed(3),
          `${s.entropy.toFixed(2)} bits`,
          s.edgeDensity.toFixed(3),
          `(${s.centreX.toFixed(2)}, ${s.centreY.toFixed(2)})`,
          s.highFreqRatio.toFixed(3),
          s.pngBytes.toLocaleString(),
        ]);
      });
      if (imgRows.length > 0) {
        tables.push({
          title: "Per-step image stats",
          caption:
            "Cultural-analytics-style measurements computed in-browser per preview. Mean RGB drift reveals colour bias; entropy falls from ~8 bits (noise) toward 6-7 (structure); edge density typically peaks mid-trajectory; saturation inflates at high CFG; centre-of-mass shows where the model 'decided' the subject sits; HF energy collapses early as diffusion denoises high frequencies first; PNG bytes is a Kolmogorov-complexity proxy.",
          table: { headers: IMG_HEADERS, rows: imgRows },
        });
      }

      return {
        title: `Layer ${li + 1} · ${layer.label}`,
        caption: `seed ${layer.seed} · ${layer.steps} steps · CFG ${layer.cfg} · ${layer.latents.length} captured · ${layer.modelId}` +
          (layer.responseTimeMs !== null ? ` · ${(layer.responseTimeMs / 1000).toFixed(1)}s` : ""),
        images,
        tables,
      };
    });

    const headLayer = layers[0];
    downloadPdf(`${stamp}.pdf`, {
      meta: {
        title: "Denoise Trajectory",
        subtitle: layers.length === 1
          ? `local · ${headLayer.modelId}`
          : `${layers.length} layers compared (newest first)`,
        fields:
          layers.length === 1
            ? [
                { label: "Prompt", value: headLayer.prompt },
                { label: "Seed", value: headLayer.seed },
                { label: "Steps", value: headLayer.steps },
                { label: "CFG", value: `${headLayer.cfg} (${cfgCaption(headLayer.cfg)})` },
                { label: "Latent dim", value: headLayer.latents[0]?.length ?? 0 },
                ...(headLayer.responseTimeMs !== null
                  ? [{ label: "Total time", value: `${(headLayer.responseTimeMs / 1000).toFixed(1)}s` }]
                  : []),
              ]
            : [
                { label: "Layers", value: layers.length },
                ...layers.map((l, i) => ({ label: `Layer ${i + 1}`, value: `${l.label} (${l.latents.length} steps)` })),
              ],
      },
      groups,
      glossary: termsFor([
        "Prompt", "Seed", "Steps", "CFG", "Preview every",
        "step", "‖z‖", "Δ to prev", "cos→final", "cos→start", "preview",
        "timestep", "sigma", "mean", "std", "min", "max",
      ]),
    });
  }

  return (
    <DeepDive actions={<ExportButtons onCsv={exportCsv} onPdf={exportPdf} onJson={exportJson} />}>
      <div className="space-y-8">
        {layers.map((layer) => {
          const richRows = buildRichRows(layer);
          const tableRows = rowsToTableArray(richRows);
          const cameraEntries = buildCameraEntries(layer);
          return (
            <section key={layer.id} className="space-y-4">
              <header className="flex items-center gap-2 border-b border-parchment pb-1">
                <span className="inline-block w-2 h-2 rounded-full flex-shrink-0" style={{ background: layer.colour }} />
                <h3 className="font-sans text-caption uppercase tracking-wider font-medium text-burgundy">{layer.label}</h3>
                <span className="font-sans text-[10px] text-muted-foreground ml-auto">
                  {layer.latents.length} steps · seed {layer.seed} · {layer.steps} req · CFG {layer.cfg} · {layer.modelId.split("/").pop()}
                </span>
              </header>
              {cameraEntries.length > 0 && (
                <CameraRoll entries={cameraEntries} title="Camera roll · denoising progression" />
              )}
              <div>
                <Table
                  headers={TRAJ_HEADERS}
                  rows={tableRows}
                  numericColumns={[0, 1, 2, 3, 4, 5, 6, 7, 8]}
                  caption="Per-step latent geometry. The inspector below lets you expand any step for the preview thumbnail and full distribution stats."
                />
              </div>
              <StepInspector richRows={richRows} cfg={layer.cfg} />
            </section>
          );
        })}
      </div>
    </DeepDive>
  );
}

/**
 * Slider + preview panel for retrospectively inspecting any step's image
 * and stats. Default position is the latest received step; dragging
 * lets the user scrub backwards and forwards through the trajectory.
 */
function StepScrubberModal({
  captured,
  requestedSteps,
  selected,
  onChange,
  onClose,
  previews,
  stepStats,
  cfg,
}: {
  /** Number of latents captured so far (slider range = 0..captured-1). */
  captured: number;
  /** Total steps requested in the form — what the denominator should show. */
  requestedSteps: number;
  selected: number;
  onChange: (n: number) => void;
  onClose: () => void;
  previews: Array<string | null>;
  stepStats: Array<Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">>;
  cfg: number;
}) {
  const idx = Math.max(0, Math.min(captured - 1, selected));
  const preview = previews[idx];
  const stat = stepStats[idx];

  // RGB histogram for the current step's preview thumbnail. Computed
  // lazily on selection change; computeImageStats is cached per src so
  // scrubbing back to a previously-viewed step is instant.
  const [hist, setHist] = useState<{ r: number[]; g: number[]; b: number[] } | null>(null);
  useEffect(() => {
    let cancelled = false;
    setHist(null);
    if (!preview) return;
    void computeImageStats(preview)
      .then((s) => {
        if (cancelled) return;
        setHist({ r: s.histR, g: s.histG, b: s.histB });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [preview]);

  // Auto-playback: advance the selected step every `playbackMs`,
  // wrapping to 0 when we hit the last captured step. Lets the user
  // watch the denoising as a small animation rather than dragging the
  // slider by hand. Dragging the slider doesn't pause — it just seeks
  // to that index and playback continues from there.
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackMs, setPlaybackMs] = useState(250);
  useEffect(() => {
    if (!isPlaying || captured < 2) return;
    const t = setInterval(() => {
      onChange((idx + 1) % captured);
    }, playbackMs);
    return () => clearInterval(t);
    // We deliberately depend on `idx` so the next tick is computed
    // from the current selected index, not a stale snapshot.
  }, [isPlaying, captured, idx, playbackMs, onChange]);
  // If the run is still in flight or otherwise didn't capture every
  // requested step, show that explicitly so "Step 12 of 20 · capturing"
  // reads correctly mid-stream and "Step 16 of 20 (capture stopped)"
  // reads correctly after a user-initiated abort.
  const partial = captured < requestedSteps;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4"
      onClick={onClose}
    >
      <div
        className="card-editorial max-w-3xl w-full p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="font-display text-display-md font-bold text-burgundy">
              Step inspector
            </h3>
            <p className="font-sans text-caption text-muted-foreground">
              Drag the slider to retrospectively walk through every captured
              latent. Step 1 is near-pure noise; the final step is the
              destination latent decoded into the result image.
            </p>
          </div>
          <button onClick={onClose} className="btn-editorial-ghost px-2 py-1" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-4 items-start">
          <div
            className="bg-cream/40 border border-parchment rounded-sm overflow-hidden flex items-center justify-center"
            style={{ width: "200px", height: "200px" }}
          >
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt={`step ${idx + 1}`} className="w-full h-full object-cover" />
            ) : (
              <span className="font-sans text-caption text-muted-foreground text-center px-3">
                no preview at this step
                <br />
                <span className="italic">
                  (set Preview every = 1 in the form for full coverage)
                </span>
              </span>
            )}
          </div>
          <div className="min-w-0">
            <div className="flex items-center justify-between mb-2">
              <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">
                Step <span className="text-foreground">{idx + 1}</span> of {requestedSteps}
                {partial && (
                  <span className="ml-1 italic normal-case tracking-normal text-burgundy">
                    · {captured} captured
                  </span>
                )}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(0, captured - 1)}
              value={idx}
              onChange={(e) => onChange(parseInt(e.target.value, 10))}
              className="w-full"
              title={`Drag to scrub through ${captured} captured steps.`}
            />
            <div className="flex justify-between font-sans text-[10px] text-muted-foreground mt-0.5 mb-2">
              <span>step 1 · noise</span>
              <span>
                step {requestedSteps} · final
                {partial && " (not yet reached)"}
              </span>
            </div>

            {/* Play / speed controls. Auto-loops when reaching the
                final captured step. Dragging the slider while playing
                seeks but doesn't pause. */}
            <div className="flex items-center gap-3 mb-3 font-sans text-caption">
              <button
                type="button"
                onClick={() => setIsPlaying((p) => !p)}
                disabled={captured < 2}
                className={
                  isPlaying
                    ? "px-3 py-1 border border-burgundy bg-burgundy text-cream rounded-sm flex items-center gap-1.5"
                    : "btn-editorial-secondary px-3 py-1 flex items-center gap-1.5"
                }
                title={isPlaying ? "Pause auto-playback" : "Play the denoising trajectory at the chosen speed"}
              >
                {isPlaying ? (
                  <>
                    <Square size={11} fill="currentColor" /> Stop
                  </>
                ) : (
                  <>
                    <Play size={11} fill="currentColor" /> Play
                  </>
                )}
              </button>
              <label
                className="flex items-center gap-1.5 text-muted-foreground"
                title="Time between frames during auto-playback. Faster = jumpier; slower = more time per step to read scalars and the histogram."
              >
                <span className="uppercase tracking-wider cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">
                  Speed
                </span>
                <select
                  value={playbackMs}
                  onChange={(e) => setPlaybackMs(parseInt(e.target.value, 10))}
                  className="input-editorial py-0.5 text-caption"
                >
                  <option value={100}>fast (100 ms)</option>
                  <option value={250}>normal (250 ms)</option>
                  <option value={500}>slow (500 ms)</option>
                  <option value={1000}>study (1 s)</option>
                  <option value={2000}>dwell (2 s)</option>
                </select>
              </label>
              <span className="text-muted-foreground italic ml-auto">
                {isPlaying ? "looping · click Stop to pause" : "looping playback · drag slider to seek"}
              </span>
            </div>

            <dl className="grid grid-cols-[100px_1fr] gap-y-1 font-sans text-caption">
              <dt className="text-muted-foreground">CFG</dt>
              <dd className="text-foreground">{cfg}</dd>
              {stat?.timestep != null && (
                <>
                  <dt className="text-muted-foreground">timestep (t)</dt>
                  <dd className="text-foreground">{stat.timestep.toFixed(2)}</dd>
                </>
              )}
              {stat?.sigma != null && (
                <>
                  <dt className="text-muted-foreground">sigma (σ)</dt>
                  <dd className="text-foreground">{stat.sigma.toFixed(4)}</dd>
                </>
              )}
              {stat?.latentNorm != null && (
                <>
                  <dt className="text-muted-foreground">norm (‖z‖)</dt>
                  <dd className="text-foreground">{stat.latentNorm.toFixed(3)}</dd>
                </>
              )}
              {stat?.latentMean != null && (
                <>
                  <dt className="text-muted-foreground">mean (μ)</dt>
                  <dd className="text-foreground">{stat.latentMean.toFixed(4)}</dd>
                </>
              )}
              {stat?.latentStd != null && (
                <>
                  <dt className="text-muted-foreground">std</dt>
                  <dd className="text-foreground">{stat.latentStd.toFixed(4)}</dd>
                </>
              )}
              {stat?.latentMin != null && (
                <>
                  <dt className="text-muted-foreground">min</dt>
                  <dd className="text-foreground">{stat.latentMin.toFixed(4)}</dd>
                </>
              )}
              {stat?.latentMax != null && (
                <>
                  <dt className="text-muted-foreground">max</dt>
                  <dd className="text-foreground">{stat.latentMax.toFixed(4)}</dd>
                </>
              )}
            </dl>

            {/* RGB histogram of the current step's preview — Photoshop /
                Lightroom style overlapping-channel curves with mix-blend
                multiply, same primitive used in the contact-sheet
                frames. Empty white box if no preview was captured for
                this step (set Preview every = 1 in the form). */}
            <div className="mt-4">
              <div className="flex items-center gap-3 font-sans text-[10px] text-muted-foreground mb-1">
                <span className="uppercase tracking-wider">RGB histogram</span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-2 h-2 rounded-sm" style={{ background: "#cd2650" }} />R
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-2 h-2 rounded-sm" style={{ background: "#3b7d4f" }} />G
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-2 h-2 rounded-sm" style={{ background: "#2e5d8a" }} />B
                </span>
                <span className="ml-auto italic">
                  {preview ? "of the decoded preview" : "no preview at this step"}
                </span>
              </div>
              <RgbHistogramMini width={400} height={80} hist={hist} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * 35mm-style film strip: every captured step in order, scrollable
 * horizontally, with per-step scalars under each frame. Sprockets and
 * background styled to evoke a contact sheet rather than a UI panel.
 */
function buildFilmReelSvg(
  previews: Array<string | null>,
  latents: Float32Array[],
  stepStats: Array<Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">>,
  cfg: number,
  prompt: string,
  modelId: string,
  seed: number,
): string {
  // Layout constants. Match the on-screen FilmStrip proportions but render
  // at higher fidelity (each frame 220 px wide).
  const FRAME_W = 220;
  const FRAME_H = 220;
  const CAPTION_H = 110;
  const GAP = 12;
  const PAD_X = 16;
  const SPROCKET_H = 14;
  const SPROCKET_W = 14;
  const SPROCKET_GAP = 14;
  const TITLE_H = 30;
  const FOOTER_H = 22;

  const innerH = SPROCKET_H + 8 + FRAME_H + 6 + CAPTION_H + 8 + SPROCKET_H;
  const W = PAD_X * 2 + latents.length * (FRAME_W + GAP) - GAP;
  const H = TITLE_H + innerH + FOOTER_H;

  const frames = latents
    .map((_, i) => {
      const x = PAD_X + i * (FRAME_W + GAP);
      const y = TITLE_H + SPROCKET_H + 8;
      const preview = previews[i];
      const stat = stepStats[i] ?? {};

      // Frame image (or "no preview" placeholder).
      const imageEl = preview
        ? `<image href="${escXml(preview)}" x="${x}" y="${y}" width="${FRAME_W}" height="${FRAME_H}" preserveAspectRatio="xMidYMid slice" />`
        : `<rect x="${x}" y="${y}" width="${FRAME_W}" height="${FRAME_H}" fill="#1a1a1a" />` +
          `<text x="${x + FRAME_W / 2}" y="${y + FRAME_H / 2}" fill="#666" font-family="ui-monospace, monospace" font-size="10" text-anchor="middle" dominant-baseline="middle">no preview</text>`;
      // Frame border (slight inset highlight).
      const border = `<rect x="${x}" y="${y}" width="${FRAME_W}" height="${FRAME_H}" fill="none" stroke="#0a0a0a" stroke-width="2" />`;

      // Caption box below the frame.
      const cy = y + FRAME_H + 6;
      const captionBg = `<rect x="${x}" y="${cy}" width="${FRAME_W}" height="${CAPTION_H}" fill="#0d0d0d" rx="2" />`;

      // Caption rows.
      const lines: Array<{ k: string; v: string }> = [
        { k: "step", v: String(i + 1) },
      ];
      if (stat.timestep != null) lines.push({ k: "t", v: stat.timestep.toFixed(0) });
      if (stat.sigma != null) lines.push({ k: "σ", v: stat.sigma.toFixed(2) });
      if (stat.latentNorm != null) lines.push({ k: "‖z‖", v: stat.latentNorm.toFixed(2) });
      if (stat.latentMean != null) lines.push({ k: "μ", v: stat.latentMean.toFixed(3) });
      if (stat.latentStd != null) lines.push({ k: "std", v: stat.latentStd.toFixed(2) });
      lines.push({ k: "cfg", v: String(cfg) });

      const lineH = 14;
      const captionEls = lines
        .map((line, j) => {
          const ly = cy + 14 + j * lineH;
          return (
            `<text x="${x + 10}" y="${ly}" fill="#888" font-family="ui-monospace, monospace" font-size="10">${escXml(line.k)}</text>` +
            `<text x="${x + FRAME_W - 10}" y="${ly}" fill="#d4d4d4" font-family="ui-monospace, monospace" font-size="10" text-anchor="end">${escXml(line.v)}</text>`
          );
        })
        .join("");

      return imageEl + border + captionBg + captionEls;
    })
    .join("");

  // Sprocket holes — top + bottom rows.
  const sprocketCount = Math.max(latents.length * 2 + 4, 16);
  const sprocketSpacing = (W - PAD_X * 2) / sprocketCount;
  const sprocketsTop = Array.from({ length: sprocketCount })
    .map((_, i) => {
      const sx = PAD_X + i * sprocketSpacing;
      const sy = TITLE_H + 0;
      return `<rect x="${sx}" y="${sy}" width="${SPROCKET_W}" height="${SPROCKET_H * 0.7}" fill="#222" rx="1" />`;
    })
    .join("");
  const sprocketsBot = Array.from({ length: sprocketCount })
    .map((_, i) => {
      const sx = PAD_X + i * sprocketSpacing;
      const sy = TITLE_H + innerH - SPROCKET_H * 0.7;
      return `<rect x="${sx}" y="${sy}" width="${SPROCKET_W}" height="${SPROCKET_H * 0.7}" fill="#222" rx="1" />`;
    })
    .join("");

  const title =
    `<text x="${PAD_X}" y="${TITLE_H - 10}" fill="#a36b3a" font-family="ui-monospace, monospace" font-size="11" letter-spacing="3">DIFFUSION ATLAS · CONTACT SHEET</text>` +
    `<text x="${W - PAD_X}" y="${TITLE_H - 10}" fill="#888" font-family="ui-monospace, monospace" font-size="10" text-anchor="end">${escXml(modelId)} · seed ${escXml(seed)} · ${latents.length} frames</text>`;

  const footer =
    `<text x="${PAD_X}" y="${H - 6}" fill="#666" font-family="ui-monospace, monospace" font-size="9">${escXml(prompt)}</text>` +
    `<text x="${W - PAD_X}" y="${H - 6}" fill="#666" font-family="ui-monospace, monospace" font-size="9" text-anchor="end">vector-lab-tools.github.io</text>`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="sans-serif">
  <rect width="${W}" height="${H}" fill="#111" />
  ${title}
  ${sprocketsTop}
  ${frames}
  ${sprocketsBot}
  ${footer}
</svg>`;
}

function FilmStrip({
  previews,
  latents,
  stepStats,
  cfg,
  prompt,
  modelId,
  seed,
}: {
  previews: Array<string | null>;
  latents: Float32Array[];
  stepStats: Array<Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">>;
  cfg: number;
  prompt: string;
  modelId: string;
  seed: number;
}) {
  const [openFrame, setOpenFrame] = useState<{ index: number; entry: CameraRollEntry } | null>(null);

  function exportReelSvg() {
    const svg = buildFilmReelSvg(previews, latents, stepStats, cfg, prompt, modelId, seed);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    downloadSvg(`film-reel-seed${seed}-${stamp}.svg`, svg);
  }
  async function exportReelPng() {
    const svg = buildFilmReelSvg(previews, latents, stepStats, cfg, prompt, modelId, seed);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    // 2× supersample so the rasterised reel reads sharply on Retina /
    // print. Bump to 3 for genuinely-print-quality output.
    await downloadPngFromSvg(`film-reel-seed${seed}-${stamp}.png`, svg, 2);
  }
  // Build CameraRollEntry for a frame, used when its metadata block is clicked.
  function entryForFrame(i: number): CameraRollEntry | null {
    const url = previews[i];
    if (!url) return null;
    const stat = stepStats[i] ?? {};
    const details: Array<{ label: string; value: string | number }> = [
      { label: "Step", value: i + 1 },
      { label: "CFG", value: `${cfg} (${cfgCaption(cfg)})` },
      { label: "Seed", value: seed },
      { label: "Prompt", value: prompt },
    ];
    if (stat.timestep != null) details.push({ label: "Timestep", value: stat.timestep.toFixed(2) });
    if (stat.sigma != null) details.push({ label: "Sigma", value: stat.sigma.toFixed(4) });
    if (stat.latentNorm != null) details.push({ label: "‖z‖", value: stat.latentNorm.toFixed(4) });
    if (stat.latentMean != null) details.push({ label: "Mean", value: stat.latentMean.toFixed(4) });
    if (stat.latentStd != null) details.push({ label: "Std", value: stat.latentStd.toFixed(4) });
    if (stat.latentMin != null) details.push({ label: "Min", value: stat.latentMin.toFixed(4) });
    if (stat.latentMax != null) details.push({ label: "Max", value: stat.latentMax.toFixed(4) });
    return { src: url, caption: `step ${i + 1}`, subcaption: "preview · click for stats", details };
  }

  const FRAME_PX = 150;
  const COL_GAP_PX = 12;
  // Joint width keeps the dark strip and white metadata strip aligned column-by-column.
  const totalCols = latents.length;
  const stripMinWidth = totalCols * FRAME_PX + (totalCols - 1) * COL_GAP_PX;

  return (
    <div className="rounded-sm border border-parchment overflow-hidden">
      {/* Export button — sits on the page surface, above the dark film strip. */}
      <div className="bg-card px-3 py-1.5 flex items-center justify-between border-b border-parchment">
        <span
          className="font-mono text-[9px] uppercase tracking-[0.18em]"
          style={{ color: "#a36b3a" }}
        >
          DIFFUSION ATLAS · CONTACT SHEET
        </span>
        <span className="inline-flex items-stretch gap-1">
          <button
            onClick={exportReelSvg}
            className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1 border border-parchment-dark hover:border-burgundy rounded-sm px-2 py-0.5"
            title="Download this film reel as a self-contained SVG. Embeds all preview thumbnails and per-step scalars. Best for editorial layout — vector, scales to any size, but some apps refuse SVG imports."
          >
            <Download size={10} /> svg
          </button>
          <button
            onClick={() => void exportReelPng()}
            className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1 border border-parchment-dark hover:border-burgundy rounded-sm px-2 py-0.5"
            title="Download this film reel as a 2× rasterised PNG. Universal compatibility — every app imports PNG. Pixel-grained at the reel's native size; sharpness depends on the source thumbnail resolution."
          >
            <Download size={10} /> png
          </button>
        </span>
      </div>

      {/* Single horizontally-scrolling container so dark strip and white metadata
          strip stay column-aligned while scrolling. */}
      <div className="overflow-x-auto">
        <div style={{ minWidth: `${stripMinWidth + 24}px` }}>
          {/* Dark film strip: sprockets + frames + sprockets only. No numbers. */}
          <div className="bg-[#111] py-2">
            {/* Top sprocket row */}
            <div className="flex gap-3 px-3 mb-2">
              {Array.from({ length: Math.max(latents.length * 2, 12) }).map((_, i) => (
                <span
                  key={`top-${i}`}
                  className="block w-3 h-2 bg-[#222] rounded-[1px] flex-shrink-0"
                />
              ))}
            </div>

            <div className="px-3">
              <div className="flex items-start" style={{ gap: `${COL_GAP_PX}px` }}>
                {latents.map((_, i) => {
                  const preview = previews[i];
                  return (
                    <div
                      key={i}
                      className="flex-shrink-0 bg-black border-2 border-[#0a0a0a] rounded-sm overflow-hidden"
                      style={{ width: `${FRAME_PX}px`, height: `${FRAME_PX}px` }}
                    >
                      {preview ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={preview} alt={`step ${i + 1}`} className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-[10px] text-[#666] font-mono uppercase tracking-wider">
                          no preview
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Bottom sprocket row */}
            <div className="flex gap-3 px-3 mt-2">
              {Array.from({ length: Math.max(latents.length * 2, 12) }).map((_, i) => (
                <span
                  key={`bot-${i}`}
                  className="block w-3 h-2 bg-[#222] rounded-[1px] flex-shrink-0"
                />
              ))}
            </div>
          </div>

          {/* White edge-print metadata strip — below the film, on the page. */}
          <div className="bg-card px-3 py-2 border-t border-parchment">
            <div className="flex items-start" style={{ gap: `${COL_GAP_PX}px` }}>
              {latents.map((_, i) => (
                <FilmFrameMetadata
                  key={i}
                  width={FRAME_PX}
                  step={i + 1}
                  stat={stepStats[i]}
                  cfg={cfg}
                  preview={previews[i] ?? null}
                  onOpen={() => {
                    const e = entryForFrame(i);
                    if (e) setOpenFrame({ index: i, entry: e });
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {openFrame && (
        <FilmFrameModal
          openFrame={openFrame}
          onClose={() => setOpenFrame(null)}
          onPrev={() => {
            for (let j = openFrame.index - 1; j >= 0; j--) {
              const e = entryForFrame(j);
              if (e) {
                setOpenFrame({ index: j, entry: e });
                return;
              }
            }
          }}
          onNext={() => {
            for (let j = openFrame.index + 1; j < latents.length; j++) {
              const e = entryForFrame(j);
              if (e) {
                setOpenFrame({ index: j, entry: e });
                return;
              }
            }
          }}
          totalFrames={latents.length}
          hasPrev={Array.from({ length: openFrame.index }).some((_, j) => previews[j])}
          hasNext={previews.slice(openFrame.index + 1).some(Boolean)}
        />
      )}
    </div>
  );
}

/**
 * Click-to-expand per-step inspector. Each row shows a one-line summary;
 * clicking expands a panel with the preview thumbnail (if any) and the
 * full set of per-step scalars side by side.
 */
function StepInspector({
  richRows,
  cfg,
}: {
  richRows: Array<{
    step: number;
    norm: number;
    stepSize: number | null;
    cosToFinal: number;
    cosToStart: number;
    timestep: number | null | undefined;
    sigma: number | null | undefined;
    latentMean: number | undefined;
    latentStd: number | undefined;
    latentMin: number | undefined;
    latentMax: number | undefined;
    preview: string | null;
  }>;
  cfg: number;
}) {
  const [expanded, setExpanded] = useState<number | null>(null);
  if (richRows.length === 0) return null;
  return (
    <div>
      <h4 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
        Per-step inspector
      </h4>
      <div className="border border-parchment rounded-sm divide-y divide-parchment">
        {richRows.map((r) => {
          const isOpen = expanded === r.step;
          return (
            <div key={r.step}>
              <button
                onClick={() => setExpanded(isOpen ? null : r.step)}
                className={`w-full text-left px-3 py-1.5 flex items-center gap-3 font-sans text-caption hover:bg-cream/40 transition-colors ${isOpen ? "bg-cream/60" : ""}`}
              >
                <span className="font-medium text-foreground w-12">step {r.step}</span>
                <span className="text-muted-foreground w-24">cos→final {r.cosToFinal.toFixed(2)}</span>
                <span className="text-muted-foreground w-20">‖z‖ {r.norm.toFixed(2)}</span>
                {r.timestep != null && <span className="text-muted-foreground w-20">t {r.timestep.toFixed(0)}</span>}
                {r.sigma != null && <span className="text-muted-foreground w-20">σ {r.sigma.toFixed(2)}</span>}
                {r.preview && <span className="text-burgundy text-[10px] ml-auto">preview ↓</span>}
              </button>
              {isOpen && (
                <div className="px-3 py-3 bg-cream/30 grid grid-cols-1 sm:grid-cols-[120px_1fr] gap-3">
                  {r.preview ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={r.preview}
                      alt={`step ${r.step}`}
                      className="w-30 h-30 object-cover rounded-sm border border-parchment"
                      style={{ width: "120px", height: "120px" }}
                    />
                  ) : (
                    <div className="w-30 h-30 bg-cream/60 border border-parchment rounded-sm flex items-center justify-center font-sans text-caption text-muted-foreground" style={{ width: "120px", height: "120px" }}>
                      no preview
                    </div>
                  )}
                  <dl className="grid grid-cols-2 gap-x-4 gap-y-1 font-sans text-caption">
                    <dt className="text-muted-foreground">step</dt><dd className="text-foreground">{r.step}</dd>
                    <dt className="text-muted-foreground">CFG</dt><dd className="text-foreground">{cfg}</dd>
                    {r.timestep != null && <><dt className="text-muted-foreground" title="Scheduler timestep value at this step.">timestep</dt><dd className="text-foreground">{r.timestep.toFixed(2)}</dd></>}
                    {r.sigma != null && <><dt className="text-muted-foreground" title="Noise level the model is targeting at this step.">sigma</dt><dd className="text-foreground">{r.sigma.toFixed(4)}</dd></>}
                    <dt className="text-muted-foreground" title="L2 norm of the per-step latent.">‖z‖</dt><dd className="text-foreground">{r.norm.toFixed(4)}</dd>
                    <dt className="text-muted-foreground" title="L2 distance from the previous step's latent.">Δ to prev</dt><dd className="text-foreground">{r.stepSize != null ? r.stepSize.toFixed(4) : "—"}</dd>
                    <dt className="text-muted-foreground" title="Cosine similarity to the final latent.">cos→final</dt><dd className="text-foreground">{r.cosToFinal.toFixed(4)}</dd>
                    <dt className="text-muted-foreground" title="Cosine similarity to the first step's latent.">cos→start</dt><dd className="text-foreground">{r.cosToStart.toFixed(4)}</dd>
                    {r.latentMean != null && <><dt className="text-muted-foreground" title="Mean of latent values.">mean</dt><dd className="text-foreground">{r.latentMean.toFixed(4)}</dd></>}
                    {r.latentStd != null && <><dt className="text-muted-foreground" title="Standard deviation across latent values.">std</dt><dd className="text-foreground">{r.latentStd.toFixed(4)}</dd></>}
                    {r.latentMin != null && <><dt className="text-muted-foreground" title="Min latent value.">min</dt><dd className="text-foreground">{r.latentMin.toFixed(4)}</dd></>}
                    {r.latentMax != null && <><dt className="text-muted-foreground" title="Max latent value.">max</dt><dd className="text-foreground">{r.latentMax.toFixed(4)}</dd></>}
                  </dl>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * White edge-print metadata block sitting under each film frame. Black mono
 * type on cream/white, like text printed under a contact sheet. Includes a
 * tiny overlapping-channel RGB histogram (Photoshop-style) computed lazily
 * from the preview thumbnail. Clicking opens the full frame modal — same
 * larger view we use elsewhere in the app.
 */
function FilmFrameMetadata({
  width,
  step,
  stat,
  cfg,
  preview,
  onOpen,
}: {
  width: number;
  step: number;
  stat: Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl"> | undefined;
  cfg: number;
  preview: string | null;
  onOpen: () => void;
}) {
  const [hist, setHist] = useState<{ r: number[]; g: number[]; b: number[] } | null>(null);
  useEffect(() => {
    let cancelled = false;
    if (!preview) return;
    void computeImageStats(preview).then((s) => {
      if (cancelled) return;
      setHist({ r: s.histR, g: s.histG, b: s.histB });
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [preview]);

  return (
    <button
      onClick={onOpen}
      disabled={!preview}
      className="flex-shrink-0 bg-white border border-parchment-dark rounded-sm hover:border-burgundy hover:shadow-editorial transition-all text-left disabled:opacity-60 disabled:cursor-default disabled:hover:border-parchment-dark disabled:hover:shadow-none"
      style={{ width: `${width}px` }}
      title={preview ? "Click for full preview + image stats" : "No preview captured at this step"}
    >
      <div className="px-1.5 py-1 font-mono text-[9px] text-black leading-tight">
        <div className="flex items-baseline justify-between mb-0.5">
          <span className="font-bold">#{step}</span>
          <span className="text-[8px] text-neutral-500">cfg {cfg}</span>
        </div>
        <div className="grid grid-cols-2 gap-x-1 gap-y-[1px]">
          {stat?.timestep != null && (<><span className="text-neutral-500">t</span><span className="text-right">{stat.timestep.toFixed(0)}</span></>)}
          {stat?.sigma != null && (<><span className="text-neutral-500">σ</span><span className="text-right">{stat.sigma.toFixed(2)}</span></>)}
          {stat?.latentNorm != null && (<><span className="text-neutral-500">‖z‖</span><span className="text-right">{stat.latentNorm.toFixed(1)}</span></>)}
          {stat?.latentMean != null && (<><span className="text-neutral-500">μ</span><span className="text-right">{stat.latentMean.toFixed(2)}</span></>)}
          {stat?.latentStd != null && (<><span className="text-neutral-500">std</span><span className="text-right">{stat.latentStd.toFixed(2)}</span></>)}
        </div>
        <div className="mt-1">
          <RgbHistogramMini width={width - 12} height={28} hist={hist} />
        </div>
      </div>
    </button>
  );
}

/**
 * Tiny overlapping-channel RGB histogram, drawn as three semi-transparent
 * filled polylines on a near-white background. Mirrors the Photoshop /
 * Lightroom style. Renders a placeholder rectangle until stats arrive.
 */
function RgbHistogramMini({
  width,
  height,
  hist,
}: {
  width: number;
  height: number;
  hist: { r: number[]; g: number[]; b: number[] } | null;
}) {
  if (!hist) {
    return (
      <div
        className="bg-neutral-100 border border-neutral-200 rounded-[2px]"
        style={{ width, height }}
      />
    );
  }
  const bins = hist.r.length;
  const max = Math.max(
    ...hist.r, ...hist.g, ...hist.b,
    1,
  );
  const stepX = width / bins;
  const path = (channel: number[]) => {
    const pts: string[] = [`0,${height}`];
    for (let i = 0; i < bins; i++) {
      const x = i * stepX;
      const y = height - (channel[i] / max) * height;
      pts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
    }
    pts.push(`${width},${height}`);
    return pts.join(" ");
  };
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="block bg-neutral-50 border border-neutral-200 rounded-[2px]"
      style={{ mixBlendMode: "multiply" }}
    >
      <polygon points={path(hist.r)} fill="rgba(220, 38, 38, 0.55)" stroke="rgba(180, 20, 20, 0.9)" strokeWidth="0.5" />
      <polygon points={path(hist.g)} fill="rgba(34, 160, 80, 0.55)" stroke="rgba(20, 120, 60, 0.9)" strokeWidth="0.5" />
      <polygon points={path(hist.b)} fill="rgba(37, 99, 235, 0.55)" stroke="rgba(20, 60, 180, 0.9)" strokeWidth="0.5" />
    </svg>
  );
}

/**
 * Thin wrapper around the shared FrameModal — opens a big version of a
 * film-strip frame with full stats + RGB histogram, with prev/next
 * navigation between frames that have previews.
 */
function FilmFrameModal({
  openFrame,
  onClose,
  onPrev,
  onNext,
  totalFrames,
  hasPrev,
  hasNext,
}: {
  openFrame: { index: number; entry: CameraRollEntry };
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  totalFrames: number;
  hasPrev: boolean;
  hasNext: boolean;
}) {
  return (
    <FrameModal
      entry={openFrame.entry}
      onClose={onClose}
      onPrev={hasPrev ? onPrev : undefined}
      onNext={hasNext ? onNext : undefined}
      index={openFrame.index}
      total={totalFrames}
    />
  );
}
