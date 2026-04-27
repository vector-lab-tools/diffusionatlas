"use client";

import { useEffect, useRef, useState } from "react";
import { UMAP } from "umap-js";
import { X, Plus, Lock, Unlock } from "lucide-react";
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
import { downloadSvg, escXml } from "@/lib/export/svg";
import { Download } from "lucide-react";
import { lookup as lookupTerm, termsFor } from "@/lib/docs/glossary";
import { PromptChips, STARTER_PRESETS } from "@/components/shared/PromptChips";
import { RandomSeedButton, nextSeed, type SeedMode } from "@/components/shared/RandomSeedButton";

type ProjectionKind = "pca" | "umap" | "film";

type StepStat = Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">;

interface TrajectoryLayer {
  id: string;
  label: string;
  colour: string;
  visible: boolean;
  /**
   * `false` = a temporary layer that will be replaced by the next run.
   * `true`  = the user has locked this layer in place; subsequent runs
   *           leave it alone and add their own (initially temporary)
   *           layer alongside it.
   */
  locked: boolean;
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
  // 12 is the modern "fast and good" sweet spot for SD 1.5 with the
  // DPMSolverMultistepScheduler ("DPM++ 2M Karras") the backend pins —
  // converges noticeably faster than Euler or PNDM, so 12 steps now
  // produces a coherent image where 12 with Euler would be mushy. Bump
  // to 20-30 for research-grade fidelity once you know what you're after.
  const [steps, setSteps] = useState(12);
  const [cfg, setCfg] = useState(7.5);

  const [previewEvery, setPreviewEvery] = useState(1);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

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
  const [savedLayers, setSavedLayers] = useState<TrajectoryLayer[]>([]);

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

  async function run() {
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
      });
    } catch (err) {
      setError(`Cannot reach local backend at ${settings.localBaseUrl}. Is uvicorn running? (${err instanceof Error ? err.message : String(err)})`);
      setRunning(false);
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
      streamErr = err instanceof Error ? err.message : String(err);
    }

    if (streamErr) {
      setError(streamErr);
      setRunning(false);
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

    // Auto-save the just-finished run as a temporary (unlocked) layer.
    // It will be replaced if the user runs again, unless they lock it
    // first. Computes its own projection so the layer is renderable
    // without depending on the live state vars.
    if (collected.length >= 2 && finalUrl) {
      const newLayer: TrajectoryLayer = {
        id: `layer-${Date.now()}`,
        label: `${prompt.slice(0, 32)}${prompt.length > 32 ? "…" : ""} · seed ${activeSeed}`,
        // Temporary layers always use a neutral ink colour so locked
        // layers retain their bright palette colours.
        colour: "#1a1a1a",
        visible: true,
        locked: false,
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
      setSavedLayers((prev) => [newLayer, ...prev.filter((l) => l.locked)]);
    }

    setRunning(false);
  }

  function updateLayer(id: string, patch: Partial<TrajectoryLayer>) {
    setSavedLayers((prev) => prev.map((l) => (l.id === id ? { ...l, ...patch } : l)));
  }

  function deleteLayer(id: string) {
    setSavedLayers((prev) => prev.filter((l) => l.id !== id));
  }

  /**
   * Lock a layer in place: gives it a stable colour from the palette so
   * future temp runs (which use neutral ink) don't visually clash, and
   * flips its `locked` flag.
   */
  function lockLayer(id: string) {
    setSavedLayers((prev) => {
      // Pick the next palette colour based on how many already-locked
      // layers we have, so colours cycle predictably.
      const lockedCount = prev.filter((l) => l.locked).length;
      return prev.map((l) =>
        l.id === id ? { ...l, locked: true, colour: nextColour(lockedCount) } : l,
      );
    });
  }

  function unlockLayer(id: string) {
    setSavedLayers((prev) => prev.map((l) => (l.id === id ? { ...l, locked: false, colour: "#1a1a1a" } : l)));
  }

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
            gridTemplateColumns:
              "minmax(220px, 1.6fr) 72px 72px 88px minmax(108px, 1fr) minmax(108px, 1fr)",
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
            <input
              type="number"
              step="0.5"
              value={cfg}
              onChange={(e) => setCfg(parseFloat(e.target.value) || 0)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block" title="Decode a thumbnail every N steps. 1 = capture every step (richer Deep Dive + step scrubber, ~0.5s extra per step). 0 disables previews entirely. Higher values make the run faster at the cost of less data in the inspector.">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Preview every</span>
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
      </div>

      {savedLayers.length > 0 && (
        <div className="border border-parchment rounded-sm bg-cream/20 p-3 mb-4">
          <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
            Layers · {savedLayers.filter((l) => l.locked).length} locked
            {savedLayers.some((l) => !l.locked) && <span className="ml-1">· 1 temp</span>}
            <span className="ml-2 italic normal-case tracking-normal">
              click the padlock to keep a temp layer; locked layers stay through future runs.
            </span>
          </h3>
          <div className="space-y-1.5">
            {savedLayers.map((layer) => (
              <div
                key={layer.id}
                className={
                  "flex items-center gap-2 text-caption rounded-sm " +
                  (layer.locked
                    ? ""
                    : "border border-dashed border-parchment-dark bg-cream/40 px-1 py-0.5")
                }
              >
                <input
                  type="checkbox"
                  checked={layer.visible}
                  onChange={(e) => updateLayer(layer.id, { visible: e.target.checked })}
                  title="Show / hide this layer"
                />
                <span
                  className="inline-block w-3 h-3 rounded-full flex-shrink-0"
                  style={{ background: layer.colour }}
                  title={layer.locked ? "Locked layer · stays across runs" : "Temporary layer · will be replaced by the next run"}
                />
                <input
                  type="text"
                  value={layer.label}
                  onChange={(e) => updateLayer(layer.id, { label: e.target.value })}
                  className={"input-editorial py-0.5 text-caption flex-1 min-w-0" + (layer.locked ? "" : " italic")}
                />
                <span className="font-sans text-caption text-muted-foreground">
                  {!layer.locked && <span className="text-burgundy not-italic mr-2 font-medium uppercase tracking-wider text-[10px]">temp</span>}
                  {layer.latents.length} steps · seed {layer.seed} · {layer.modelId.split("/").pop()}
                </span>
                <button
                  onClick={() => (layer.locked ? unlockLayer(layer.id) : lockLayer(layer.id))}
                  className={
                    layer.locked
                      ? "btn-editorial-ghost p-1 text-burgundy"
                      : "btn-editorial-ghost p-1 text-muted-foreground hover:text-burgundy"
                  }
                  title={layer.locked
                    ? "Locked: this layer stays across future runs · click to unlock (will be removed on next run)"
                    : "Temporary: this layer will be replaced by the next run · click to lock it in place"}
                  aria-pressed={layer.locked}
                  aria-label={layer.locked ? "Unlock layer" : "Lock layer"}
                >
                  {layer.locked ? <Lock size={12} /> : <Unlock size={12} />}
                </button>
                <button
                  onClick={() => deleteLayer(layer.id)}
                  className="btn-editorial-ghost p-1"
                  title="Delete layer"
                >
                  <X size={12} />
                </button>
              </div>
            ))}
            <button
              onClick={() => setSavedLayers([])}
              className="font-sans text-caption text-burgundy hover:text-burgundy-900 underline underline-offset-2 mt-1"
            >
              Clear all layers
            </button>
          </div>
        </div>
      )}

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <button
          onClick={() => void run()}
          disabled={running || !isLocal}
          className={running || !isLocal ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running
            ? statusMsg && progress === null
              ? statusMsg
              : `Streaming step ${progress?.done ?? 0}/${progress?.total ?? steps}…`
            : `Run trajectory (${steps} steps)`}
        </button>
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

          {/* Step scrubber: drag to retrospectively inspect any step. */}
          {latents.length > 1 && (
            <StepScrubber
              total={latents.length}
              selected={selectedStep ?? latents.length - 1}
              onChange={setSelectedStep}
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
      { label: "CFG", value: layer.cfg },
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
        { label: "CFG", value: layer.cfg },
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
                { label: "CFG", value: headLayer.cfg },
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
function StepScrubber({
  total,
  selected,
  onChange,
  previews,
  stepStats,
  cfg,
}: {
  total: number;
  selected: number;
  onChange: (n: number) => void;
  previews: Array<string | null>;
  stepStats: Array<Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">>;
  cfg: number;
}) {
  const idx = Math.max(0, Math.min(total - 1, selected));
  const preview = previews[idx];
  const stat = stepStats[idx];
  return (
    <div className="border border-parchment rounded-sm bg-cream/20 p-3 mb-3 grid grid-cols-1 sm:grid-cols-[60px_1fr] gap-3 items-center">
      <div className="bg-cream/40 border border-parchment rounded-sm overflow-hidden flex items-center justify-center" style={{ width: "60px", height: "60px" }}>
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt={`step ${idx + 1}`} className="w-full h-full object-cover" />
        ) : (
          <span className="font-sans text-[10px] text-muted-foreground text-center px-2">
            no preview at this step (set Preview every = 1 for full coverage)
          </span>
        )}
      </div>
      <div className="min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">
            Step <span className="text-foreground">{idx + 1}</span> of {total}
          </span>
          <div className="flex items-center gap-1.5 font-sans text-[10px]">
            {stat?.timestep != null && <span className="text-muted-foreground">t <span className="text-foreground">{stat.timestep.toFixed(0)}</span></span>}
            {stat?.sigma != null && <span className="text-muted-foreground">σ <span className="text-foreground">{stat.sigma.toFixed(2)}</span></span>}
            {stat?.latentNorm != null && <span className="text-muted-foreground">‖z‖ <span className="text-foreground">{stat.latentNorm.toFixed(1)}</span></span>}
            {stat?.latentMean != null && <span className="text-muted-foreground">μ <span className="text-foreground">{stat.latentMean.toFixed(2)}</span></span>}
            {stat?.latentStd != null && <span className="text-muted-foreground">std <span className="text-foreground">{stat.latentStd.toFixed(2)}</span></span>}
            <span className="text-muted-foreground">cfg <span className="text-foreground">{cfg}</span></span>
          </div>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, total - 1)}
          value={idx}
          onChange={(e) => onChange(parseInt(e.target.value, 10))}
          className="w-full"
          title={`Drag to scrub through ${total} captured steps. Step 1 is near pure noise; step ${total} is the final latent.`}
        />
        <div className="flex justify-between font-sans text-[9px] text-muted-foreground mt-0.5">
          <span>step 1 · noise</span>
          <span>step {total} · final</span>
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

  function exportReel() {
    const svg = buildFilmReelSvg(previews, latents, stepStats, cfg, prompt, modelId, seed);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    downloadSvg(`film-reel-seed${seed}-${stamp}.svg`, svg);
  }
  // Build CameraRollEntry for a frame, used when its metadata block is clicked.
  function entryForFrame(i: number): CameraRollEntry | null {
    const url = previews[i];
    if (!url) return null;
    const stat = stepStats[i] ?? {};
    const details: Array<{ label: string; value: string | number }> = [
      { label: "Step", value: i + 1 },
      { label: "CFG", value: cfg },
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
        <button
          onClick={exportReel}
          className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1 border border-parchment-dark hover:border-burgundy rounded-sm px-2 py-0.5"
          title="Download this film reel as a self-contained SVG. Embeds all preview thumbnails and per-step scalars."
        >
          <Download size={10} /> export reel · svg
        </button>
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
