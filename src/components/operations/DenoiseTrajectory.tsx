"use client";

import { useEffect, useState } from "react";
import { UMAP } from "umap-js";
import { X, Plus } from "lucide-react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import { saveRun } from "@/lib/cache/runs";
import { pca3D, type Point3 } from "@/lib/geometry/pca";
import { TrajectoryThree } from "@/components/viz/TrajectoryThree";
import type { Run, RunSampleRef } from "@/types/run";
import { DeepDive } from "@/components/shared/DeepDive";
import { Table } from "@/components/shared/Table";
import { ExportButtons } from "@/components/shared/ExportButtons";
import { CameraRoll } from "@/components/shared/CameraRoll";
import { downloadCsv } from "@/lib/export/csv";
import { downloadPdf } from "@/lib/export/pdf";
import { downloadJson } from "@/lib/export/json";
import { lookup as lookupTerm, termsFor } from "@/lib/docs/glossary";

type ProjectionKind = "pca" | "umap" | "film";

type StepStat = Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">;

interface TrajectoryLayer {
  id: string;
  label: string;
  colour: string;
  visible: boolean;
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

  const [prompt, setPrompt] = useState("a cat sitting on a wooden chair, photorealistic");
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(7.5);

  const [previewEvery, setPreviewEvery] = useState(4);
  const [projection, setProjection] = useState<ProjectionKind>("pca");
  // SD 1.5 was trained at 512×512 — running it at 1024 produces black images.
  // SDXL and FLUX want 1024. Default to 512 here since the local default is SD 1.5.
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);

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
          seed,
          steps,
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
      const imageKey = `traj::local::${settings.modelId}::${seed}::${steps}::${cfg}::${Date.now()}`;
      await cacheImage(imageKey, dataUrlToBlob(finalUrl));

      const samples: RunSampleRef[] = [{ imageKey, variable: "final", responseTimeMs: respMs ?? undefined }];
      const run: Run = {
        id: `traj::${Date.now()}`,
        kind: "single",
        createdAt: new Date().toISOString(),
        providerId: "local",
        modelId: settings.modelId,
        prompt,
        seed,
        steps,
        cfg,
        width: settings.defaults.width,
        height: settings.defaults.height,
        samples,
        extra: { trajectory: true, stepCount: collected.length },
      };
      await saveRun(run);
    }

    setRunning(false);
  }

  function saveCurrentAsLayer() {
    if (latents.length === 0) return;
    const id = `layer-${Date.now()}`;
    const idx = savedLayers.length;
    const layer: TrajectoryLayer = {
      id,
      label: `${prompt.slice(0, 32)}${prompt.length > 32 ? "…" : ""} · seed ${seed}`,
      colour: nextColour(idx),
      visible: true,
      prompt, seed, steps, cfg, width, height,
      modelId: settings.modelId,
      latents: latents.slice(),
      previews: previews.slice(),
      stepStats: stepStats.slice(),
      finalImage,
      responseTimeMs,
      points: points.slice(),
    };
    setSavedLayers((prev) => [...prev, layer]);
  }

  function updateLayer(id: string, patch: Partial<TrajectoryLayer>) {
    setSavedLayers((prev) => prev.map((l) => (l.id === id ? { ...l, ...patch } : l)));
  }

  function deleteLayer(id: string) {
    setSavedLayers((prev) => prev.filter((l) => l.id !== id));
  }

  // Build the "active" layer (the most recent run, possibly still in flight)
  // so render and deep-dive code can treat it the same as saved layers.
  const activeLayer: TrajectoryLayer | null =
    latents.length > 0
      ? {
          id: "active",
          label: "active run",
          colour: "#1a1a1a",
          visible: true,
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

  const visibleLayers = [
    ...savedLayers.filter((l) => l.visible),
    ...(activeLayer ? [activeLayer] : []),
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
        <label className="block">
          <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">Prompt</span>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="input-editorial mt-1"
          />
        </label>
        <div className="grid grid-cols-4 gap-3">
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
              min={2}
              max={100}
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
          <label className="block" title="Decode a thumbnail every N steps. 0 disables previews. Each thumbnail adds one VAE decode.">
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
        </div>
        <div className="grid grid-cols-2 gap-3">
          <label className="block" title="Image width in pixels. SD 1.5 was trained at 512; SDXL / FLUX expect 1024. Mismatched sizes produce black images.">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Width</span>
            <input
              type="number"
              step={64}
              min={256}
              max={2048}
              value={width}
              onChange={(e) => setWidth(parseInt(e.target.value, 10) || 512)}
              className="input-editorial mt-1"
            />
          </label>
          <label className="block" title="Image height in pixels. Match Width and the model's training resolution.">
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4">Height</span>
            <input
              type="number"
              step={64}
              min={256}
              max={2048}
              value={height}
              onChange={(e) => setHeight(parseInt(e.target.value, 10) || 512)}
              className="input-editorial mt-1"
            />
          </label>
        </div>
      </div>

      {savedLayers.length > 0 && (
        <div className="border border-parchment rounded-sm bg-cream/20 p-3 mb-4">
          <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
            Layers · {savedLayers.length} saved
            <span className="ml-2 italic normal-case tracking-normal">toggle, rename, or remove. Visible layers overlay in the 3D / Film views.</span>
          </h3>
          <div className="space-y-1.5">
            {savedLayers.map((layer) => (
              <div key={layer.id} className="flex items-center gap-2 text-caption">
                <input
                  type="checkbox"
                  checked={layer.visible}
                  onChange={(e) => updateLayer(layer.id, { visible: e.target.checked })}
                  title="Show / hide this layer"
                />
                <span
                  className="inline-block w-3 h-3 rounded-full flex-shrink-0"
                  style={{ background: layer.colour }}
                  title="Layer colour"
                />
                <input
                  type="text"
                  value={layer.label}
                  onChange={(e) => updateLayer(layer.id, { label: e.target.value })}
                  className="input-editorial py-0.5 text-caption flex-1 min-w-0"
                />
                <span className="font-sans text-caption text-muted-foreground">
                  {layer.latents.length} steps · seed {layer.seed} · {layer.modelId.split("/").pop()}
                </span>
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
        {latents.length > 0 && !running && (
          <button
            onClick={saveCurrentAsLayer}
            className="btn-editorial-secondary px-3 py-2 flex items-center gap-1"
            title="Save the current run as a comparison layer. Run another prompt and they will overlay in the 3D / Film views."
          >
            <Plus size={14} /> Save as layer
          </button>
        )}
        <span className="font-sans text-caption text-muted-foreground">
          {settings.backend === "local" ? "Local" : "Hosted"} · {settings.providerId} · {settings.modelId}
        </span>
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
          {projection === "film" && visibleLayers.length > 0 ? (
            <div className="space-y-3">
              {visibleLayers.map((l) => (
                <div key={l.id}>
                  {visibleLayers.length > 1 && (
                    <div className="font-sans text-caption text-muted-foreground mb-1 flex items-center gap-2">
                      <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ background: l.colour }} />
                      {l.label}
                    </div>
                  )}
                  <FilmStrip
                    previews={l.previews}
                    latents={l.latents}
                    stepStats={l.stepStats}
                    cfg={l.cfg}
                  />
                </div>
              ))}
            </div>
          ) : visibleLayers.some((l) => l.points.length >= 2) ? (
            <TrajectoryThree
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
          {finalImage && (
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-[1fr_240px] gap-4 items-start">
              <dl className="grid grid-cols-[120px_1fr] gap-y-1 font-sans text-caption">
                <dt className="text-muted-foreground">Provider</dt><dd className="text-foreground">local</dd>
                <dt className="text-muted-foreground">Model</dt><dd className="text-foreground">{settings.modelId}</dd>
                <dt className="text-muted-foreground">Seed</dt><dd className="text-foreground">{seed}</dd>
                <dt className="text-muted-foreground">Steps</dt><dd className="text-foreground">{steps}</dd>
                <dt className="text-muted-foreground">CFG</dt><dd className="text-foreground">{cfg}</dd>
                <dt className="text-muted-foreground">Latent dim</dt><dd className="text-foreground">{latents[0]?.length ?? 0}</dd>
              </dl>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={finalImage} alt="final" className="rounded-sm border border-parchment shadow-editorial w-full" />
            </div>
          )}
        </div>
      )}

      {latents.length > 0 && (
        <TrajectoryDeepDive
          prompt={prompt}
          seed={seed}
          steps={steps}
          cfg={cfg}
          modelId={settings.modelId}
          latents={latents}
          previews={previews}
          stepStats={stepStats}
          points={points}
          finalImage={finalImage}
          responseTimeMs={responseTimeMs}
        />
      )}
    </div>
  );
}

interface TrajectoryDeepDiveProps {
  prompt: string;
  seed: number;
  steps: number;
  cfg: number;
  modelId: string;
  latents: Float32Array[];
  previews: Array<string | null>;
  stepStats: Array<Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">>;
  points: Point3[];
  finalImage: string | null;
  responseTimeMs: number | null;
}

function TrajectoryDeepDive({ prompt, seed, steps, cfg, modelId, latents, previews, stepStats, points, finalImage, responseTimeMs }: TrajectoryDeepDiveProps) {
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

  const final = latents[latents.length - 1];
  // Per-step rows include both client-side derived fields (step size, cosine
  // to final / start) and the server-side scalars (timestep, sigma, mean,
  // std) so a researcher gets one comprehensive table.
  const richRows = latents.map((l, i) => {
    const norm = Math.sqrt(l.reduce((a, x) => a + x * x, 0));
    const stepSize = i > 0 ? l2(l, latents[i - 1]) : 0;
    const cosToFinal = cosine(l, final);
    const cosToStart = cosine(l, latents[0]);
    const stat = stepStats[i] ?? {};
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
      preview: previews[i] ?? null,
    };
  });

  const tableRows: Array<Array<string | number>> = richRows.map((r) => [
    r.step,
    r.timestep != null ? r.timestep.toFixed(1) : "—",
    r.sigma != null ? r.sigma.toFixed(3) : "—",
    r.norm.toFixed(3),
    r.stepSize != null ? r.stepSize.toFixed(3) : "—",
    r.cosToFinal.toFixed(3),
    r.cosToStart.toFixed(3),
    r.latentMean != null ? r.latentMean.toFixed(3) : "—",
    r.latentStd != null ? r.latentStd.toFixed(3) : "—",
    r.preview ? "yes" : "—",
  ]);

  const headers = ["step", "timestep", "sigma", "‖z‖", "Δ to prev", "cos→final", "cos→start", "mean", "std", "preview"];

  const stamp = `traj-${seed}-${Date.now()}`;
  function exportCsv() { downloadCsv(`${stamp}.csv`, headers, tableRows); }
  function exportJson() {
    downloadJson(`${stamp}.json`, {
      operation: "denoise-trajectory",
      modelId, prompt, seed, steps, cfg,
      latentDim: latents[0]?.length ?? 0,
      stepCount: latents.length,
      points,
      perStep: tableRows.map((r) => ({ step: r[0], norm: parseFloat(String(r[1])), deltaToPrev: r[2] === "—" ? null : parseFloat(String(r[2])), cosToFinal: parseFloat(String(r[3])), hasPreview: r[4] === "yes" })),
      responseTimeMs,
    });
  }
  function exportPdf() {
    const images: Array<{ dataUrl: string; caption?: string }> = [];
    previews.forEach((p, i) => {
      if (p) images.push({ dataUrl: p, caption: `step ${i + 1}` });
    });
    if (finalImage) images.push({ dataUrl: finalImage, caption: "final" });
    downloadPdf(`${stamp}.pdf`, {
      meta: {
        title: "Denoise Trajectory",
        subtitle: `local · ${modelId}`,
        fields: [
          { label: "Prompt", value: prompt },
          { label: "Seed", value: seed },
          { label: "Steps", value: steps },
          { label: "CFG", value: cfg },
          { label: "Latent dim", value: latents[0]?.length ?? 0 },
          ...(responseTimeMs !== null ? [{ label: "Total time", value: `${(responseTimeMs / 1000).toFixed(1)}s` }] : []),
        ],
      },
      images,
      appendix: [
        {
          title: "Per-step latent geometry",
          caption: "Δ to prev is the L2 distance between successive latents (length of each denoising step). cos→final is the cosine similarity between each step's latent and the final latent (rises from ~0 to 1 as the trajectory converges).",
          table: { headers, rows: tableRows },
        },
      ],
      glossary: termsFor(["Prompt", "Seed", "Steps", "CFG", "Preview every", "step", "‖z‖", "Δ to prev", "cos→final", "preview"]),
    });
  }

  // Camera roll: each preview thumbnail + the final image, with rich
  // per-step metadata attached so clicking a frame opens the full detail.
  const cameraEntries: Array<{
    src: string;
    caption?: string;
    subcaption?: string;
    details?: Array<{ label: string; value: string | number }>;
  }> = [];
  previews.forEach((url, i) => {
    if (!url) return;
    const stat = stepStats[i] ?? {};
    const details: Array<{ label: string; value: string | number }> = [
      { label: "Step", value: i + 1 },
      { label: "CFG", value: cfg },
    ];
    if (stat.timestep != null) details.push({ label: "Timestep", value: stat.timestep.toFixed(2) });
    if (stat.sigma != null) details.push({ label: "Sigma", value: stat.sigma.toFixed(4) });
    if (stat.latentNorm != null) details.push({ label: "‖z‖", value: stat.latentNorm.toFixed(4) });
    if (stat.latentMean != null) details.push({ label: "Mean", value: stat.latentMean.toFixed(4) });
    if (stat.latentStd != null) details.push({ label: "Std", value: stat.latentStd.toFixed(4) });
    if (stat.latentMin != null) details.push({ label: "Min", value: stat.latentMin.toFixed(4) });
    if (stat.latentMax != null) details.push({ label: "Max", value: stat.latentMax.toFixed(4) });
    cameraEntries.push({
      src: url,
      caption: `step ${i + 1}`,
      subcaption: "preview · click for stats",
      details,
    });
  });
  if (finalImage) {
    cameraEntries.push({
      src: finalImage,
      caption: "final",
      subcaption: `${steps} steps`,
      details: [
        { label: "Prompt", value: prompt },
        { label: "Seed", value: seed },
        { label: "Steps", value: steps },
        { label: "CFG", value: cfg },
        { label: "Model", value: modelId },
        ...(responseTimeMs !== null ? [{ label: "Time", value: `${(responseTimeMs / 1000).toFixed(1)}s` }] : []),
      ],
    });
  }

  return (
    <DeepDive actions={<ExportButtons onCsv={exportCsv} onPdf={exportPdf} onJson={exportJson} />}>
      <div className="space-y-6">
        {cameraEntries.length > 0 && <CameraRoll entries={cameraEntries} title="Camera roll · denoising progression" />}
        <div>
          <Table
            headers={headers}
            rows={tableRows}
            numericColumns={[0, 1, 2, 3, 4, 5, 6, 7, 8]}
            caption="Per-step latent geometry. Click any step row below for richer inspection (preview thumbnail, full distribution stats)."
          />
        </div>
        <StepInspector richRows={richRows} cfg={cfg} />
      </div>
    </DeepDive>
  );
}

/**
 * 35mm-style film strip: every captured step in order, scrollable
 * horizontally, with per-step scalars under each frame. Sprockets and
 * background styled to evoke a contact sheet rather than a UI panel.
 */
function FilmStrip({
  previews,
  latents,
  stepStats,
  cfg,
}: {
  previews: Array<string | null>;
  latents: Float32Array[];
  stepStats: Array<Omit<StepEvent, "event" | "shape" | "latentB64" | "previewDataUrl">>;
  cfg: number;
}) {
  return (
    <div className="rounded-sm border border-parchment overflow-hidden bg-[#111] py-3">
      {/* Top sprocket holes */}
      <div className="flex gap-3 px-3 mb-2">
        {Array.from({ length: Math.max(latents.length * 2, 12) }).map((_, i) => (
          <span
            key={`top-${i}`}
            className="block w-3 h-2 bg-[#222] rounded-[1px] flex-shrink-0"
          />
        ))}
      </div>

      <div className="overflow-x-auto px-3">
        <div className="flex gap-3 items-start">
          {latents.map((_, i) => {
            const preview = previews[i];
            const stat = stepStats[i];
            return (
              <div
                key={i}
                className="flex-shrink-0 flex flex-col items-stretch"
                style={{ width: "150px" }}
              >
                <div className="bg-black aspect-square border-2 border-[#0a0a0a] rounded-sm overflow-hidden flex items-center justify-center">
                  {preview ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={preview} alt={`step ${i + 1}`} className="w-full h-full object-cover" />
                  ) : (
                    <div className="text-[10px] text-[#666] font-mono uppercase tracking-wider">
                      no preview
                    </div>
                  )}
                </div>
                <div className="bg-[#0d0d0d] text-[#d4d4d4] font-mono text-[10px] leading-tight px-2 py-1.5 mt-1 rounded-sm">
                  <div className="flex justify-between">
                    <span className="text-[#888]">step</span>
                    <span>{i + 1}</span>
                  </div>
                  {stat?.timestep != null && (
                    <div className="flex justify-between">
                      <span className="text-[#888]">t</span>
                      <span>{stat.timestep.toFixed(0)}</span>
                    </div>
                  )}
                  {stat?.sigma != null && (
                    <div className="flex justify-between">
                      <span className="text-[#888]">σ</span>
                      <span>{stat.sigma.toFixed(2)}</span>
                    </div>
                  )}
                  {stat?.latentNorm != null && (
                    <div className="flex justify-between">
                      <span className="text-[#888]">‖z‖</span>
                      <span>{stat.latentNorm.toFixed(2)}</span>
                    </div>
                  )}
                  {stat?.latentMean != null && (
                    <div className="flex justify-between">
                      <span className="text-[#888]">μ</span>
                      <span>{stat.latentMean.toFixed(3)}</span>
                    </div>
                  )}
                  {stat?.latentStd != null && (
                    <div className="flex justify-between">
                      <span className="text-[#888]">std</span>
                      <span>{stat.latentStd.toFixed(2)}</span>
                    </div>
                  )}
                  <div className="flex justify-between mt-0.5">
                    <span className="text-[#666]">cfg</span>
                    <span className="text-[#888]">{cfg}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom sprocket holes */}
      <div className="flex gap-3 px-3 mt-2">
        {Array.from({ length: Math.max(latents.length * 2, 12) }).map((_, i) => (
          <span
            key={`bot-${i}`}
            className="block w-3 h-2 bg-[#222] rounded-[1px] flex-shrink-0"
          />
        ))}
      </div>

      <div className="px-3 mt-2 font-mono text-[9px] text-[#555] uppercase tracking-wider flex items-center justify-between">
        <span>diffusion-atlas · contact sheet</span>
        <span>{latents.length} frames · scroll →</span>
      </div>
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
