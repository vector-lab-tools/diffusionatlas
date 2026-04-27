"use client";

import { useEffect, useState } from "react";
import { UMAP } from "umap-js";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import { saveRun } from "@/lib/cache/runs";
import { pca3D, type Point3 } from "@/lib/geometry/pca";
import { TrajectoryThree } from "@/components/viz/TrajectoryThree";
import type { Run, RunSampleRef } from "@/types/run";
import { DeepDive } from "@/components/shared/DeepDive";
import { Table } from "@/components/shared/Table";
import { ExportButtons } from "@/components/shared/ExportButtons";
import { downloadCsv } from "@/lib/export/csv";
import { downloadPdf } from "@/lib/export/pdf";
import { downloadJson } from "@/lib/export/json";
import { lookup as lookupTerm, termsFor } from "@/lib/docs/glossary";

type ProjectionKind = "pca" | "umap";

interface StartEvent { event: "start"; meta: Record<string, unknown> }
interface StepEvent { event: "step"; step: number; totalSteps: number; shape: number[]; latentB64: string; previewDataUrl?: string }
interface DoneEvent { event: "done"; imageDataUrl: string; responseTimeMs: number }
interface ErrorEvent { event: "error"; message: string }

type StreamEvent = StartEvent | StepEvent | DoneEvent | ErrorEvent;

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

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [latents, setLatents] = useState<Float32Array[]>([]);
  const [previews, setPreviews] = useState<Array<string | null>>([]);
  const [points, setPoints] = useState<Point3[]>([]);
  const [finalImage, setFinalImage] = useState<string | null>(null);
  const [responseTimeMs, setResponseTimeMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    setLatents([]);
    setPreviews([]);
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
          width: settings.defaults.width,
          height: settings.defaults.height,
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
          if (event.event === "step") {
            collected.push(decodeLatent(event.latentB64));
            collectedPreviews.push(event.previewDataUrl ?? null);
            setProgress({ done: event.step, total: event.totalSteps });
            setLatents([...collected]);
            setPreviews([...collectedPreviews]);
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

  return (
    <div className="card-editorial p-6 max-w-5xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">Denoise Trajectory</h2>
      <p className="font-body text-body-sm text-foreground mb-1">
        Trace the iterative denoising path through latent space. The local backend streams per-step latents; the client projects them to 3D via PCA (or UMAP) and renders the curve in Three.js with the start (gold) and end (burgundy) marked.
      </p>
      <p className="font-sans text-caption italic text-muted-foreground mb-4">
        Read the curve as: gold marker is pure noise; burgundy is the final image. Bends in the curve are where the denoising made a "decision" about composition. Thumbnails along the path show the image emerging from noise — the gap between an early-step blur and the final image is where most of the generative work happens.
      </p>

      {!isLocal && (
        <div className="border border-burgundy/40 bg-burgundy/5 text-burgundy p-3 mb-4 font-sans text-body-sm rounded-sm">
          Local FastAPI backend required. Per-step latents are not exposed by hosted providers.{" "}
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
            <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">Preview every</span>
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
      </div>

      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={() => void run()}
          disabled={running || !isLocal}
          className={running || !isLocal ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running ? `Streaming step ${progress?.done ?? 0}/${progress?.total ?? steps}…` : `Run trajectory (${steps} steps)`}
        </button>
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
          <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
            <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground">
              Trajectory · {latents.length} step{latents.length !== 1 ? "s" : ""} captured
            </h3>
            <div className="flex items-center gap-2">
              <span className="font-sans text-caption text-muted-foreground">Projection</span>
              <button
                onClick={() => setProjection("pca")}
                className={projection === "pca" ? "btn-editorial-primary px-3 py-1 text-caption" : "btn-editorial-secondary px-3 py-1 text-caption"}
              >
                PCA
              </button>
              <button
                onClick={() => setProjection("umap")}
                className={projection === "umap" ? "btn-editorial-primary px-3 py-1 text-caption" : "btn-editorial-secondary px-3 py-1 text-caption"}
                title={latents.length < 4 ? "UMAP needs ≥4 steps; falls back to PCA" : undefined}
              >
                UMAP
              </button>
              {responseTimeMs !== null && (
                <span className="font-sans text-caption text-muted-foreground ml-2">
                  {(responseTimeMs / 1000).toFixed(1)}s total
                </span>
              )}
            </div>
          </div>
          {points.length >= 2 ? (
            <TrajectoryThree points={points} previews={previews} />
          ) : (
            <div className="border border-parchment bg-cream/30 p-8 text-center font-sans text-body-sm text-muted-foreground rounded-sm">
              {running ? "Receiving latents… 3D render appears once the stream completes." : "Need at least 2 steps to render a trajectory."}
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
  points: Point3[];
  finalImage: string | null;
  responseTimeMs: number | null;
}

function TrajectoryDeepDive({ prompt, seed, steps, cfg, modelId, latents, previews, points, finalImage, responseTimeMs }: TrajectoryDeepDiveProps) {
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
  const tableRows: Array<Array<string | number>> = latents.map((l, i) => {
    const norm = Math.sqrt(l.reduce((a, x) => a + x * x, 0));
    const stepSize = i > 0 ? l2(l, latents[i - 1]) : 0;
    const cosToFinal = cosine(l, final);
    return [
      i + 1,
      norm.toFixed(3),
      i > 0 ? stepSize.toFixed(3) : "—",
      cosToFinal.toFixed(3),
      previews[i] ? "yes" : "—",
    ];
  });

  const headers = ["step", "‖z‖", "Δ to prev", "cos→final", "preview"];

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

  return (
    <DeepDive actions={<ExportButtons onCsv={exportCsv} onPdf={exportPdf} onJson={exportJson} />}>
      <Table headers={headers} rows={tableRows} numericColumns={[0, 1, 2, 3]} caption="Per-step latent geometry. Δ to prev is the L2 step length; cos→final shows how directionally similar each step is to the final latent (rises from ~0 to 1 as denoising progresses)." />
    </DeepDive>
  );
}
