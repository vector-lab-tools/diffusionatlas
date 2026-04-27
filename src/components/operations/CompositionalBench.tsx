"use client";

import { useMemo, useState } from "react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { DiffusionRequest, DiffusionResultMeta } from "@/lib/providers/types";
import {
  CATEGORIES,
  TASKS,
  type BenchTask,
  type TaskCategoryId,
} from "@/lib/bench/tasks";

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
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? "image/png";
  const bytes = atob(b64);
  const buf = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
  return new Blob([buf], { type: mime });
}

export function CompositionalBench() {
  const { settings } = useSettings();
  const { set: cacheImage } = useImageBlobCache();

  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(settings.defaults.steps);
  const [rows, setRows] = useState<BenchRow[]>(() =>
    TASKS.map((task) => ({ task, status: "pending" as RowStatus, verdict: null })),
  );
  const [running, setRunning] = useState(false);
  const [topError, setTopError] = useState<string | null>(null);
  const [errorLink, setErrorLink] = useState<{ href: string; label: string } | null>(null);

  async function callOnce(prompt: string): Promise<{ ok: boolean; data?: DiffuseResponse; err?: GenError; status?: number }> {
    const request: DiffusionRequest = {
      modelId: settings.modelId,
      prompt,
      seed,
      steps,
      cfg: settings.defaults.cfg,
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
        return { ok: false, err, status: res.status };
      }
      const data: DiffuseResponse = await res.json();
      return { ok: true, data };
    } catch (err) {
      return { ok: false, err: { error: "network", message: err instanceof Error ? err.message : String(err) } };
    }
  }

  function setRowAt(idx: number, patch: Partial<BenchRow>) {
    setRows((prev) => prev.map((r, j) => (j === idx ? { ...r, ...patch } : r)));
  }

  async function runOne(idx: number): Promise<{ abortAll?: boolean }> {
    const task = rows[idx].task;
    const MAX_ATTEMPTS = 4;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const r = await callOnce(task.prompt);
      if (r.ok && r.data) {
        const dataUrl = r.data.images[0];
        const runId = `bench::${r.data.meta.providerId}::${r.data.meta.modelId}::${task.id}::${seed}::${Date.now()}`;
        await cacheImage(runId, dataUrlToBlob(dataUrl));
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

  async function runBench() {
    setRunning(true);
    setTopError(null);
    setErrorLink(null);
    setRows((prev) => prev.map((r) => ({ ...r, status: "pending" as RowStatus, verdict: null, imageDataUrl: undefined, meta: undefined, errorMessage: undefined })));

    let aborted = false;
    for (let i = 0; i < TASKS.length; i++) {
      if (aborted) {
        setRows((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Skipped" } : r)));
        break;
      }
      setRowAt(i, { status: "running" });
      const { abortAll } = await runOne(i);
      if (abortAll) aborted = true;
      if (i < TASKS.length - 1) await new Promise((res) => setTimeout(res, 1500));
    }
    setRunning(false);
  }

  function setVerdict(idx: number, verdict: Verdict) {
    setRowAt(idx, { verdict });
  }

  // Per-category and overall scores, computed live as user marks pass/fail.
  const scores = useMemo(() => {
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
  }, [rows]);

  const overallPct = scores.totalScored === 0 ? null : (scores.totalPass / scores.totalScored) * 100;

  return (
    <div className="card-editorial p-6 max-w-5xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">Compositional Bench</h2>
      <p className="font-body text-body-sm text-foreground mb-4">
        GenEval-style scored tasks: single object, two objects, counting, colour binding. Generate the pack at a fixed seed, mark each result pass or fail; per-category accuracy aggregates live.
      </p>

      <div className="border border-cream/80 bg-cream/30 text-foreground p-3 mb-4 font-sans text-caption rounded-sm">
        Scoring is manual in v0.1 (you mark pass/fail per image). CLIP-based auto-scoring is queued for v0.2 once the local backend lands.
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
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
        <div className="block">
          <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground">Pack size</span>
          <div className="font-sans text-body-sm mt-2">{TASKS.length} tasks · {CATEGORIES.length} categories</div>
        </div>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={() => void runBench()}
          disabled={running}
          className={running ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running ? "Running bench…" : `Run bench (${TASKS.length} images)`}
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

      {/* Score panel */}
      {scores.totalScored > 0 && (
        <div className="border-t border-parchment pt-4 mb-4">
          <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
            Score
          </h3>
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

      {/* Tasks */}
      <div className="border-t border-parchment pt-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {rows.map((row, i) => (
            <div key={row.task.id} className="border border-parchment rounded-sm overflow-hidden flex flex-col">
              <div className="aspect-square bg-cream/50 flex items-center justify-center">
                {row.status === "ok" && row.imageDataUrl && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={row.imageDataUrl} alt={row.task.prompt} className="w-full h-full object-cover" />
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
              <div className="p-2 flex-1 flex flex-col gap-2">
                <div className="font-sans text-caption uppercase tracking-wider text-muted-foreground">
                  {CATEGORIES.find((c) => c.id === row.task.category)?.label}
                </div>
                <div className="font-body text-body-sm text-foreground line-clamp-2">{row.task.prompt}</div>
                <div className="font-sans text-caption text-muted-foreground italic line-clamp-2">{row.task.criterion}</div>
                <div className="flex gap-2 mt-auto">
                  <button
                    onClick={() => setVerdict(i, row.verdict === "pass" ? null : "pass")}
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
                    onClick={() => setVerdict(i, row.verdict === "fail" ? null : "fail")}
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
    </div>
  );
}
