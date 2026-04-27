"use client";

import { useMemo, useState } from "react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { DiffusionRequest, DiffusionResultMeta, ProviderId } from "@/lib/providers/types";
import {
  CATEGORIES,
  TASKS,
  type BenchTask,
  type TaskCategoryId,
} from "@/lib/bench/tasks";
import { saveRun } from "@/lib/cache/runs";
import type { Run, RunSampleRef } from "@/types/run";
import { ALL_PROVIDERS, PROVIDER_DEFAULT_MODEL, providerLabel } from "@/lib/providers/defaults";
import { DeepDive } from "@/components/shared/DeepDive";
import { Table } from "@/components/shared/Table";
import { ExportButtons } from "@/components/shared/ExportButtons";
import { downloadCsv } from "@/lib/export/csv";
import { downloadPdf } from "@/lib/export/pdf";
import { downloadJson } from "@/lib/export/json";

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

function freshRows(): BenchRow[] {
  return TASKS.map((task) => ({ task, status: "pending" as RowStatus, verdict: null }));
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
  const { settings } = useSettings();
  const { set: cacheImage } = useImageBlobCache();

  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(settings.defaults.steps);
  const [rows, setRows] = useState<BenchRow[]>(freshRows);
  const [rowsB, setRowsB] = useState<BenchRow[]>(freshRows);
  const [running, setRunning] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [clipThreshold, setClipThreshold] = useState(0.25);
  const [topError, setTopError] = useState<string | null>(null);
  const [errorLink, setErrorLink] = useState<{ href: string; label: string } | null>(null);

  // Cross-backend comparison state
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

  async function callOnce(prompt: string, cfg_: ProviderConfig): Promise<{ ok: boolean; data?: DiffuseResponse; err?: GenError; status?: number }> {
    const request: DiffusionRequest = {
      modelId: cfg_.modelId,
      prompt,
      seed,
      steps,
      cfg: settings.defaults.cfg,
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
        const imageKey = `${keyPrefix}::${r.data.meta.providerId}::${r.data.meta.modelId}::${task.id}::${seed}::${Date.now()}`;
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
    setter(freshRows());
    const setRowAt = makeSetRowAt(setter);
    const samples: RunSampleRef[] = [];
    let providerIdSeen: string | undefined;
    let modelIdSeen: string | undefined;
    let aborted = false;
    for (let i = 0; i < TASKS.length; i++) {
      if (aborted) {
        setter((prev) => prev.map((r, j) => (j >= i ? { ...r, status: "error", errorMessage: "Skipped" } : r)));
        break;
      }
      setRowAt(i, { status: "running" });
      const ref: { key?: string; meta?: DiffusionResultMeta } = {};
      const { abortAll } = await runOne(i, TASKS[i], cfg_, setRowAt, keyPrefix, ref);
      if (ref.key && ref.meta) {
        samples.push({ imageKey: ref.key, variable: TASKS[i].id, responseTimeMs: ref.meta.responseTimeMs });
        providerIdSeen = ref.meta.providerId;
        modelIdSeen = ref.meta.modelId;
      }
      if (abortAll) aborted = true;
      if (i < TASKS.length - 1) await new Promise((res) => setTimeout(res, 1500));
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
      seed,
      steps,
      cfg: settings.defaults.cfg,
      width: settings.defaults.width,
      height: settings.defaults.height,
      samples,
      extra: { taskCount: TASKS.length, categoryCount: CATEGORIES.length, ...(lane ? { lane } : {}) },
    };
    await saveRun(run);
  }

  async function runBench() {
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
      setRowsB(freshRows());
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
          onClick={() => void runBench()}
          disabled={running}
          className={running ? "btn-editorial-secondary opacity-50" : "btn-editorial-primary"}
        >
          {running
            ? "Running bench…"
            : compareEnabled
              ? `Run bench × 2 (${TASKS.length * 2} images)`
              : `Run bench (${TASKS.length} images)`}
        </button>
        <button
          onClick={() => void autoScore()}
          disabled={running || scoring}
          className={running || scoring ? "btn-editorial-secondary opacity-50" : "btn-editorial-secondary"}
          title="Score every generated image against its prompt with CLIP (local backend required)"
        >
          {scoring ? "Scoring…" : "Auto-score (CLIP)"}
        </button>
        <label className="flex items-center gap-2 font-sans text-caption text-muted-foreground">
          Threshold
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
      />
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
              {row.clipScore !== undefined && (
                <div className="font-sans text-caption text-muted-foreground">
                  CLIP score:{" "}
                  <span className={row.clipScore >= clipThreshold ? "text-burgundy font-medium" : "text-foreground"}>
                    {row.clipScore.toFixed(3)}
                  </span>
                  <span className="ml-1">
                    ({row.clipScore >= clipThreshold ? "≥" : "<"} {clipThreshold.toFixed(2)})
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
}

function BenchDeepDive({ seed, steps, clipThreshold, primaryLabel, rowsA, scoresA, compareEnabled, compareLabel, rowsB, scoresB }: BenchDeepDiveProps) {
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
    downloadPdf(`${stamp}.pdf`, {
      meta: {
        title: "Compositional Bench",
        subtitle: compareEnabled ? `${primaryLabel}  ↔  ${compareLabel}` : primaryLabel,
        fields: [
          { label: "Seed", value: seed },
          { label: "Steps", value: steps },
          { label: "CLIP threshold", value: clipThreshold.toFixed(2) },
          { label: "Tasks", value: TASKS.length },
        ],
      },
      table: { headers: ["lane", ...taskHeaders], rows: taskTableRows },
      images,
    });
  }

  if (rowsA.every((r) => r.status === "pending")) return null;

  return (
    <DeepDive actions={<ExportButtons onCsv={exportCsv} onPdf={exportPdf} onJson={exportJson} />}>
      <div className="space-y-6">
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
      </div>
    </DeepDive>
  );
}
