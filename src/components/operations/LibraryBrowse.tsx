"use client";

import { useEffect, useMemo, useState } from "react";
import { Trash2 } from "lucide-react";
import { listRuns, deleteRun, clearRuns } from "@/lib/cache/runs";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import type { Run, RunKind } from "@/types/run";

const KIND_LABEL: Record<RunKind, string> = {
  sweep: "Guidance Sweep",
  neighbourhood: "Latent Neighbourhood",
  bench: "Compositional Bench",
  single: "Single Generate",
};

function formatRelative(iso: string): string {
  const then = new Date(iso).getTime();
  const now = Date.now();
  const seconds = Math.round((now - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

interface RunCardProps {
  run: Run;
  onDelete: (id: string) => void;
}

function RunCard({ run, onDelete }: RunCardProps) {
  const { get } = useImageBlobCache();
  const [thumbs, setThumbs] = useState<Array<string | null>>([]);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const urls: string[] = [];
    void (async () => {
      const next: Array<string | null> = await Promise.all(
        run.samples.map(async (s) => {
          const blob = await get(s.imageKey);
          if (!blob) return null;
          const url = URL.createObjectURL(blob);
          urls.push(url);
          return url;
        }),
      );
      if (!cancelled) setThumbs(next);
    })();
    return () => {
      cancelled = true;
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [run.id, get]);

  const visibleSamples = expanded ? thumbs : thumbs.slice(0, 6);

  return (
    <div className="border border-parchment rounded-sm p-3 bg-card">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex-1 min-w-0">
          <div className="font-sans text-caption uppercase tracking-wider text-muted-foreground">
            {KIND_LABEL[run.kind]} · {run.samples.length} {run.samples.length === 1 ? "image" : "images"}
          </div>
          <div className="font-body text-body-sm text-foreground line-clamp-2 mt-0.5">
            {run.prompt}
          </div>
          <div className="font-sans text-caption text-muted-foreground mt-1">
            {formatRelative(run.createdAt)} · {run.providerId} · {run.modelId}
            <span> · seed {run.seed} · steps {run.steps}</span>
            {run.kind === "sweep" && run.extra && Array.isArray(run.extra.cfgList) && (
              <span> · cfg [{(run.extra.cfgList as number[]).join(", ")}]</span>
            )}
            {run.kind === "neighbourhood" && run.extra && (
              <span> · k {String(run.extra.k)} · radius {String(run.extra.radius)}</span>
            )}
          </div>
        </div>
        <button
          onClick={() => onDelete(run.id)}
          className="btn-editorial-ghost px-2 py-1 flex-shrink-0"
          title="Delete run"
        >
          <Trash2 size={14} />
        </button>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
        {visibleSamples.map((url, i) => (
          <div key={i} className="aspect-square bg-cream/50 border border-parchment rounded-sm overflow-hidden">
            {url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={url} alt={`sample ${i}`} className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center font-sans text-caption text-muted-foreground">
                missing
              </div>
            )}
          </div>
        ))}
      </div>

      {thumbs.length > 6 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 font-sans text-caption text-burgundy hover:text-burgundy-900"
        >
          {expanded ? "Show fewer" : `Show all ${thumbs.length}`}
        </button>
      )}
    </div>
  );
}

export function LibraryBrowse() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);

  async function refresh() {
    setLoading(true);
    const all = await listRuns();
    setRuns(all);
    setLoading(false);
  }

  useEffect(() => {
    void refresh();
  }, []);

  const grouped = useMemo(() => {
    const out: Record<RunKind, Run[]> = { sweep: [], neighbourhood: [], bench: [], single: [] };
    for (const r of runs) out[r.kind].push(r);
    return out;
  }, [runs]);

  async function handleDelete(id: string) {
    await deleteRun(id);
    void refresh();
  }

  async function handleClearAll() {
    if (!confirm("Delete all saved runs? Image blobs in the image cache are not removed.")) return;
    await clearRuns();
    void refresh();
  }

  return (
    <div className="card-editorial p-6 max-w-5xl">
      <div className="flex items-start justify-between mb-2">
        <div>
          <h2 className="font-display text-display-md font-bold text-burgundy">Library</h2>
          <p className="font-body text-body-sm text-foreground mt-1">
            Saved runs from Sweep, Neighbourhood, and Bench. Stored locally in IndexedDB; nothing leaves the browser.
          </p>
        </div>
        {runs.length > 0 && (
          <button onClick={() => void handleClearAll()} className="btn-editorial-secondary px-3 py-2 text-caption">
            Clear all
          </button>
        )}
      </div>

      {loading ? (
        <div className="font-sans text-caption text-muted-foreground py-6">Loading…</div>
      ) : runs.length === 0 ? (
        <div className="border-t border-parchment pt-6 mt-4">
          <p className="font-body text-body-sm text-muted-foreground">
            No saved runs yet. Run a Guidance Sweep, Latent Neighbourhood, or Compositional Bench and they will appear here.
          </p>
        </div>
      ) : (
        <div className="border-t border-parchment pt-4 space-y-6">
          {(["sweep", "neighbourhood", "bench", "single"] as RunKind[]).map((kind) => {
            const list = grouped[kind];
            if (list.length === 0) return null;
            return (
              <section key={kind}>
                <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
                  {KIND_LABEL[kind]} ({list.length})
                </h3>
                <div className="grid grid-cols-1 gap-3">
                  {list.map((run) => (
                    <RunCard key={run.id} run={run} onDelete={handleDelete} />
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}
