"use client";

import { useEffect, useState } from "react";
import { X, Trash2 } from "lucide-react";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import { CameraRoll, type CameraRollEntry } from "@/components/shared/CameraRoll";

interface CachedImagesModalProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Browse every image in the IndexedDB image cache. Keys are structured
 * `kind::providerId::modelId::seed::steps::cfg::timestamp` (or
 * `traj::local::modelId::seed::steps::cfg::timestamp` for trajectory finals,
 * `bench::providerId::modelId::taskId::seed::timestamp` for bench), so we
 * parse them best-effort to surface a useful caption + details.
 */
export function CachedImagesModal({ open, onClose }: CachedImagesModalProps) {
  const { list, count, approximateBytes, clear } = useImageBlobCache();
  const [entries, setEntries] = useState<CameraRollEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const objectUrls: string[] = [];
    setLoading(true);
    void (async () => {
      try {
        const items = await list();
        // Newest first — keys end with a timestamp.
        items.sort((a, b) => b.key.localeCompare(a.key));
        const built = items.map(({ key, blob }) => {
          const url = URL.createObjectURL(blob);
          objectUrls.push(url);
          return entryFromKey(key, url, blob.size);
        });
        if (!cancelled) {
          setEntries(built);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setLoading(false);
          console.error("Failed to list cached images:", err);
        }
      }
    })();
    return () => {
      cancelled = true;
      objectUrls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [open, list]);

  if (!open) return null;

  async function handleClear() {
    if (!confirm(`Delete all ${count} cached images? This cannot be undone.`)) return;
    await clear();
    setEntries([]);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/50 p-4" onClick={onClose}>
      <div
        className="card-editorial w-full max-w-5xl max-h-[90vh] overflow-y-auto p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="font-display text-display-md font-bold text-burgundy">Cached images</h2>
            <p className="font-sans text-caption text-muted-foreground">
              Every image generated this session, kept locally in IndexedDB.{" "}
              {count} total · approximately {formatBytes(approximateBytes)} on disk.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {count > 0 && (
              <button
                onClick={() => void handleClear()}
                className="btn-editorial-secondary px-3 py-1 text-caption flex items-center gap-1"
                title="Delete every cached image"
              >
                <Trash2 size={12} /> Clear all
              </button>
            )}
            <button onClick={onClose} className="btn-editorial-ghost px-2 py-1" aria-label="Close">
              <X size={16} />
            </button>
          </div>
        </div>

        {loading ? (
          <p className="font-sans text-caption text-muted-foreground italic">Loading cached images…</p>
        ) : entries.length === 0 ? (
          <p className="font-sans text-caption text-muted-foreground italic">No cached images yet.</p>
        ) : (
          <CameraRoll entries={entries} title="" />
        )}
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function entryFromKey(key: string, url: string, blobSize: number): CameraRollEntry {
  // Best-effort parse of the structured key.
  const parts = key.split("::");
  const kind = parts[0] ?? "image";
  const details: Array<{ label: string; value: string | number }> = [
    { label: "Kind", value: kind },
    { label: "Size", value: formatBytes(blobSize) },
  ];

  let caption = kind;
  let subcaption = "";

  if (kind === "sweep" || kind === "sweep-cmp") {
    // sweep::provider::model::seed::steps::cfg::timestamp
    if (parts[1]) details.push({ label: "Provider", value: parts[1] });
    if (parts[2]) details.push({ label: "Model", value: parts[2] });
    if (parts[3]) details.push({ label: "Seed", value: parts[3] });
    if (parts[4]) details.push({ label: "Steps", value: parts[4] });
    if (parts[5]) details.push({ label: "CFG", value: parts[5] });
    if (parts[6]) details.push({ label: "When", value: formatTimestamp(parts[6]) });
    caption = `CFG ${parts[5] ?? "?"}`;
    subcaption = `${kind === "sweep-cmp" ? "compare · " : ""}${parts[2]?.split("/").pop() ?? ""} · seed ${parts[3] ?? "?"}`;
  } else if (kind === "nbr" || kind === "nbr-cmp") {
    // nbr::provider::model::seed::steps::cfg::timestamp
    if (parts[1]) details.push({ label: "Provider", value: parts[1] });
    if (parts[2]) details.push({ label: "Model", value: parts[2] });
    if (parts[3]) details.push({ label: "Seed", value: parts[3] });
    if (parts[4]) details.push({ label: "Steps", value: parts[4] });
    if (parts[5]) details.push({ label: "CFG", value: parts[5] });
    if (parts[6]) details.push({ label: "When", value: formatTimestamp(parts[6]) });
    caption = `seed ${parts[3] ?? "?"}`;
    subcaption = `${kind === "nbr-cmp" ? "compare · " : "neighbourhood · "}${parts[2]?.split("/").pop() ?? ""}`;
  } else if (kind === "bench" || kind === "bench-cmp") {
    // bench::provider::model::taskId::seed::timestamp
    if (parts[1]) details.push({ label: "Provider", value: parts[1] });
    if (parts[2]) details.push({ label: "Model", value: parts[2] });
    if (parts[3]) details.push({ label: "Task", value: parts[3] });
    if (parts[4]) details.push({ label: "Seed", value: parts[4] });
    if (parts[5]) details.push({ label: "When", value: formatTimestamp(parts[5]) });
    caption = parts[3] ?? "task";
    subcaption = `${kind === "bench-cmp" ? "compare · " : "bench · "}${parts[2]?.split("/").pop() ?? ""}`;
  } else if (kind === "traj") {
    // traj::local::model::seed::steps::cfg::timestamp
    if (parts[1]) details.push({ label: "Provider", value: parts[1] });
    if (parts[2]) details.push({ label: "Model", value: parts[2] });
    if (parts[3]) details.push({ label: "Seed", value: parts[3] });
    if (parts[4]) details.push({ label: "Steps", value: parts[4] });
    if (parts[5]) details.push({ label: "CFG", value: parts[5] });
    if (parts[6]) details.push({ label: "When", value: formatTimestamp(parts[6]) });
    caption = "trajectory final";
    subcaption = `${parts[2]?.split("/").pop() ?? ""} · seed ${parts[3] ?? "?"}`;
  } else {
    // Generic: just show parts.
    parts.slice(1).forEach((p, i) => details.push({ label: `Part ${i + 1}`, value: p }));
    caption = kind;
    subcaption = parts.slice(1, 3).join(" · ");
  }

  return { src: url, caption, subcaption, details };
}

function formatTimestamp(s: string): string {
  const n = parseInt(s, 10);
  if (!Number.isFinite(n)) return s;
  return new Date(n).toLocaleString();
}
