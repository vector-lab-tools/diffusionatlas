"use client";

import { useState, type ReactNode } from "react";
import { X } from "lucide-react";

export interface CameraRollEntry {
  src: string;
  caption?: string;
  /** Optional secondary caption (smaller, shown under the first). */
  subcaption?: string;
  /** Free-form metadata shown when the thumbnail is clicked. */
  details?: Array<{ label: string; value: string | number | ReactNode }>;
}

interface CameraRollProps {
  title?: string;
  entries: CameraRollEntry[];
  /** Aspect ratio of each thumbnail. Default 'square'. */
  aspect?: "square" | "auto";
}

/**
 * Compact thumbnail grid for the Deep Dive — every generated image from a
 * run laid out side-by-side with provider/model/seed/cfg captions. Click
 * any thumbnail to open a modal with the full-size image plus the rich
 * metadata for that frame.
 */
export function CameraRoll({ title = "Camera roll", entries, aspect = "square" }: CameraRollProps) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  if (entries.length === 0) {
    return (
      <div className="font-sans text-caption text-muted-foreground italic">
        No images yet — run the operation to populate the camera roll.
      </div>
    );
  }

  const open = openIdx !== null ? entries[openIdx] : null;

  return (
    <div>
      <h4 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
        {title} <span className="text-foreground">· {entries.length} {entries.length === 1 ? "image" : "images"}</span>
        <span className="ml-2 italic normal-case tracking-normal text-muted-foreground">click any frame for full-size + metadata</span>
      </h4>
      <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-2">
        {entries.map((entry, i) => (
          <button
            key={i}
            onClick={() => setOpenIdx(i)}
            className="flex flex-col text-left group"
            title={`${entry.caption ?? ""}${entry.subcaption ? " · " + entry.subcaption : ""}`}
          >
            <div className={`${aspect === "square" ? "aspect-square" : ""} bg-cream/50 border border-parchment rounded-sm overflow-hidden group-hover:border-burgundy transition-colors`}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={entry.src} alt={entry.caption ?? `frame ${i}`} className="w-full h-full object-cover" />
            </div>
            {(entry.caption || entry.subcaption) && (
              <div className="mt-1 font-sans text-[10px] text-muted-foreground text-center leading-tight">
                {entry.caption && <div className="text-foreground">{entry.caption}</div>}
                {entry.subcaption && <div>{entry.subcaption}</div>}
              </div>
            )}
          </button>
        ))}
      </div>

      {open && openIdx !== null && (
        <FrameModal
          entry={open}
          onClose={() => setOpenIdx(null)}
          onPrev={openIdx > 0 ? () => setOpenIdx(openIdx - 1) : undefined}
          onNext={openIdx < entries.length - 1 ? () => setOpenIdx(openIdx + 1) : undefined}
          index={openIdx}
          total={entries.length}
        />
      )}
    </div>
  );
}

interface FrameModalProps {
  entry: CameraRollEntry;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  index: number;
  total: number;
}

function FrameModal({ entry, onClose, onPrev, onNext, index, total }: FrameModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/60 p-4" onClick={onClose}>
      <div
        className="card-editorial max-w-4xl w-full max-h-[90vh] overflow-y-auto p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-3">
          <div>
            {entry.caption && (
              <h3 className="font-display text-display-md font-bold text-burgundy">{entry.caption}</h3>
            )}
            {entry.subcaption && (
              <p className="font-sans text-caption text-muted-foreground">{entry.subcaption}</p>
            )}
            <p className="font-sans text-caption text-muted-foreground mt-1">
              Frame {index + 1} of {total}
            </p>
          </div>
          <button onClick={onClose} className="btn-editorial-ghost px-2 py-1" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-[1fr_220px] gap-4">
          <div className="bg-cream/40 border border-parchment rounded-sm overflow-hidden">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={entry.src} alt={entry.caption ?? "frame"} className="w-full h-auto" />
          </div>

          {entry.details && entry.details.length > 0 ? (
            <dl className="grid grid-cols-[80px_1fr] gap-y-1 font-sans text-caption">
              {entry.details.map((d, i) => (
                <div key={i} className="contents">
                  <dt className="text-muted-foreground">{d.label}</dt>
                  <dd className="text-foreground break-words">{d.value}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="font-sans text-caption text-muted-foreground italic">
              No additional metadata for this frame.
            </p>
          )}
        </div>

        <div className="flex items-center justify-between mt-3 pt-3 border-t border-parchment">
          <button
            onClick={onPrev}
            disabled={!onPrev}
            className={onPrev ? "btn-editorial-secondary px-3 py-1 text-caption" : "btn-editorial-secondary px-3 py-1 text-caption opacity-30 cursor-not-allowed"}
          >
            ← previous
          </button>
          <button
            onClick={onNext}
            disabled={!onNext}
            className={onNext ? "btn-editorial-secondary px-3 py-1 text-caption" : "btn-editorial-secondary px-3 py-1 text-caption opacity-30 cursor-not-allowed"}
          >
            next →
          </button>
        </div>
      </div>
    </div>
  );
}
