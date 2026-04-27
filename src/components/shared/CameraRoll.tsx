"use client";

import { useEffect, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import { computeImageStats, type ImageStats } from "@/lib/image/stats";

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
        className="card-editorial max-w-3xl w-full max-h-[90vh] overflow-y-auto p-4"
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

        <div className="grid grid-cols-1 sm:grid-cols-[280px_1fr] gap-4">
          <div className="bg-cream/40 border border-parchment rounded-sm overflow-hidden flex items-center justify-center" style={{ maxHeight: "320px" }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={entry.src} alt={entry.caption ?? "frame"} className="w-full h-auto object-contain" style={{ maxHeight: "320px" }} />
          </div>

          <div className="space-y-4">
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

            <ImageStatsPanel src={entry.src} />
          </div>
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

function ImageStatsPanel({ src }: { src: string }) {
  const [stats, setStats] = useState<ImageStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);
    computeImageStats(src)
      .then((s) => {
        if (!cancelled) {
          setStats(s);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setErr(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [src]);

  if (loading) {
    return (
      <div className="border-t border-parchment pt-3">
        <p className="font-sans text-caption text-muted-foreground italic">Computing image stats…</p>
      </div>
    );
  }
  if (err || !stats) {
    return (
      <div className="border-t border-parchment pt-3">
        <p className="font-sans text-caption text-burgundy">Stats failed: {err ?? "unknown error"}</p>
      </div>
    );
  }

  const rows: Array<{ label: string; value: ReactNode; hint: string }> = [
    {
      label: "Dimensions",
      value: `${stats.width} × ${stats.height}`,
      hint: "Pixel size of the image as analysed (capped at 256 px to keep stat computation fast).",
    },
    {
      label: "Mean RGB",
      value: (
        <span className="flex items-center gap-2">
          <span
            className="inline-block w-3 h-3 rounded-sm border border-parchment-dark/30"
            style={{ background: `rgb(${Math.round(stats.meanR)}, ${Math.round(stats.meanG)}, ${Math.round(stats.meanB)})` }}
          />
          {stats.meanR.toFixed(0)} · {stats.meanG.toFixed(0)} · {stats.meanB.toFixed(0)}
        </span>
      ),
      hint: "Per-channel mean intensity (0-255). The swatch shows the actual mean colour. Drift across denoising steps reveals colour bias.",
    },
    {
      label: "Hue",
      value: (
        <span className="flex items-center gap-2">
          <span
            className="inline-block w-3 h-3 rounded-sm border border-parchment-dark/30"
            style={{ background: `hsl(${stats.meanHue.toFixed(0)}, 80%, 50%)` }}
          />
          {stats.meanHue.toFixed(0)}°
        </span>
      ),
      hint: "Chroma-weighted mean hue in degrees, computed as a circular mean to handle the 0/360 wrap. Manovich's cultural-analytics work uses hue distributions to fingerprint visual style.",
    },
    {
      label: "Brightness",
      value: stats.meanLuma.toFixed(1),
      hint: "Mean luminance (0.299·R + 0.587·G + 0.114·B). Tends to settle around mid-grey as the model converges.",
    },
    {
      label: "Contrast",
      value: stats.stdLuma.toFixed(1),
      hint: "Std-dev of luminance — a one-number measure of contrast. Inflates at high CFG.",
    },
    {
      label: "Saturation",
      value: stats.saturation.toFixed(3),
      hint: "Mean chroma (max-channel − min-channel) normalised. Rises as the model commits to colours; mode-collapse images are over-saturated.",
    },
    {
      label: "Entropy",
      value: `${stats.entropy.toFixed(2)} bits`,
      hint: "Shannon entropy of the luminance histogram. Pure noise approaches 8 bits; structured images settle at 6-7. The fall across steps is denoising as information reduction.",
    },
    {
      label: "Edge density",
      value: stats.edgeDensity.toFixed(3),
      hint: "Mean Sobel-gradient magnitude (0-1). Noise has uniform high edges everywhere; structured images concentrate them at object boundaries.",
    },
    {
      label: "Centre of mass",
      value: `(${stats.centreX.toFixed(2)}, ${stats.centreY.toFixed(2)})`,
      hint: "Luma-weighted centroid in fractional coordinates (0,0 top-left → 1,1 bottom-right). Shows where the brightness — and usually the subject — sits in the frame.",
    },
    {
      label: "HF energy",
      value: stats.highFreqRatio.toFixed(3),
      hint: "Fraction of total energy in the (image − Gaussian-blurred image) residual. Diffusion famously denoises high frequencies first; this should fall sharply early in the trajectory.",
    },
    {
      label: "PNG bytes",
      value: stats.pngBytes.toLocaleString(),
      hint: "Size of the canvas re-encoded as PNG. A practical proxy for Kolmogorov complexity: more compressible (smaller) images are more structured; less compressible (larger) ones are closer to noise.",
    },
  ];

  return (
    <div className="border-t border-parchment pt-3 space-y-3">
      <h4 className="font-sans text-caption uppercase tracking-wider text-muted-foreground">
        Image stats
      </h4>

      <Histograms stats={stats} />

      <dl className="grid grid-cols-[100px_1fr] gap-y-1 font-sans text-caption">
        {rows.map((r) => (
          <div key={r.label} className="contents">
            <dt
              className="text-muted-foreground cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4"
              title={r.hint}
            >
              {r.label}
            </dt>
            <dd className="text-foreground tabular-nums">{r.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Compact 32-bin histograms — three RGB channels and one luma — rendered as
 * inline SVGs. Diffusion-trajectory-relevant: their shape moves from flat
 * (noise) to peaky (structure) as denoising progresses. */
function Histograms({ stats }: { stats: ImageStats }) {
  const W = 240;
  const H = 36;

  function bars(values: number[], colour: string, label: string, hint: string) {
    const max = Math.max(1, ...values);
    const barW = W / values.length;
    return (
      <div title={hint} className="cursor-help">
        <div className="flex items-center gap-2 font-sans text-[10px] text-muted-foreground mb-0.5">
          <span className="inline-block w-2 h-2 rounded-sm" style={{ background: colour }} />
          <span>{label}</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block">
          {values.map((v, i) => {
            const h = (v / max) * H;
            return (
              <rect
                key={i}
                x={i * barW}
                y={H - h}
                width={Math.max(1, barW - 0.5)}
                height={h}
                fill={colour}
                opacity={0.85}
              />
            );
          })}
        </svg>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-3">
      {bars(stats.histR, "#cd2650", "R", "Red-channel histogram (32 bins). A flat distribution → noise; peaks → committed colour structure.")}
      {bars(stats.histG, "#3b7d4f", "G", "Green-channel histogram (32 bins).")}
      {bars(stats.histB, "#2e5d8a", "B", "Blue-channel histogram (32 bins).")}
      {bars(stats.histLuma, "#444", "Luma", "Luminance histogram. The shape moves from flat (noise) to a Gaussian-ish peak as denoising progresses; bimodal shapes suggest distinct light/dark regions in the image.")}
    </div>
  );
}
