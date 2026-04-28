"use client";

/**
 * Shared contact-sheet primitives. The same dark-strip + edge-print
 * metadata + clickable RGB histogram pattern that DenoiseTrajectory
 * uses for FilmStrip is generic enough to host any operation's
 * results — Guidance Sweep frames its CFG values, Latent
 * Neighbourhood frames its seed offsets, Trajectory frames its
 * denoising steps. The frame metadata block + tiny RGB histogram are
 * identical across all three; only the scalars differ.
 */

import { useEffect, useState } from "react";
import { computeImageStats } from "@/lib/image/stats";

/**
 * Tiny overlapping-channel RGB histogram (Photoshop / Lightroom style).
 * Three semi-transparent filled polylines on a near-white background.
 * Used as the bottom strip of every contact-sheet frame; clicking the
 * frame parent opens the full FrameModal with the larger version.
 */
export function RgbHistogramMini({
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
  const max = Math.max(...hist.r, ...hist.g, ...hist.b, 1);
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
 * White edge-print metadata block that sits below each dark-strip
 * frame. Shows a primary heading (CFG value, seed, step number),
 * an optional secondary heading (right-aligned annotation), a
 * 2-column scalar grid, and a tiny clickable RGB histogram computed
 * from the preview thumbnail. Clicking anywhere on the block fires
 * `onOpen` so the full FrameModal pops up with the bigger image and
 * the full image-stats panel.
 */
export function ContactSheetFrame({
  width,
  primary,
  secondary,
  caption,
  scalars,
  preview,
  onOpen,
  disabledTitle = "No image captured for this frame",
  enabledTitle = "Click for full preview + image stats",
}: {
  width: number;
  primary: string;
  secondary?: string;
  /**
   * Optional tiny italic line under the primary/secondary heading, used
   * for short annotations like the CFG band ("balanced default",
   * "oversaturated", etc.) that explain what the primary value means
   * without forcing the user to recall the convention.
   */
  caption?: string;
  scalars: Array<{ key: string; value: string }>;
  preview: string | null;
  onOpen: () => void;
  disabledTitle?: string;
  enabledTitle?: string;
}) {
  const [hist, setHist] = useState<{ r: number[]; g: number[]; b: number[] } | null>(null);
  useEffect(() => {
    let cancelled = false;
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

  return (
    <button
      onClick={onOpen}
      disabled={!preview}
      className="flex-shrink-0 bg-white border border-parchment-dark rounded-sm hover:border-burgundy hover:shadow-editorial transition-all text-left disabled:opacity-60 disabled:cursor-default disabled:hover:border-parchment-dark disabled:hover:shadow-none"
      style={{ width: `${width}px` }}
      title={preview ? enabledTitle : disabledTitle}
    >
      <div className="px-1.5 py-1 font-mono text-[9px] text-black leading-tight">
        <div className="flex items-baseline justify-between mb-0.5">
          <span className="font-bold">{primary}</span>
          {secondary && <span className="text-[8px] text-neutral-500">{secondary}</span>}
        </div>
        {caption && (
          <div className="text-[7px] italic text-neutral-500 mb-0.5 leading-tight">
            {caption}
          </div>
        )}
        {scalars.length > 0 && (
          <div className="grid grid-cols-2 gap-x-1 gap-y-[1px]">
            {scalars.map((s) => (
              <span key={s.key} className="contents">
                <span className="text-neutral-500">{s.key}</span>
                <span className="text-right">{s.value}</span>
              </span>
            ))}
          </div>
        )}
        <div className="mt-1">
          <RgbHistogramMini width={width - 12} height={28} hist={hist} />
        </div>
      </div>
    </button>
  );
}

/**
 * Sprocket-hole row used at the top and bottom of the dark contact-sheet
 * strip. Length adapts to the frame count so the strip looks like a real
 * piece of 35mm regardless of whether you're looking at 5 CFG cells or
 * 24 trajectory steps.
 */
export function SprocketRow({ frameCount }: { frameCount: number }) {
  const count = Math.max(frameCount * 2, 12);
  return (
    <div className="flex gap-3 px-3">
      {Array.from({ length: count }).map((_, i) => (
        <span key={i} className="block w-3 h-2 bg-[#222] rounded-[1px] flex-shrink-0" />
      ))}
    </div>
  );
}
