"use client";

interface CameraRollEntry {
  src: string;
  caption?: string;
  /** Optional secondary caption (smaller, shown under the first). */
  subcaption?: string;
}

interface CameraRollProps {
  title?: string;
  entries: CameraRollEntry[];
  /** Aspect ratio of each thumbnail. Default 'square'. */
  aspect?: "square" | "auto";
}

/**
 * Compact thumbnail grid for the Deep Dive — every generated image from a
 * run laid out side-by-side with provider/model/seed/cfg captions. The
 * same image set is included in the PDF export's image grid, so what the
 * user sees here is what they'd get in the report.
 */
export function CameraRoll({ title = "Camera roll", entries, aspect = "square" }: CameraRollProps) {
  if (entries.length === 0) {
    return (
      <div className="font-sans text-caption text-muted-foreground italic">
        No images yet — run the operation to populate the camera roll.
      </div>
    );
  }
  return (
    <div>
      <h4 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
        {title} <span className="text-foreground">· {entries.length} {entries.length === 1 ? "image" : "images"}</span>
      </h4>
      <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-2">
        {entries.map((entry, i) => (
          <div key={i} className="flex flex-col">
            <div className={`${aspect === "square" ? "aspect-square" : ""} bg-cream/50 border border-parchment rounded-sm overflow-hidden`}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={entry.src} alt={entry.caption ?? `frame ${i}`} className="w-full h-full object-cover" />
            </div>
            {(entry.caption || entry.subcaption) && (
              <div className="mt-1 font-sans text-[10px] text-muted-foreground text-center leading-tight">
                {entry.caption && <div className="text-foreground">{entry.caption}</div>}
                {entry.subcaption && <div>{entry.subcaption}</div>}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
