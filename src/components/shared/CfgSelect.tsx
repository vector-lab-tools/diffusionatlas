"use client";

/**
 * Shared CFG (classifier-free guidance) value picker. A constrained
 * dropdown rather than a free-form number input: the canonical values
 * plus their plain-English annotations cover what 99% of users will
 * ever want, and refusing arbitrary numbers keeps users out of the
 * "I typed 100 and got a 422" failure mode. Researchers who genuinely
 * need an off-list CFG can still set it via Settings → Defaults.
 */

export interface CfgPreset {
  value: number;
  label: string;
  annotation?: string;
}

export const CFG_PRESETS: CfgPreset[] = [
  { value: 0, label: "0", annotation: "prompt off (unconditional)" },
  { value: 1, label: "1", annotation: "no amplification" },
  { value: 2.5, label: "2.5", annotation: "atmospheric" },
  { value: 4, label: "4", annotation: "soft amplification" },
  { value: 5, label: "5" },
  { value: 6, label: "6" },
  { value: 7, label: "7" },
  { value: 7.5, label: "7.5", annotation: "balanced default" },
  { value: 8, label: "8" },
  { value: 9, label: "9" },
  { value: 10, label: "10" },
  { value: 12, label: "12", annotation: "aggressive" },
  { value: 15, label: "15", annotation: "mode collapse risk" },
  { value: 20, label: "20", annotation: "oversaturated" },
];

/**
 * Fallback resolver: if the parent has a CFG that isn't in the preset
 * list (e.g. carried over from an older settings.defaults), snap to
 * the nearest preset value rather than rendering a broken select.
 */
export function nearestCfgPreset(value: number): number {
  let best = CFG_PRESETS[0];
  let bestDist = Math.abs(value - best.value);
  for (const p of CFG_PRESETS) {
    const d = Math.abs(value - p.value);
    if (d < bestDist) {
      best = p;
      bestDist = d;
    }
  }
  return best.value;
}

/**
 * Plain-English band for any CFG value — used as a tiny annotation
 * under the CFG number in contact-sheet frames so the user doesn't
 * have to recall the convention. Bands are inclusive on the lower
 * bound, so e.g. cfg ∈ [7, 10) falls into "balanced".
 */
export function cfgCaption(value: number): string {
  if (value <= 0) return "prompt off";
  if (value < 1.5) return "no amplification";
  if (value < 3) return "atmospheric";
  if (value < 5) return "soft amplification";
  if (value < 10) return "balanced default";
  if (value < 14) return "aggressive";
  if (value < 18) return "mode collapse risk";
  return "oversaturated";
}

export function CfgSelect({
  value,
  onChange,
  className = "input-editorial mt-1",
  id,
}: {
  value: number;
  onChange: (next: number) => void;
  className?: string;
  id?: string;
}) {
  // If the current value is off the preset grid, snap it visually so
  // the select renders cleanly. We don't mutate parent state — the
  // first onChange the user fires will reconcile.
  const safeValue = CFG_PRESETS.some((p) => p.value === value)
    ? value
    : nearestCfgPreset(value);

  return (
    <select
      id={id}
      value={safeValue}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className={className}
    >
      {CFG_PRESETS.map((p) => (
        <option key={p.value} value={p.value} title={p.annotation ?? undefined}>
          {p.annotation ? `${p.label} — ${p.annotation}` : p.label}
        </option>
      ))}
    </select>
  );
}
