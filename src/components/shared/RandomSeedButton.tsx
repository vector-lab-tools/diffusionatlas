"use client";

import { Dices, Shuffle, Plus } from "lucide-react";

/**
 * The three behaviours offered by `RandomSeedButton`:
 *
 *   - `off`       : the dice does nothing automatically; the user only
 *                   rolls when they click it.
 *   - `shuffle`   : a fresh random seed is rolled before every run.
 *   - `increment` : the seed is bumped by +1 before every run, so a series
 *                   of runs walks deterministically through neighbouring
 *                   seeds (the standard "compare adjacent seeds" pattern).
 */
export type SeedMode = "off" | "shuffle" | "increment";

/**
 * Roll a fresh 32-bit-ish seed (range `[0, 2^31 - 1]`, matching the
 * AUTOMATIC1111 / fal / Replicate convention).
 */
export function rollSeed(): number {
  return Math.floor(Math.random() * 2_147_483_647);
}

/**
 * Apply the current seed mode and return the seed that should actually be
 * used for the next run. `mode === "off"` returns `current` unchanged.
 */
export function nextSeed(mode: SeedMode, current: number): number {
  if (mode === "shuffle") return rollSeed();
  if (mode === "increment") return current + 1;
  return current;
}

/**
 * Seed-control cluster. Two stacked half-height mode chips (shuffle on top,
 * increment beneath) sit next to a single full-height dice button — the
 * whole control is the same overall height as the seed input, so it slots
 * in cleanly without wasting a column. The mode chips are mutually
 * exclusive: clicking the active one turns it off; clicking the other
 * switches between them.
 *
 * Animations differ by mode so the user can read at a glance which kind of
 * roll just happened: shuffle spins, increment bumps upward; an idle
 * shuffle gently pulses the dice as a "this will move on run" hint.
 */
export function RandomSeedButton({
  onPick,
  title = "Roll a fresh random seed (range 0 – 2,147,483,647)",
  mode = "off",
  onModeChange,
  spinning = false,
}: {
  onPick: (seed: number) => void;
  title?: string;
  mode?: SeedMode;
  onModeChange?: (next: SeedMode) => void;
  spinning?: boolean;
}) {
  const isShuffle = mode === "shuffle";
  const isIncrement = mode === "increment";
  const isActive = isShuffle || isIncrement;

  const diceAnim = spinning
    ? isIncrement
      ? "animate-[seed-bump_0.45s_ease-out]"
      : "animate-[spin_0.45s_linear]"
    : isShuffle
      ? "animate-[pulse_1.6s_ease-in-out_infinite]"
      : "";

  function toggle(next: SeedMode) {
    if (!onModeChange) return;
    onModeChange(mode === next ? "off" : next);
  }

  // Each chip takes half the cluster height (`flex-1` inside a `flex-col`
  // parent that is itself stretched by the surrounding `items-stretch`
  // row). That makes the whole cluster track whatever the seed input's
  // height happens to be — no magic numbers.
  const chipBase = "flex-1 w-12 inline-flex items-center justify-center border transition-colors";

  return (
    <>
      <style jsx>{`
        @keyframes seed-bump {
          0%   { transform: translateY(0); }
          40%  { transform: translateY(-3px); }
          100% { transform: translateY(0); }
        }
      `}</style>
      <span className="inline-flex items-stretch gap-1 flex-shrink-0 self-stretch">
        {onModeChange && (
          <span className="inline-flex flex-col gap-0 self-stretch">
            <button
              type="button"
              onClick={() => toggle("shuffle")}
              className={
                (isShuffle
                  ? `${chipBase} border-burgundy bg-burgundy text-cream`
                  : `${chipBase} border-parchment-dark bg-transparent text-muted-foreground hover:text-burgundy hover:border-burgundy`) +
                " rounded-t-sm border-b-0"
              }
              title={isShuffle
                ? "Shuffle on: a fresh random seed is rolled before every run · click to turn off"
                : "Shuffle: roll a fresh random seed automatically before every run"}
              aria-label={isShuffle ? "Turn shuffle off" : "Turn shuffle on (random seed each run)"}
              aria-pressed={isShuffle}
            >
              <Shuffle size={13} />
            </button>
            <button
              type="button"
              onClick={() => toggle("increment")}
              className={
                (isIncrement
                  ? `${chipBase} border-burgundy bg-burgundy text-cream`
                  : `${chipBase} border-parchment-dark bg-transparent text-muted-foreground hover:text-burgundy hover:border-burgundy`) +
                " rounded-b-sm"
              }
              title={isIncrement
                ? "Increment on: seed is bumped by +1 before every run · click to turn off"
                : "Increment: bump seed by +1 before every run (walk neighbouring seeds)"}
              aria-label={isIncrement ? "Turn increment off" : "Turn increment on (+1 each run)"}
              aria-pressed={isIncrement}
            >
              <Plus size={13} />
            </button>
          </span>
        )}
        <button
          type="button"
          onClick={() => onPick(rollSeed())}
          className={
            isActive
              ? "self-stretch w-10 inline-flex items-center justify-center rounded-sm border border-burgundy bg-burgundy/5 text-burgundy hover:bg-burgundy/10 transition-colors flex-shrink-0"
              : "self-stretch w-10 inline-flex items-center justify-center rounded-sm border border-parchment-dark bg-transparent text-muted-foreground hover:text-burgundy hover:border-burgundy transition-colors flex-shrink-0"
          }
          title={
            isShuffle
              ? `${title} · shuffle on: random seed before every run`
              : isIncrement
                ? `${title} · increment on: +1 before every run`
                : title
          }
          aria-label="Pick a random seed"
        >
          <Dices size={14} className={diceAnim} />
        </button>
      </span>
    </>
  );
}
