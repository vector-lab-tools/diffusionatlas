"use client";

/**
 * Generic temp-vs-locked layer stack with auto-persistence.
 *
 * Every Diffusion Atlas operation that produces inspectable output
 * (Trajectory, Sweep, Neighbourhood, Bench) wants the same shape:
 *
 *   • Each completed run becomes a *temporary* (unlocked) layer with a
 *     neutral colour. Running again replaces the temp.
 *   • The user clicks a padlock to *lock* a layer in place — locked
 *     layers gain a real palette colour and survive future runs.
 *   • The whole stack persists to IDB so it survives page reloads.
 *   • Reset wipes everything and clears the persisted snapshot.
 *
 * `useLayerStack` is the hook; `<LayerStackPanel>` is the matching UI.
 * Operation-specific code only needs to define the payload shape and
 * the per-row render — nothing about persistence, lock-state, or the
 * temp/locked palette logic should be duplicated again.
 */

import { useEffect, useState } from "react";
import { loadOpState, saveOpState, clearOpState } from "@/lib/cache/op_state";

/** Fields every layer carries, regardless of operation. */
export interface BaseLayer {
  id: string;
  label: string;
  /** Hex colour. Locked layers get a palette colour; temp uses neutral ink. */
  colour: string;
  /** Toggle for showing in 3D / contact-sheet views. */
  visible: boolean;
  /** `false` = temporary, will be replaced by next run. `true` = pinned. */
  locked: boolean;
  /** Unix ms — used for newest-first sorting. */
  createdAt: number;
}

/** Burgundy / gold / blue / green / purple / slate — used for locked layers. */
export const LAYER_COLOURS = ["#7c2d36", "#c9a227", "#2e5d8a", "#3b7d4f", "#8a3b6e", "#5e5e5e"];
/** The neutral ink temp layers use so locked-layer colours stay distinct. */
export const TEMP_COLOUR = "#1a1a1a";

export interface UseLayerStackResult<T extends BaseLayer> {
  layers: T[];
  /** True once the IDB hydrate has settled. Skip persisting until this is true. */
  hydrated: boolean;
  /** Replace the entire stack (rare — used by hydrate / reset). */
  setLayers: React.Dispatch<React.SetStateAction<T[]>>;
  /** Drop every unlocked layer. Call at the start of a new run. */
  dropUnlocked: () => void;
  /** Prepend a layer. Locked layers stay; existing unlocked layers are dropped. */
  pushNew: (layer: T) => void;
  /** Mutate a single layer's fields. */
  updateLayer: (id: string, patch: Partial<T>) => void;
  /** Lock a layer — grants it a palette colour based on locked-count. */
  lockLayer: (id: string) => void;
  /** Unlock a layer — flips it back to temp / neutral. */
  unlockLayer: (id: string) => void;
  /** Remove a layer entirely. */
  deleteLayer: (id: string) => void;
  /** Wipe the stack and the persisted IDB snapshot. */
  reset: () => void;
  /** Allocate the next palette colour for a fresh locked layer. */
  nextLockedColour: () => string;
}

/**
 * @param persistKey  IDB key under the `op_state` store. Bump the version
 *                    suffix in this string when the layer shape changes
 *                    incompatibly so old snapshots don't crash the loader.
 * @param validate    Optional sanity-check on hydrated data — reject
 *                    snapshots that don't look right (e.g. payload shape
 *                    changed). Defaults to "trust the array".
 */
export function useLayerStack<T extends BaseLayer>(
  persistKey: string,
  validate?: (stored: unknown) => T[] | null,
): UseLayerStackResult<T> {
  const [layers, setLayers] = useState<T[]>([]);
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from IDB once on mount.
  useEffect(() => {
    let cancelled = false;
    void loadOpState<T[]>(persistKey).then((stored) => {
      if (cancelled) return;
      if (Array.isArray(stored)) {
        const accepted = validate ? validate(stored) : (stored as T[]);
        if (accepted && accepted.length > 0) setLayers(accepted);
      }
      setHydrated(true);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [persistKey]);

  // Debounced write-back. Skips the initial render before hydration to
  // avoid clobbering a real snapshot with the empty initial state.
  useEffect(() => {
    if (!hydrated) return;
    const t = setTimeout(() => {
      void saveOpState(persistKey, layers);
    }, 500);
    return () => clearTimeout(t);
  }, [layers, hydrated, persistKey]);

  function dropUnlocked() {
    setLayers((prev) => prev.filter((l) => l.locked));
  }

  function pushNew(layer: T) {
    setLayers((prev) => [layer, ...prev.filter((l) => l.locked)]);
  }

  function updateLayer(id: string, patch: Partial<T>) {
    setLayers((prev) => prev.map((l) => (l.id === id ? { ...l, ...patch } : l)));
  }

  function lockLayer(id: string) {
    setLayers((prev) => {
      const lockedCount = prev.filter((l) => l.locked).length;
      return prev.map((l) =>
        l.id === id
          ? { ...l, locked: true, colour: LAYER_COLOURS[lockedCount % LAYER_COLOURS.length] }
          : l,
      );
    });
  }

  function unlockLayer(id: string) {
    setLayers((prev) => prev.map((l) => (l.id === id ? { ...l, locked: false, colour: TEMP_COLOUR } : l)));
  }

  function deleteLayer(id: string) {
    setLayers((prev) => prev.filter((l) => l.id !== id));
  }

  function reset() {
    setLayers([]);
    void clearOpState(persistKey);
  }

  function nextLockedColour(): string {
    return LAYER_COLOURS[layers.filter((l) => l.locked).length % LAYER_COLOURS.length];
  }

  return {
    layers,
    hydrated,
    setLayers,
    dropUnlocked,
    pushNew,
    updateLayer,
    lockLayer,
    unlockLayer,
    deleteLayer,
    reset,
    nextLockedColour,
  };
}
