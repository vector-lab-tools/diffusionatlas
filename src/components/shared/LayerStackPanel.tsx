"use client";

/**
 * Shared UI for the layer-stack rail at the top of an operation —
 * one row per layer with: visibility checkbox, colour dot, editable
 * label, optional metadata text on the right, lock/unlock toggle,
 * delete button. Plus a footer with a Reset button + persistence note.
 *
 * Pair with `useLayerStack` from `@/lib/operations/useLayerStack`.
 * Operation-specific code only needs to render the metadata string.
 */

import { Lock, Unlock, X } from "lucide-react";
import type { BaseLayer } from "@/lib/operations/useLayerStack";

export interface LayerStackPanelProps<T extends BaseLayer> {
  layers: T[];
  /** Render the right-hand metadata string for one layer (e.g. "5 cells · seed 42"). */
  renderMetadata?: (layer: T) => React.ReactNode;
  /** Called when the user edits a layer's label inline. */
  onRename: (id: string, label: string) => void;
  /** Called when the user toggles visibility. */
  onToggleVisible: (id: string, visible: boolean) => void;
  onLock: (id: string) => void;
  onUnlock: (id: string) => void;
  onDelete: (id: string) => void;
  onReset: () => void;
  /** "Trajectory" / "Sweep" / "Neighbourhood" / "Bench" — used in the Reset confirm. */
  operationName: string;
}

export function LayerStackPanel<T extends BaseLayer>(props: LayerStackPanelProps<T>) {
  const {
    layers,
    renderMetadata,
    onRename,
    onToggleVisible,
    onLock,
    onUnlock,
    onDelete,
    onReset,
    operationName,
  } = props;
  if (layers.length === 0) return null;
  const lockedCount = layers.filter((l) => l.locked).length;
  const tempCount = layers.length - lockedCount;
  return (
    <div className="border border-parchment rounded-sm bg-cream/20 p-3 mb-4">
      <h3 className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-2">
        Layers · {lockedCount} locked
        {tempCount > 0 && <span className="ml-1">· {tempCount} temp</span>}
        <span className="ml-2 italic normal-case tracking-normal">
          click the padlock to keep a temp layer; locked layers stay through future runs.
        </span>
      </h3>
      <div className="space-y-1.5">
        {layers.map((layer) => (
          <div
            key={layer.id}
            className={
              "flex items-center gap-2 text-caption rounded-sm " +
              (layer.locked
                ? ""
                : "border border-dashed border-parchment-dark bg-cream/40 px-1 py-0.5")
            }
          >
            <input
              type="checkbox"
              checked={layer.visible}
              onChange={(e) => onToggleVisible(layer.id, e.target.checked)}
              title="Show / hide this layer"
            />
            <span
              className="inline-block w-3 h-3 rounded-full flex-shrink-0"
              style={{ background: layer.colour }}
              title={layer.locked ? "Locked layer · stays across runs" : "Temporary layer · will be replaced by the next run"}
            />
            <input
              type="text"
              value={layer.label}
              onChange={(e) => onRename(layer.id, e.target.value)}
              className={"input-editorial py-0.5 text-caption flex-1 min-w-0" + (layer.locked ? "" : " italic")}
            />
            <span className="font-sans text-caption text-muted-foreground">
              {!layer.locked && (
                <span className="text-burgundy not-italic mr-2 font-medium uppercase tracking-wider text-[10px]">
                  temp
                </span>
              )}
              {renderMetadata?.(layer)}
            </span>
            <button
              onClick={() => (layer.locked ? onUnlock(layer.id) : onLock(layer.id))}
              className={
                layer.locked
                  ? "btn-editorial-ghost p-1 text-burgundy"
                  : "btn-editorial-ghost p-1 text-muted-foreground hover:text-burgundy"
              }
              title={layer.locked
                ? "Locked: this layer stays across future runs · click to unlock (will be removed on next run)"
                : "Temporary: this layer will be replaced by the next run · click to lock it in place"}
              aria-pressed={layer.locked}
              aria-label={layer.locked ? "Unlock layer" : "Lock layer"}
            >
              {layer.locked ? <Lock size={12} /> : <Unlock size={12} />}
            </button>
            <button
              onClick={() => onDelete(layer.id)}
              className="btn-editorial-ghost p-1"
              title="Delete layer"
            >
              <X size={12} />
            </button>
          </div>
        ))}
        <div className="flex items-center gap-3 mt-1">
          <button
            onClick={() => {
              if (
                !confirm(
                  `Reset all ${operationName} layers? This clears in-session state and the persisted snapshot in IndexedDB. The cached image blobs in StatusBar are unaffected.`,
                )
              )
                return;
              onReset();
            }}
            className="font-sans text-caption text-burgundy hover:text-burgundy-900 underline underline-offset-2"
            title={`Clear every saved layer (locked + temp) and wipe the persisted snapshot for ${operationName}. Cannot be undone.`}
          >
            Reset {operationName}
          </button>
          <span className="font-sans text-caption text-muted-foreground italic">
            · layers persist across reloads
          </span>
        </div>
      </div>
    </div>
  );
}
