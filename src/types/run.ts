/**
 * A "run" is a single completed operation: one generate call, one sweep,
 * one neighbourhood sample, one bench pack. Stored in the `runs` IDB
 * object store keyed by `id`. Image blobs are stored separately in the
 * `images` store and referenced by the keys in `imageKeys`.
 */

export type RunKind = "sweep" | "neighbourhood" | "bench" | "single";

export interface RunSampleRef {
  /** IDB key under which the image blob is stored (in `images` store). */
  imageKey: string;
  /** Per-sample variable: cfg for sweep, seed for neighbourhood, taskId for bench. */
  variable: number | string;
  /** Optional response time, milliseconds. */
  responseTimeMs?: number;
}

export interface Run {
  id: string;
  kind: RunKind;
  createdAt: string; // ISO
  providerId: string;
  modelId: string;
  prompt: string;
  /** Generation params held flat; extra per-kind fields under `extra`. */
  seed: number;
  steps: number;
  cfg: number;
  width: number;
  height: number;
  /** Per-sample image references. Single-image runs have one entry. */
  samples: RunSampleRef[];
  /** Free-form per-kind extras (e.g. cfgList for sweep, k/radius for neighbourhood). */
  extra?: Record<string, unknown>;
}
