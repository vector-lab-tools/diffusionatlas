/**
 * Per-operation persistent state — saves a complete operation snapshot
 * (Trajectory layers, Sweep stack, etc.) into the `op_state` IDB store
 * so it survives page reloads and tab switches.
 *
 * Values flow through the structured-clone algorithm, so Float32Array
 * latents, Blob image data, and plain objects all serialise without
 * manual conversion. `null` is a valid stored value (means "user hit
 * Reset"); `undefined` from `loadOpState` means "nothing saved yet".
 */
import { withDB } from "./idb";

const STORE = "op_state";

export async function loadOpState<T>(key: string): Promise<T | undefined> {
  try {
    const value = await withDB(async (db) => (await db.get(STORE, key)) as T | undefined);
    if (typeof window !== "undefined") {
      // Visible breadcrumb so a missing-snapshot bug doesn't have to be
      // diagnosed by manually opening Application → IndexedDB.
      console.info(`[op_state] loaded "${key}":`, value === undefined ? "(no snapshot)" : `${Array.isArray(value) ? `${value.length} entries` : typeof value}`);
    }
    return value;
  } catch (err) {
    console.warn(`[op_state] load failed for "${key}":`, err);
    return undefined;
  }
}

export async function saveOpState<T>(key: string, value: T): Promise<void> {
  try {
    await withDB(async (db) => {
      await db.put(STORE, value as unknown as object, key);
    });
    if (typeof window !== "undefined") {
      console.debug(`[op_state] saved "${key}":`, Array.isArray(value) ? `${value.length} entries` : typeof value);
    }
  } catch (err) {
    // Persistence is a nice-to-have — never crash a run for it. But
    // surface the failure so the user/dev can see why nothing
    // persisted (quota, structured-clone, version mismatch, …).
    console.warn(`[op_state] save failed for "${key}":`, err);
  }
}

export async function clearOpState(key: string): Promise<void> {
  try {
    await withDB(async (db) => {
      await db.delete(STORE, key);
    });
  } catch (err) {
    console.warn(`[op_state] clear failed for "${key}":`, err);
  }
}
