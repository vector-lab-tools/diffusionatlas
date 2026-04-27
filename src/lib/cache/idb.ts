import { openDB, type IDBPDatabase } from "idb";

const DB_NAME = "diffusion-atlas";
const DB_VERSION = 2;

let dbPromise: Promise<IDBPDatabase> | null = null;

/**
 * Opens (or reuses) a connection to the IndexedDB. The connection can be
 * closed out from under us by the browser — most commonly when another tab
 * triggers a version change, but also when devtools storage is cleared,
 * during HMR, or after long idle periods. We listen for `close` and
 * `versionchange` and drop the cached promise so the next call opens a
 * fresh connection rather than handing back a dead one (which throws
 * `InvalidStateError: The database connection is closing`).
 */
export function getDB(): Promise<IDBPDatabase> {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains("latents")) db.createObjectStore("latents");
        if (!db.objectStoreNames.contains("images")) db.createObjectStore("images");
        if (!db.objectStoreNames.contains("runs")) db.createObjectStore("runs");
        if (!db.objectStoreNames.contains("bench")) db.createObjectStore("bench");
      },
      blocked() {
        // Another connection has the old version open and is blocking
        // our upgrade. Drop the cache so we retry cleanly.
        dbPromise = null;
      },
      blocking() {
        // We are blocking another tab's upgrade — close so it can proceed.
        dbPromise = null;
      },
      terminated() {
        // Browser killed the connection (idle close, devtools clear,
        // private-mode quota, etc.). Drop the cache.
        dbPromise = null;
      },
    }).then((db) => {
      // `close` fires whenever the connection ends for any reason.
      db.addEventListener("close", () => {
        dbPromise = null;
      });
      // If a version change is requested elsewhere, close ourselves so
      // the upgrade can run, then drop the cache.
      db.addEventListener("versionchange", () => {
        try { db.close(); } catch {}
        dbPromise = null;
      });
      return db;
    }).catch((err) => {
      // Don't cache a rejected promise — next call should retry.
      dbPromise = null;
      throw err;
    });
  }
  return dbPromise;
}

/**
 * Run a function against the database, retrying once if the connection
 * was closing. Use this for any operation that might race a tab/HMR
 * close. Pass through other errors unchanged.
 */
export async function withDB<T>(fn: (db: IDBPDatabase) => Promise<T>): Promise<T> {
  try {
    return await fn(await getDB());
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("database connection is closing") || msg.includes("InvalidStateError")) {
      dbPromise = null;
      return fn(await getDB());
    }
    throw err;
  }
}
