"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { withDB } from "@/lib/cache/idb";

interface ImageBlobCacheCtx {
  get: (key: string) => Promise<Blob | undefined>;
  set: (key: string, blob: Blob) => Promise<void>;
  list: () => Promise<Array<{ key: string; blob: Blob }>>;
  count: number;
  approximateBytes: number;
  clear: () => Promise<void>;
}

const Ctx = createContext<ImageBlobCacheCtx | null>(null);

export function ImageBlobCacheProvider({ children }: { children: ReactNode }) {
  const [count, setCount] = useState(0);
  const [approximateBytes, setApproximateBytes] = useState(0);

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    try {
      await withDB(async (db) => {
        const c = await db.count("images");
        setCount(c);
        // Sum blob sizes from the images store directly. Previously this
        // used navigator.storage.estimate() which reports whole-origin
        // usage (latents, runs, bench combined) — that produced numbers
        // that disagreed with the actual image-store contents and
        // confused the CachedImagesModal.
        const blobs = (await db.getAll("images")) as Blob[];
        let total = 0;
        for (const b of blobs) total += b.size;
        setApproximateBytes(total);
      });
    } catch {}
  }

  const value: ImageBlobCacheCtx = {
    async get(key) {
      return withDB((db) => db.get("images", key));
    },
    async set(key, blob) {
      await withDB(async (db) => {
        await db.put("images", blob, key);
      });
      void refresh();
    },
    async list() {
      // getAllKeys + getAll is more reliable across browsers/idb versions
      // than walking a cursor — the previous cursor implementation was
      // silently returning an empty array on Safari + recent Chromium,
      // producing the "37 total · No cached images yet" contradiction.
      return withDB(async (db) => {
        const keys = (await db.getAllKeys("images")) as IDBValidKey[];
        const blobs = (await db.getAll("images")) as Blob[];
        const out: Array<{ key: string; blob: Blob }> = [];
        const n = Math.min(keys.length, blobs.length);
        for (let i = 0; i < n; i++) {
          out.push({ key: String(keys[i]), blob: blobs[i] });
        }
        return out;
      });
    },
    count,
    approximateBytes,
    async clear() {
      await withDB(async (db) => {
        await db.clear("images");
      });
      void refresh();
    },
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useImageBlobCache() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useImageBlobCache must be used within ImageBlobCacheProvider");
  return ctx;
}
