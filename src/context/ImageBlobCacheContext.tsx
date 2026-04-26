"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getDB } from "@/lib/cache/idb";

interface ImageBlobCacheCtx {
  get: (key: string) => Promise<Blob | undefined>;
  set: (key: string, blob: Blob) => Promise<void>;
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
      const db = await getDB();
      const c = await db.count("images");
      setCount(c);
      if (navigator.storage?.estimate) {
        const est = await navigator.storage.estimate();
        setApproximateBytes(est.usage ?? 0);
      }
    } catch {}
  }

  const value: ImageBlobCacheCtx = {
    async get(key) {
      const db = await getDB();
      return db.get("images", key);
    },
    async set(key, blob) {
      const db = await getDB();
      await db.put("images", blob, key);
      void refresh();
    },
    count,
    approximateBytes,
    async clear() {
      const db = await getDB();
      await db.clear("images");
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
