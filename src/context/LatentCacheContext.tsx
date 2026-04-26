"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getDB } from "@/lib/cache/idb";

interface LatentCacheCtx {
  get: (key: string) => Promise<Float32Array | undefined>;
  set: (key: string, value: Float32Array) => Promise<void>;
  count: number;
  clear: () => Promise<void>;
}

const Ctx = createContext<LatentCacheCtx | null>(null);

export function LatentCacheProvider({ children }: { children: ReactNode }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    void refreshCount();
  }, []);

  async function refreshCount() {
    const db = await getDB();
    setCount(await db.count("latents"));
  }

  const value: LatentCacheCtx = {
    async get(key) {
      const db = await getDB();
      return db.get("latents", key);
    },
    async set(key, value) {
      const db = await getDB();
      await db.put("latents", value, key);
      void refreshCount();
    },
    count,
    async clear() {
      const db = await getDB();
      await db.clear("latents");
      void refreshCount();
    },
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLatentCache() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useLatentCache must be used within LatentCacheProvider");
  return ctx;
}
