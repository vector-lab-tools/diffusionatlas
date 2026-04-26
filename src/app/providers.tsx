"use client";

import { DiffusionSettingsProvider } from "@/context/DiffusionSettingsContext";
import { LatentCacheProvider } from "@/context/LatentCacheContext";
import { ImageBlobCacheProvider } from "@/context/ImageBlobCacheContext";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <DiffusionSettingsProvider>
      <LatentCacheProvider>
        <ImageBlobCacheProvider>{children}</ImageBlobCacheProvider>
      </LatentCacheProvider>
    </DiffusionSettingsProvider>
  );
}
