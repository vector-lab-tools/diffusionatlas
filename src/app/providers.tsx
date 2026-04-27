"use client";

import { DiffusionSettingsProvider } from "@/context/DiffusionSettingsContext";
import { LatentCacheProvider } from "@/context/LatentCacheContext";
import { ImageBlobCacheProvider } from "@/context/ImageBlobCacheContext";
import { BackendHealthProvider } from "@/context/BackendHealthContext";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <DiffusionSettingsProvider>
      <BackendHealthProvider>
        <LatentCacheProvider>
          <ImageBlobCacheProvider>{children}</ImageBlobCacheProvider>
        </LatentCacheProvider>
      </BackendHealthProvider>
    </DiffusionSettingsProvider>
  );
}
