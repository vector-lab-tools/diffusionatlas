"use client";

import { useState } from "react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useLatentCache } from "@/context/LatentCacheContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import { VERSION } from "@/lib/version";
import { CachedImagesModal } from "./CachedImagesModal";
import { BackendHealth } from "./BackendHealth";

interface StatusBarProps {
  lastQueryTime?: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function StatusBar({ lastQueryTime }: StatusBarProps) {
  const { settings } = useSettings();
  const { count: latentCount } = useLatentCache();
  const { count: imageCount, approximateBytes } = useImageBlobCache();
  const [imagesOpen, setImagesOpen] = useState(false);

  return (
    <>
    <footer className="sticky bottom-0 z-30 bg-cream border-t border-parchment-dark px-6 py-2 flex items-center gap-6 font-sans text-caption text-slate">
      <span>
        <span className="text-ink font-medium">v{VERSION}</span>
      </span>
      <span className="h-3 w-px bg-parchment-dark" />
      <span>
        {settings.backend === "local" ? "Local" : "Hosted"} · {settings.providerId}
      </span>
      <span className="h-3 w-px bg-parchment-dark" />
      <span>{settings.modelId}</span>
      <span className="h-3 w-px bg-parchment-dark" />
      <span>
        {latentCount} latent{latentCount !== 1 ? "s" : ""}
      </span>
      <span className="h-3 w-px bg-parchment-dark" />
      {imageCount > 0 ? (
        <button
          onClick={() => setImagesOpen(true)}
          className="text-burgundy hover:text-burgundy-900 underline underline-offset-2 decoration-dotted"
          title="Browse every cached image (IndexedDB)"
        >
          {imageCount} image{imageCount !== 1 ? "s" : ""} ({formatBytes(approximateBytes)})
        </button>
      ) : (
        <span>
          {imageCount} images ({formatBytes(approximateBytes)})
        </span>
      )}
      {lastQueryTime !== undefined && (
        <>
          <span className="h-3 w-px bg-parchment-dark" />
          <span>Last: {lastQueryTime.toFixed(1)}s</span>
        </>
      )}
      <span className="h-3 w-px bg-parchment-dark" />
      <BackendHealth />

      <a
        href="https://vector-lab-tools.github.io"
        target="_blank"
        rel="noopener noreferrer"
        title="Part of the Vector Lab"
        className="ml-auto flex items-center gap-1.5 hover:text-foreground transition-colors"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/icons/vector-lab-logo-mark.svg"
          alt=""
          width={14}
          height={14}
          aria-hidden="true"
          className="block opacity-80"
        />
        <span>Part of the Vector Lab</span>
      </a>
    </footer>
    <CachedImagesModal open={imagesOpen} onClose={() => setImagesOpen(false)} />
    </>
  );
}
