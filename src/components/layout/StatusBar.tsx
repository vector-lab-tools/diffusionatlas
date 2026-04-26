"use client";

import { useSettings } from "@/context/DiffusionSettingsContext";
import { useLatentCache } from "@/context/LatentCacheContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import { VERSION } from "@/lib/version";

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

  return (
    <footer className="border-t border-parchment-dark px-6 py-2 flex items-center gap-6 font-sans text-caption text-slate">
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
      <span>
        {imageCount} image{imageCount !== 1 ? "s" : ""} ({formatBytes(approximateBytes)})
      </span>
      {lastQueryTime !== undefined && (
        <>
          <span className="h-3 w-px bg-parchment-dark" />
          <span>Last: {lastQueryTime.toFixed(1)}s</span>
        </>
      )}

      <a
        href="https://vector-lab-tools.github.io"
        target="_blank"
        rel="noopener noreferrer"
        title="Part of the Vector Lab"
        className="ml-auto flex items-center gap-1.5 hover:text-foreground transition-colors"
      >
        <span>Part of the Vector Lab</span>
      </a>
    </footer>
  );
}
