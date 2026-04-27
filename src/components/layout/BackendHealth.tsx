"use client";

import { useState } from "react";
import { X } from "lucide-react";
import { useSettings } from "@/context/DiffusionSettingsContext";
import { useBackendHealth } from "@/context/BackendHealthContext";

/**
 * Renders the green/amber/red dot in the StatusBar that reflects the
 * shared backend-health poll. The actual fetching lives in
 * `BackendHealthContext` so other components (e.g. the Width/Height
 * selects in DenoiseTrajectory, which want to default to the loaded
 * model's native resolution) can read the same report.
 */
export function BackendHealth() {
  const { settings } = useSettings();
  const { status, report, lastCheck } = useBackendHealth();
  const [open, setOpen] = useState(false);

  const dotColour =
    status === "ok"
      ? "bg-green-500"
      : status === "error"
        ? "bg-red-500"
        : "bg-amber-400";
  const dotTitle =
    status === "ok"
      ? `Local backend reachable at ${settings.localBaseUrl}${
          report?.currentModelId ? ` · ${report.currentModelId} loaded` : ""
        } (click for details)`
      : status === "error"
        ? `Cannot reach local backend at ${settings.localBaseUrl} (click for details)`
        : "Checking local backend…";

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-1.5 hover:text-foreground transition-colors"
        title={dotTitle}
      >
        <span
          className={`inline-block w-2 h-2 rounded-full ${dotColour} ${status === "checking" ? "animate-pulse" : ""}`}
        />
        <span className="font-sans text-caption">backend</span>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="card-editorial max-w-md w-full p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <h2 className="font-display text-display-md font-bold text-burgundy flex items-center gap-2">
                  <span className={`inline-block w-3 h-3 rounded-full ${dotColour}`} />
                  Local backend
                </h2>
                <p className="font-sans text-caption text-muted-foreground">
                  {settings.localBaseUrl}
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="btn-editorial-ghost px-2 py-1"
                aria-label="Close"
              >
                <X size={16} />
              </button>
            </div>

            {status === "error" ? (
              <div className="border border-burgundy/40 bg-burgundy/5 text-burgundy p-3 mb-3 font-sans text-body-sm rounded-sm">
                Cannot reach the backend. Start uvicorn with:
                <pre className="mt-2 bg-cream/40 p-2 rounded-sm font-mono text-[11px] whitespace-pre-wrap">
                  cd backend{"\n"}.venv/bin/uvicorn main:app --port 8000 --host 127.0.0.1
                </pre>
              </div>
            ) : null}

            {report && (
              <dl className="grid grid-cols-[120px_1fr] gap-y-1 font-sans text-caption">
                <dt className="text-muted-foreground">Device</dt>
                <dd className="text-foreground">{report.device ?? "—"}</dd>
                <dt className="text-muted-foreground">dtype</dt>
                <dd className="text-foreground">{report.dtype ?? "—"}</dd>
                <dt className="text-muted-foreground">PyTorch</dt>
                <dd className="text-foreground">{report.torchVersion ?? "—"}</dd>
                <dt className="text-muted-foreground">Apple Silicon</dt>
                <dd className="text-foreground">{report.appleSilicon ? "yes" : "no"}</dd>
                <dt className="text-muted-foreground">Total memory</dt>
                <dd className="text-foreground">
                  {report.totalMemoryGb != null ? `${report.totalMemoryGb.toFixed(1)} GB` : "—"}
                </dd>
                <dt className="text-muted-foreground">Loaded model</dt>
                <dd className="text-foreground">{report.currentModelId ?? "(none yet)"}</dd>
                <dt className="text-muted-foreground" title="Native pixel resolution derived from the loaded pipeline's UNet/transformer sample_size × VAE scale factor.">Native size</dt>
                <dd className="text-foreground">
                  {report.nativeWidth && report.nativeHeight
                    ? `${report.nativeWidth} × ${report.nativeHeight}`
                    : "—"}
                </dd>
                <dt className="text-muted-foreground">Ready</dt>
                <dd className="text-foreground">{report.ready ? "yes" : "no"}</dd>
                <dt className="text-muted-foreground">Last check</dt>
                <dd className="text-foreground">
                  {lastCheck ? `${Math.round((Date.now() - lastCheck) / 1000)}s ago` : "—"}
                </dd>
              </dl>
            )}
          </div>
        </div>
      )}
    </>
  );
}
