"use client";

/**
 * Single shared poller for the local backend's /health endpoint.
 *
 * Lifts what was an internal hook of the StatusBar's BackendHealth dot into
 * a context, so other parts of the app (notably DenoiseTrajectory's
 * Width/Height selects, which want to default to the loaded model's native
 * resolution) can read the same report without each spawning its own 5-s
 * interval.
 */
import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useSettings } from "@/context/DiffusionSettingsContext";

export interface HealthReport {
  device?: string;
  dtype?: string;
  torchVersion?: string;
  appleSilicon?: boolean;
  totalMemoryGb?: number | null;
  currentModelId?: string | null;
  /** Native pixel width derived from the loaded pipeline's UNet/transformer sample_size × vae_scale_factor. */
  nativeWidth?: number | null;
  nativeHeight?: number | null;
  ready?: boolean;
}

export type HealthStatus = "checking" | "ok" | "error";

interface BackendHealthValue {
  status: HealthStatus;
  report: HealthReport | null;
  lastCheck: number | null;
}

const Ctx = createContext<BackendHealthValue>({ status: "checking", report: null, lastCheck: null });

export function BackendHealthProvider({ children }: { children: ReactNode }) {
  const { settings } = useSettings();
  const [status, setStatus] = useState<HealthStatus>("checking");
  const [report, setReport] = useState<HealthReport | null>(null);
  const [lastCheck, setLastCheck] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const ctrl = new AbortController();
        const timeout = setTimeout(() => ctrl.abort(), 3000);
        const res = await fetch(`${settings.localBaseUrl}/health`, { method: "GET", signal: ctrl.signal });
        clearTimeout(timeout);
        if (!res.ok) throw new Error(`status ${res.status}`);
        const data: HealthReport = await res.json();
        if (cancelled) return;
        setReport(data);
        setStatus("ok");
        setLastCheck(Date.now());
      } catch {
        if (cancelled) return;
        setStatus("error");
        setLastCheck(Date.now());
      }
      if (!cancelled) timerRef.current = setTimeout(check, 5000);
    }

    void check();
    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [settings.localBaseUrl]);

  const value = useMemo<BackendHealthValue>(() => ({ status, report, lastCheck }), [status, report, lastCheck]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useBackendHealth(): BackendHealthValue {
  return useContext(Ctx);
}
