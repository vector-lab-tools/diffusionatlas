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
export type WarmupStatus = "idle" | "warming" | "warm" | "failed" | "too-large";

/**
 * 413 payload from the backend when the loaded model wouldn't fit in
 * available memory. The user can override per-request, but the
 * StatusBar should warn before they spend time on a run that's likely
 * to push the OS into encrypted swap.
 */
export interface MemoryWarning {
  modelId: string;
  footprintGb: number;
  availableGb: number;
  headroomGb: number;
  message: string;
}

interface BackendHealthValue {
  status: HealthStatus;
  report: HealthReport | null;
  lastCheck: number | null;
  /**
   * Pipeline-warmup state. `warming` while a /warmup call is in flight,
   * `warm` once it has completed (or once /health reports the model is
   * already loaded — meaning a previous session warmed it). `failed` if
   * the warmup endpoint errored. `too-large` when the model wouldn't
   * fit in available RAM and the backend refused to load.
   */
  warmup: WarmupStatus;
  /** Set when warmup returned a 413 model-too-large response. */
  memoryWarning: MemoryWarning | null;
}

const Ctx = createContext<BackendHealthValue>({
  status: "checking",
  report: null,
  lastCheck: null,
  warmup: "idle",
  memoryWarning: null,
});

export function BackendHealthProvider({ children }: { children: ReactNode }) {
  const { settings } = useSettings();
  const [status, setStatus] = useState<HealthStatus>("checking");
  const [report, setReport] = useState<HealthReport | null>(null);
  const [lastCheck, setLastCheck] = useState<number | null>(null);
  const [warmup, setWarmup] = useState<WarmupStatus>("idle");
  const [memoryWarning, setMemoryWarning] = useState<MemoryWarning | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Track which (baseUrl, modelId) we've already warmed so we don't
  // refire warmup on every poll. Stored as a key so a model change
  // invalidates and re-warms.
  const warmedKeyRef = useRef<string | null>(null);

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

  // Auto-warmup: when the user is on Local backend, the health poll has
  // succeeded, and we haven't warmed this (baseUrl, modelId) before,
  // fire `/warmup` in the background. The first /generate after this
  // returns at full speed instead of paying the ~1-2 min MPS warmup
  // cost. If /health reports the model is already loaded (warm carry-
  // over from a prior session) we skip the call and mark warm directly.
  useEffect(() => {
    if (settings.backend !== "local") {
      setWarmup("idle");
      warmedKeyRef.current = null;
      return;
    }
    if (status !== "ok") return;
    if (!settings.modelId) return;

    const key = `${settings.localBaseUrl}::${settings.modelId}`;
    if (warmedKeyRef.current === key) return;

    // If /health already reports this model loaded, treat as warm — a
    // previous session warmed the kernels and they are still hot.
    if (report?.currentModelId === settings.modelId && report?.ready) {
      warmedKeyRef.current = key;
      setWarmup("warm");
      return;
    }

    let cancelled = false;
    setWarmup("warming");
    setMemoryWarning(null);
    void (async () => {
      try {
        const res = await fetch(`${settings.localBaseUrl}/warmup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ modelId: settings.modelId }),
        });
        if (res.status === 413) {
          // Pre-load fit check refused the model. Surface as a structured
          // warning so the StatusBar can prompt for an override.
          const body = await res.json().catch(() => ({}));
          const detail = (body && body.detail) || body;
          if (cancelled) return;
          setMemoryWarning({
            modelId: detail?.modelId ?? settings.modelId,
            footprintGb: detail?.footprintGb ?? 0,
            availableGb: detail?.availableGb ?? 0,
            headroomGb: detail?.headroomGb ?? 0,
            message: detail?.message ?? "Model wouldn't fit in available memory.",
          });
          warmedKeyRef.current = key;
          setWarmup("too-large");
          return;
        }
        if (!res.ok) throw new Error(`status ${res.status}`);
        await res.json();
        if (cancelled) return;
        warmedKeyRef.current = key;
        setWarmup("warm");
      } catch {
        if (cancelled) return;
        setWarmup("failed");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [settings.backend, settings.localBaseUrl, settings.modelId, status, report?.currentModelId, report?.ready]);

  const value = useMemo<BackendHealthValue>(
    () => ({ status, report, lastCheck, warmup, memoryWarning }),
    [status, report, lastCheck, warmup, memoryWarning],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useBackendHealth(): BackendHealthValue {
  return useContext(Ctx);
}
