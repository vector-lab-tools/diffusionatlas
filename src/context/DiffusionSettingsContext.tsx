"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { type Backend, type ProviderId } from "@/lib/providers/types";

export type { Backend, ProviderId };

export interface DiffusionSettings {
  backend: Backend;
  providerId: ProviderId;
  modelId: string;
  defaults: {
    steps: number;
    cfg: number;
    width: number;
    height: number;
    scheduler: string;
  };
  localBaseUrl: string;
  apiKeys: Partial<Record<ProviderId, string>>;
  darkMode: boolean;
  /** Global override: when on, every operation clamps its step count to fastModeMaxSteps. */
  fastMode: boolean;
  fastModeMaxSteps: number;
}

/** Resolve the effective step count given a form value and current settings. */
export function effectiveSteps(formSteps: number, settings: DiffusionSettings): number {
  if (settings.fastMode) return Math.min(formSteps, settings.fastModeMaxSteps);
  return formSteps;
}

const DEFAULT_SETTINGS: DiffusionSettings = {
  backend: "hosted",
  providerId: "replicate",
  modelId: "black-forest-labs/flux-schnell",
  // 512×512 is the safe default — SD 1.5 (the local default) is trained at
  // 512 and fp32 attention at 1024 needs ~4 GB per forward, blowing the
  // MPS watermark cap on a 24 GB box. SDXL/FLUX users can bump to 1024 in
  // Settings; DenoiseTrajectory already auto-snaps to the loaded model's
  // native size via /health, so this default mainly affects the hosted
  // lanes of Sweep / Neighbourhood / Bench.
  defaults: { steps: 4, cfg: 0, width: 512, height: 512, scheduler: "DPMSolverMultistep" },
  localBaseUrl: "http://localhost:8000",
  apiKeys: {},
  darkMode: false,
  fastMode: false,
  fastModeMaxSteps: 8,
};

interface SettingsCtx {
  settings: DiffusionSettings;
  setSettings: (s: DiffusionSettings) => void;
  toggleDarkMode: () => void;
  settingsOpen: boolean;
  setSettingsOpen: (open: boolean) => void;
}

const Ctx = createContext<SettingsCtx | null>(null);
const STORAGE_KEY = "diffusion-atlas:settings";

export function DiffusionSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettingsState] = useState<DiffusionSettings>(DEFAULT_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const stored = JSON.parse(raw) as Partial<DiffusionSettings>;
      // Migrate archived/deprecated model IDs to the current default.
      const ARCHIVED = new Set(["stability-ai/sdxl", "stability-ai/stable-diffusion"]);
      if (stored.modelId && ARCHIVED.has(stored.modelId)) {
        stored.modelId = DEFAULT_SETTINGS.modelId;
      }
      setSettingsState({ ...DEFAULT_SETTINGS, ...stored });
    } catch {}
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", settings.darkMode);
  }, [settings.darkMode]);

  const setSettings = (s: DiffusionSettings) => {
    setSettingsState(s);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
    } catch {}
  };

  const toggleDarkMode = () => setSettings({ ...settings, darkMode: !settings.darkMode });

  return (
    <Ctx.Provider value={{ settings, setSettings, toggleDarkMode, settingsOpen, setSettingsOpen }}>
      {children}
    </Ctx.Provider>
  );
}

export function useSettings() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useSettings must be used within DiffusionSettingsProvider");
  return ctx;
}
