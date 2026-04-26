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
}

const DEFAULT_SETTINGS: DiffusionSettings = {
  backend: "hosted",
  providerId: "replicate",
  modelId: "black-forest-labs/flux-schnell",
  defaults: { steps: 4, cfg: 0, width: 1024, height: 1024, scheduler: "DPMSolverMultistep" },
  localBaseUrl: "http://localhost:8000",
  apiKeys: {},
  darkMode: false,
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
      if (raw) setSettingsState({ ...DEFAULT_SETTINGS, ...JSON.parse(raw) });
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
