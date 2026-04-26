"use client";

import { X } from "lucide-react";
import { useSettings, type Backend, type ProviderId } from "@/context/DiffusionSettingsContext";
import { useLatentCache } from "@/context/LatentCacheContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";

const HOSTED_PROVIDERS: ProviderId[] = ["replicate", "fal", "together", "stability"];

export function SettingsPanel() {
  const { settings, setSettings, settingsOpen, setSettingsOpen } = useSettings();
  const { count: latentCount, clear: clearLatents } = useLatentCache();
  const { count: imageCount, clear: clearImages } = useImageBlobCache();

  if (!settingsOpen) return null;

  const setBackend = (backend: Backend) => setSettings({ ...settings, backend, providerId: backend === "local" ? "local" : "replicate" });
  const setProvider = (providerId: ProviderId) => setSettings({ ...settings, providerId });
  const setApiKey = (provider: ProviderId, key: string) =>
    setSettings({ ...settings, apiKeys: { ...settings.apiKeys, [provider]: key } });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40">
      <div className="card-editorial w-full max-w-2xl max-h-[85vh] overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="font-display text-display-md font-bold text-burgundy">Settings</h2>
          <button
            onClick={() => setSettingsOpen(false)}
            className="btn-editorial-ghost px-2 py-2"
            aria-label="Close"
          >
            <X size={16} />
          </button>
        </div>

        <section className="mb-6">
          <h3 className="font-sans text-body-sm font-semibold mb-2 uppercase tracking-wider text-muted-foreground">
            Backend
          </h3>
          <div className="flex gap-2">
            {(["hosted", "local"] as Backend[]).map((b) => (
              <button
                key={b}
                onClick={() => setBackend(b)}
                className={
                  settings.backend === b ? "btn-editorial-primary" : "btn-editorial-secondary"
                }
              >
                {b === "hosted" ? "Hosted" : "Local FastAPI"}
              </button>
            ))}
          </div>
        </section>

        {settings.backend === "hosted" ? (
          <section className="mb-6">
            <h3 className="font-sans text-body-sm font-semibold mb-2 uppercase tracking-wider text-muted-foreground">
              Provider
            </h3>
            <select
              value={settings.providerId}
              onChange={(e) => setProvider(e.target.value as ProviderId)}
              className="input-editorial"
            >
              {HOSTED_PROVIDERS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>

            <h3 className="font-sans text-body-sm font-semibold mt-4 mb-2 uppercase tracking-wider text-muted-foreground">
              API Keys
            </h3>
            {HOSTED_PROVIDERS.map((p) => (
              <div key={p} className="mb-2">
                <label className="font-sans text-caption text-muted-foreground block mb-1">
                  {p}
                </label>
                <input
                  type="password"
                  value={settings.apiKeys[p] ?? ""}
                  onChange={(e) => setApiKey(p, e.target.value)}
                  className="input-editorial"
                  placeholder={`${p} API key`}
                />
              </div>
            ))}
          </section>
        ) : (
          <section className="mb-6">
            <h3 className="font-sans text-body-sm font-semibold mb-2 uppercase tracking-wider text-muted-foreground">
              Local FastAPI URL
            </h3>
            <input
              type="text"
              value={settings.localBaseUrl}
              onChange={(e) => setSettings({ ...settings, localBaseUrl: e.target.value })}
              className="input-editorial"
              placeholder="http://localhost:8000"
            />
          </section>
        )}

        <section className="mb-6">
          <h3 className="font-sans text-body-sm font-semibold mb-2 uppercase tracking-wider text-muted-foreground">
            Model
          </h3>
          <input
            type="text"
            value={settings.modelId}
            onChange={(e) => setSettings({ ...settings, modelId: e.target.value })}
            className="input-editorial"
            placeholder="model id"
          />
        </section>

        <section className="mb-6">
          <h3 className="font-sans text-body-sm font-semibold mb-2 uppercase tracking-wider text-muted-foreground">
            Cache
          </h3>
          <div className="flex items-center gap-3 mb-2 font-sans text-body-sm">
            <span>{latentCount} latents</span>
            <button onClick={() => void clearLatents()} className="btn-editorial-ghost px-3 py-1">
              Clear
            </button>
          </div>
          <div className="flex items-center gap-3 font-sans text-body-sm">
            <span>{imageCount} images</span>
            <button onClick={() => void clearImages()} className="btn-editorial-ghost px-3 py-1">
              Clear
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
