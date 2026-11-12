"use client";

import { useEffect, useState } from "react";
import { X, Eye, EyeOff } from "lucide-react";
import {
  useSettings,
  type Backend,
  type DiffusionSettings,
  type ProviderId,
} from "@/context/DiffusionSettingsContext";
import { useLatentCache } from "@/context/LatentCacheContext";
import { useImageBlobCache } from "@/context/ImageBlobCacheContext";
import {
  HOSTED_PROVIDERS,
  PROVIDER_DEFAULT_MODEL,
  PROVIDER_SUGGESTED_MODELS,
  providerLabel,
} from "@/lib/providers/defaults";

const SECTION_HEADING =
  "font-sans text-body-sm font-semibold mb-2 uppercase tracking-wider text-muted-foreground";

export function SettingsPanel() {
  const { settings, setSettings, settingsOpen, setSettingsOpen } = useSettings();
  const { count: latentCount, clear: clearLatents } = useLatentCache();
  const { count: imageCount, clear: clearImages } = useImageBlobCache();
  const [keyStatus, setKeyStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [visibleKeys, setVisibleKeys] = useState<Set<ProviderId>>(new Set());

  // Edits go to a draft taken when the panel opens and reach the saved
  // settings only on Save. Closing with unsaved changes asks first.
  const [draft, setDraft] = useState<DiffusionSettings>(settings);
  const [confirmClose, setConfirmClose] = useState(false);

  useEffect(() => {
    if (settingsOpen) {
      setDraft(settings);
      setConfirmClose(false);
      setKeyStatus("idle");
    }
  }, [settingsOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keys saved in .env.local fill gaps in both the saved settings and the
  // draft, so restoring them does not count as an unsaved change.
  useEffect(() => {
    if (!settingsOpen) return;
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch("/api/keys");
        if (!res.ok) return;
        const stored: Partial<Record<ProviderId, string>> = await res.json();
        if (cancelled) return;
        const missing: Partial<Record<ProviderId, string>> = {};
        for (const [pid, val] of Object.entries(stored) as [ProviderId, string][]) {
          if (!settings.apiKeys[pid] && val) missing[pid] = val;
        }
        if (Object.keys(missing).length > 0) {
          setSettings({ ...settings, apiKeys: { ...settings.apiKeys, ...missing } });
          setDraft((prev) => ({ ...prev, apiKeys: { ...missing, ...prev.apiKeys } }));
        }
      } catch {
        /* offline or no .env.local */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [settingsOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  const dirty = JSON.stringify(draft) !== JSON.stringify(settings);

  const close = () => {
    setConfirmClose(false);
    setSettingsOpen(false);
  };
  const saveAndClose = () => {
    setSettings(draft);
    close();
  };
  const requestClose = () => {
    if (dirty) setConfirmClose(true);
    else close();
  };

  useEffect(() => {
    if (!settingsOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (confirmClose) setConfirmClose(false);
      else if (dirty) setConfirmClose(true);
      else setSettingsOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [settingsOpen, confirmClose, dirty, setSettingsOpen]);

  if (!settingsOpen) return null;

  const update = (patch: Partial<DiffusionSettings>) => setDraft((prev) => ({ ...prev, ...patch }));

  const toggleKeyVisibility = (pid: ProviderId) => {
    setVisibleKeys((prev) => {
      const next = new Set(prev);
      if (next.has(pid)) next.delete(pid);
      else next.add(pid);
      return next;
    });
  };

  const setBackend = (backend: Backend) => {
    const providerId: ProviderId = backend === "local" ? "local" : "replicate";
    update({ backend, providerId, modelId: PROVIDER_DEFAULT_MODEL[providerId] });
  };
  const setProvider = (providerId: ProviderId) =>
    update({ providerId, modelId: PROVIDER_DEFAULT_MODEL[providerId] });
  const setApiKey = (provider: ProviderId, key: string) =>
    setDraft((prev) => ({ ...prev, apiKeys: { ...prev.apiKeys, [provider]: key } }));

  async function persistKeys() {
    setKeyStatus("saving");
    try {
      const res = await fetch("/api/keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft.apiKeys),
      });
      setKeyStatus(res.ok ? "saved" : "error");
      if (res.ok) setTimeout(() => setKeyStatus("idle"), 2000);
    } catch {
      setKeyStatus("error");
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4" onClick={requestClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        className="relative card-editorial shadow-editorial-lg w-full max-w-2xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-parchment shrink-0">
          <h2 id="settings-title" className="font-display text-display-md font-bold text-burgundy">
            Settings
          </h2>
          <button onClick={requestClose} className="btn-editorial-ghost px-2 py-2" aria-label="Close settings">
            <X size={16} />
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-3 px-6 py-3 border-b border-parchment shrink-0">
          <span className="font-sans text-caption uppercase tracking-wider text-muted-foreground font-semibold">
            Backend
          </span>
          {(["hosted", "local"] as Backend[]).map((b) => (
            <button
              key={b}
              onClick={() => setBackend(b)}
              className={`${draft.backend === b ? "btn-editorial-primary" : "btn-editorial-secondary"} px-3 py-1.5`}
            >
              {b === "hosted" ? "Hosted" : "Local FastAPI"}
            </button>
          ))}
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto px-6 py-5">
          {draft.backend === "hosted" ? (
            <section className="mb-6">
              <h3 className={SECTION_HEADING}>Provider</h3>
              <select
                value={draft.providerId}
                onChange={(e) => setProvider(e.target.value as ProviderId)}
                className="input-editorial"
              >
                {HOSTED_PROVIDERS.map((p) => (
                  <option key={p} value={p}>
                    {providerLabel(p)}
                  </option>
                ))}
              </select>

              <h3 className={`${SECTION_HEADING} mt-4`}>API Keys</h3>
              {HOSTED_PROVIDERS.map((p) => (
                <div key={p} className="mb-2">
                  <label className="font-sans text-caption text-muted-foreground block mb-1">
                    {providerLabel(p)}
                  </label>
                  <div className="relative">
                    <input
                      type={visibleKeys.has(p) ? "text" : "password"}
                      value={draft.apiKeys[p] ?? ""}
                      onChange={(e) => setApiKey(p, e.target.value)}
                      className="input-editorial pr-10"
                      placeholder={`${providerLabel(p)} API key`}
                    />
                    <button
                      type="button"
                      onClick={() => toggleKeyVisibility(p)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors p-1"
                      title={visibleKeys.has(p) ? "Hide API key" : "Show API key"}
                    >
                      {visibleKeys.has(p) ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
              ))}
              <div className="mt-3 flex items-center gap-3">
                <button onClick={() => void persistKeys()} className="btn-editorial-secondary px-3 py-2">
                  Save to .env.local
                </button>
                <span className="font-sans text-caption text-muted-foreground">
                  {keyStatus === "saving" && "Saving…"}
                  {keyStatus === "saved" && "Saved"}
                  {keyStatus === "error" && "Save failed"}
                  {keyStatus === "idle" && "Persists keys across dev restarts"}
                </span>
              </div>
            </section>
          ) : (
            <section className="mb-6">
              <h3 className={SECTION_HEADING}>Local FastAPI URL</h3>
              <input
                type="text"
                value={draft.localBaseUrl}
                onChange={(e) => update({ localBaseUrl: e.target.value })}
                className="input-editorial"
                placeholder="http://localhost:8000"
              />
            </section>
          )}

          <section className="mb-6">
            <h3 className={SECTION_HEADING}>Model</h3>
            <input
              type="text"
              list="model-suggestions"
              value={draft.modelId}
              onChange={(e) => update({ modelId: e.target.value.trim() })}
              className="input-editorial"
              placeholder="model id"
              spellCheck={false}
              autoComplete="off"
            />
            <datalist id="model-suggestions">
              {PROVIDER_SUGGESTED_MODELS[draft.providerId].map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
            <p className="mt-1 font-sans text-caption text-muted-foreground italic">
              Pick a suggestion or type any model id from the provider&apos;s catalogue
              {draft.providerId === "local" ? " (a Hugging Face id)" : ""}. Suggestions can go out of date; a typed id is used as entered.
            </p>
            <button
              onClick={() => update({ modelId: PROVIDER_DEFAULT_MODEL[draft.providerId] })}
              className="btn-editorial-ghost px-3 py-1 mt-2 font-sans text-caption"
            >
              Reset to default
            </button>
          </section>

          <section className="mb-6">
            <h3 className={SECTION_HEADING}>Fast mode</h3>
            <label className="flex items-start gap-2 font-sans text-body-sm mb-2">
              <input
                type="checkbox"
                checked={draft.fastMode}
                onChange={(e) => update({ fastMode: e.target.checked })}
                className="mt-1"
              />
              <span>
                Cap denoising steps globally
                <span className="block font-sans text-caption text-muted-foreground italic">
                  When on, every operation clamps its <code>Steps</code> to the cap below regardless of the form value. Useful for fast iteration on prompts and UX before committing to a quality run.
                </span>
              </span>
            </label>
            <label className="flex items-center gap-2 font-sans text-caption">
              <span className="text-muted-foreground">Max steps</span>
              <input
                type="number"
                min={1}
                max={100}
                value={draft.fastModeMaxSteps}
                onChange={(e) => update({ fastModeMaxSteps: parseInt(e.target.value, 10) || 8 })}
                className="input-editorial w-24 py-1"
                disabled={!draft.fastMode}
              />
            </label>
          </section>

          <section>
            <h3 className={SECTION_HEADING}>Cache</h3>
            <p className="font-sans text-caption text-muted-foreground italic mb-2">
              Clearing a cache takes effect at once and is not undone by Cancel.
            </p>
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

        <div className="px-6 py-3 border-t border-parchment flex flex-wrap items-center justify-between gap-3 shrink-0">
          <span className="font-sans text-caption text-muted-foreground">API keys are stored in your browser only.</span>
          <div className="flex items-center gap-2">
            {dirty && <span className="font-sans text-caption text-burgundy">Unsaved changes</span>}
            <button onClick={requestClose} className="btn-editorial-ghost px-3 py-1.5 font-sans text-caption">
              Cancel
            </button>
            <button
              onClick={saveAndClose}
              disabled={!dirty}
              className="btn-editorial-primary px-3 py-1.5 font-sans text-caption disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
          </div>
        </div>

        {confirmClose && (
          <div
            className="absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-ink/30"
            onClick={() => setConfirmClose(false)}
          >
            <div
              role="alertdialog"
              aria-labelledby="settings-unsaved-title"
              className="card-editorial shadow-editorial-lg p-5 w-full max-w-sm mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 id="settings-unsaved-title" className="font-display text-body-lg font-semibold mb-1.5">
                Save your changes?
              </h3>
              <p className="font-sans text-body-sm text-muted-foreground mb-4">
                You changed the settings. Save them, or discard and keep the previous settings.
              </p>
              <div className="flex justify-end gap-2">
                <button onClick={() => setConfirmClose(false)} className="btn-editorial-ghost px-3 py-1.5 font-sans text-caption">
                  Keep editing
                </button>
                <button onClick={close} className="btn-editorial-ghost px-3 py-1.5 font-sans text-caption">
                  Discard
                </button>
                <button onClick={saveAndClose} className="btn-editorial-primary px-3 py-1.5 font-sans text-caption" autoFocus>
                  Save
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
