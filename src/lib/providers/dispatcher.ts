/**
 * Provider dispatcher: maps a provider id to its DiffusionProvider impl.
 */

import { type DiffusionProvider, type ProviderId } from "./types";
import { replicateProvider } from "./replicate";

const REGISTRY: Partial<Record<ProviderId, DiffusionProvider>> = {
  replicate: replicateProvider,
  // fal, together, stability, local — to follow
};

export function getProvider(id: ProviderId): DiffusionProvider {
  const provider = REGISTRY[id];
  if (!provider) {
    throw new Error(`Provider '${id}' not yet implemented`);
  }
  return provider;
}

export function listImplementedProviders(): ProviderId[] {
  return Object.keys(REGISTRY) as ProviderId[];
}
