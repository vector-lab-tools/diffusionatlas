/**
 * Shared term definitions used across operations and exports.
 *
 * Two consumers:
 *   1. Table headers and form-field labels render `title` attributes for hover.
 *   2. PDF exports draw a "Key" section listing the terms used in that doc.
 *
 * Definitions kept short; the help-dropdown panel covers the longer arguments.
 */

export interface Term {
  term: string;
  definition: string;
}

/** Lookup by short label/header text (case-sensitive). */
export const GLOSSARY: Record<string, string> = {
  // Form fields
  Prompt: "The text input describing what the model should generate.",
  Seed: "Random seed for reproducibility. Same seed + same parameters + same model = same image.",
  Steps: "Number of denoising steps. More steps usually means cleaner output but slower generation.",
  CFG: "Classifier-Free Guidance scale. Higher values push the trajectory more aggressively toward the prompt; too high causes oversaturated mode collapse, too low drifts off-prompt.",
  "CFG list": "Comma-separated list of CFG values to sweep. The drift curve plots distance from the baseline (CFG nearest 7.5).",
  "Anchor seed": "Centre of the neighbourhood. Other seeds are deterministic offsets within ±radius.",
  "k samples": "Number of seeds in the neighbourhood (anchor included). Larger k probes more of the local manifold but costs more API calls.",
  Radius: "Maximum integer offset from the anchor seed. Larger values probe how connected the manifold is at distance.",
  "Preview every": "How often to decode an intermediate latent through the VAE for a thumbnail. 0 disables previews. Each thumbnail adds one VAE decode to the trajectory time.",
  Threshold: "CLIP cosine similarity cutoff. Above this is pass, below is fail. 0.20 is permissive, 0.30 is strict — what counts as 'matching' is a methodological choice.",

  // Table column headers
  lane: "Which provider lane this row belongs to: primary (the main backend) or compare (the second backend in cross-backend mode).",
  status: "Pending → queued; Running → in flight; Ok → returned successfully; Error → failed (see error column).",
  provider: "The provider that handled this call (e.g. replicate, fal, local).",
  model: "The model identifier the provider was given.",
  time: "Wall-clock time from request to response, in seconds.",
  drift: "Normalised perceptual-hash distance from the baseline (CFG nearest 7.5). 0 = identical, 1 = completely different.",
  error: "Error message if the call failed.",
  task: "Short id of the bench task (e.g. so-1, cb-3).",
  category: "Compositional category: single object, two objects, counting, or colour binding.",
  verdict: "Pass/fail mark. Set manually or from CLIP auto-score.",
  CLIP: "Cosine similarity between the image and prompt as embedded by CLIP (typically 0.18 – 0.35 for matching pairs).",
  pass: "Number of tasks marked pass.",
  fail: "Number of tasks marked fail.",
  pending: "Number of tasks not yet scored.",
  accuracy: "Pass / (pass + fail) as a percentage. Pending tasks excluded.",
  role: "Whether the seed is the anchor (centre of the neighbourhood) or a neighbour offset from it.",
  step: "Denoising step number, 1-indexed.",
  preview: "Whether a VAE-decoded thumbnail was emitted for this step.",

  // Latent-geometry symbols
  "‖z‖": "L2 norm of the per-step latent. Tends to be largest at the noisy start and shrinks as denoising converges.",
  "Δ to prev": "L2 distance between this step's latent and the previous step's latent. The length of one denoising step in latent space.",
  "cos→final": "Cosine similarity between this step's latent and the final latent. Rises from near 0 (pure noise) to 1 (the destination) as denoising progresses.",
  "image_size": "Sweep / Neighbourhood: width × height. Trajectory: latent dim × scale factor.",
};

export function lookup(term: string): string | undefined {
  return GLOSSARY[term];
}

/** Build a glossary slice covering the headers a table actually uses. */
export function termsFor(headers: string[]): Term[] {
  const out: Term[] = [];
  const seen = new Set<string>();
  for (const h of headers) {
    if (seen.has(h)) continue;
    const def = GLOSSARY[h];
    if (def) {
      out.push({ term: h, definition: def });
      seen.add(h);
    }
  }
  return out;
}
