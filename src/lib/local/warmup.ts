/**
 * Shared copy for the "first local /generate is slow" UX. Sweep,
 * Neighbourhood, and Bench all show the same short message in their
 * first cell while the MPS pipeline warms; hovering it reveals the
 * full explanation. Keeping the strings here means future tweaks land
 * in one place rather than three.
 */

export const WARMUP_LABEL = "Generating · first call may warm the MPS pipeline (~1–2 min)";

export const WARMUP_TOOLTIP =
  "Apple Silicon (MPS) compiles GPU kernels lazily on the first inference, fills attention buffers, " +
  "and hits the VAE decoder for the first time — that one-off cost is usually 1–2 minutes for SD 1.5 " +
  "at 512×512. Subsequent cells reuse the warmed pipeline and typically take 1–3 seconds each. " +
  "If the backend was already warmed (the StatusBar shows a green 'warm' dot), this label won't appear " +
  "and the first cell runs at full speed.";

/** True when an in-flight cell's status text is the warmup notice. */
export function isWarmupMessage(s: string | undefined | null): boolean {
  return typeof s === "string" && s.startsWith("Generating · first call");
}

/**
 * Translate a verbose backend error into a short cell-sized label,
 * keeping the full original message available for a hover tooltip.
 * The dark contact-sheet frames are 150 px wide — anything longer
 * than ~5 words wraps badly and looks like a wall of red caps.
 */
export function shortenBackendError(msg: string | undefined | null): { short: string; full: string } {
  const full = msg ?? "Failed";
  if (!msg) return { short: "failed", full };
  // NaN-in-VAE all-black detection.
  if (/NaN in the VAE/i.test(msg) || /all-black image/i.test(msg)) {
    const nanExplanation =
      "NaN = \"Not a Number\" — a special value in IEEE 754 floating-point " +
      "that represents the result of an undefined or unrepresentable " +
      "arithmetic operation. It shows up when the hardware computes " +
      "something like 0/0, ∞ − ∞, √(negative), log(0), or when a " +
      "multiplication or summation overflows the representable range and " +
      "the result has no meaningful numeric value. In a diffusion model, " +
      "once one position in a tensor goes NaN, the next attention layer's " +
      "matrix multiplication propagates it across every position, so by " +
      "the time the VAE decodes the final latent every pixel is NaN — " +
      "PIL clamps NaN to 0 when saving, producing the all-black image " +
      "you're seeing.\n" +
      "─────────────────────────\n" +
      "» Backend response (verbatim) «\n\n" +
      full;
    return { short: "NaN — model couldn't render", full: nanExplanation };
  }
  // Memory pressure.
  if (/out of memory/i.test(msg) || /MPS backend out of memory/i.test(msg)) {
    return { short: "out of memory", full };
  }
  // Resolution too large for model.
  if (/Resolution.*is too large/i.test(msg)) {
    return { short: "resolution too large", full };
  }
  // Auth / payment.
  if (/auth/i.test(msg) || /api.*key/i.test(msg)) {
    return { short: "auth failed", full };
  }
  if (/payment_required|insufficient credit/i.test(msg)) {
    return { short: "out of credit", full };
  }
  // Rate limit.
  if (/rate.?limit/i.test(msg)) {
    return { short: "rate limited", full };
  }
  // Aborted by user.
  if (/Stopped by user|aborted/i.test(msg)) {
    return { short: "stopped", full };
  }
  // Generic short fallback — first 4 words, lowercased.
  const trimmed = msg.split(/[.!?\n]/)[0].split(/\s+/).slice(0, 4).join(" ").toLowerCase();
  return { short: trimmed || "failed", full };
}
