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
