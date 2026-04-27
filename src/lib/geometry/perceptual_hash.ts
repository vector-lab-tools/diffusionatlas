/**
 * Tiny perceptual hash (aHash variant) for measuring "how different are
 * these two images?" without needing CLIP. Suitable for the Sweep drift
 * curve: cheap, deterministic, runs in the browser via Canvas.
 *
 * 16x16 grayscale, mean threshold, 256-bit hash. Hamming distance between
 * two hashes is the drift score. Normalised to [0, 1] by dividing by 256.
 */

const HASH_SIDE = 16;
const HASH_BITS = HASH_SIDE * HASH_SIDE;

export type Hash = Uint8Array; // 256 bits packed into 32 bytes

export async function ahash(dataUrl: string): Promise<Hash> {
  const img = await loadImage(dataUrl);
  const canvas = document.createElement("canvas");
  canvas.width = HASH_SIDE;
  canvas.height = HASH_SIDE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(img, 0, 0, HASH_SIDE, HASH_SIDE);
  const { data } = ctx.getImageData(0, 0, HASH_SIDE, HASH_SIDE);

  const grays = new Float32Array(HASH_BITS);
  let sum = 0;
  for (let i = 0; i < HASH_BITS; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const v = 0.299 * r + 0.587 * g + 0.114 * b;
    grays[i] = v;
    sum += v;
  }
  const mean = sum / HASH_BITS;

  const hash = new Uint8Array(HASH_BITS / 8);
  for (let i = 0; i < HASH_BITS; i++) {
    if (grays[i] >= mean) {
      hash[i >> 3] |= 1 << (i & 7);
    }
  }
  return hash;
}

export function hammingDistance(a: Hash, b: Hash): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    let x = a[i] ^ b[i];
    while (x) {
      x &= x - 1;
      d++;
    }
  }
  return d;
}

export function normalisedDrift(a: Hash, b: Hash): number {
  return hammingDistance(a, b) / HASH_BITS;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image load failed"));
    img.src = src;
  });
}
