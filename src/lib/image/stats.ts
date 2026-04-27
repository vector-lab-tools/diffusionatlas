/**
 * Per-image quantitative stats computed in-browser via Canvas.
 *
 * Designed for diffusion-trajectory analysis: every metric here is one
 * a researcher would expect to *change* across denoising steps. Tracking
 * how each scalar evolves is itself a finding.
 *
 *   - meanR/G/B — colour cast. Drift across steps reveals channel bias
 *     in the training data or VAE.
 *   - meanLuma + stdLuma — brightness, contrast. CFG sweeps tend to
 *     amplify both at high guidance.
 *   - saturation — mean chroma. Mode collapse at high CFG inflates this.
 *   - entropy — Shannon entropy of the luma histogram, in bits.
 *     Pure noise → ~8 bits; structured images → 6-7. The drop across
 *     steps is the signature of denoising as information reduction.
 *   - edgeDensity — mean Sobel magnitude. Noise: uniform high edges
 *     everywhere; structured images: edges concentrated at object
 *     boundaries. The total often rises mid-trajectory then falls.
 *   - centreX / centreY — luma-weighted centre of mass, normalised to
 *     [0,1]. Shows where the model "decided" the object should sit.
 *   - highFreqRatio — energy in (image − Gaussian-blur(image)) divided
 *     by total energy. Diffusion famously denoises high frequencies
 *     first; this should fall sharply early on.
 *   - pngBytes — size of the canvas re-encoded as PNG. A practical
 *     proxy for Kolmogorov complexity: compressible images have
 *     structure, incompressible ones are noise.
 */

export interface ImageStats {
  width: number;
  height: number;
  meanR: number;
  meanG: number;
  meanB: number;
  meanLuma: number;
  stdLuma: number;
  saturation: number;
  entropy: number;
  edgeDensity: number;
  centreX: number;
  centreY: number;
  highFreqRatio: number;
  pngBytes: number;
  /** 32-bin histograms per channel + luma. Each entry is the count in that bin. */
  histR: number[];
  histG: number[];
  histB: number[];
  histLuma: number[];
  /** Mean hue, in degrees [0, 360). Cultural-analytics indicator a la Manovich. */
  meanHue: number;
}

const cache = new Map<string, ImageStats>();

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image load failed"));
    img.src = src;
  });
}

/**
 * Box blur (3x3) — cheap stand-in for Gaussian blur for the
 * high-frequency residual computation. Only applied on a downscaled
 * canvas so the cost stays in the millisecond range.
 */
function boxBlur(data: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(data.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, count = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          const idx = (ny * w + nx) * 4;
          r += data[idx];
          g += data[idx + 1];
          b += data[idx + 2];
          count++;
        }
      }
      const oIdx = (y * w + x) * 4;
      out[oIdx] = r / count;
      out[oIdx + 1] = g / count;
      out[oIdx + 2] = b / count;
      out[oIdx + 3] = data[oIdx + 3];
    }
  }
  return out;
}

export async function computeImageStats(src: string): Promise<ImageStats> {
  const cached = cache.get(src);
  if (cached) return cached;

  const img = await loadImage(src);
  // Cap size so stats are fast even on large images.
  const MAX = 256;
  const scale = Math.min(MAX / img.width, MAX / img.height, 1);
  const W = Math.max(1, Math.round(img.width * scale));
  const H = Math.max(1, Math.round(img.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(img, 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H);
  const N = W * H;

  // Mean RGB, mean luma, mean chroma (saturation proxy), per-channel histograms,
  // mean hue. One pass, accumulating everything we need for the stats panel.
  let sumR = 0, sumG = 0, sumB = 0, sumLuma = 0, sumLuma2 = 0, sumChroma = 0;
  const histogram = new Uint32Array(256);
  // 32-bin coarse histograms for compact display.
  const hR = new Array(32).fill(0);
  const hG = new Array(32).fill(0);
  const hB = new Array(32).fill(0);
  const hL = new Array(32).fill(0);
  // Sum hue as a unit vector to handle the 0/360 wrap-around correctly.
  let hueX = 0, hueY = 0, hueW = 0;
  let cmX = 0, cmY = 0, cmW = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const chroma = max - min;
      sumR += r;
      sumG += g;
      sumB += b;
      sumLuma += luma;
      sumLuma2 += luma * luma;
      sumChroma += chroma;
      histogram[Math.round(luma) | 0]++;
      hR[(r >> 3) & 0x1f]++;
      hG[(g >> 3) & 0x1f]++;
      hB[(b >> 3) & 0x1f]++;
      hL[(Math.round(luma) >> 3) & 0x1f]++;
      // Hue (degrees) — only meaningful when there's some chroma.
      if (chroma > 0) {
        let hue = 0;
        if (max === r) hue = ((g - b) / chroma) % 6;
        else if (max === g) hue = (b - r) / chroma + 2;
        else hue = (r - g) / chroma + 4;
        hue *= 60;
        if (hue < 0) hue += 360;
        const rad = (hue * Math.PI) / 180;
        // Weight by chroma so achromatic pixels don't drag the mean.
        hueX += Math.cos(rad) * chroma;
        hueY += Math.sin(rad) * chroma;
        hueW += chroma;
      }
      cmX += x * luma;
      cmY += y * luma;
      cmW += luma;
    }
  }
  const meanHueRad = hueW > 0 ? Math.atan2(hueY / hueW, hueX / hueW) : 0;
  let meanHue = (meanHueRad * 180) / Math.PI;
  if (meanHue < 0) meanHue += 360;
  const meanR = sumR / N;
  const meanG = sumG / N;
  const meanB = sumB / N;
  const meanLuma = sumLuma / N;
  const varLuma = sumLuma2 / N - meanLuma * meanLuma;
  const stdLuma = Math.sqrt(Math.max(0, varLuma));
  const saturation = sumChroma / N / 255;

  // Shannon entropy of luma histogram (bits)
  let entropy = 0;
  for (let i = 0; i < 256; i++) {
    if (histogram[i] === 0) continue;
    const p = histogram[i] / N;
    entropy -= p * Math.log2(p);
  }

  const centreX = cmW > 0 ? cmX / cmW / W : 0.5;
  const centreY = cmW > 0 ? cmY / cmW / H : 0.5;

  // Edge density via 3x3 Sobel on the luma channel.
  const luma = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const j = i * 4;
    luma[i] = 0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2];
  }
  let edgeSum = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const gx =
        -luma[i - W - 1] - 2 * luma[i - 1] - luma[i + W - 1] +
        luma[i - W + 1] + 2 * luma[i + 1] + luma[i + W + 1];
      const gy =
        -luma[i - W - 1] - 2 * luma[i - W] - luma[i - W + 1] +
        luma[i + W - 1] + 2 * luma[i + W] + luma[i + W + 1];
      edgeSum += Math.sqrt(gx * gx + gy * gy);
    }
  }
  const edgeDensity = edgeSum / ((W - 2) * (H - 2)) / 255;

  // High-frequency ratio: variance of (image − blurred) over variance of image.
  const blurred = boxBlur(data, W, H);
  let hfSumSq = 0, totalSumSq = 0;
  for (let i = 0; i < N; i++) {
    const j = i * 4;
    const r = data[j], g = data[j + 1], b = data[j + 2];
    const lr = blurred[j], lg = blurred[j + 1], lb = blurred[j + 2];
    const dr = r - lr, dg = g - lg, db = b - lb;
    hfSumSq += dr * dr + dg * dg + db * db;
    totalSumSq += r * r + g * g + b * b;
  }
  const highFreqRatio = totalSumSq > 0 ? hfSumSq / totalSumSq : 0;

  // Compressibility proxy: PNG byte count
  const pngBytes = await new Promise<number>((resolve) => {
    canvas.toBlob((blob) => resolve(blob ? blob.size : 0), "image/png");
  });

  const stats: ImageStats = {
    width: W,
    height: H,
    meanR,
    meanG,
    meanB,
    meanLuma,
    stdLuma,
    saturation,
    entropy,
    edgeDensity,
    centreX,
    centreY,
    highFreqRatio,
    pngBytes,
    histR: hR,
    histG: hG,
    histB: hB,
    histLuma: hL,
    meanHue,
  };
  cache.set(src, stats);
  return stats;
}
