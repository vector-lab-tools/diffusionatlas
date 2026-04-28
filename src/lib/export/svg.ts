/**
 * SVG export helper. Self-contained file, embeds images as data URLs.
 */

export function downloadSvg(filename: string, svg: string): void {
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Tiny XML-attribute escaper for any free-form text we drop into SVG. */
export function escXml(s: string | number | undefined | null): string {
  if (s === undefined || s === null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Read the natural pixel dimensions of an SVG string. Looks at the
 * root element's `width`/`height` first, falls back to `viewBox`.
 */
function svgDimensions(svg: string): { width: number; height: number } {
  const parser = new DOMParser();
  const doc = parser.parseFromString(svg, "image/svg+xml");
  const root = doc.documentElement;
  let w = parseFloat(root.getAttribute("width") || "0");
  let h = parseFloat(root.getAttribute("height") || "0");
  if (!w || !h) {
    const vb = root.getAttribute("viewBox");
    if (vb) {
      const parts = vb.split(/\s+/);
      if (parts.length >= 4) {
        if (!w) w = parseFloat(parts[2]) || 0;
        if (!h) h = parseFloat(parts[3]) || 0;
      }
    }
  }
  return { width: w || 800, height: h || 600 };
}

/**
 * Rasterise an SVG string to a PNG blob via Canvas. The SVG must be
 * fully self-contained — any referenced assets need to be embedded as
 * data URLs (which our `buildFilmReelSvg` etc. already do, since we
 * inline preview thumbnails for SVG portability anyway).
 *
 * `scale` controls supersampling: 1 = native, 2 = 2× pixel density
 * (the default — looks crisp on Retina screens and prints), 3 = print
 * resolution. Higher values cost more memory + time.
 */
export async function svgToPngBlob(svg: string, scale: number = 2): Promise<Blob> {
  const { width, height } = svgDimensions(svg);
  const svgBlob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("Failed to load SVG into Image element"));
      img.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    // White background so transparent SVG areas aren't black PNG.
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    return await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("Canvas.toBlob returned null"))),
        "image/png",
      );
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Convenience: rasterise an SVG to PNG and trigger a browser download. */
export async function downloadPngFromSvg(filename: string, svg: string, scale: number = 2): Promise<void> {
  const blob = await svgToPngBlob(svg, scale);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
