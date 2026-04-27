/**
 * Tiny PCA via the X X^T trick: when N << D, the top eigenvectors of the
 * N×N Gram matrix give us coordinates in PC space directly. Fast enough
 * for trajectory-shaped inputs (20–50 steps × ~16k latent dims).
 *
 * Returns N points in 3D, suitable for plotting.
 */

export type Point3 = [number, number, number];

export function pca3D(samples: Float32Array[]): Point3[] {
  const N = samples.length;
  if (N === 0) return [];
  const D = samples[0].length;
  if (N === 1) return [[0, 0, 0]];

  // 1. Mean-centre
  const mean = new Float64Array(D);
  for (const s of samples) {
    for (let i = 0; i < D; i++) mean[i] += s[i];
  }
  for (let i = 0; i < D; i++) mean[i] /= N;

  const centred: Float64Array[] = samples.map((s) => {
    const c = new Float64Array(D);
    for (let i = 0; i < D; i++) c[i] = s[i] - mean[i];
    return c;
  });

  // 2. Gram matrix K = X X^T (N×N, symmetric PSD)
  const K: number[][] = [];
  for (let i = 0; i < N; i++) K.push(new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    const xi = centred[i];
    for (let j = i; j < N; j++) {
      const xj = centred[j];
      let dot = 0;
      for (let k = 0; k < D; k++) dot += xi[k] * xj[k];
      K[i][j] = dot;
      K[j][i] = dot;
    }
  }

  // 3. Top 3 eigenvectors via power iteration with deflation.
  const ITERS = 80;
  const components: number[][] = [];

  for (let p = 0; p < 3; p++) {
    // Init with a deterministic but well-distributed seed (mix index + p).
    let v = new Array<number>(N).fill(0).map((_, i) => Math.sin((i + 1) * (p + 1) * 1.7) + 0.001);
    let lambda = 0;
    for (let iter = 0; iter < ITERS; iter++) {
      const w = new Array<number>(N).fill(0);
      for (let i = 0; i < N; i++) {
        const Ki = K[i];
        let s = 0;
        for (let j = 0; j < N; j++) s += Ki[j] * v[j];
        w[i] = s;
      }
      lambda = Math.sqrt(w.reduce((a, b) => a + b * b, 0)) || 1;
      v = w.map((x) => x / lambda);
    }
    components.push(v);

    // Deflate: K -= lambda * v v^T
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        K[i][j] -= lambda * v[i] * v[j];
      }
    }
  }

  // 4. Coordinates: row i is (c0[i], c1[i], c2[i]).
  const out: Point3[] = [];
  for (let i = 0; i < N; i++) {
    out.push([components[0][i], components[1][i], components[2][i]]);
  }
  return out;
}
