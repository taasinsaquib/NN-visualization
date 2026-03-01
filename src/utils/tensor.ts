export function get3D(
  data: Float32Array,
  shape: number[],
  c: number,
  r: number,
  col: number
): number {
  const [, H, W] = shape;
  const index = c * (H * W) + r * W + col;
  return data[index];
}

export function getChannel(
  data: Float32Array,
  shape: number[],
  channel: number
): Float32Array {
  if (shape.length === 1) {
    return data;
  }
  const [, H, W] = shape;
  const size = H * W;
  const out = new Float32Array(size);
  const offset = channel * size;
  for (let i = 0; i < size; i++) {
    out[i] = data[offset + i];
  }
  return out;
}

export function minMax(data: Float32Array): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

export function normalize(
  data: Float32Array,
  min: number,
  max: number
): Float32Array {
  const out = new Float32Array(data.length);
  const range = max - min || 1;
  for (let i = 0; i < data.length; i++) {
    out[i] = (data[i] - min) / range;
  }
  return out;
}

export function tensorToRGBA(
  data: Float32Array,
  shape: [number, number, number]
): Uint8ClampedArray {
  const [C, H, W] = shape;
  const size = H * W;
  const { min, max } = minMax(data);
  const range = max - min || 1;
  const out = new Uint8ClampedArray(size * 4);
  for (let r = 0; r < H; r++) {
    for (let col = 0; col < W; col++) {
      const idx = r * W + col;
      const base = idx * 4;
      const rVal = C > 0 ? Math.round(((data[0 * size + idx] - min) / range) * 255) : 0;
      const gVal = C > 1 ? Math.round(((data[1 * size + idx] - min) / range) * 255) : 0;
      const bVal = C > 2 ? Math.round(((data[2 * size + idx] - min) / range) * 255) : 0;
      out[base] = Math.max(0, Math.min(255, rVal));
      out[base + 1] = Math.max(0, Math.min(255, gVal));
      out[base + 2] = Math.max(0, Math.min(255, bVal));
      out[base + 3] = 255;
    }
  }
  return out;
}
