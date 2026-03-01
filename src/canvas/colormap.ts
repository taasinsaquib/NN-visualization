const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 35, 116],
  [64, 67, 135],
  [52, 94, 141],
  [33, 145, 140],
  [94, 201, 98],
  [190, 223, 38],
  [253, 231, 37],
];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function valueToColor(value: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, value));
  const scaled = v * (VIRIDIS.length - 1);
  const i0 = Math.floor(scaled);
  const i1 = Math.min(i0 + 1, VIRIDIS.length - 1);
  const frac = scaled - i0;
  return [
    Math.round(lerp(VIRIDIS[i0][0], VIRIDIS[i1][0], frac)),
    Math.round(lerp(VIRIDIS[i0][1], VIRIDIS[i1][1], frac)),
    Math.round(lerp(VIRIDIS[i0][2], VIRIDIS[i1][2], frac)),
  ];
}

export function valueToRGBA(value: number, alpha: number = 1): [number, number, number, number] {
  const [r, g, b] = valueToColor(value);
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255);
  return [r, g, b, a];
}

export function divergingColor(value: number): [number, number, number] {
  const v = Math.max(-1, Math.min(1, value));
  const t = (v + 1) / 2;
  if (t <= 0.5) {
    const s = t * 2;
    return [
      Math.round(lerp(33, 255, s)),
      Math.round(lerp(102, 255, s)),
      Math.round(lerp(172, 255, s)),
    ];
  } else {
    const s = (t - 0.5) * 2;
    return [
      Math.round(lerp(255, 213, s)),
      Math.round(lerp(255, 94, s)),
      Math.round(lerp(255, 0, s)),
    ];
  }
}
