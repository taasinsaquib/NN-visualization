import type { ModelMetadata, ModelData, LoadedLayer, TensorData } from '../types';

const BASE = import.meta.env.BASE_URL;

async function fetchBinary(path: string): Promise<Float32Array> {
  const response = await fetch(`${BASE}data/${path}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength % 4 !== 0) {
    throw new Error(`${path}: buffer size ${buffer.byteLength} is not a multiple of 4 — file may be corrupted or an LFS pointer`);
  }
  return new Float32Array(buffer);
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}data/${path}`);
  return response.json();
}

export async function loadModelData(): Promise<ModelData> {
  const metadata = await fetchJson<ModelMetadata>('metadata.json');
  const layers = new Map<string, LoadedLayer>();

  await Promise.all(
    metadata.layers.map(async (def) => {
      const activations: TensorData = {
        data: await fetchBinary(def.activations_file),
        shape: def.shape,
      };

      let weights: TensorData | undefined;
      let bias: TensorData | undefined;

      if (def.weights_file && def.weights_shape) {
        weights = {
          data: await fetchBinary(def.weights_file),
          shape: def.weights_shape,
        };
      }

      if (def.bias_file) {
        bias = {
          data: await fetchBinary(def.bias_file),
          shape: [def.shape[0]],
        };
      }

      layers.set(def.id, { def, activations, weights, bias });
    })
  );

  return {
    metadata,
    layers,
    imageUrl: `${BASE}data/images/${metadata.image}.jpg`,
  };
}
