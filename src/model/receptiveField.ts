import type { LayerDef } from '../types';

export interface ReceptiveFieldRegion {
  layerId: string;
  startRow: number;
  startCol: number;
  endRow: number;
  endCol: number;
}

export function computeReceptiveFieldChain(
  layers: LayerDef[],
  clickedLayerId: string,
  row: number,
  col: number,
): ReceptiveFieldRegion[] {
  const layerMap = new Map<string, LayerDef>();
  for (const l of layers) layerMap.set(l.id, l);

  const regions: ReceptiveFieldRegion[] = [];

  let currentId = clickedLayerId;
  let startRow = row;
  let startCol = col;
  let endRow = row + 1;
  let endCol = col + 1;

  while (true) {
    const layer = layerMap.get(currentId);
    if (!layer || !layer.input_layer) break;

    const inputLayer = layerMap.get(layer.input_layer);
    if (!inputLayer) break;

    if (layer.type === 'conv' || layer.type === 'maxpool') {
      const stride = layer.stride ?? 1;
      const padding = layer.padding ?? 0;
      const kernelSize = layer.kernel_size ?? 1;

      const newStartRow = startRow * stride - padding;
      const newStartCol = startCol * stride - padding;
      const newEndRow = (endRow - 1) * stride - padding + kernelSize;
      const newEndCol = (endCol - 1) * stride - padding + kernelSize;

      const inH = inputLayer.shape.length >= 3 ? inputLayer.shape[1]! : 1;
      const inW = inputLayer.shape.length >= 3 ? inputLayer.shape[2]! : 1;
      startRow = Math.max(0, newStartRow);
      startCol = Math.max(0, newStartCol);
      endRow = Math.min(inH, newEndRow);
      endCol = Math.min(inW, newEndCol);
    }

    regions.push({
      layerId: layer.input_layer,
      startRow,
      startCol,
      endRow,
      endCol,
    });

    currentId = layer.input_layer;
  }

  return regions;
}
