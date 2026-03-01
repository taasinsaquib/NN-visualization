import type { LayerDef, LayerType } from '../types';

export interface LayerBlock {
  id: string;
  type: LayerType;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  depthPx: number;
  depthOffsetX: number;
  depthOffsetY: number;
  channels: number;
  spatialH: number;
  spatialW: number;
  isKernel?: boolean;
  parentConvId?: string;
}

export interface SceneLayout {
  blocks: LayerBlock[];
  totalWidth: number;
  totalHeight: number;
}

const SPATIAL_SCALE = 2;
const KERNEL_SCALE = 8;
const CHANNEL_DEPTH_SCALE = 0.5;
const OBLIQUE_ANGLE = Math.PI / 4;
const LAYER_GAP = 60;
const VERTICAL_CENTER = 400;

const COS_45 = Math.cos(OBLIQUE_ANGLE);

export function computeLayout(layers: LayerDef[]): SceneLayout {
  const blocks: LayerBlock[] = [];
  let x = 80;

  for (const layer of layers) {
    if (layer.type === 'conv' && layer.weights_shape) {
      const [, inCh, kH, kW] = layer.weights_shape;
      const kWidth = kW * KERNEL_SCALE;
      const kHeight = kH * KERNEL_SCALE;
      const kDepthPx = inCh * CHANNEL_DEPTH_SCALE;
      const kDepthOffsetX = -(kDepthPx * COS_45);
      const kDepthOffsetY = -(kDepthPx * COS_45);
      blocks.push({
        id: `${layer.id}-kernel`,
        type: layer.type,
        label: `${layer.id} kernel\n${inCh}×${kH}×${kW}`,
        x,
        y: VERTICAL_CENTER - kHeight / 2,
        width: kWidth,
        height: kHeight,
        depthPx: kDepthPx,
        depthOffsetX: kDepthOffsetX,
        depthOffsetY: kDepthOffsetY,
        channels: inCh,
        spatialH: kH,
        spatialW: kW,
        isKernel: true,
        parentConvId: layer.id,
      });
      x += Math.abs(kDepthOffsetX) + kWidth + LAYER_GAP;
    }
    const [channels, spatialH, spatialW] = layer.shape;
    const width = spatialW * SPATIAL_SCALE;
    const height = spatialH * SPATIAL_SCALE;
    const depthPx = channels * CHANNEL_DEPTH_SCALE;
    const depthOffsetX = -(depthPx * COS_45);
    const depthOffsetY = -(depthPx * COS_45);
    const y = VERTICAL_CENTER - height / 2;
    const label = `${layer.id}\n${channels}×${spatialH}×${spatialW}`;

    blocks.push({
      id: layer.id,
      type: layer.type,
      label,
      x,
      y,
      width,
      height,
      depthPx,
      depthOffsetX,
      depthOffsetY,
      channels,
      spatialH,
      spatialW,
    });

    x += Math.abs(depthOffsetX) + width + LAYER_GAP;
  }

  const totalWidth =
    blocks.length > 0
      ? blocks[blocks.length - 1].x + blocks[blocks.length - 1].width
      : 0;

  const minY = Math.min(
    ...blocks.map((b) => b.y + b.depthOffsetY)
  );
  const maxY = Math.max(
    ...blocks.map((b) => b.y + b.height)
  );
  const totalHeight = maxY - minY;

  return { blocks, totalWidth, totalHeight };
}
