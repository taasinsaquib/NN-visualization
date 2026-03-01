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
  is1D?: boolean;
  shape1D?: number;
}

export interface SceneLayout {
  blocks: LayerBlock[];
  totalWidth: number;
  totalHeight: number;
}

const SPATIAL_SCALE = 2;
const KERNEL_SCALE = 8;
const CHANNEL_DEPTH_SCALE = 0.5;
const MAX_DEPTH_PX = 64;
const OBLIQUE_ANGLE = Math.PI / 4;
const LAYER_GAP = 28;
const VERTICAL_CENTER = 400;
const BAR_WIDTH_1D = 30;
const BAR_HEIGHT_1D_MAX = 300;
const BAR_HEIGHT_1D_SCALE = 0.1;
const FC_WIDTH = 120;
const FC_HEIGHT = 200;
const SOFTMAX_WIDTH = 180;
const SOFTMAX_HEIGHT = 300;

const COS_45 = Math.cos(OBLIQUE_ANGLE);

export function computeLayout(layers: LayerDef[]): SceneLayout {
  const blocks: LayerBlock[] = [];
  let x = 80;

  for (const layer of layers) {
    const is1D = layer.shape.length === 1;
    const shape1D = is1D ? layer.shape[0] : undefined;

    if (layer.type === 'conv' && layer.weights_shape && layer.weights_shape.length === 4) {
      const [, inCh, kH, kW] = layer.weights_shape;
      const kWidth = kW * KERNEL_SCALE;
      const kHeight = kH * KERNEL_SCALE;
      const kDepthPx = Math.min(inCh * CHANNEL_DEPTH_SCALE, MAX_DEPTH_PX);
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

    if (is1D && shape1D !== undefined) {
      let width: number;
      let height: number;
      let label: string;

      if (layer.type === 'linear') {
        width = FC_WIDTH;
        height = FC_HEIGHT;
        const ws = layer.weights_shape;
        label = ws ? `${layer.id}\n${ws[0]}×${ws[1]}` : `${layer.id}\n${shape1D}`;
      } else if (layer.type === 'softmax') {
        width = SOFTMAX_WIDTH;
        height = SOFTMAX_HEIGHT;
        label = `${layer.id}\n${shape1D} classes`;
      } else {
        width = BAR_WIDTH_1D;
        height = Math.min(shape1D * BAR_HEIGHT_1D_SCALE, BAR_HEIGHT_1D_MAX);
        label = `${layer.id}\n${shape1D}`;
      }

      const y = VERTICAL_CENTER - height / 2;
      blocks.push({
        id: layer.id,
        type: layer.type,
        label,
        x,
        y,
        width,
        height,
        depthPx: 0,
        depthOffsetX: 0,
        depthOffsetY: 0,
        channels: shape1D,
        spatialH: 1,
        spatialW: 1,
        is1D: true,
        shape1D,
      });
    } else {
      const [channels, spatialH, spatialW] = layer.shape as [number, number, number];
      const width = spatialW * SPATIAL_SCALE;
      const height = spatialH * SPATIAL_SCALE;
      const depthPx = Math.min(channels * CHANNEL_DEPTH_SCALE, MAX_DEPTH_PX);
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
    }

    const lastBlock = blocks[blocks.length - 1];
    x += Math.abs(lastBlock.depthOffsetX) + lastBlock.width + LAYER_GAP;
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
