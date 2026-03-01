import { useRef, useEffect, useState, useCallback } from 'react';
import type { ModelData } from '../types';
import { computeLayout, type SceneLayout, type LayerBlock } from './layout';
import { valueToColor, divergingColor } from './colormap';
import { getChannel, minMax, normalize, tensorToRGBA } from '../utils/tensor';
import { computeReceptiveFieldChain, type ReceptiveFieldRegion } from '../model/receptiveField';

interface Transform {
  offsetX: number;
  offsetY: number;
  scale: number;
}

export interface RfSelection {
  layerId: string;
  channel: number;
  row: number;
  col: number;
}

interface CanvasViewProps {
  modelData: ModelData;
  channelMap: Record<string, number>;
  rfMiniChannelMap: Record<string, number>;
  explodedBlock: string | null;
  rfSelection: RfSelection | null;
  onBlockClick?: (blockId: string) => void;
  onExplode?: (blockId: string | null) => void;
  onChannelSelect?: (blockId: string, channel: number) => void;
  onRfSelect?: (sel: RfSelection | null) => void;
  onRfMiniChannelChange?: (layerId: string, delta: number) => void;
  showPreviews: boolean;
}

const SOFTMAX_ROW_H = 16;
const SOFTMAX_VISIBLE_ROWS = 18;

const LAYER_COLORS: Record<string, string> = {
  input: '#4a9eff',
  conv: '#ff6b6b',
  relu: '#51cf66',
  maxpool: '#ffd43b',
  linear: '#a78bfa',
  adaptive_avg_pool: '#f472b6',
  flatten: '#34d399',
  softmax: '#fbbf24',
};

const THUMB_SIZE = 40;
const THUMB_GAP = 4;

function getRfMiniBlockBounds(
  block: LayerBlock,
  region: ReceptiveFieldRegion
): { x: number; y: number; w: number; h: number } {
  const rfH = region.endRow - region.startRow;
  const rfW = region.endCol - region.startCol;
  const maxDim = Math.max(rfH, rfW);
  const miniScale = Math.min(80 / maxDim, 4);
  const miniW = rfW * miniScale;
  const miniH = rfH * miniScale;
  const miniX = block.x + block.width / 2 - miniW / 2;
  const miniY = block.y - miniH - 30;
  return { x: miniX, y: miniY, w: miniW, h: miniH };
}

function getFilter(
  weights: Float32Array,
  weightsShape: number[],
  outChannel: number
): Float32Array {
  const [, inCh, kH, kW] = weightsShape;
  const filterSize = inCh * kH * kW;
  return weights.slice(outChannel * filterSize, (outChannel + 1) * filterSize);
}

function drawLinearWeightHeatmap(
  ctx: CanvasRenderingContext2D,
  block: LayerBlock,
  modelData: ModelData,
) {
  const layer = modelData.layers.get(block.id);
  if (!layer?.weights) return false;

  const { x, y, width, height } = block;
  const w = layer.weights.data;
  const [outF, inF] = layer.weights.shape;
  const { min: wMin, max: wMax } = minMax(w);
  const range = wMax - wMin || 1;
  const sampleH = Math.min(128, outF);
  const sampleW = Math.min(128, inF);
  const imgData = ctx.createImageData(sampleW, sampleH);
  for (let r = 0; r < sampleH; r++) {
    for (let c = 0; c < sampleW; c++) {
      const srcRow = Math.floor((r / sampleH) * outF);
      const srcCol = Math.floor((c / sampleW) * inF);
      const norm = (w[srcRow * inF + srcCol] - wMin) / range;
      const [rv, gv, bv] = valueToColor(norm);
      const idx = (r * sampleW + c) * 4;
      imgData.data[idx] = rv;
      imgData.data[idx + 1] = gv;
      imgData.data[idx + 2] = bv;
      imgData.data[idx + 3] = 255;
    }
  }
  const offscreen = new OffscreenCanvas(sampleW, sampleH);
  const offCtx = offscreen.getContext('2d')!;
  offCtx.putImageData(imgData, 0, 0);
  ctx.drawImage(offscreen, x, y, width, height);
  return true;
}

function drawSoftmaxBlock(
  ctx: CanvasRenderingContext2D,
  block: LayerBlock,
  modelData: ModelData,
  scrollOffset: number,
) {
  const { x, y, width, height } = block;
  const layer = modelData.layers.get(block.id);
  if (!layer) return;

  ctx.fillStyle = '#111';
  ctx.fillRect(x, y, width, height);

  const data = layer.activations.data;
  const n = data.length;
  const indexed: { idx: number; prob: number }[] = [];
  for (let i = 0; i < n; i++) indexed.push({ idx: i, prob: data[i] });
  indexed.sort((a, b) => b.prob - a.prob);

  const classNames = new Map<number, string>();
  if (modelData.metadata.categories) {
    const cats = modelData.metadata.categories;
    for (let i = 0; i < cats.length; i++) classNames.set(i, cats[i]);
  } else if (modelData.metadata.predictions) {
    for (const p of modelData.metadata.predictions) classNames.set(p.class_idx, p.class_name);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, width, height);
  ctx.clip();

  const visibleRows = Math.floor(height / SOFTMAX_ROW_H);
  const startIdx = Math.max(0, Math.min(scrollOffset, n - visibleRows));
  const barMaxW = width * 0.45;

  for (let i = 0; i < visibleRows && startIdx + i < n; i++) {
    const entry = indexed[startIdx + i];
    const ry = y + i * SOFTMAX_ROW_H;
    const barW = entry.prob * barMaxW;

    const intensity = Math.max(0.15, entry.prob * 3);
    ctx.fillStyle = `rgba(251, 191, 36, ${intensity})`;
    ctx.fillRect(x + 2, ry + 2, barW, SOFTMAX_ROW_H - 4);

    const rank = startIdx + i;
    const name = classNames.get(entry.idx) ?? `class ${entry.idx}`;
    const pctStr = entry.prob >= 0.01
      ? `${(entry.prob * 100).toFixed(1)}%`
      : entry.prob >= 0.001
        ? `${(entry.prob * 100).toFixed(2)}%`
        : `${(entry.prob * 100).toExponential(0)}`;

    ctx.fillStyle = rank < 5 ? '#fff' : '#aaa';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    const label = `${name}`;
    ctx.fillText(label, x + barMaxW + 6, ry + SOFTMAX_ROW_H - 4, width - barMaxW - 10);

    ctx.fillStyle = '#888';
    ctx.textAlign = 'right';
    ctx.fillText(pctStr, x + barMaxW + 2, ry + SOFTMAX_ROW_H - 4);
  }

  ctx.restore();

  if (n > visibleRows) {
    const trackH = height - 4;
    const thumbH = Math.max(20, (visibleRows / n) * trackH);
    const thumbY = y + 2 + ((startIdx / (n - visibleRows)) * (trackH - thumbH));
    ctx.fillStyle = 'rgba(255,255,255,0.1)';
    ctx.fillRect(x + width - 4, y + 2, 3, trackH);
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.fillRect(x + width - 4, thumbY, 3, thumbH);
  }
}

function drawBlock(
  ctx: CanvasRenderingContext2D,
  block: LayerBlock,
  modelData: ModelData,
  channelMap: Record<string, number>,
  imageRef: React.RefObject<HTMLImageElement | null>,
  softmaxScroll: number,
) {
  const { x, y, width, height, depthOffsetX, depthOffsetY } = block;
  const baseColor = LAYER_COLORS[block.type] || '#888';

  if (block.type === 'linear' && block.is1D) {
    if (!drawLinearWeightHeatmap(ctx, block, modelData)) {
      ctx.fillStyle = baseColor;
      ctx.fillRect(x, y, width, height);
    }
  } else if (block.type === 'softmax' && block.is1D) {
    drawSoftmaxBlock(ctx, block, modelData, softmaxScroll);
  } else if (block.isKernel && block.parentConvId) {
    const parentLayer = modelData.layers.get(block.parentConvId);
    if (parentLayer?.weights) {
      const outCh = Math.min(
        channelMap[block.parentConvId] ?? 0,
        (parentLayer.weights.shape[0] ?? 1) - 1
      );
      const filter = getFilter(
        parentLayer.weights.data,
        parentLayer.weights.shape,
        outCh
      );
      const [inCh, kH, kW] = parentLayer.weights.shape.slice(1) as [
        number,
        number,
        number,
      ];

      if (inCh === 3) {
        const rgba = tensorToRGBA(filter, [3, kH, kW]);
        const imgData = ctx.createImageData(kW, kH);
        imgData.data.set(rgba);
        const offscreen = new OffscreenCanvas(kW, kH);
        const offCtx = offscreen.getContext('2d')!;
        offCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(offscreen, x, y, width, height);
      } else {
        const channelData = getChannel(filter, [inCh, kH, kW], 0);
        const { min, max } = minMax(channelData);
        const normalized = normalize(channelData, min, max);
        const imgData = ctx.createImageData(kW, kH);
        for (let i = 0; i < normalized.length; i++) {
          const [r, g, b] = valueToColor(normalized[i]);
          imgData.data[i * 4] = r;
          imgData.data[i * 4 + 1] = g;
          imgData.data[i * 4 + 2] = b;
          imgData.data[i * 4 + 3] = 255;
        }
        const offscreen = new OffscreenCanvas(kW, kH);
        const offCtx = offscreen.getContext('2d')!;
        offCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(offscreen, x, y, width, height);
      }
    } else {
      ctx.fillStyle = baseColor;
      ctx.fillRect(x, y, width, height);
    }
  } else if (block.type === 'input') {
    const inputCh = channelMap[block.id] ?? 0;
    if (inputCh === 0 && imageRef.current?.complete) {
      ctx.drawImage(imageRef.current, x, y, width, height);
    } else {
      const inputLayer = modelData.layers.get('input');
      if (inputLayer && inputCh >= 1 && inputCh <= 3) {
        const ch = inputCh - 1;
        const channelData = getChannel(
          inputLayer.activations.data,
          inputLayer.activations.shape,
          ch
        );
        const { min, max } = minMax(channelData);
        const normalized = normalize(channelData, min, max);
        const imgData = ctx.createImageData(block.spatialW, block.spatialH);
        for (let i = 0; i < normalized.length; i++) {
          const [r, g, b] = valueToColor(normalized[i]);
          imgData.data[i * 4] = r;
          imgData.data[i * 4 + 1] = g;
          imgData.data[i * 4 + 2] = b;
          imgData.data[i * 4 + 3] = 255;
        }
        const offscreen = new OffscreenCanvas(block.spatialW, block.spatialH);
        const offCtx = offscreen.getContext('2d')!;
        offCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(offscreen, x, y, width, height);
      } else if (imageRef.current?.complete) {
        ctx.drawImage(imageRef.current, x, y, width, height);
      }
    }
  } else {
    const layer = modelData.layers.get(block.id);
    if (layer) {
      const channel = Math.min(channelMap[block.id] ?? 0, block.channels - 1);
      const channelData = getChannel(
        layer.activations.data,
        layer.activations.shape,
        channel
      );
      const { min, max } = minMax(channelData);
      const normalized = normalize(channelData, min, max);

      const imgData = ctx.createImageData(block.spatialW, block.spatialH);
      for (let i = 0; i < normalized.length; i++) {
        const [r, g, b] = valueToColor(normalized[i]);
        imgData.data[i * 4] = r;
        imgData.data[i * 4 + 1] = g;
        imgData.data[i * 4 + 2] = b;
        imgData.data[i * 4 + 3] = 255;
      }

      const offscreen = new OffscreenCanvas(block.spatialW, block.spatialH);
      const offCtx = offscreen.getContext('2d')!;
      offCtx.putImageData(imgData, 0, 0);

      ctx.drawImage(offscreen, x, y, width, height);
    } else {
      ctx.fillStyle = baseColor;
      ctx.fillRect(x, y, width, height);
    }
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.4)';
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, width, height);

  if (block.depthPx > 0) {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + depthOffsetX, y + depthOffsetY);
    ctx.lineTo(x + width + depthOffsetX, y + depthOffsetY);
    ctx.lineTo(x + width, y);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + depthOffsetX, y + depthOffsetY);
    ctx.lineTo(x + depthOffsetX, y + height + depthOffsetY);
    ctx.lineTo(x, y + height);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,0,0,0.15)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.stroke();
  }

  const selCh = channelMap[block.id] ?? 0;
  let t = -1;
  if (!block.isKernel && block.channels > 0 && !block.is1D) {
    if (block.type === 'input') {
      if (selCh >= 1 && selCh <= 3) t = (selCh - 1) / 3;
    } else if (selCh < block.channels) {
      t = selCh / block.channels;
    }
  }
  if (t >= 0) {
    const lx = x + t * depthOffsetX;
    const ly = y + t * depthOffsetY;
    const rx = x + width + t * depthOffsetX;
    const ry = y + t * depthOffsetY;
    ctx.strokeStyle = '#ff0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(rx, ry);
    ctx.stroke();
  }

  ctx.fillStyle = '#ccc';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  const lines = block.label.split('\n');
  lines.forEach((line, i) => {
    ctx.fillStyle = i === 0 ? '#fff' : '#999';
    ctx.fillText(line, x + width / 2, y + height + 18 + i * 16);
  });
}

function drawExplodedBlock(
  ctx: CanvasRenderingContext2D,
  block: LayerBlock,
  modelData: ModelData,
  channelMap: Record<string, number>,
  _imageRef: React.RefObject<HTMLImageElement | null>,
) {
  if (block.is1D) return;
  const { x } = block;
  const gridStartY = block.y + block.height + 40;
  const layer = modelData.layers.get(block.id);
  if (!layer) return;

  const cols = Math.ceil(Math.sqrt(block.channels));
  const rows = Math.ceil(block.channels / cols);
  const gridWidth = cols * (THUMB_SIZE + THUMB_GAP) - THUMB_GAP;
  const gridHeight = rows * (THUMB_SIZE + THUMB_GAP) - THUMB_GAP;

  for (let ch = 0; ch < block.channels; ch++) {
    const col = ch % cols;
    const row = Math.floor(ch / cols);
    const tx = x + col * (THUMB_SIZE + THUMB_GAP);
    const ty = gridStartY + row * (THUMB_SIZE + THUMB_GAP);

    const channelData = getChannel(
      layer.activations.data,
      layer.activations.shape,
      ch
    );
    const { min, max } = minMax(channelData);
    const normalized = normalize(channelData, min, max);

    const imgData = ctx.createImageData(block.spatialW, block.spatialH);
    for (let i = 0; i < normalized.length; i++) {
      const [r, g, b] = valueToColor(normalized[i]);
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }

    const offscreen = new OffscreenCanvas(block.spatialW, block.spatialH);
    const offCtx = offscreen.getContext('2d')!;
    offCtx.putImageData(imgData, 0, 0);
    ctx.drawImage(offscreen, tx, ty, THUMB_SIZE, THUMB_SIZE);

    const selectedCh = channelMap[block.id] ?? 0;
    if (ch === selectedCh) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.strokeRect(tx, ty, THUMB_SIZE, THUMB_SIZE);
    } else {
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.strokeRect(tx, ty, THUMB_SIZE, THUMB_SIZE);
    }

    ctx.fillStyle = '#fff';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(String(ch), tx + THUMB_SIZE / 2, ty + THUMB_SIZE - 4);
  }

  ctx.fillStyle = '#ccc';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(
    `${block.id} (exploded)`,
    x + gridWidth / 2,
    gridStartY + gridHeight + 18
  );
}

function drawKernelOverlay(
  ctx: CanvasRenderingContext2D,
  block: LayerBlock,
  region: ReceptiveFieldRegion,
  weights: Float32Array,
  weightsShape: number[],
  channel: number,
) {
  const [, inCh, kH, kW] = weightsShape;
  const filterSize = inCh * kH * kW;
  const filterWeights = weights.slice(channel * filterSize, (channel + 1) * filterSize);
  const pxPerRow = block.height / block.spatialH;
  const pxPerCol = block.width / block.spatialW;
  const baseX = block.x + region.startCol * pxPerCol;
  const baseY = block.y + region.startRow * pxPerRow;
  const cellW = (region.endCol - region.startCol) * pxPerCol / kW;
  const cellH = (region.endRow - region.startRow) * pxPerRow / kH;
  let minW = Infinity;
  let maxW = -Infinity;
  for (let i = 0; i < filterWeights.length; i++) {
    minW = Math.min(minW, filterWeights[i]);
    maxW = Math.max(maxW, filterWeights[i]);
  }
  const range = Math.max(maxW - minW, 1e-6);
  for (let r = 0; r < kH; r++) {
    for (let c = 0; c < kW; c++) {
      let val = 0;
      for (let ic = 0; ic < inCh; ic++) {
        val += filterWeights[ic * (kH * kW) + r * kW + c];
      }
      val /= inCh;
      const norm = (val - minW) / range * 2 - 1;
      const [cr, cg, cb] = divergingColor(norm);
      ctx.fillStyle = `rgba(${cr},${cg},${cb},0.5)`;
      ctx.fillRect(
        baseX + c * cellW,
        baseY + r * cellH,
        cellW + 0.5,
        cellH + 0.5
      );
    }
  }
}

function drawRfMiniBlock(
  ctx: CanvasRenderingContext2D,
  block: LayerBlock,
  region: ReceptiveFieldRegion,
  modelData: ModelData,
  channelMap: Record<string, number>,
  rfMiniChannelMap: Record<string, number>,
) {
  const layer = modelData.layers.get(region.layerId);
  if (!layer) return;

  const { x: miniX, y: miniY, w: miniW, h: miniH } = getRfMiniBlockBounds(block, region);
  const channel = Math.min(
    rfMiniChannelMap[region.layerId] ?? channelMap[region.layerId] ?? 0,
    layer.def.shape[0] - 1
  );
  const channelData = getChannel(layer.activations.data, layer.activations.shape, channel);
  const [, H, W] = layer.activations.shape;

  const cropH = region.endRow - region.startRow;
  const cropW = region.endCol - region.startCol;
  const crop = new Float32Array(cropH * cropW);
  for (let r = 0; r < cropH; r++) {
    for (let c = 0; c < cropW; c++) {
      const srcR = region.startRow + r;
      const srcC = region.startCol + c;
      if (srcR >= 0 && srcR < H && srcC >= 0 && srcC < W) {
        crop[r * cropW + c] = channelData[srcR * W + srcC];
      }
    }
  }

  const { min, max } = minMax(crop);
  const normalized = normalize(crop, min, max);

  const imgData = ctx.createImageData(cropW, cropH);
  for (let i = 0; i < normalized.length; i++) {
    const [r, g, b] = valueToColor(normalized[i]);
    imgData.data[i * 4] = r;
    imgData.data[i * 4 + 1] = g;
    imgData.data[i * 4 + 2] = b;
    imgData.data[i * 4 + 3] = 255;
  }
  const offscreen = new OffscreenCanvas(cropW, cropH);
  const offCtx = offscreen.getContext('2d')!;
  offCtx.putImageData(imgData, 0, 0);

  ctx.drawImage(offscreen, miniX, miniY, miniW, miniH);

  ctx.strokeStyle = 'rgba(255, 200, 100, 0.8)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(miniX, miniY, miniW, miniH);

  const miniDepth = 4;
  const dox = -(miniDepth * 0.707);
  const doy = -(miniDepth * 0.707);

  ctx.beginPath();
  ctx.moveTo(miniX, miniY);
  ctx.lineTo(miniX + dox, miniY + doy);
  ctx.lineTo(miniX + miniW + dox, miniY + doy);
  ctx.lineTo(miniX + miniW, miniY);
  ctx.closePath();
  ctx.fillStyle = 'rgba(255,255,255,0.08)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 200, 100, 0.5)';
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(miniX, miniY);
  ctx.lineTo(miniX + dox, miniY + doy);
  ctx.lineTo(miniX + dox, miniY + miniH + doy);
  ctx.lineTo(miniX, miniY + miniH);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,0,0,0.15)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 200, 100, 0.5)';
  ctx.stroke();

  ctx.fillStyle = '#ffd700';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`${cropH}×${cropW}`, miniX + miniW / 2, miniY - 4);

  const pxPerRow = block.height / block.spatialH;
  const pxPerCol = block.width / block.spatialW;
  const highlightCenterX = block.x + ((region.startCol + region.endCol) / 2) * pxPerCol;
  const highlightCenterY = block.y + ((region.startRow + region.endRow) / 2) * pxPerRow;

  ctx.strokeStyle = 'rgba(255, 200, 100, 0.4)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(miniX + miniW / 2, miniY + miniH);
  ctx.lineTo(highlightCenterX, highlightCenterY);
  ctx.stroke();
  ctx.setLineDash([]);
}

const PREVIEW_TARGET_PX = 200;

function getPreviewScale(layout: SceneLayout): number {
  let maxSpatial = 0;
  for (const block of layout.blocks) {
    if (block.isKernel || block.type === 'input') continue;
    maxSpatial = Math.max(maxSpatial, block.spatialH, block.spatialW);
  }
  return maxSpatial > 0 ? PREVIEW_TARGET_PX / maxSpatial : 1;
}

function getPreviewBounds(
  block: LayerBlock,
  previewScale: number,
): { x: number; y: number; w: number; h: number } {
  const w = block.spatialW * previewScale;
  const h = block.spatialH * previewScale;
  const px = block.x + block.width / 2 - w / 2;
  const py = block.y + block.depthOffsetY - h - 20;
  return { x: px, y: py, w, h };
}

function drawPreviews(
  ctx: CanvasRenderingContext2D,
  layout: SceneLayout,
  modelData: ModelData,
  channelMap: Record<string, number>,
) {
  const previewScale = getPreviewScale(layout);
  for (const block of layout.blocks) {
    if (block.isKernel || block.type === 'input' || block.is1D) continue;
    const layer = modelData.layers.get(block.id);
    if (!layer) continue;

    const ch = Math.min(channelMap[block.id] ?? 0, block.channels - 1);
    const channelData = getChannel(layer.activations.data, layer.activations.shape, ch);
    const { min, max } = minMax(channelData);
    const normalized = normalize(channelData, min, max);

    const { x: px, y: py, w: pw, h: ph } = getPreviewBounds(block, previewScale);

    const imgData = ctx.createImageData(block.spatialW, block.spatialH);
    for (let i = 0; i < normalized.length; i++) {
      const [r, g, b] = valueToColor(normalized[i]);
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }
    const offscreen = new OffscreenCanvas(block.spatialW, block.spatialH);
    const offCtx = offscreen.getContext('2d')!;
    offCtx.putImageData(imgData, 0, 0);
    ctx.drawImage(offscreen, px, py, pw, ph);

    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(px, py, pw, ph);

    ctx.fillStyle = '#999';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${block.spatialH}x${block.spatialW}`, px + pw / 2, py - 4);
  }
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  layout: SceneLayout,
  modelData: ModelData,
  transform: Transform,
  channelMap: Record<string, number>,
  rfMiniChannelMap: Record<string, number>,
  explodedBlock: string | null,
  imageRef: React.RefObject<HTMLImageElement | null>,
  rfSelection: RfSelection | null,
  showPreviews: boolean,
  softmaxScroll: number,
) {
  const { width: cw, height: ch } = ctx.canvas;
  ctx.clearRect(0, 0, cw, ch);

  ctx.save();
  ctx.translate(transform.offsetX, transform.offsetY);
  ctx.scale(transform.scale, transform.scale);

  for (const block of layout.blocks) {
    if (block.isKernel) {
      drawBlock(ctx, block, modelData, channelMap, imageRef, softmaxScroll);
    } else if (block.id === explodedBlock) {
      drawBlock(ctx, block, modelData, channelMap, imageRef, softmaxScroll);
      drawExplodedBlock(ctx, block, modelData, channelMap, imageRef);
    } else {
      drawBlock(ctx, block, modelData, channelMap, imageRef, softmaxScroll);
    }
  }

  if (showPreviews) {
    drawPreviews(ctx, layout, modelData, channelMap);
  }

  if (rfSelection) {
    const regions = computeReceptiveFieldChain(
      modelData.metadata.layers,
      rfSelection.layerId,
      rfSelection.row,
      rfSelection.col
    );

    for (const region of regions) {
      const block = layout.blocks.find((b) => b.id === region.layerId && !b.isKernel);
      if (!block || block.is1D) continue;

      const pxPerRow = block.height / block.spatialH;
      const pxPerCol = block.width / block.spatialW;

      const rx = block.x + region.startCol * pxPerCol;
      const ry = block.y + region.startRow * pxPerRow;
      const rw = (region.endCol - region.startCol) * pxPerCol;
      const rh = (region.endRow - region.startRow) * pxPerRow;

      ctx.fillStyle = 'rgba(255, 100, 100, 0.3)';
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = 'rgba(255, 100, 100, 0.8)';
      ctx.lineWidth = 2;
      ctx.strokeRect(rx, ry, rw, rh);

      if (
        region.layerId === 'input' &&
        rfSelection.layerId === 'conv1' &&
        regions[regions.length - 1]?.layerId === 'input'
      ) {
        const conv1Layer = modelData.layers.get('conv1');
        if (conv1Layer?.weights) {
          drawKernelOverlay(
            ctx,
            block,
            region,
            conv1Layer.weights.data,
            conv1Layer.weights.shape,
            channelMap['conv1'] ?? 0
          );
        }
      }
    }

    const sourceBlock = layout.blocks.find(
      (b) => b.id === rfSelection.layerId && !b.isKernel
    );
    if (sourceBlock) {
      const pxPerRow = sourceBlock.height / sourceBlock.spatialH;
      const pxPerCol = sourceBlock.width / sourceBlock.spatialW;
      const px = sourceBlock.x + rfSelection.col * pxPerCol;
      const py = sourceBlock.y + rfSelection.row * pxPerRow;
      ctx.fillStyle = 'rgba(255, 255, 0, 0.5)';
      ctx.fillRect(px, py, pxPerCol, pxPerRow);
      ctx.strokeStyle = 'rgba(255, 255, 0, 1)';
      ctx.lineWidth = 2;
      ctx.strokeRect(px, py, pxPerCol, pxPerRow);
    }

    ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    const points: { x: number; y: number }[] = [];
    if (sourceBlock) {
      const pxPerRow = sourceBlock.height / sourceBlock.spatialH;
      const pxPerCol = sourceBlock.width / sourceBlock.spatialW;
      points.push({
        x: sourceBlock.x + (rfSelection.col + 0.5) * pxPerCol,
        y: sourceBlock.y + (rfSelection.row + 0.5) * pxPerRow,
      });
    }
    for (const region of regions) {
      const block = layout.blocks.find((b) => b.id === region.layerId && !b.isKernel);
      if (!block) continue;
      const pxPerRow = block.height / block.spatialH;
      const pxPerCol = block.width / block.spatialW;
      const cx = (region.startCol + region.endCol) / 2;
      const cy = (region.startRow + region.endRow) / 2;
      points.push({ x: block.x + cx * pxPerCol, y: block.y + cy * pxPerRow });
    }
    for (let i = 0; i < points.length - 1; i++) {
      ctx.beginPath();
      ctx.moveTo(points[i].x, points[i].y);
      ctx.lineTo(points[i + 1].x, points[i + 1].y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    for (const region of regions) {
      const block = layout.blocks.find((b) => b.id === region.layerId && !b.isKernel);
      const layer = modelData.layers.get(region.layerId);
      if (block && layer && layer.activations.shape.length === 3) {
        drawRfMiniBlock(ctx, block, region, modelData, channelMap, rfMiniChannelMap);
      }
    }
  }

  ctx.restore();
}

export default function CanvasView({
  modelData,
  channelMap,
  rfMiniChannelMap,
  explodedBlock,
  rfSelection,
  onBlockClick,
  onExplode,
  onChannelSelect,
  onRfSelect,
  onRfMiniChannelChange,
  showPreviews,
}: CanvasViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [transform, setTransform] = useState<Transform>({
    offsetX: 40,
    offsetY: 40,
    scale: 0.8,
  });
  const transformRef = useRef(transform);
  transformRef.current = transform;
  const isPanning = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });
  const [softmaxScroll, setSoftmaxScroll] = useState(0);

  const layout = computeLayout(modelData.metadata.layers);

  useEffect(() => {
    const img = new Image();
    img.src = modelData.imageUrl;
    img.onload = () => {
      imageRef.current = img;
    };
    return () => {
      imageRef.current = null;
    };
  }, [modelData.imageUrl]);

  const getRfMiniBlockAtPoint = useCallback(
    (rx: number, ry: number): { layerId: string; maxCh: number } | null => {
      if (!rfSelection) return null;
      const regions = computeReceptiveFieldChain(
        modelData.metadata.layers,
        rfSelection.layerId,
        rfSelection.row,
        rfSelection.col
      );
      for (const region of regions) {
        const block = layout.blocks.find((b) => b.id === region.layerId && !b.isKernel);
        if (!block) continue;
        const { x, y, w, h } = getRfMiniBlockBounds(block, region);
        if (rx >= x && rx <= x + w && ry >= y && ry <= y + h) {
          const layer = modelData.layers.get(region.layerId);
          const maxCh = layer ? layer.def.shape[0] - 1 : 0;
          return { layerId: region.layerId, maxCh };
        }
      }
      return null;
    },
    [layout, modelData, rfSelection]
  );

  const getSoftmaxBlockAtPoint = useCallback(
    (rx: number, ry: number): LayerBlock | null => {
      for (const block of layout.blocks) {
        if (block.type === 'softmax' && block.is1D) {
          if (rx >= block.x && rx <= block.x + block.width &&
              ry >= block.y && ry <= block.y + block.height) {
            return block;
          }
        }
      }
      return null;
    },
    [layout]
  );

  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const rx = (mx - transformRef.current.offsetX) / transformRef.current.scale;
      const ry = (my - transformRef.current.offsetY) / transformRef.current.scale;

      const miniHit = getRfMiniBlockAtPoint(rx, ry);
      if (miniHit && onRfMiniChannelChange) {
        const delta = e.deltaY > 0 ? 1 : -1;
        onRfMiniChannelChange(miniHit.layerId, delta);
        return;
      }

      const smBlock = getSoftmaxBlockAtPoint(rx, ry);
      if (smBlock) {
        const scrollStep = e.deltaY > 0 ? 3 : -3;
        const totalClasses = smBlock.channels;
        const visibleRows = Math.floor(smBlock.height / SOFTMAX_ROW_H);
        setSoftmaxScroll((prev) =>
          Math.max(0, Math.min(totalClasses - visibleRows, prev + scrollStep))
        );
        return;
      }

      const t = transformRef.current;
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(0.05, Math.min(5, t.scale * zoomFactor));
      const ratio = newScale / t.scale;
      setTransform({
        offsetX: mx - (mx - t.offsetX) * ratio,
        offsetY: my - (my - t.offsetY) * ratio,
        scale: newScale,
      });
    },
    [getRfMiniBlockAtPoint, onRfMiniChannelChange, getSoftmaxBlockAtPoint]
  );

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) {
      isPanning.current = true;
      lastMouse.current = { x: e.clientX, y: e.clientY };
    }
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setTransform((prev) => ({
      ...prev,
      offsetX: prev.offsetX + dx,
      offsetY: prev.offsetY + dy,
    }));
  }, []);

  const handleMouseUp = useCallback(() => {
    isPanning.current = false;
  }, []);

  const getBlockAtPoint = useCallback(
    (mx: number, my: number): { block: LayerBlock; channel?: number } | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rx = (mx - transform.offsetX) / transform.scale;
      const ry = (my - transform.offsetY) / transform.scale;

      for (const block of layout.blocks) {
        if (block.id === explodedBlock && block.id !== 'input') {
          const gridStartY = block.y + block.height + 40;
          const cols = Math.ceil(Math.sqrt(block.channels));
          const rows = Math.ceil(block.channels / cols);
          const gridWidth = cols * (THUMB_SIZE + THUMB_GAP) - THUMB_GAP;
          const gridHeight = rows * (THUMB_SIZE + THUMB_GAP) - THUMB_GAP;
          if (
            rx >= block.x &&
            rx <= block.x + gridWidth &&
            ry >= gridStartY &&
            ry <= gridStartY + gridHeight
          ) {
            const col = Math.floor((rx - block.x) / (THUMB_SIZE + THUMB_GAP));
            const row = Math.floor((ry - gridStartY) / (THUMB_SIZE + THUMB_GAP));
            const ch = row * cols + col;
            if (ch >= 0 && ch < block.channels) return { block, channel: ch };
            return { block };
          }
        } else if (block.isKernel) {
          if (
            rx >= block.x &&
            rx <= block.x + block.width &&
            ry >= block.y &&
            ry <= block.y + block.height
          ) {
            return { block };
          }
        } else {
          if (
            rx >= block.x &&
            rx <= block.x + block.width &&
            ry >= block.y &&
            ry <= block.y + block.height
          ) {
            return { block };
          }
        }
      }
      return null;
    },
    [layout, transform, explodedBlock]
  );

  const getPreviewAtPoint = useCallback(
    (rx: number, ry: number): { block: LayerBlock; row: number; col: number } | null => {
      if (!showPreviews) return null;
      const previewScale = getPreviewScale(layout);
      for (const block of layout.blocks) {
        if (block.isKernel || block.type === 'input') continue;
        const { x, y, w, h } = getPreviewBounds(block, previewScale);
        if (rx >= x && rx <= x + w && ry >= y && ry <= y + h) {
          const row = Math.floor(((ry - y) / h) * block.spatialH);
          const col = Math.floor(((rx - x) / w) * block.spatialW);
          return {
            block,
            row: Math.max(0, Math.min(block.spatialH - 1, row)),
            col: Math.max(0, Math.min(block.spatialW - 1, col)),
          };
        }
      }
      return null;
    },
    [layout, showPreviews]
  );

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const rx = (mx - transform.offsetX) / transform.scale;
      const ry = (my - transform.offsetY) / transform.scale;
      const miniHit = getRfMiniBlockAtPoint(rx, ry);
      if (miniHit) {
        onBlockClick?.(miniHit.layerId);
        return;
      }
      const previewHit = getPreviewAtPoint(rx, ry);
      if (previewHit) {
        onRfSelect?.({
          layerId: previewHit.block.id,
          channel: channelMap[previewHit.block.id] ?? 0,
          row: previewHit.row,
          col: previewHit.col,
        });
        onBlockClick?.(previewHit.block.id);
        return;
      }
      const hit = getBlockAtPoint(mx, my);

      if (!hit) {
        onRfSelect?.(null);
        onExplode?.(null);
        return;
      }

      if (hit.channel !== undefined && onChannelSelect) {
        onChannelSelect(hit.block.id, hit.channel);
        return;
      }
      const clickedLayerId = hit.block.isKernel && hit.block.parentConvId ? hit.block.parentConvId : hit.block.id;
      if (clickedLayerId !== explodedBlock) {
        onExplode?.(null);
      }
      if (hit.block.isKernel && hit.block.parentConvId && onBlockClick) {
        onBlockClick(hit.block.parentConvId);
        return;
      }
      if (
        !hit.block.isKernel &&
        hit.block.id !== 'input' &&
        hit.block.id !== explodedBlock
      ) {
        const rx = (mx - transform.offsetX) / transform.scale;
        const ry = (my - transform.offsetY) / transform.scale;
        const row = Math.floor(
          ((ry - hit.block.y) / hit.block.height) * hit.block.spatialH
        );
        const col = Math.floor(
          ((rx - hit.block.x) / hit.block.width) * hit.block.spatialW
        );
        const rowClamped = Math.max(0, Math.min(hit.block.spatialH - 1, row));
        const colClamped = Math.max(0, Math.min(hit.block.spatialW - 1, col));
        onRfSelect?.({
          layerId: hit.block.id,
          channel: channelMap[hit.block.id] ?? 0,
          row: rowClamped,
          col: colClamped,
        });
      }
      if (onBlockClick) {
        onBlockClick(hit.block.isKernel && hit.block.parentConvId ? hit.block.parentConvId : hit.block.id);
      }
    },
    [
      getBlockAtPoint,
      getPreviewAtPoint,
      onBlockClick,
      onExplode,
      onChannelSelect,
      onRfSelect,
      transform,
      channelMap,
      explodedBlock,
    ]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const resizeObserver = new ResizeObserver(() => {
      canvas.width = container.clientWidth * devicePixelRatio;
      canvas.height = container.clientHeight * devicePixelRatio;
      canvas.style.width = `${container.clientWidth}px`;
      canvas.style.height = `${container.clientHeight}px`;
    });
    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    drawScene(
      ctx,
      layout,
      modelData,
      transform,
      channelMap,
      rfMiniChannelMap,
      explodedBlock,
      imageRef,
      rfSelection,
      showPreviews,
      softmaxScroll,
    );
  }, [layout, modelData, transform, channelMap, rfMiniChannelMap, explodedBlock, rfSelection, showPreviews, softmaxScroll]);

  return (
    <div ref={containerRef} className="absolute inset-0">
      <canvas
        ref={canvasRef}
        className="cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
      />
    </div>
  );
}
