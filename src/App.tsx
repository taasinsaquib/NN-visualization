import { useEffect, useState, useCallback } from 'react';
import type { ModelData } from './types';
import { loadModelData } from './data/loader';
import CanvasView from './canvas/CanvasView';
import { computeReceptiveFieldChain } from './model/receptiveField';
import { get3D } from './utils/tensor';

const CHANNEL_GROUPS: string[][] = [
  ['conv1', 'relu1', 'pool1'],
  ['conv2', 'relu2', 'pool2'],
];

function App() {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [channelMap, setChannelMap] = useState<Record<string, number>>({});
  const [selectedBlock, setSelectedBlock] = useState<string | null>(null);
  const [explodedBlock, setExplodedBlock] = useState<string | null>(null);
  const [rfSelection, setRfSelection] = useState<{
    layerId: string;
    channel: number;
    row: number;
    col: number;
  } | null>(null);
  const [rfMiniChannelMap, setRfMiniChannelMap] = useState<Record<string, number>>({});

  useEffect(() => {
    loadModelData()
      .then(setModelData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const getChannel = useCallback(
    (layerId: string) => channelMap[layerId] ?? 0,
    [channelMap]
  );
  const setSyncedChannel = useCallback((layerId: string, ch: number) => {
    const group = CHANNEL_GROUPS.find((g) => g.includes(layerId));
    if (group) {
      setChannelMap((prev) => {
        const next = { ...prev };
        for (const id of group) next[id] = ch;
        return next;
      });
    } else {
      setChannelMap((prev) => ({ ...prev, [layerId]: ch }));
    }
  }, []);

  const selectedLayer =
    selectedBlock && modelData ? modelData.layers.get(selectedBlock) : null;
  const isInput = selectedBlock === 'input';
  const maxChannel = isInput ? 3 : (selectedLayer ? selectedLayer.def.shape[0] - 1 : 0);
  const channelLabel = isInput ? (['RGB', 'Red', 'Green', 'Blue'][getChannel(selectedBlock ?? '')] ?? 'Channel') : 'Channel';
  const isConv = selectedLayer?.def.type === 'conv';

  const handleExplodeToggle = useCallback(() => {
    if (!selectedBlock) return;
    setExplodedBlock((prev) => (prev === selectedBlock ? null : selectedBlock));
  }, [selectedBlock]);

  const handleChannelSelect = useCallback((blockId: string, channel: number) => {
    setSyncedChannel(blockId, channel);
    setExplodedBlock(null);
  }, [setSyncedChannel]);

  const handleRfSelect = useCallback((sel: typeof rfSelection) => {
    setRfSelection(sel);
    if (!sel) setRfMiniChannelMap({});
  }, []);

  const handleRfMiniChannelChange = useCallback(
    (layerId: string, delta: number) => {
      if (!modelData) return;
      const layer = modelData.layers.get(layerId);
      if (!layer) return;
      const maxCh = layer.def.shape[0] - 1;
      setRfMiniChannelMap((prev) => {
        const current = prev[layerId] ?? 0;
        let next = current + delta;
        if (next > maxCh) next = 0;
        else if (next < 0) next = maxCh;
        return { ...prev, [layerId]: next };
      });
    },
    [modelData]
  );

  return (
    <div className="flex h-screen w-screen bg-gray-950 text-white">
      <main className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-400">
            Loading model data...
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center text-red-400">
            Error: {error}
          </div>
        )}
        {modelData && (
          <CanvasView
            modelData={modelData}
            channelMap={channelMap}
            rfMiniChannelMap={rfMiniChannelMap}
            explodedBlock={explodedBlock}
            rfSelection={rfSelection}
            onBlockClick={setSelectedBlock}
            onExplode={setExplodedBlock}
            onChannelSelect={handleChannelSelect}
            onRfSelect={handleRfSelect}
            onRfMiniChannelChange={handleRfMiniChannelChange}
          />
        )}
      </main>
      <aside className="w-80 border-l border-gray-800 p-4 overflow-y-auto shrink-0">
        <h2 className="text-lg font-semibold mb-4">AlexNet Visualizer</h2>

        {rfSelection && modelData && (
          <div className="space-y-3 mb-4 p-3 rounded bg-gray-800/50">
            <p className="text-sm font-medium">Receptive Field</p>
            <p className="text-sm text-gray-300">
              Clicked: {rfSelection.layerId} [{rfSelection.row}, {rfSelection.col}] ch {rfSelection.channel}
            </p>
            {(() => {
              const regions = computeReceptiveFieldChain(
                modelData.metadata.layers,
                rfSelection.layerId,
                rfSelection.row,
                rfSelection.col
              );
              const layer = modelData.layers.get(rfSelection.layerId);
              const activation =
                layer && rfSelection.row < layer.activations.shape[1] && rfSelection.col < layer.activations.shape[2]
                  ? get3D(layer.activations.data, layer.activations.shape, rfSelection.channel, rfSelection.row, rfSelection.col)
                  : null;
              return (
                <>
                  {activation !== null && (
                    <p className="text-sm text-gray-300">Activation: {activation.toFixed(4)}</p>
                  )}
                  <div className="mt-2 space-y-1">
                    <p className="text-xs text-gray-500 uppercase tracking-wider">RF per layer (scroll on mini-block to change channel)</p>
                    {regions.map((r) => {
                      const layer = modelData.layers.get(r.layerId);
                      const maxCh = layer ? layer.def.shape[0] - 1 : 0;
                      const ch = rfMiniChannelMap[r.layerId] ?? channelMap[r.layerId] ?? 0;
                      return (
                        <p key={r.layerId} className="text-sm text-gray-300 font-mono">
                          {r.layerId}: {r.endRow - r.startRow}×{r.endCol - r.startCol} ch {ch}/{maxCh}
                        </p>
                      );
                    })}
                  </div>
                </>
              );
            })()}
          </div>
        )}
        {selectedLayer ? (
          <div className="space-y-4">
            <div>
              <p className="text-sm text-gray-400">Layer</p>
              <p className="font-mono">{selectedLayer.def.id}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Type</p>
              <p className="font-mono">{selectedLayer.def.type}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Shape</p>
              <p className="font-mono">{selectedLayer.def.shape.join(' × ')}</p>
            </div>
            {isConv && selectedLayer.def.weights_shape && (
              <div>
                <p className="text-sm text-gray-400">Kernel</p>
                <p className="font-mono">
                  {selectedLayer.def.weights_shape.join(' × ')}
                </p>
              </div>
            )}
            <div>
              <label className="text-sm text-gray-400 block mb-1">
                {isInput ? channelLabel : `Channel: ${getChannel(selectedLayer.def.id)}`}
              </label>
              <input
                type="range"
                min={0}
                max={maxChannel}
                value={Math.min(getChannel(selectedLayer.def.id), maxChannel)}
                onChange={(e) =>
                  setSyncedChannel(selectedLayer.def.id, Number(e.target.value))
                }
                className="w-full"
              />
            </div>
            <div>
              <button
                onClick={handleExplodeToggle}
                className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
              >
                {explodedBlock === selectedLayer.def.id ? 'Collapse' : 'Explode View'}
              </button>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-400">Click a layer to inspect</p>
        )}
      </aside>
    </div>
  );
}

export default App;
