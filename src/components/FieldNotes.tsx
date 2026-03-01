import { Link } from 'react-router-dom';

const COLORS = {
  input: '#4a9eff',
  kernel: '#ff6b6b',
  padding: '#51cf66',
  stride: '#ffd43b',
  output: '#b197fc',
} as const;

type LayerParams = {
  name: string;
  input: number;
  kernel: number;
  stride: number;
  padding: number;
  output: number;
};

const ALEXNET_LAYERS: LayerParams[] = [
  { name: 'Conv1', input: 227, kernel: 11, stride: 4, padding: 2, output: 56 },
  { name: 'Pool1', input: 56, kernel: 3, stride: 2, padding: 0, output: 27 },
  { name: 'Conv2', input: 27, kernel: 5, stride: 1, padding: 2, output: 27 },
  { name: 'Pool2', input: 27, kernel: 3, stride: 2, padding: 0, output: 13 },
  { name: 'Conv3', input: 13, kernel: 3, stride: 1, padding: 1, output: 13 },
  { name: 'Conv4', input: 13, kernel: 3, stride: 1, padding: 1, output: 13 },
  { name: 'Conv5', input: 13, kernel: 3, stride: 1, padding: 1, output: 13 },
  { name: 'Pool5', input: 13, kernel: 3, stride: 2, padding: 0, output: 6 },
];

function ColoredNumber({
  value,
  type,
}: {
  value: number;
  type: keyof typeof COLORS;
}) {
  return <span style={{ color: COLORS[type] }}>{value}</span>;
}

function LayerBreakdown({ layer }: { layer: LayerParams }) {
  const inner = layer.input - layer.kernel + 2 * layer.padding;
  const divided = Math.floor(inner / layer.stride) + 1;

  return (
    <div className="font-mono text-sm space-y-1 text-gray-300">
      <p>
        (<ColoredNumber value={layer.input} type="input" /> −{' '}
        <ColoredNumber value={layer.kernel} type="kernel" /> + 2×
        <ColoredNumber value={layer.padding} type="padding" />) ÷{' '}
        <ColoredNumber value={layer.stride} type="stride" /> + 1
      </p>
      <p>
        = (<ColoredNumber value={layer.input} type="input" /> −{' '}
        <ColoredNumber value={layer.kernel} type="kernel" /> +{' '}
        <ColoredNumber value={2 * layer.padding} type="padding" />) ÷{' '}
        <ColoredNumber value={layer.stride} type="stride" /> + 1
      </p>
      <p>
        = {inner} ÷ <ColoredNumber value={layer.stride} type="stride" /> + 1
      </p>
      <p>= {Math.floor(inner / layer.stride)} + 1</p>
      <p>
        = <ColoredNumber value={divided} type="output" />
      </p>
    </div>
  );
}

function Conv1Diagram() {
  const cellSize = 12;
  const paddingCells = 2;
  const inputCells = 8;
  const kernelCells = 3;
  const strideCells = 2;
  const totalCells = inputCells + 2 * paddingCells;
  const diagramW = totalCells * cellSize + 60;
  const diagramH = totalCells * cellSize + 60;

  return (
    <div className="rounded-lg border border-gray-700 p-4 bg-gray-900/50">
      <p className="text-sm text-gray-400 mb-3">
        Simplified view: input (blue) 227×227, padding (green) 2px each side,
        kernel (red) 11×11 slides with stride 4. Each step the kernel moves 4
        pixels.
      </p>
      <svg width={diagramW} height={diagramH} className="overflow-visible">
        <defs>
          <marker
            id="arrow"
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="4"
            orient="auto"
          >
            <polygon points="0 0, 8 4, 0 8" fill={COLORS.stride} />
          </marker>
          <pattern
            id="inputGrid"
            width={cellSize}
            height={cellSize}
            patternUnits="userSpaceOnUse"
          >
            <rect
              width={cellSize}
              height={cellSize}
              fill="none"
              stroke={COLORS.input}
              strokeWidth="0.6"
              opacity="0.7"
            />
          </pattern>
          <pattern
            id="paddingGrid"
            width={cellSize}
            height={cellSize}
            patternUnits="userSpaceOnUse"
          >
            <rect width={cellSize} height={cellSize} fill={COLORS.padding} opacity="0.2" />
            <rect
              width={cellSize}
              height={cellSize}
              fill="none"
              stroke={COLORS.padding}
              strokeWidth="0.4"
              opacity="0.6"
            />
          </pattern>
        </defs>
        {/* Padding (green) */}
        <rect
          x={30}
          y={30}
          width={totalCells * cellSize}
          height={totalCells * cellSize}
          fill="url(#paddingGrid)"
          stroke={COLORS.padding}
          strokeWidth="1.5"
        />
        {/* Input (blue) */}
        <rect
          x={30 + paddingCells * cellSize}
          y={30 + paddingCells * cellSize}
          width={inputCells * cellSize}
          height={inputCells * cellSize}
          fill="url(#inputGrid)"
          stroke={COLORS.input}
          strokeWidth="2"
        />
        {/* Kernel position 1 */}
        <rect
          x={30 + paddingCells * cellSize}
          y={30 + paddingCells * cellSize}
          width={kernelCells * cellSize}
          height={kernelCells * cellSize}
          fill={COLORS.kernel}
          fillOpacity="0.4"
          stroke={COLORS.kernel}
          strokeWidth="2"
        />
        {/* Kernel position 2 (stride step right) */}
        <rect
          x={30 + paddingCells * cellSize + strideCells * cellSize}
          y={30 + paddingCells * cellSize}
          width={kernelCells * cellSize}
          height={kernelCells * cellSize}
          fill={COLORS.kernel}
          fillOpacity="0.25"
          stroke={COLORS.kernel}
          strokeWidth="1.5"
          strokeDasharray="3"
        />
        {/* Arrow showing stride */}
        <line
          x1={30 + paddingCells * cellSize + (kernelCells * cellSize) / 2}
          y1={30 + paddingCells * cellSize + kernelCells * cellSize + 6}
          x2={30 + paddingCells * cellSize + strideCells * cellSize + (kernelCells * cellSize) / 2}
          y2={30 + paddingCells * cellSize + kernelCells * cellSize + 6}
          stroke={COLORS.stride}
          strokeWidth="2"
          markerEnd="url(#arrow)"
        />
        <text
          x={30 + paddingCells * cellSize + (strideCells * cellSize) / 2 + (kernelCells * cellSize) / 2}
          y={30 + paddingCells * cellSize + kernelCells * cellSize + 20}
          fill={COLORS.stride}
          fontSize="10"
          textAnchor="middle"
        >
          stride S
        </text>
        <text x={30} y={25} fill={COLORS.padding} fontSize="10">
          P (padding)
        </text>
        <text
          x={30 + (totalCells * cellSize) / 2}
          y={30 + totalCells * cellSize + 22}
          fill={COLORS.input}
          fontSize="11"
          textAnchor="middle"
        >
          input W×H
        </text>
      </svg>
    </div>
  );
}

export default function FieldNotes() {
  return (
    <div className="min-h-screen bg-gray-950 text-white overflow-y-auto">
      <div className="max-w-3xl mx-auto px-6 py-12 pb-24">
        {/* Back link */}
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-gray-400 hover:text-white mb-10 text-sm transition-colors"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10 19l-7-7m0 0l7-7m-7 7h18"
            />
          </svg>
          Back to visualizer
        </Link>

        {/* Title */}
        <h1 className="text-3xl font-bold tracking-tight mb-2">
          Field Notes: Convolution Output Size
        </h1>
        <p className="text-gray-400 mb-12">
          How output dimensions are computed for conv and pool layers in AlexNet
        </p>

        {/* Legend */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-5 mb-12">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            Variable Legend
          </h2>
          <div className="flex flex-wrap gap-x-6 gap-y-3">
            <span style={{ color: COLORS.input }}>●</span>
            <span className="text-gray-300">input_size (W/H)</span>
            <span style={{ color: COLORS.kernel }}>●</span>
            <span className="text-gray-300">kernel_size (K)</span>
            <span style={{ color: COLORS.padding }}>●</span>
            <span className="text-gray-300">padding (P)</span>
            <span style={{ color: COLORS.stride }}>●</span>
            <span className="text-gray-300">stride (S)</span>
            <span style={{ color: COLORS.output }}>●</span>
            <span className="text-gray-300">output_size</span>
          </div>
        </div>

        {/* Formula */}
        <section className="mb-16">
          <h2 className="text-xl font-semibold mb-6">The Formula</h2>
          <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-8 font-mono text-lg md:text-xl">
            <span style={{ color: COLORS.output }}>output</span>
            <span className="text-gray-400"> = floor((</span>
            <span style={{ color: COLORS.input }}>input</span>
            <span className="text-gray-400"> − </span>
            <span style={{ color: COLORS.kernel }}>kernel</span>
            <span className="text-gray-400"> + 2 × </span>
            <span style={{ color: COLORS.padding }}>padding</span>
            <span className="text-gray-400">) ÷ </span>
            <span style={{ color: COLORS.stride }}>stride</span>
            <span className="text-gray-400">) + 1</span>
          </div>
        </section>

        {/* Visual diagram */}
        <section className="mb-16">
          <h2 className="text-xl font-semibold mb-6">Visual Diagram (Conv1)</h2>
          <Conv1Diagram />
        </section>

        {/* Interactive examples */}
        <section className="mb-16">
          <h2 className="text-xl font-semibold mb-6">AlexNet Layer Examples</h2>
          <div className="space-y-10">
            {ALEXNET_LAYERS.map((layer) => (
              <div
                key={layer.name}
                className="rounded-xl border border-gray-800 bg-gray-900/30 p-6"
              >
                <h3 className="text-lg font-semibold mb-4">{layer.name}</h3>
                <p className="font-mono text-base mb-4">
                  <ColoredNumber value={layer.output} type="output" />
                  <span className="text-gray-400"> = floor((</span>
                  <ColoredNumber value={layer.input} type="input" />
                  <span className="text-gray-400"> − </span>
                  <ColoredNumber value={layer.kernel} type="kernel" />
                  <span className="text-gray-400"> + 2 × </span>
                  <ColoredNumber value={layer.padding} type="padding" />
                  <span className="text-gray-400">) ÷ </span>
                  <ColoredNumber value={layer.stride} type="stride" />
                  <span className="text-gray-400">) + 1</span>
                </p>
                <div className="pl-4 border-l-2 border-gray-700">
                  <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
                    Step-by-step
                  </p>
                  <LayerBreakdown layer={layer} />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Summary table */}
        <section>
          <h2 className="text-xl font-semibold mb-6">Quick Reference</h2>
          <div className="overflow-x-auto rounded-xl border border-gray-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 font-medium">Layer</th>
                  <th className="text-left py-3 px-4 font-medium" style={{ color: COLORS.input }}>
                    Input
                  </th>
                  <th className="text-left py-3 px-4 font-medium" style={{ color: COLORS.kernel }}>
                    Kernel
                  </th>
                  <th className="text-left py-3 px-4 font-medium" style={{ color: COLORS.stride }}>
                    Stride
                  </th>
                  <th className="text-left py-3 px-4 font-medium" style={{ color: COLORS.padding }}>
                    Pad
                  </th>
                  <th className="text-left py-3 px-4 font-medium" style={{ color: COLORS.output }}>
                    Output
                  </th>
                </tr>
              </thead>
              <tbody>
                {ALEXNET_LAYERS.map((layer) => (
                  <tr key={layer.name} className="border-b border-gray-800 last:border-0">
                    <td className="py-3 px-4 font-mono">{layer.name}</td>
                    <td className="py-3 px-4" style={{ color: COLORS.input }}>
                      {layer.input}
                    </td>
                    <td className="py-3 px-4" style={{ color: COLORS.kernel }}>
                      {layer.kernel}
                    </td>
                    <td className="py-3 px-4" style={{ color: COLORS.stride }}>
                      {layer.stride}
                    </td>
                    <td className="py-3 px-4" style={{ color: COLORS.padding }}>
                      {layer.padding}
                    </td>
                    <td className="py-3 px-4" style={{ color: COLORS.output }}>
                      {layer.output}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
