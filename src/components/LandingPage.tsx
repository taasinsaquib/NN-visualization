import { Link } from 'react-router-dom';

const LAYER_STEPS = [
  { label: 'Input', color: '#4a9eff', sub: '227×227×3' },
  { label: 'Conv1', color: '#ff6b6b', sub: '56×56×64' },
  { label: 'ReLU', color: '#51cf66' },
  { label: 'Pool1', color: '#ffd43b', sub: '27×27×64' },
  { label: 'Conv2', color: '#ff6b6b', sub: '27×27×192' },
  { label: 'ReLU', color: '#51cf66' },
  { label: 'Pool2', color: '#ffd43b', sub: '13×13×192' },
  { label: 'Conv3', color: '#ff6b6b', sub: '13×13×384' },
  { label: 'ReLU', color: '#51cf66' },
  { label: 'Conv4', color: '#ff6b6b', sub: '13×13×256' },
  { label: 'ReLU', color: '#51cf66' },
  { label: 'Conv5', color: '#ff6b6b', sub: '13×13×256' },
  { label: 'ReLU', color: '#51cf66' },
  { label: 'Pool5', color: '#ffd43b', sub: '6×6×256' },
  { label: 'FC6', color: '#a78bfa', sub: '4096' },
  { label: 'FC7', color: '#a78bfa', sub: '4096' },
  { label: 'FC8', color: '#a78bfa', sub: '1000' },
  { label: 'Classes', color: '#f472b6', sub: 'Top-5' },
];

const FEATURES = [
  {
    title: 'Layer-by-Layer Exploration',
    description: 'View activations and weights for every layer in the network. See how the input image transforms as it flows through convolutions, pooling, and fully connected layers.',
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
      </svg>
    ),
  },
  {
    title: 'Receptive Field Tracing',
    description: 'Click any neuron to trace its receptive field back to the input image. Visualize exactly which pixels contribute to each activation.',
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
  {
    title: 'Kernel Visualization',
    description: 'See convolution kernels as RGB images or heatmaps. Explore how learned filters detect edges, textures, and patterns.',
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    title: 'Output Size Formula',
    description: 'Interactive field notes on how convolution output dimensions are computed, with color-coded variables and step-by-step breakdowns.',
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
    ),
  },
];

function NetworkDiagram() {
  return (
    <svg viewBox="0 0 320 200" className="w-full max-w-md mx-auto opacity-20" aria-hidden="true">
      {[0, 1, 2].map((col) =>
        Array.from({ length: [4, 6, 3][col] }, (_, row) => {
          const cx = 80 + col * 80;
          const count = [4, 6, 3][col];
          const cy = 100 - ((count - 1) * 25) / 2 + row * 25;
          return <circle key={`${col}-${row}`} cx={cx} cy={cy} r="6" fill="currentColor" opacity={0.6} />;
        })
      )}
      {[0, 1].map((fromCol) => {
        const fromCount = [4, 6, 3][fromCol];
        const toCount = [4, 6, 3][fromCol + 1];
        return Array.from({ length: fromCount }, (_, fromRow) =>
          Array.from({ length: toCount }, (_, toRow) => {
            const x1 = 80 + fromCol * 80;
            const y1 = 100 - ((fromCount - 1) * 25) / 2 + fromRow * 25;
            const x2 = 80 + (fromCol + 1) * 80;
            const y2 = 100 - ((toCount - 1) * 25) / 2 + toRow * 25;
            return (
              <line
                key={`${fromCol}-${fromRow}-${toRow}`}
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="currentColor" strokeWidth="0.5" opacity={0.15}
              />
            );
          })
        );
      })}
    </svg>
  );
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-indigo-950/30 via-transparent to-transparent" />
        <div className="relative max-w-5xl mx-auto px-6 pt-24 pb-20 text-center">
          <NetworkDiagram />
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mt-8">
            AlexNet{' '}
            <span className="bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
              Visualizer
            </span>
          </h1>
          <p className="mt-6 text-lg text-gray-400 max-w-2xl mx-auto leading-relaxed">
            An interactive visualization of the AlexNet convolutional neural network.
            Explore activations, weights, and receptive fields across every layer — from
            the 227×227 input image through to the final 1000-class predictions.
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-4">
            <Link
              to="/visualizer"
              className="px-8 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 font-medium transition-colors text-lg"
            >
              Explore the Network
            </Link>
            <Link
              to="/field-notes"
              className="px-8 py-3 rounded-lg border border-gray-700 hover:border-gray-500 hover:bg-gray-900 font-medium transition-colors text-lg"
            >
              Field Notes
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-5xl mx-auto px-6 py-20">
        <h2 className="text-2xl font-semibold text-center mb-12">What you can explore</h2>
        <div className="grid sm:grid-cols-2 gap-6">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="rounded-xl border border-gray-800 bg-gray-900/30 p-6 hover:border-gray-700 transition-colors"
            >
              <div className="text-indigo-400 mb-4">{f.icon}</div>
              <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
              <p className="text-sm text-gray-400 leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Architecture */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="text-2xl font-semibold text-center mb-10">Full Architecture at a Glance</h2>
        <div className="overflow-x-auto pb-4">
          <div className="flex items-center gap-1 min-w-max mx-auto justify-center">
            {LAYER_STEPS.map((step, i) => (
              <div key={i} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div
                    className="rounded-md px-2 py-1 text-xs font-mono font-medium border"
                    style={{
                      borderColor: step.color + '60',
                      backgroundColor: step.color + '15',
                      color: step.color,
                    }}
                  >
                    {step.label}
                  </div>
                  {step.sub && (
                    <span className="text-[10px] text-gray-500 mt-1 font-mono">{step.sub}</span>
                  )}
                </div>
                {i < LAYER_STEPS.length - 1 && (
                  <svg className="w-4 h-4 text-gray-600 mx-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-8 text-center text-sm text-gray-500">
        <p>
          Based on <span className="text-gray-400">"ImageNet Classification with Deep Convolutional Neural Networks"</span>
        </p>
        <p className="mt-1">Krizhevsky, Sutskever &amp; Hinton (2012)</p>
      </footer>
    </div>
  );
}
