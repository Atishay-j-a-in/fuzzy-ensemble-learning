import React, { useState, useMemo } from 'react';
import { 
  Layers, 
  Workflow, 
  Cpu, 
  Sparkles, 
  CheckCircle2, 
  Maximize2, 
  Info, 
  GitMerge, 
  Scale, 
  ShieldAlert, 
  ShieldCheck, 
  ChevronRight,
  Eye,
  Sliders,
  Calculator,
  Binary,
  Microscope,
  Database
} from 'lucide-react';

interface PipelineNode {
  id: string;
  stageNumber: string;
  title: string;
  badge: string;
  color: string;
  summary: string;
  details: Record<string, string>;
  codeSnippet?: string;
}

export const ArchitectureSection: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'pipeline' | 'harvesting' | 'simulator' | 'taxonomy' | 'figures'>('pipeline');
  const [selectedNodeId, setSelectedNodeId] = useState<string>('harvesting');
  const [selectedImageModal, setSelectedImageModal] = useState<string | null>(null);

  // Interactive Fuzzy Simulator State (default confidences for 5 models)
  const [simXception, setSimXception] = useState<number>(0.92);
  const [simIRV2, setSimIRV2] = useState<number>(0.89);
  const [simIV3, setSimIV3] = useState<number>(0.88);
  const [simVGG19, setSimVGG19] = useState<number>(0.84);
  const [simVGG16, setSimVGG16] = useState<number>(0.86);

  // Shapley weights from publication
  const shapleyWeights = {
    'Xception': 0.23,
    'InceptionResNetV2': 0.22,
    'InceptionV3': 0.20,
    'VGG19': 0.18,
    'VGG16': 0.17,
  };

  // Real-time Choquet Integral calculation
  const calculatedChoquet = useMemo(() => {
    const models = [
      { name: 'Xception', val: simXception, weight: shapleyWeights['Xception'] },
      { name: 'InceptionResNetV2', val: simIRV2, weight: shapleyWeights['InceptionResNetV2'] },
      { name: 'InceptionV3', val: simIV3, weight: shapleyWeights['InceptionV3'] },
      { name: 'VGG19', val: simVGG19, weight: shapleyWeights['VGG19'] },
      { name: 'VGG16', val: simVGG16, weight: shapleyWeights['VGG16'] },
    ];

    // Sort ascending: f(x_(1)) <= f(x_(2)) <= ... <= f(x_(n))
    const sorted = [...models].sort((a, b) => a.val - b.val);

    let integral = 0;
    const steps: Array<{ stepIndex: number; delta: number; subsetWeight: number; contribution: number; subset: string[] }> = [];

    for (let i = 0; i < sorted.length; i++) {
      const prevVal = i === 0 ? 0 : sorted[i - 1].val;
      const currentVal = sorted[i].val;
      const delta = currentVal - prevVal;

      // Subset A_(i) contains elements from index i to n-1
      const subsetElements = sorted.slice(i);
      const subsetWeight = subsetElements.reduce((acc, el) => acc + el.weight, 0);
      const contribution = delta * subsetWeight;
      integral += contribution;

      steps.push({
        stepIndex: i + 1,
        delta,
        subsetWeight,
        contribution,
        subset: subsetElements.map(el => el.name),
      });
    }

    const simpleAvg = models.reduce((a, b) => a + b.val, 0) / models.length;

    return {
      sorted,
      integral: Math.min(1.0, Math.max(0, integral)),
      simpleAvg,
      steps,
      boost: integral - simpleAvg,
    };
  }, [simXception, simIRV2, simIV3, simVGG19, simVGG16]);

  const pipelineNodes: PipelineNode[] = [
    {
      id: 'input',
      stageNumber: '01',
      title: 'Biopsy Image Acquisition',
      badge: '512 × 512 × 3 RGB',
      color: 'border-blue-500/50 text-blue-400',
      summary: 'High-resolution microscopic breast biopsy patches digitized from Whole Slide Images (WSI) at 200× magnification.',
      details: {
        'Image Dimensions': '(512, 512, 3) Float32 Tensor',
        'Dataset Benchmark': 'ICIAR 2018 Grand Challenge on Breast Cancer Histology (BACH)',
        'Stain Chemistry': 'Hematoxylin and Eosin (H&E) histological staining',
        'Spatial Resolution': '0.42 μm/pixel at 200× optical zoom',
      },
      codeSnippet: `image = Image.open(path).convert('RGB')\nimage = image.resize((512, 512))\ntensor = np.array(image, dtype=np.float32) / 255.0`
    },
    {
      id: 'macenko',
      stageNumber: '02',
      title: 'Macenko Optical Density Normalization',
      badge: 'SVD Deconvolution',
      color: 'border-indigo-500/50 text-indigo-400',
      summary: 'Converts RGB color space into Optical Density (OD) vectors to isolate Hematoxylin and Eosin stain vectors via Singular Value Decomposition (SVD).',
      details: {
        'Mathematical Space': 'OD = -log10((I + 1) / 255)',
        'Deconvolution': 'SVD decomposition on optical density vectors with OD > 0.15 threshold',
        'Purpose': 'Eliminates staining batch variability across different pathology laboratories',
        'Output': 'Normalized RGB tensor calibrated against reference clinical staining matrix',
      },
      codeSnippet: `# Convert RGB to Optical Density\nOD = -np.log10((I.astype(np.float32) + 1) / 255.0)\nODhat = OD[~np.any(OD < beta, axis=1)]\n_, V = np.linalg.eigh(np.cov(ODhat, rowvar=False))`
    },
    {
      id: 'backbones',
      stageNumber: '03',
      title: '5 Pre-Trained Deep Backbones',
      badge: 'Parallel Feature Extraction',
      color: 'border-purple-500/50 text-purple-400',
      summary: 'Parallel execution across 5 deep convolutional architectures initialized with pre-trained ImageNet representations.',
      details: {
        'Architectures': 'Xception (20.8M), InceptionResNetV2 (54.3M), InceptionV3 (21.8M), VGG19 (20.0M), VGG16 (14.7M)',
        'Fine-Tuning': 'First 95 layers frozen; subsequent convolutional stages fine-tuned on normalized BACH dataset',
        'Transfer Strategy': 'Multi-scale intermediate receptive field harvesting',
      },
      codeSnippet: `xception = tf.keras.applications.Xception(include_top=False, weights="imagenet")\nirv2 = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet")\niv3 = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")`
    },
    {
      id: 'harvesting',
      stageNumber: '04',
      title: 'Multi-Scale Layer Harvesting',
      badge: '2,520-dim Vector (Xception)',
      color: 'border-teal-500/50 text-teal-400',
      summary: 'Instead of taking only final bottleneck features, intermediate activation maps are tapped across Early, Middle, and Exit flows to capture cytology at multiple scales.',
      details: {
        'Low-Level Tap': 'block4_sepconv1_act -> GlobalAveragePooling2D (728 dims) -> Nuclear chromatin texture',
        'Mid-Level Tap': 'block5_sepconv1_act -> GlobalAveragePooling2D (728 dims) -> Glandular architecture',
        'High-Level Tap': 'block14_sepconv1 -> GlobalAveragePooling2D (1064 dims) -> Stromal invasion patterns',
        'Concatenation': 'Concatenate([x1, x2, x3]) = 2,520-dimensional composite feature vector',
      },
      codeSnippet: `taps = ['block4_sepconv1_act', 'block5_sepconv1_act', 'block14_sepconv1']\npooled = [GlobalAveragePooling2D()(xception.get_layer(name).output) for name in taps]\ncomposite_features = Concatenate()(pooled) # shape: (batch, 2520)`
    },
    {
      id: 'heads',
      stageNumber: '05',
      title: 'Dense Classification Heads',
      badge: 'Dense(512) -> Dropout(0.5)',
      color: 'border-amber-500/50 text-amber-400',
      summary: 'Task-specific multi-layer perceptron heads trained with Dropout and Adam optimizer to generate calibrated class probability vectors.',
      details: {
        'Layers': 'Dense(512, activation="relu") -> BatchNormalization -> Dropout(0.5) -> Dense(4, activation="softmax")',
        'Loss Function': 'Categorical Cross-Entropy with label smoothing',
        'Output': '4-element calibrated probability distribution vector',
      },
      codeSnippet: `x = Dense(512, activation='relu')(composite_features)\nx = BatchNormalization()(x)\nx = Dropout(0.5)(x)\noutput = Dense(4, activation='softmax')(x)`
    },
    {
      id: 'choquet',
      stageNumber: '06',
      title: 'Choquet Fuzzy Integral Fusion',
      badge: 'Shapley Game Theory (95% Acc)',
      color: 'border-teal-400 text-teal-300',
      summary: 'Non-additive decision fusion combining the individual model probability vectors using Shapley coalition game theory and the Choquet Fuzzy Integral.',
      details: {
        'Mathematical Formula': 'C_μ(f) = ∑_{i=1}^{n} (f(x_(i)) - f(x_(i-1))) · μ(A_(i))',
        'Coalition Weights': 'μ(Xception)=0.23, μ(IRV2)=0.22, μ(IV3)=0.20, μ(VGG19)=0.18, μ(VGG16)=0.17',
        'Diagnostic Boost': 'Elevates 4-class test accuracy from 91.0% to 95.0% and binary accuracy to 96.0%',
      },
      codeSnippet: `# Choquet Integral implementation\nsorted_idx = np.argsort(model_probs)\nintegral = 0.0\nfor i in range(len(sorted_idx)):\n    delta = sorted_probs[i] - (sorted_probs[i-1] if i > 0 else 0)\n    subset_weight = sum(shapley_weights[m] for m in sorted_models[i:])\n    integral += delta * subset_weight`
    },
  ];

  const publicationFigures = [
    {
      src: '/assets/Method Flowchart.png',
      fallback: 'Method Flowchart.png',
      title: 'Figure 1: Method Flowchart of the Proposed Framework',
      desc: 'Complete overview schematic illustrating Macenko stain normalization, parallel deep feature extraction across 5 models, and Choquet fuzzy integral consensus.',
    },
    {
      src: '/assets/VGG19.png',
      fallback: 'VGG19.png',
      title: 'Figure 2: Multi-Scale Layer Harvesting Architecture',
      desc: 'Detailed diagram illustrating how multi-depth activation maps are pooled and concatenated before the dense classification head.',
    },
    {
      src: '/assets/images.PNG',
      fallback: 'images.PNG',
      title: 'Figure 3: Histological Biopsy Taxonomy Samples',
      desc: 'Microscopic patches from ICIAR 2018 BACH: (A) Normal tissue, (B) Benign lesion, (C) In-situ carcinoma, and (D) Invasive carcinoma.',
    },
    {
      src: '/assets/dataset.png',
      fallback: 'dataset.png',
      title: 'Figure 4: ICIAR 2018 BACH Dataset Distribution',
      desc: 'Distribution summary of training, validation, and testing patches across all 4 histopathological categories.',
    },
    {
      src: '/assets/cover.png',
      fallback: 'cover.png',
      title: 'Figure 5: System Research Cover Banner',
      desc: 'Computational pathology AI framework banner highlighting the fuzzy multi-model coalition paradigm.',
    },
  ];

  return (
    <div className="py-8 sm:py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      
      {/* Section Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-teal-400 font-semibold mb-1">
            <Workflow className="w-4 h-4" />
            <span>SYSTEM ARCHITECTURE & METHODOLOGY</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Breast Cancer Deep Learning Pipeline
          </h1>
          <p className="text-xs sm:text-sm text-slate-300 mt-1 max-w-2xl">
            Explore the multi-scale intermediate feature harvesting network, Macenko optical density deconvolution, and game-theoretic Choquet fuzzy integral consensus.
          </p>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex flex-wrap gap-1 bg-slate-900/90 p-1.5 rounded-xl border border-slate-800 shrink-0">
          {[
            { id: 'pipeline', label: '1. Pipeline Flow', icon: Workflow },
            { id: 'harvesting', label: '2. Layer Harvesting', icon: Cpu },
            { id: 'simulator', label: '3. Fuzzy Simulator', icon: Calculator },
            { id: 'taxonomy', label: '4. Pathology Taxonomy', icon: ShieldAlert },
            { id: 'figures', label: '5. Research Figures', icon: Layers },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  isActive
                    ? 'bg-teal-600 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* TAB 1: END-TO-END PIPELINE DIAGRAM */}
      {activeTab === 'pipeline' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column: Interactive 6-Stage Node Pipeline */}
          <div className="lg:col-span-7 space-y-3">
            <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
              <span>Interactive Pipeline Nodes (Click to inspect)</span>
              <span className="text-teal-400 text-[11px]">Selected: {selectedNodeId}</span>
            </div>

            {pipelineNodes.map((node, index) => {
              const isSelected = selectedNodeId === node.id;
              return (
                <div
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  className={`p-4 rounded-xl cursor-pointer transition-all border ${
                    isSelected
                      ? 'bg-slate-900 border-teal-500 shadow-lg shadow-teal-950/40 translate-x-1'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/90'
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2.5">
                      <span className="font-mono text-xs font-bold text-teal-400 bg-teal-950/60 px-2 py-0.5 rounded border border-teal-500/30">
                        {node.stageNumber}
                      </span>
                      <h3 className="text-sm font-bold text-white">{node.title}</h3>
                    </div>
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-950 text-slate-300 border border-slate-800">
                      {node.badge}
                    </span>
                  </div>

                  <p className="mt-2 text-xs text-slate-300 leading-relaxed">
                    {node.summary}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Right Column: Node Inspector & Python Code Box */}
          <div className="lg:col-span-5 sticky top-24">
            {(() => {
              const node = pipelineNodes.find((n) => n.id === selectedNodeId) || pipelineNodes[3];
              return (
                <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-5 shadow-2xl">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div className="flex items-center gap-2 text-xs font-mono font-bold text-teal-400 uppercase tracking-wider">
                      <Info className="w-4 h-4" />
                      <span>Node Specifications</span>
                    </div>
                    <span className="font-mono text-xs text-slate-400">Stage {node.stageNumber}</span>
                  </div>

                  <div>
                    <h2 className="text-lg font-bold text-white">{node.title}</h2>
                    <p className="mt-1.5 text-xs text-slate-300 leading-relaxed">{node.summary}</p>
                  </div>

                  {/* Technical Key-Value Specs */}
                  <div className="space-y-2">
                    {Object.entries(node.details).map(([key, val]) => (
                      <div key={key} className="p-3 rounded-lg bg-slate-950 border border-slate-850">
                        <span className="text-[10px] font-mono text-teal-400 font-bold uppercase tracking-wider block">
                          {key}
                        </span>
                        <span className="text-xs text-slate-200 font-mono mt-0.5 block">
                          {val}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Implementation Code */}
                  {node.codeSnippet && (
                    <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                      <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider font-semibold">
                        Backend Python / TensorFlow Implementation:
                      </div>
                      <pre className="text-[11px] font-mono text-emerald-400 overflow-x-auto leading-relaxed">
                        {node.codeSnippet}
                      </pre>
                    </div>
                  )}

                </div>
              );
            })()}
          </div>

        </div>
      )}

      {/* TAB 2: MULTI-SCALE HARVESTING ANATOMY */}
      {activeTab === 'harvesting' && (
        <div className="space-y-6">
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
              <div>
                <h3 className="text-lg font-bold text-white">
                  Multi-Scale Intermediate Layer Harvesting Scheme
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Extracts hierarchical representations across shallow, mid, and deep receptive fields to retain both cytology texture and invasive tissue topography.
                </p>
              </div>
              <span className="font-mono text-xs font-bold px-3 py-1 rounded-lg bg-teal-500/20 text-teal-300 border border-teal-500/30">
                Composite Vector: 2,520 Features
              </span>
            </div>

            {/* 3 Harvest Taps Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono font-bold text-indigo-400 uppercase">Stage 1: Early Flow</span>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-500/30">728 Dims</span>
                </div>
                <h4 className="text-sm font-bold font-mono text-white">block4_sepconv1_act</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Tapped from entry flow. Preserves fine cellular cytology: nuclear chromatin borders, nuclear-to-cytoplasmic ratio, and basement membrane clarity.
                </p>
                <div className="p-2 rounded bg-slate-900 font-mono text-[11px] text-slate-400">
                  GlobalAveragePooling2D() &rarr; 728
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono font-bold text-teal-400 uppercase">Stage 2: Middle Flow</span>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-500/30">728 Dims</span>
                </div>
                <h4 className="text-sm font-bold font-mono text-white">block5_sepconv1_act</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Tapped from intermediate residual blocks. Preserves glandular morphology: ductal lumina contours, epithelial cell layer stratification, and cribriform patterns.
                </p>
                <div className="p-2 rounded bg-slate-900 font-mono text-[11px] text-slate-400">
                  GlobalAveragePooling2D() &rarr; 728
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono font-bold text-amber-400 uppercase">Stage 3: Exit Flow</span>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-500/30">1,064 Dims</span>
                </div>
                <h4 className="text-sm font-bold font-mono text-white">block14_sepconv1</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Tapped from final convolutional module. Captures macro-tissue semantics: stromal desmoplasia, angiogenesis, and infiltrative neoplastic nests.
                </p>
                <div className="p-2 rounded bg-slate-900 font-mono text-[11px] text-slate-400">
                  GlobalAveragePooling2D() &rarr; 1064
                </div>
              </div>

            </div>
          </div>

          {/* All 5 Model Vector Breakdown Comparison */}
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-4">
            <h3 className="text-sm font-mono font-bold text-slate-300 uppercase tracking-wider">
              Harvesting Feature Dimensions Across All 5 Backbones
            </h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
              {[
                { name: 'Xception', dims: '2,520 dims', taps: '3 Taps (Block 4, 5, 14)', testAcc: '91.0%', color: 'border-teal-500/40 text-teal-300' },
                { name: 'InceptionV3', dims: '2,320 dims', taps: '13 Mixed Conv Taps', testAcc: '90.0%', color: 'border-indigo-500/40 text-indigo-300' },
                { name: 'VGG16', dims: '1,152 dims', taps: 'Block 2, 4, 5 Conv1', testAcc: '86.0%', color: 'border-sky-500/40 text-sky-300' },
                { name: 'VGG19', dims: '896 dims', taps: 'Block 2, 3, 5 Conv1', testAcc: '83.0%', color: 'border-purple-500/40 text-purple-300' },
                { name: 'InceptionResNetV2', dims: '464 dims', taps: '4 Intermediate Layers', testAcc: '91.0%', color: 'border-amber-500/40 text-amber-300' },
              ].map((m) => (
                <div key={m.name} className={`p-4 rounded-xl bg-slate-950 border ${m.color} space-y-2`}>
                  <div className="font-mono font-bold text-white text-xs">{m.name}</div>
                  <div className="text-base font-bold font-mono text-teal-400">{m.dims}</div>
                  <div className="text-[11px] text-slate-400">{m.taps}</div>
                  <div className="pt-2 border-t border-slate-800 text-[10px] font-mono text-slate-500">
                    Standalone: {m.testAcc}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: INTERACTIVE FUZZY SIMULATOR */}
      {activeTab === 'simulator' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left: Interactive Model Sliders */}
          <div className="lg:col-span-6 rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-teal-400 uppercase tracking-wider">
                <Sliders className="w-4 h-4" />
                <span>Adjust Individual Model Confidence Scores (f_i)</span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">Interactive Math Sandbox</span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Drag the sliders below to simulate different model confidence scores. Observe how the Choquet Fuzzy Integral aggregates non-additive Shapley coalition weights to produce a consensus that overcomes single-model error!
            </p>

            {/* Sliders */}
            <div className="space-y-4 pt-2">
              {[
                { name: 'Xception', state: simXception, set: setSimXception, weight: 0.23, color: 'accent-teal-500' },
                { name: 'InceptionResNetV2', state: simIRV2, set: setSimIRV2, weight: 0.22, color: 'accent-indigo-500' },
                { name: 'InceptionV3', state: simIV3, set: setSimIV3, weight: 0.20, color: 'accent-sky-500' },
                { name: 'VGG19', state: simVGG19, set: setSimVGG19, weight: 0.18, color: 'accent-purple-500' },
                { name: 'VGG16', state: simVGG16, set: setSimVGG16, weight: 0.17, color: 'accent-amber-500' },
              ].map((s) => (
                <div key={s.name} className="p-3 rounded-xl bg-slate-950 border border-slate-850 space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-bold text-white">{s.name} (Shapley &mu; = {s.weight})</span>
                    <span className="font-bold text-teal-300">{(s.state * 100).toFixed(1)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={s.state}
                    onChange={(e) => s.set(parseFloat(e.target.value))}
                    className={`w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer ${s.color}`}
                  />
                </div>
              ))}
            </div>

            <button
              onClick={() => {
                setSimXception(0.92);
                setSimIRV2(0.89);
                setSimIV3(0.88);
                setSimVGG19(0.84);
                setSimVGG16(0.86);
              }}
              className="text-xs text-slate-400 hover:text-teal-300 underline underline-offset-4 font-mono transition-colors"
            >
              Reset to publication baseline values
            </button>
          </div>

          {/* Right: Live Choquet Math Output */}
          <div className="lg:col-span-6 space-y-5">
            
            {/* Main Result Tile */}
            <div className="rounded-2xl bg-slate-900 border border-teal-500/40 p-6 space-y-4 shadow-xl">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-bold text-teal-400 uppercase tracking-wider">
                  Calculated Choquet Fuzzy Consensus
                </span>
                <span className="px-2 py-0.5 rounded bg-teal-500/20 text-teal-300 border border-teal-500/30 font-bold">
                  Game-Theoretic Integral
                </span>
              </div>

              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-extrabold font-mono text-white">
                  {(calculatedChoquet.integral * 100).toFixed(2)}%
                </span>
                <span className="text-xs font-mono text-slate-400">
                  vs Simple Mean: {(calculatedChoquet.simpleAvg * 100).toFixed(2)}%
                </span>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-300 leading-relaxed font-mono">
                Choquet Advantage: <strong className="text-teal-300">{(calculatedChoquet.boost >= 0 ? '+' : '') + (calculatedChoquet.boost * 100).toFixed(2)}%</strong> over arithmetic average due to non-additive coalition weighting.
              </div>
            </div>

            {/* Step-by-Step Integral Expansion */}
            <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-3">
              <h4 className="text-xs font-mono font-bold text-slate-300 uppercase tracking-wider">
                Step-by-Step Choquet Summation: C_μ(f) = ∑ (f_(i) - f_(i-1)) · μ(A_(i))
              </h4>

              <div className="space-y-2 text-xs font-mono">
                {calculatedChoquet.steps.map((step) => (
                  <div key={step.stepIndex} className="p-2.5 rounded-lg bg-slate-950 border border-slate-850 flex items-center justify-between">
                    <div>
                      <span className="text-teal-400 font-bold">Step {step.stepIndex}: </span>
                      <span className="text-slate-300">Δ = {(step.delta * 100).toFixed(1)}% &times; μ = {step.subsetWeight.toFixed(2)}</span>
                    </div>
                    <span className="font-bold text-slate-200">+{(step.contribution * 100).toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>

          </div>

        </div>
      )}

      {/* TAB 4: PATHOLOGY TAXONOMY REFERENCE */}
      {activeTab === 'taxonomy' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            {
              id: 'Normal',
              name: 'Normal tissue',
              tag: 'Physiological',
              color: 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300',
              summary: 'Healthy mammary parenchyma with preserved regular lobular and ductal architecture.',
              features: [
                'Continuous basement membrane',
                'Dual cell layer: inner secretory epithelial, outer contractile myoepithelial',
                'Zero cellular atypia, uniform nuclear chromatin',
              ],
              risk: 'Normal physiological baseline',
            },
            {
              id: 'Benign',
              name: 'Benign lesion',
              tag: 'Non-Malignant',
              color: 'border-sky-500/40 bg-sky-950/20 text-sky-300',
              summary: 'Adenomatous or fibromatous hyperplastic proliferation without basement membrane breach.',
              features: [
                'Preserved myoepithelial cell layer',
                'Low mitotic rate, absence of nuclear hyperchromatism',
                'Circumscribed lesion boundaries (e.g. Fibroadenoma)',
              ],
              risk: 'Low clinical urgency / Non-metastatic',
            },
            {
              id: 'InSitu',
              name: 'In-situ carcinoma',
              tag: 'Pre-Invasive (DCIS)',
              color: 'border-amber-500/40 bg-amber-950/20 text-amber-300',
              summary: 'Ductal Carcinoma In Situ (DCIS). Neoplastic proliferation confined inside the duct lumen.',
              features: [
                'Basement membrane remains intact',
                'Atypical epithelial proliferation expanding and filling the ductal lumen',
                'Substantial risk of invasive transition if untreated',
              ],
              risk: 'Moderate-High urgency / Pre-invasive',
            },
            {
              id: 'Invasive',
              name: 'Invasive carcinoma',
              tag: 'Malignant (IDC/ILC)',
              color: 'border-rose-500/40 bg-rose-950/20 text-rose-300',
              summary: 'Invasive Ductal / Lobular Carcinoma. Neoplastic cells infiltrate past basement membrane.',
              features: [
                'Complete breach of ductal basement membrane into surrounding stroma',
                'Marked cellular pleomorphism, nuclear hyperchromasia, high mitotic count',
                'Desmoplastic stromal reaction and metastatic potential',
              ],
              risk: 'Critical urgency / Invasive Malignant',
            },
          ].map((cls) => (
            <div key={cls.id} className={`p-5 rounded-2xl border ${cls.color} flex flex-col justify-between space-y-4`}>
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono font-bold text-slate-400">Class {cls.id}</span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-950 border border-slate-800">
                    {cls.tag}
                  </span>
                </div>

                <h3 className="text-base font-bold text-white mt-2">{cls.name}</h3>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">{cls.summary}</p>

                <div className="mt-4 space-y-1.5">
                  <div className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">
                    Histological Criteria:
                  </div>
                  <ul className="space-y-1 text-xs text-slate-300">
                    {cls.features.map((f, idx) => (
                      <li key={idx} className="flex items-start gap-1.5">
                        <ChevronRight className="w-3.5 h-3.5 text-teal-400 shrink-0 mt-0.5" />
                        <span>{f}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800 text-[11px] font-mono">
                <span className="text-slate-400">Status: </span>
                <span className="font-semibold text-white">{cls.risk}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* TAB 5: PUBLICATION FIGURES & ASSETS */}
      {activeTab === 'figures' && (
        <div className="space-y-6">
          <div className="text-xs text-slate-400 font-mono">
            Click any figure to open the high-resolution inspection modal:
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {publicationFigures.map((fig, idx) => (
              <div 
                key={idx} 
                onClick={() => setSelectedImageModal(fig.src)}
                className="rounded-2xl bg-slate-900 border border-slate-800 overflow-hidden cursor-pointer group hover:border-teal-500/50 transition-all"
              >
                <div className="aspect-video bg-slate-950 flex items-center justify-center p-4 relative overflow-hidden">
                  <img
                    src={fig.src}
                    alt={fig.title}
                    className="max-h-full max-w-full object-contain group-hover:scale-105 transition-transform duration-300"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = `/assets/${fig.fallback}`;
                    }}
                  />
                  <div className="absolute bottom-3 right-3 p-1.5 rounded-lg bg-slate-950/80 text-white opacity-0 group-hover:opacity-100 transition-opacity">
                    <Eye className="w-4 h-4" />
                  </div>
                </div>

                <div className="p-4 space-y-1">
                  <h4 className="text-xs font-bold font-mono text-white">{fig.title}</h4>
                  <p className="text-[11px] text-slate-400 leading-relaxed">{fig.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Enlarged Image Modal */}
      {selectedImageModal && (
        <div
          onClick={() => setSelectedImageModal(null)}
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-xl flex items-center justify-center p-4 cursor-pointer"
        >
          <div 
            onClick={(e) => e.stopPropagation()}
            className="max-w-5xl max-h-[90vh] bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden p-4 relative"
          >
            <button
              onClick={() => setSelectedImageModal(null)}
              className="absolute top-4 right-4 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-white z-10"
            >
              Close (ESC)
            </button>
            <img
              src={selectedImageModal}
              alt="Enlarged publication figure"
              className="max-h-[80vh] max-w-full object-contain mx-auto rounded-lg"
            />
          </div>
        </div>
      )}

    </div>
  );
};
