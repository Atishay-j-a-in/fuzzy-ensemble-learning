import React, { useState } from 'react';
import { Activity, Trophy, Zap, Scale, CheckCircle2, ArrowRight } from 'lucide-react';

export const BenchmarksSection: React.FC = () => {
  const [taskMode, setTaskMode] = useState<'four_class' | 'two_class'>('four_class');

  const fourClassData = [
    { 
      model: 'VGG16 Backbone', 
      features: '1,152-dim Vector', 
      valAcc: 97.0, 
      testAcc: 86.0, 
      rank: 'Rank #5', 
      isEnsemble: false,
      color: 'bg-slate-700' 
    },
    { 
      model: 'VGG19 Backbone', 
      features: '896-dim Vector', 
      valAcc: 98.0, 
      testAcc: 83.0, 
      rank: 'Rank #6', 
      isEnsemble: false,
      color: 'bg-slate-700' 
    },
    { 
      model: 'InceptionV3 Backbone', 
      features: '2,320-dim Vector', 
      valAcc: 99.0, 
      testAcc: 90.0, 
      rank: 'Rank #4', 
      isEnsemble: false,
      color: 'bg-indigo-600' 
    },
    { 
      model: 'InceptionResNetV2 Backbone', 
      features: '464-dim Vector', 
      valAcc: 99.0, 
      testAcc: 91.0, 
      rank: 'Rank #2', 
      isEnsemble: false,
      color: 'bg-indigo-600' 
    },
    { 
      model: 'Xception Backbone (Deployed Solo)', 
      features: '2,520-dim Vector', 
      valAcc: 99.0, 
      testAcc: 91.0, 
      rank: 'Rank #2 (Top Single)', 
      isEnsemble: false, 
      highlight: true,
      color: 'bg-indigo-500' 
    },
    { 
      model: 'Fuzzy Choquet Ensemble (Proposed)', 
      features: 'Decision-Level Coalition', 
      valAcc: null, 
      testAcc: 95.0, 
      rank: 'Rank #1 (State of the Art)', 
      isEnsemble: true,
      color: 'bg-teal-500' 
    },
  ];

  const twoClassData = [
    { 
      model: 'VGG16 Backbone', 
      features: '1,152-dim Vector', 
      valAcc: 100.0, 
      testAcc: 89.0, 
      rank: 'Rank #5', 
      isEnsemble: false,
      color: 'bg-slate-700' 
    },
    { 
      model: 'VGG19 Backbone', 
      features: '896-dim Vector', 
      valAcc: 99.8, 
      testAcc: 94.0, 
      rank: 'Rank #3', 
      isEnsemble: false,
      color: 'bg-slate-700' 
    },
    { 
      model: 'InceptionResNetV2 Backbone', 
      features: '464-dim Vector', 
      valAcc: 99.7, 
      testAcc: 93.0, 
      rank: 'Rank #4', 
      isEnsemble: false,
      color: 'bg-indigo-600' 
    },
    { 
      model: 'InceptionV3 Backbone', 
      features: '2,320-dim Vector', 
      valAcc: 100.0, 
      testAcc: 94.0, 
      rank: 'Rank #3', 
      isEnsemble: false,
      color: 'bg-indigo-600' 
    },
    { 
      model: 'Xception Backbone (Deployed Solo)', 
      features: '2,520-dim Vector', 
      valAcc: 100.0, 
      testAcc: 95.0, 
      rank: 'Rank #2 (Top Single)', 
      isEnsemble: false, 
      highlight: true,
      color: 'bg-indigo-500' 
    },
    { 
      model: 'Fuzzy Choquet Ensemble (Proposed)', 
      features: 'Decision-Level Coalition', 
      valAcc: null, 
      testAcc: 96.0, 
      rank: 'Rank #1 (State of the Art)', 
      isEnsemble: true,
      color: 'bg-teal-500' 
    },
  ];

  const currentData = taskMode === 'four_class' ? fourClassData : twoClassData;

  return (
    <div className="py-8 sm:py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-teal-400 font-semibold mb-1">
            <Trophy className="w-4 h-4" />
            <span>EMPIRICAL VALIDATION & CHALLENGE BENCHMARKS</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Classification Performance Evaluation
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 mt-1 max-w-2xl">
            Comparison on the ICIAR 2018 Grand Challenge on Breast Cancer Histology (BACH) test dataset across individual convolutional backbones and the proposed Choquet Fuzzy Ensemble.
          </p>
        </div>

        {/* Task Mode Switcher */}
        <div className="p-1 rounded-xl bg-slate-900 border border-slate-800 shrink-0 self-start sm:self-center">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setTaskMode('four_class')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                taskMode === 'four_class'
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              4-Class Multi-Category (Deployed)
            </button>
            <button
              onClick={() => setTaskMode('two_class')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                taskMode === 'two_class'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              2-Class Binary (Malignant vs Non-Malignant)
            </button>
          </div>
        </div>
      </div>

      {/* Main Benchmark Table Card */}
      <div className="rounded-2xl overflow-hidden border border-slate-800 bg-slate-900 shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-950 text-[11px] font-mono font-bold uppercase tracking-wider text-slate-400">
                <th className="py-3.5 px-5">Model Architecture</th>
                <th className="py-3.5 px-5">Feature Harvest Vector</th>
                <th className="py-3.5 px-5 text-center">Validation Accuracy</th>
                <th className="py-3.5 px-5 text-center">Test Accuracy</th>
                <th className="py-3.5 px-5">Visual Test Benchmark</th>
                <th className="py-3.5 px-5 text-right">Standing</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {currentData.map((row) => (
                <tr
                  key={row.model}
                  className={`transition-colors ${
                    row.isEnsemble
                      ? 'bg-teal-950/20 text-teal-100 font-bold'
                      : row.highlight
                      ? 'bg-indigo-950/20 text-indigo-100 font-medium'
                      : 'hover:bg-slate-850/60 text-slate-300'
                  }`}
                >
                  {/* Model Name */}
                  <td className="py-3.5 px-5 font-medium flex items-center gap-2.5">
                    {row.isEnsemble ? (
                      <Trophy className="w-4 h-4 text-teal-400 shrink-0" />
                    ) : row.highlight ? (
                      <Zap className="w-4 h-4 text-indigo-400 shrink-0" />
                    ) : (
                      <div className="w-2 h-2 rounded-full bg-slate-600 shrink-0" />
                    )}
                    <span className="font-mono text-xs">{row.model}</span>
                  </td>

                  {/* Feature Vector */}
                  <td className="py-3.5 px-5 font-mono text-[11px] text-slate-400">
                    {row.features}
                  </td>

                  {/* Validation Accuracy */}
                  <td className="py-3.5 px-5 text-center font-mono font-medium">
                    {row.valAcc !== null ? `${row.valAcc.toFixed(1)}%` : '—'}
                  </td>

                  {/* Test Accuracy */}
                  <td className="py-3.5 px-5 text-center font-mono font-bold text-sm text-white">
                    {row.testAcc.toFixed(1)}%
                  </td>

                  {/* Visual Test Benchmark Meter */}
                  <td className="py-3.5 px-5 min-w-[160px]">
                    <div className="space-y-1">
                      <div className="h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${row.color}`}
                          style={{ width: `${(row.testAcc / 100) * 100}%` }}
                        />
                      </div>
                    </div>
                  </td>

                  {/* Rank Badge */}
                  <td className="py-3.5 px-5 text-right font-mono">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                        row.isEnsemble
                          ? 'bg-teal-500/20 text-teal-300 border border-teal-500/40'
                          : row.highlight
                          ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                          : 'bg-slate-800 text-slate-400 border border-slate-700'
                      }`}
                    >
                      {row.rank}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Analysis Callout Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="font-mono font-bold text-teal-400 uppercase tracking-wider flex items-center gap-1.5">
            <Scale className="w-4 h-4" />
            <span>Why Choquet Fuzzy Fusion Achieves 95.0% Accuracy</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            In standard voting or averaging ensembles, independent classifier errors on ambiguous tissue boundaries (such as differentiating In-situ DCIS from early Invasive IDC) dilute diagnostic confidence. Choquet fuzzy integration models the non-linear interaction and mutual redundancy of classifier coalitions, assigning high game-theoretic weight to complementary features and achieving state-of-the-art consensus.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="font-mono font-bold text-indigo-400 uppercase tracking-wider flex items-center gap-1.5">
            <Zap className="w-4 h-4" />
            <span>Why Intermediate Layer Harvesting Beats Bottlenecks</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            Standard transfer learning flattens only the terminal convolutional layer, discarding spatial details like nuclear chromatin borders and duct basement membrane continuity. By extracting and concatenating multi-depth representations across Early (block 4), Middle (block 5), and Exit (block 14) flows, the model retains both sub-cellular texture and macro-architectural stromal topology.
          </p>
        </div>
      </div>

    </div>
  );
};
