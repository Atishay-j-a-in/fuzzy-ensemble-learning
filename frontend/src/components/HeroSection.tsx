import React from 'react';
import { Layers, Sparkles, ArrowRight, Database, ShieldCheck, Microscope, Award } from 'lucide-react';

interface HeroSectionProps {
  onExploreDiagrams: () => void;
  onLaunchStudio: () => void;
  onLoadDemoDataset: () => void;
  isLoadingDemo: boolean;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  onExploreDiagrams,
  onLaunchStudio,
  onLoadDemoDataset,
  isLoadingDemo,
}) => {
  return (
    <div className="relative overflow-hidden pt-8 pb-16 md:pt-12 md:pb-24 border-b border-slate-800/60 bg-gradient-to-b from-slate-950 via-slate-900/90 to-slate-950">
      
      {/* Background ambient lighting effects */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-600/15 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-purple-600/15 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-teal-600/10 rounded-full blur-3xl pointer-events-none -z-10" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Top Research Pill */}
        <div className="flex justify-center">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 text-xs font-medium shadow-inner">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>ICIAR 2018 Grand Challenge (BACH) &bull; Expert Systems with Applications</span>
          </div>
        </div>

        {/* Main Headline */}
        <div className="text-center mt-6 max-w-4xl mx-auto">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white leading-tight">
            Fuzzy Ensemble & <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-300 to-teal-300">Xception Deep AI</span> for Breast Cancer Histology
          </h1>
          <p className="mt-6 text-base sm:text-lg text-slate-300 leading-relaxed font-normal">
            Automated 4-class histopathology diagnostic platform powered by deep multi-scale feature harvesting, Macenko stain normalization, and fuzzy game-theoretic decision fusion.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <button
            onClick={onLaunchStudio}
            className="flex items-center gap-2.5 px-6 py-3.5 rounded-xl font-semibold text-sm bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-xl shadow-indigo-600/30 hover:shadow-indigo-600/50 hover:scale-[1.02] active:scale-[0.98] transition-all"
          >
            <Database className="w-4 h-4" />
            <span>Upload Biopsy Dataset / ZIP</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={onLoadDemoDataset}
            disabled={isLoadingDemo}
            className="flex items-center gap-2.5 px-6 py-3.5 rounded-xl font-semibold text-sm bg-slate-800/80 hover:bg-slate-700/80 text-slate-100 border border-slate-700/80 hover:border-slate-600 shadow-lg shadow-black/20 hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50"
          >
            <Microscope className="w-4 h-4 text-teal-400" />
            <span>{isLoadingDemo ? 'Analyzing Demo Samples...' : 'Run Demo BACH Test Batch (10 Images)'}</span>
          </button>

          <button
            onClick={onExploreDiagrams}
            className="flex items-center gap-2.5 px-6 py-3.5 rounded-xl font-semibold text-sm bg-slate-900/60 hover:bg-slate-800/60 text-indigo-300 border border-indigo-500/30 hover:border-indigo-500/50 transition-all"
          >
            <Layers className="w-4 h-4 text-indigo-400" />
            <span>Explore Architecture Diagrams</span>
          </button>
        </div>

        {/* Highlight Metrics Cards */}
        <div className="mt-16 grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
          
          <div className="glass-card rounded-2xl p-5 border border-slate-800/80 hover:border-indigo-500/40 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Deployed Model</span>
              <Microscope className="w-5 h-5 text-indigo-400" />
            </div>
            <div className="mt-3 text-2xl font-bold text-white">Xception Deep Net</div>
            <p className="mt-1 text-xs text-slate-400">Multi-scale concatenated pooling (2520-dim)</p>
          </div>

          <div className="glass-card rounded-2xl p-5 border border-slate-800/80 hover:border-teal-500/40 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">4-Class Accuracy</span>
              <Award className="w-5 h-5 text-teal-400" />
            </div>
            <div className="mt-3 text-2xl font-bold text-teal-300">91.0% (95% Ens)</div>
            <p className="mt-1 text-xs text-slate-400">Tested on ICIAR 2018 BACH Challenge</p>
          </div>

          <div className="glass-card rounded-2xl p-5 border border-slate-800/80 hover:border-purple-500/40 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Diagnostic Scope</span>
              <ShieldCheck className="w-5 h-5 text-purple-400" />
            </div>
            <div className="mt-3 text-2xl font-bold text-white">4 Pathology Classes</div>
            <p className="mt-1 text-xs text-slate-400">Normal, Benign, In-Situ, & Invasive</p>
          </div>

          <div className="glass-card rounded-2xl p-5 border border-slate-800/80 hover:border-pink-500/40 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Stain Calibration</span>
              <Sparkles className="w-5 h-5 text-pink-400" />
            </div>
            <div className="mt-3 text-2xl font-bold text-white">Macenko Normalization</div>
            <p className="mt-1 text-xs text-slate-400">H&E color deconvolution at 512x512</p>
          </div>

        </div>

      </div>
    </div>
  );
};
