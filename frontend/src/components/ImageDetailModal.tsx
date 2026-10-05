import React, { useState, useEffect } from 'react';
import { 
  X, 
  Microscope, 
  Sparkles, 
  Layers, 
  ShieldCheck, 
  ShieldAlert, 
  Info,
  Scale,
  Maximize2,
  ZoomIn,
  ZoomOut
} from 'lucide-react';
import { PredictionItem } from '../types';

interface ImageDetailModalProps {
  item: PredictionItem | null;
  onClose: () => void;
}

const CLASS_CONFIG: Record<string, { name: string; bg: string; text: string; border: string; dot: string; risk: string }> = {
  'Normal tissue': {
    name: 'Normal tissue',
    bg: 'bg-emerald-950/40',
    text: 'text-emerald-300',
    border: 'border-emerald-500/40',
    dot: 'bg-emerald-400',
    risk: 'Physiological baseline / No atypia detected',
  },
  'Benign': {
    name: 'Benign lesion',
    bg: 'bg-sky-950/40',
    text: 'text-sky-300',
    border: 'border-sky-500/40',
    dot: 'bg-sky-400',
    risk: 'Non-malignant proliferation / Low clinical risk',
  },
  'In-situ carcinoma': {
    name: 'In-situ carcinoma (DCIS)',
    bg: 'bg-amber-950/40',
    text: 'text-amber-300',
    border: 'border-amber-500/40',
    dot: 'bg-amber-400',
    risk: 'Pre-invasive neoplastic growth confined inside ductal lumen',
  },
  'Invasive carcinoma': {
    name: 'Invasive carcinoma (IDC/ILC)',
    bg: 'bg-rose-950/40',
    text: 'text-rose-300',
    border: 'border-rose-500/40',
    dot: 'bg-rose-400',
    risk: 'Malignant stromal infiltration / High clinical urgency',
  },
};

const SHAPLEY_WEIGHTS: Record<string, { weight: number; singleAcc: string }> = {
  'Xception': { weight: 0.23, singleAcc: '91.0%' },
  'InceptionResNetV2': { weight: 0.22, singleAcc: '91.0%' },
  'InceptionV3': { weight: 0.20, singleAcc: '90.0%' },
  'VGG19': { weight: 0.18, singleAcc: '83.0%' },
  'VGG16': { weight: 0.17, singleAcc: '86.0%' },
};

export const ImageDetailModal: React.FC<ImageDetailModalProps> = ({ item, onClose }) => {
  const [isZoomed, setIsZoomed] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!item) return null;

  const cfg = CLASS_CONFIG[item.predicted_class] || CLASS_CONFIG['Benign'];

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto"
      role="dialog"
      aria-modal="true"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-3xl rounded-2xl bg-slate-950 border border-slate-800 shadow-2xl overflow-hidden my-auto animate-in fade-in zoom-in-95 duration-150"
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between p-4 sm:p-5 border-b border-slate-800 bg-slate-900/95">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-teal-500/15 border border-teal-500/30 flex items-center justify-center text-teal-400">
              <Microscope className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm sm:text-base font-bold text-white font-mono">{item.image}</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  ID: {item.predicted_class_id}
                </span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                Latency: {item.latency_ms || 0}ms &bull; {item.timestamp}
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 sm:p-6 space-y-6 max-h-[80vh] overflow-y-auto">
          
          {/* Top Row: Biopsy Image View + Consensus Diagnosis Card */}
          <div className="grid grid-cols-1 sm:grid-cols-12 gap-5 items-start">
            
            {/* Histological Patch View */}
            <div className="sm:col-span-5 relative group">
              <div 
                onClick={() => setIsZoomed(!isZoomed)}
                className={`rounded-xl bg-slate-900 overflow-hidden border border-slate-800 cursor-zoom-in relative ${
                  isZoomed ? 'aspect-square sm:aspect-auto sm:h-64' : 'aspect-square'
                }`}
              >
                <img
                  src={item.thumbnail}
                  alt={item.image}
                  className={`w-full h-full object-cover transition-transform duration-300 ${
                    isZoomed ? 'scale-150' : 'group-hover:scale-105'
                  }`}
                />
                <div className="absolute bottom-2 right-2 px-2 py-1 rounded bg-slate-950/80 text-[10px] font-mono text-slate-300 flex items-center gap-1 backdrop-blur-sm">
                  {isZoomed ? <ZoomOut className="w-3 h-3" /> : <ZoomIn className="w-3 h-3" />}
                  <span>{isZoomed ? 'Reset Zoom' : 'Click to Zoom'}</span>
                </div>
              </div>
              <div className="text-[11px] text-center font-mono text-slate-500 mt-1.5">
                512 × 512 &bull; 200× Optical Magnification
              </div>
            </div>

            {/* Final Ensemble Decision Card */}
            <div className="sm:col-span-7 space-y-3">
              <div className={`p-4 rounded-xl border ${cfg.bg} ${cfg.border} space-y-2`}>
                <div className="flex items-center justify-between text-xs font-mono font-semibold">
                  <span className="flex items-center gap-1.5 text-slate-300">
                    <Sparkles className="w-3.5 h-3.5 text-teal-400" />
                    <span>CHOQUET FUZZY CONSENSUS</span>
                  </span>
                  <span className={`px-2 py-0.5 rounded-full text-xs font-bold font-mono ${cfg.text} bg-slate-950/60 border ${cfg.border}`}>
                    {item.confidence_percentage}% Certainty
                  </span>
                </div>

                <div className="text-xl font-extrabold text-white">
                  {item.predicted_class}
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">
                  {item.description || cfg.risk}
                </p>

                <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono">
                  <span className="text-slate-400">Consensus Agreement:</span>
                  <span className="font-bold text-teal-300">{item.consensus || '5/5 Models Agree'}</span>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 leading-relaxed">
                <span className="font-semibold text-slate-200 block mb-0.5">Clinical Pathology Marker:</span>
                {cfg.risk}
              </div>
            </div>

          </div>

          {/* 5-Model Individual Votes Matrix */}
          {item.individual_votes && (
            <div className="rounded-xl bg-slate-900 border border-slate-800 p-4 space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <h4 className="text-xs font-bold font-mono text-slate-300 uppercase tracking-wider flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5 text-teal-400" />
                  <span>5 Backbone Model Voting & Shapley Weights</span>
                </h4>
                <span className="text-[11px] font-mono text-slate-400">Game-Theoretic Coalition</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                {Object.entries(item.individual_votes).map(([mName, v]) => {
                  const mCfg = CLASS_CONFIG[v.predicted_class] || CLASS_CONFIG['Benign'];
                  const meta = SHAPLEY_WEIGHTS[mName] || { weight: 0.20, singleAcc: '90%' };
                  const isWinnerClass = v.predicted_class === item.predicted_class;
                  return (
                    <div
                      key={mName}
                      className="p-3 rounded-lg bg-slate-950 border border-slate-850 flex items-center justify-between gap-2"
                    >
                      <div>
                        <div className="font-mono font-bold text-slate-200 text-xs flex items-center gap-1.5">
                          <span>{mName}</span>
                          <span className="text-[10px] text-slate-500 font-normal">(&mu;={meta.weight})</span>
                        </div>
                        <div className="flex items-center gap-1.5 mt-1">
                          <span className={`w-1.5 h-1.5 rounded-full ${mCfg.dot}`} />
                          <span className={`font-semibold ${mCfg.text}`}>
                            {v.predicted_class}
                          </span>
                        </div>
                      </div>

                      <div className="text-right shrink-0">
                        <div className="font-mono font-bold text-slate-200">
                          {(v.confidence * 100).toFixed(1)}%
                        </div>
                        <span className="text-[10px] font-mono text-slate-500">
                          Single: {meta.singleAcc}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* 4-Class Calibrated Softmax Probability Spectrum */}
          <div className="rounded-xl bg-slate-900 border border-slate-800 p-4 space-y-3">
            <h4 className="text-xs font-bold font-mono text-slate-300 uppercase tracking-wider">
              Ensemble 4-Class Probability Spectrum
            </h4>

            <div className="space-y-2.5">
              {[
                { 
                  name: 'Normal tissue', 
                  prob: item.probability_Normal_tissue || item.probabilities['Normal tissue'] || 0, 
                  color: 'bg-emerald-500',
                  textColor: 'text-emerald-400' 
                },
                { 
                  name: 'Benign', 
                  prob: item.probability_Benign || item.probabilities['Benign'] || 0, 
                  color: 'bg-sky-500',
                  textColor: 'text-sky-400' 
                },
                { 
                  name: 'In-situ carcinoma', 
                  prob: item.probability_In_situ_carcinoma || item.probabilities['In-situ carcinoma'] || 0, 
                  color: 'bg-amber-500',
                  textColor: 'text-amber-400' 
                },
                { 
                  name: 'Invasive carcinoma', 
                  prob: item.probability_Invasive_carcinoma || item.probabilities['Invasive carcinoma'] || 0, 
                  color: 'bg-rose-500',
                  textColor: 'text-rose-400' 
                },
              ].map((cls) => {
                const pct = (cls.prob * 100).toFixed(2);
                const isSelected = cls.name === item.predicted_class;
                return (
                  <div key={cls.name} className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className={isSelected ? 'text-white font-bold' : 'text-slate-400'}>
                        {cls.name} {isSelected ? '(Winner)' : ''}
                      </span>
                      <span className={isSelected ? `${cls.textColor} font-bold` : 'text-slate-300'}>
                        {pct}%
                      </span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${cls.color}`}
                        style={{ width: `${Math.max(1, Number(pct))}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/95 flex items-center justify-between">
          <span className="text-xs text-slate-400 font-mono">
            ICIAR 2018 BACH Challenge &bull; 512&times;512 RGB input (heads trained on Macenko-normalized data)
          </span>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-white transition-colors"
          >
            Close Dialog (ESC)
          </button>
        </div>

      </div>
    </div>
  );
};
