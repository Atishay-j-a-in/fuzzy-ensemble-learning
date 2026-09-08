import React from 'react';
import { 
  Activity, 
  Layers, 
  BarChart3, 
  Database, 
  FileText, 
  CheckCircle2, 
  AlertCircle,
  Microscope,
  Sparkles,
  Zap
} from 'lucide-react';

interface NavbarProps {
  backendHealthy: boolean;
  activeTab: 'studio' | 'results' | 'architecture' | 'benchmarks' | 'citation';
  setActiveTab: (tab: 'studio' | 'results' | 'architecture' | 'benchmarks' | 'citation') => void;
  resultCount: number;
  inferenceMode: 'ensemble' | 'xception';
  setInferenceMode: (mode: 'ensemble' | 'xception') => void;
}

interface NavTabItem {
  id: 'studio' | 'results' | 'architecture' | 'benchmarks' | 'citation';
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  count?: number | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  backendHealthy,
  activeTab,
  setActiveTab,
  resultCount,
  inferenceMode,
  setInferenceMode,
}) => {
  const navTabs: NavTabItem[] = [
    { id: 'studio', label: 'Inference Studio', icon: Database },
    { 
      id: 'results', 
      label: 'Diagnostics & CSV', 
      icon: BarChart3, 
      count: resultCount > 0 ? resultCount : null 
    },
    { id: 'architecture', label: 'Pipeline Architecture', icon: Layers },
    { id: 'benchmarks', label: 'Model Benchmarks', icon: Activity },
    { id: 'citation', label: 'Publication', icon: FileText },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-[#090d16]/95 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        
        {/* Brand Lockup */}
        <div 
          onClick={() => setActiveTab('studio')} 
          className="flex items-center gap-3 cursor-pointer group select-none shrink-0"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && setActiveTab('studio')}
        >
          <div className="w-9 h-9 rounded-lg bg-teal-500/15 border border-teal-500/30 flex items-center justify-center text-teal-400 group-hover:border-teal-400 transition-colors">
            <Microscope className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-base sm:text-lg tracking-tight text-white">
                fuzzy<span className="text-teal-400">BACH</span>
              </span>
              <span className="text-[10px] font-bold tracking-wider px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono hidden sm:inline">
                95.0% Acc
              </span>
            </div>
            <div className="text-[11px] text-slate-400 font-medium hidden md:block">
              Histopathological Breast Cancer AI
            </div>
          </div>
        </div>

        {/* Primary Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800" aria-label="Main Navigation">
          {navTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  isActive
                    ? 'bg-teal-600 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                }`}
                aria-current={isActive ? 'page' : undefined}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
                {tab.count !== null && (
                  <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono font-bold ${
                    isActive ? 'bg-white text-teal-900' : 'bg-teal-500/20 text-teal-300'
                  }`}>
                    {tab.count}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Right Section: Inference Mode Selector & Server Health */}
        <div className="flex items-center gap-3">
          
          {/* Active Model Switcher */}
          <div className="flex items-center p-0.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-medium">
            <button
              onClick={() => setInferenceMode('ensemble')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                inferenceMode === 'ensemble'
                  ? 'bg-teal-500/20 text-teal-300 border border-teal-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Choquet Fuzzy Integral with 5 Pre-trained Backbones (95% Accuracy)"
            >
              <Sparkles className="w-3 h-3 text-teal-400" />
              <span>5-Model Ensemble</span>
            </button>
            <button
              onClick={() => setInferenceMode('xception')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                inferenceMode === 'xception'
                  ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Single Xception Intermediate Feature Harvesting (91% Accuracy)"
            >
              <Zap className="w-3 h-3 text-indigo-400" />
              <span>Xception</span>
            </button>
          </div>

          {/* Health Status Indicator */}
          <div 
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-mono font-medium border ${
              backendHealthy 
                ? 'bg-emerald-950/40 text-emerald-300 border-emerald-500/30' 
                : 'bg-rose-950/40 text-rose-300 border-rose-500/30'
            }`}
            title={backendHealthy ? 'FastAPI Backend Online: 5 Models & Macenko Normalizer Active' : 'Connecting to FastAPI backend...'}
          >
            <span className={`w-2 h-2 rounded-full ${backendHealthy ? 'bg-emerald-400' : 'bg-rose-400 animate-pulse'}`} />
            <span className="hidden sm:inline">{backendHealthy ? '5 Models Active' : 'Offline'}</span>
          </div>

        </div>

      </div>

      {/* Mobile Tab Strip */}
      <div className="flex md:hidden overflow-x-auto px-4 py-2 bg-slate-900/90 border-t border-slate-800 gap-1 scrollbar-none">
        {navTabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap shrink-0 ${
                isActive
                  ? 'bg-teal-600 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
              {tab.count !== null && (
                <span className="px-1.5 py-0.2 rounded-full text-[10px] font-mono bg-teal-500/30 text-teal-200">
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </header>
  );
};
