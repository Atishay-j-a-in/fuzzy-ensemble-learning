import React from 'react';
import { Microscope, ShieldAlert, Heart } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="w-full bg-[#090d16] border-t border-slate-800 py-10 text-slate-400 text-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-6">
        
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400">
            <Microscope className="w-4 h-4" />
          </div>
          <div>
            <div className="font-bold text-slate-200">fuzzyBACH Histopathology AI</div>
            <div className="text-[11px] text-slate-500 font-mono">
              Fuzzy Ensemble of Deep Learning Models &bull; ICIAR 2018 Grand Challenge
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="max-w-lg text-center md:text-right text-[11px] text-slate-400 leading-relaxed">
          <p>
            <strong className="text-slate-300">Research & Educational Use Only:</strong> This computational system is designed to explore multi-backbone feature harvesting and Choquet fuzzy consensus in computational pathology. Not certified as a standalone clinical diagnostic device.
          </p>
        </div>

      </div>
    </footer>
  );
};
