import React, { useState } from 'react';
import { FileText, Copy, Check, ExternalLink, BookOpen, Users, Award, Building2 } from 'lucide-react';

export const CitationSection: React.FC = () => {
  const [copied, setCopied] = useState(false);

  const bibtex = `@article{bhowal2021fuzzy,
  title={Fuzzy ensemble of deep learning models using Choquet fuzzy integral, coalition game and information theory for breast cancer histology classification},
  author={Bhowal, Pratik and Sen, Subhankar and Silva, Juan D Velasquez and Sarkar, Ram},
  journal={Expert Systems with Applications},
  volume={190},
  pages={116167},
  year={2021},
  publisher={Elsevier},
  doi={10.1016/j.eswa.2021.116167}
}`;

  const handleCopyBibtex = () => {
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const authors = [
    { 
      name: 'Pratik Bhowal', 
      role: 'Lead Researcher & Developer', 
      affiliation: 'Jadavpur University, Kolkata',
      linkedin: 'https://www.linkedin.com/in/pratik-bhowal-1066aa198' 
    },
    { 
      name: 'Subhankar Sen', 
      role: 'Lead Researcher & Developer', 
      affiliation: 'Jadavpur University, Kolkata',
      linkedin: 'https://www.linkedin.com/in/subhankar-sen-a62457190' 
    },
    { 
      name: 'Prof. Juan D. Velásquez Silva', 
      role: 'Full Professor', 
      affiliation: 'Department of Industrial Engineering, University of Chile',
      linkedin: 'https://www.linkedin.com/in/jdvsilva/' 
    },
    { 
      name: 'Assoc. Prof. Ram Sarkar', 
      role: 'Associate Professor', 
      affiliation: 'Department of Computer Science & Engineering, Jadavpur University',
      linkedin: 'https://www.linkedin.com/in/ram-sarkar-0ba8a758' 
    },
  ];

  return (
    <div className="py-8 sm:py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-teal-400 font-semibold mb-1">
            <BookOpen className="w-4 h-4" />
            <span>PEER-REVIEWED SCIENTIFIC LITERATURE</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Academic Research & Citation
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 mt-1 max-w-2xl">
            If you utilize the fuzzy ensemble weights, layer harvesting methods, or evaluation pipeline in your academic publications, please cite the original study published in Elsevier Expert Systems with Applications.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: BibTeX Citation Block */}
        <div className="lg:col-span-7 rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-4 shadow-xl">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-teal-400 uppercase tracking-wider">
              <FileText className="w-4 h-4" />
              <span>BibTeX Reference Entry</span>
            </div>

            <button
              onClick={handleCopyBibtex}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-teal-600 hover:bg-teal-500 text-white transition-all shadow-sm active:scale-[0.98]"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-white" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied to Clipboard' : 'Copy BibTeX'}</span>
            </button>
          </div>

          <pre className="p-4 rounded-xl bg-slate-950 border border-slate-850 font-mono text-xs text-teal-300 overflow-x-auto leading-relaxed">
            {bibtex}
          </pre>

          <div className="pt-2 flex flex-wrap items-center gap-3 text-[11px] font-mono text-slate-400">
            <span>Journal: <strong>Expert Systems with Applications</strong></span>
            <span>&bull;</span>
            <span>Publisher: <strong>Elsevier</strong></span>
            <span>&bull;</span>
            <span>DOI: <strong>10.1016/j.eswa.2021.116167</strong></span>
          </div>
        </div>

        {/* Right Column: Research Authors & Affiliations */}
        <div className="lg:col-span-5 rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-4 shadow-xl">
          <div className="flex items-center gap-2 text-xs font-mono font-bold uppercase tracking-wider text-slate-400 pb-3 border-b border-slate-800">
            <Users className="w-4 h-4 text-teal-400" />
            <span>Research Authors & Affiliations</span>
          </div>

          <div className="space-y-3">
            {authors.map((author, idx) => (
              <div key={idx} className="p-3.5 rounded-xl bg-slate-950 border border-slate-850 flex items-center justify-between gap-3">
                <div>
                  <h4 className="text-xs font-bold text-white">{author.name}</h4>
                  <div className="text-[11px] text-teal-400 font-mono mt-0.5">{author.role}</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">{author.affiliation}</div>
                </div>
                <a
                  href={author.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-1.5 rounded-lg bg-slate-900 hover:bg-teal-600 text-slate-400 hover:text-white transition-colors shrink-0"
                  title="View LinkedIn Profile"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>
            ))}
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 flex items-start gap-2.5">
            <Award className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span className="leading-relaxed text-[11px]">
              Developed on the benchmark dataset from the <strong>ICIAR 2018 Grand Challenge on Breast Cancer Histology</strong>.
            </span>
          </div>
        </div>

      </div>

    </div>
  );
};
