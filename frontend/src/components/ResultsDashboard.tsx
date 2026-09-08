import React, { useState, useMemo } from 'react';
import { 
  FileSpreadsheet, 
  FileCode, 
  Search, 
  Filter, 
  Eye, 
  Sparkles,
  ArrowUpDown,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Activity,
  Microscope,
  ShieldCheck,
  ShieldAlert,
  Clock
} from 'lucide-react';
import { BatchPredictionResponse, PredictionItem, DiagnosticClass } from '../types';

interface ResultsDashboardProps {
  data: BatchPredictionResponse;
  onSelectItem: (item: PredictionItem) => void;
}

const CLASS_CONFIG: Record<string, { name: string; bg: string; text: string; border: string; dot: string }> = {
  'Normal tissue': {
    name: 'Normal tissue',
    bg: 'bg-emerald-950/40',
    text: 'text-emerald-300',
    border: 'border-emerald-500/30',
    dot: 'bg-emerald-400',
  },
  'Benign': {
    name: 'Benign lesion',
    bg: 'bg-sky-950/40',
    text: 'text-sky-300',
    border: 'border-sky-500/30',
    dot: 'bg-sky-400',
  },
  'In-situ carcinoma': {
    name: 'In-situ carcinoma',
    bg: 'bg-amber-950/40',
    text: 'text-amber-300',
    border: 'border-amber-500/30',
    dot: 'bg-amber-400',
  },
  'Invasive carcinoma': {
    name: 'Invasive carcinoma',
    bg: 'bg-rose-950/40',
    text: 'text-rose-300',
    border: 'border-rose-500/30',
    dot: 'bg-rose-400',
  },
};

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  data,
  onSelectItem,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [classFilter, setClassFilter] = useState<string>('ALL');
  const [consensusFilter, setConsensusFilter] = useState<string>('ALL');
  const [sortBy, setSortBy] = useState<'confidence' | 'name' | 'class'>('confidence');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [isExportingCsv, setIsExportingCsv] = useState(false);

  const { items, summary } = data;

  const filteredItems = useMemo(() => {
    return items
      .filter((item) => {
        const matchesSearch = item.image.toLowerCase().includes(searchTerm.toLowerCase());
        
        let matchesClass = true;
        if (classFilter === 'MALIGNANT') {
          matchesClass = item.predicted_class === 'In-situ carcinoma' || item.predicted_class === 'Invasive carcinoma';
        } else if (classFilter === 'NON_MALIGNANT') {
          matchesClass = item.predicted_class === 'Normal tissue' || item.predicted_class === 'Benign';
        } else if (classFilter !== 'ALL') {
          matchesClass = item.predicted_class === classFilter;
        }

        let matchesConsensus = true;
        if (consensusFilter === 'UNANIMOUS') {
          matchesConsensus = (item.agreeing_models_count ?? 5) === 5;
        } else if (consensusFilter === 'MAJORITY') {
          matchesConsensus = (item.agreeing_models_count ?? 5) >= 3 && (item.agreeing_models_count ?? 5) < 5;
        }

        return matchesSearch && matchesClass && matchesConsensus;
      })
      .sort((a, b) => {
        let cmp = 0;
        if (sortBy === 'name') {
          cmp = a.image.localeCompare(b.image, undefined, { numeric: true });
        } else if (sortBy === 'confidence') {
          cmp = a.confidence - b.confidence;
        } else if (sortBy === 'class') {
          cmp = a.predicted_class.localeCompare(b.predicted_class);
        }
        return sortOrder === 'asc' ? cmp : -cmp;
      });
  }, [items, searchTerm, classFilter, consensusFilter, sortBy, sortOrder]);

  const handleDownloadCsv = async () => {
    setIsExportingCsv(true);
    try {
      const response = await fetch('/api/export/csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items }),
      });

      if (!response.ok) {
        throw new Error('CSV generation endpoint failed');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `fuzzyBACH_Ensemble_Classification_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.warn('CSV export endpoint failed, executing client-side generation fallback:', err);
      generateClientCsv(items);
    } finally {
      setIsExportingCsv(false);
    }
  };

  const generateClientCsv = (dataItems: PredictionItem[]) => {
    const headers = [
      'image_filename',
      'fuzzy_ensemble_diagnosis',
      'predicted_class_id',
      'ensemble_confidence_score',
      'consensus_agreement',
      'probability_Normal_tissue',
      'probability_Benign',
      'probability_In_situ_carcinoma',
      'probability_Invasive_carcinoma',
      'latency_ms',
      'timestamp',
    ];

    const rows = dataItems.map((it) => [
      `"${it.image}"`,
      `"${it.predicted_class}"`,
      it.predicted_class_id,
      it.confidence.toFixed(6),
      `"${it.consensus || '5/5 Agree'}"`,
      (it.probability_Normal_tissue || it.probabilities['Normal tissue'] || 0).toFixed(6),
      (it.probability_Benign || it.probabilities['Benign'] || 0).toFixed(6),
      (it.probability_In_situ_carcinoma || it.probabilities['In-situ carcinoma'] || 0).toFixed(6),
      (it.probability_Invasive_carcinoma || it.probabilities['Invasive carcinoma'] || 0).toFixed(6),
      it.latency_ms || 0,
      `"${it.timestamp}"`,
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `fuzzyBACH_Classification_Export_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const handleDownloadJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `fuzzyBACH_Classification_Dataset_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const malignantCount = (summary.class_counts['In-situ carcinoma'] || 0) + (summary.class_counts['Invasive carcinoma'] || 0);
  const malignantPct = summary.total > 0 ? ((malignantCount / summary.total) * 100).toFixed(1) : '0';

  return (
    <div className="py-8 sm:py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
      
      {/* Top Header & Export Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-teal-400 font-semibold mb-1">
            <Sparkles className="w-3.5 h-3.5" />
            <span>FUZZY CHOQUET INTEGRAL DECISION REPORT</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Diagnostic Classification Results
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Evaluated {summary.total} biopsy image{summary.total > 1 ? 's' : ''} with 5 deep neural backbones & non-additive coalition fusion.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2.5 shrink-0">
          <button
            onClick={handleDownloadCsv}
            disabled={isExportingCsv}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold bg-teal-600 hover:bg-teal-500 text-white shadow-sm transition-all active:scale-[0.98] disabled:opacity-50"
          >
            <FileSpreadsheet className="w-4 h-4" />
            <span>{isExportingCsv ? 'Exporting...' : 'Export Results CSV'}</span>
          </button>

          <button
            onClick={handleDownloadJson}
            className="flex items-center gap-2 px-3 py-2.5 rounded-xl text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 transition-colors"
            title="Download complete predictions as JSON"
          >
            <FileCode className="w-4 h-4 text-indigo-400" />
            <span>JSON</span>
          </button>
        </div>
      </div>

      {/* Primary KPI Grid (6 Metric Tiles) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        
        {/* Total Samples */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-[11px] font-mono font-semibold text-slate-400">
            <span>TOTAL SAMPLES</span>
            <Microscope className="w-3.5 h-3.5 text-slate-500" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-white">
            {summary.total}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Biopsies processed</div>
        </div>

        {/* Normal Tissue */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-[11px] font-mono font-semibold text-emerald-400">
            <span>NORMAL</span>
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-white">
            {summary.class_counts['Normal tissue'] || 0}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            {summary.total > 0 ? (((summary.class_counts['Normal tissue'] || 0) / summary.total) * 100).toFixed(0) : 0}% of cohort
          </div>
        </div>

        {/* Benign */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-[11px] font-mono font-semibold text-sky-400">
            <span>BENIGN</span>
            <span className="w-2 h-2 rounded-full bg-sky-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-white">
            {summary.class_counts['Benign'] || 0}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            {summary.total > 0 ? (((summary.class_counts['Benign'] || 0) / summary.total) * 100).toFixed(0) : 0}% of cohort
          </div>
        </div>

        {/* In-situ */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-[11px] font-mono font-semibold text-amber-400">
            <span>IN-SITU (DCIS)</span>
            <span className="w-2 h-2 rounded-full bg-amber-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-white">
            {summary.class_counts['In-situ carcinoma'] || 0}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            {summary.total > 0 ? (((summary.class_counts['In-situ carcinoma'] || 0) / summary.total) * 100).toFixed(0) : 0}% of cohort
          </div>
        </div>

        {/* Invasive */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-[11px] font-mono font-semibold text-rose-400">
            <span>INVASIVE (IDC)</span>
            <span className="w-2 h-2 rounded-full bg-rose-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-white">
            {summary.class_counts['Invasive carcinoma'] || 0}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            {summary.total > 0 ? (((summary.class_counts['Invasive carcinoma'] || 0) / summary.total) * 100).toFixed(0) : 0}% of cohort
          </div>
        </div>

        {/* Malignancy Index */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-[11px] font-mono font-semibold text-indigo-400">
            <span>MALIGNANCY %</span>
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-white">
            {malignantPct}%
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">In-situ + Invasive</div>
        </div>

      </div>

      {/* Filter, Search & Sort Control Strip */}
      <div className="flex flex-col lg:flex-row items-center justify-between gap-3 p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs">
        
        {/* Search Input */}
        <div className="relative w-full lg:w-72">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search biopsy sample ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-slate-200 placeholder-slate-500 font-mono focus:outline-none focus:border-teal-500"
          />
        </div>

        {/* Filter Dropdowns & Sorters */}
        <div className="flex flex-wrap items-center gap-2.5 w-full lg:w-auto justify-between lg:justify-end">
          
          {/* Class Filter */}
          <div className="flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value)}
              className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-slate-200 font-semibold focus:outline-none"
            >
              <option value="ALL">All Diagnoses ({items.length})</option>
              <option value="Normal tissue">Normal Tissue ({summary.class_counts['Normal tissue'] || 0})</option>
              <option value="Benign">Benign Lesion ({summary.class_counts['Benign'] || 0})</option>
              <option value="In-situ carcinoma">In-situ Carcinoma ({summary.class_counts['In-situ carcinoma'] || 0})</option>
              <option value="Invasive carcinoma">Invasive Carcinoma ({summary.class_counts['Invasive carcinoma'] || 0})</option>
              <option value="MALIGNANT">Malignant Cohort ({malignantCount})</option>
              <option value="NON_MALIGNANT">Non-Malignant Cohort ({summary.total - malignantCount})</option>
            </select>
          </div>

          {/* Consensus Filter */}
          <select
            value={consensusFilter}
            onChange={(e) => setConsensusFilter(e.target.value)}
            className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-slate-200 font-semibold focus:outline-none"
          >
            <option value="ALL">All Consensus Types</option>
            <option value="UNANIMOUS">5/5 Unanimous Consensus</option>
            <option value="MAJORITY">Majority Consensus</option>
          </select>

          {/* Sort Order Toggle */}
          <button
            onClick={() => setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'))}
            className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-slate-300 font-mono hover:bg-slate-800 transition-colors"
            title="Toggle Confidence Sorting"
          >
            {sortOrder === 'asc' ? '▲ Certainty' : '▼ Certainty'}
          </button>
        </div>

      </div>

      {/* Biopsy Predictions Data Table */}
      <div className="rounded-2xl overflow-hidden border border-slate-800 bg-slate-900/60 shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900 text-[11px] font-bold font-mono uppercase tracking-wider text-slate-400">
                <th className="py-3 px-4">Biopsy Patch</th>
                <th className="py-3 px-4">Sample ID</th>
                <th className="py-3 px-4">Fuzzy Consensus Diagnosis</th>
                <th className="py-3 px-4">5-Model Agreement</th>
                <th className="py-3 px-4">Certainty</th>
                <th className="py-3 px-4">Probability Spectrum</th>
                <th className="py-3 px-4 text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredItems.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-slate-400 font-mono">
                    No matching biopsy samples found for the selected filter criteria.
                  </td>
                </tr>
              ) : (
                filteredItems.map((item, idx) => {
                  const cfg = CLASS_CONFIG[item.predicted_class] || CLASS_CONFIG['Benign'];
                  return (
                    <tr
                      key={idx}
                      onClick={() => onSelectItem(item)}
                      className="hover:bg-slate-850/90 cursor-pointer transition-colors group"
                    >
                      {/* Biopsy Thumbnail */}
                      <td className="py-2.5 px-4">
                        <div className="w-10 h-10 rounded-lg bg-slate-950 overflow-hidden border border-slate-800 group-hover:border-teal-500/50 shrink-0">
                          <img
                            src={item.thumbnail}
                            alt={item.image}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                          />
                        </div>
                      </td>

                      {/* Sample Filename */}
                      <td className="py-2.5 px-4 font-mono font-medium text-slate-200">
                        {item.image}
                      </td>

                      {/* Predicted Class Pill */}
                      <td className="py-2.5 px-4">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${cfg.bg} ${cfg.text} ${cfg.border}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
                          <span>{item.predicted_class}</span>
                        </span>
                      </td>

                      {/* 5-Model Multi-Pill Agreement */}
                      <td className="py-2.5 px-4">
                        <div className="flex items-center gap-1.5">
                          <span className="px-2 py-0.5 rounded bg-slate-800 text-[11px] font-mono font-bold text-teal-300 border border-slate-700">
                            {item.consensus || '5/5 Agree'}
                          </span>
                          {item.individual_votes && (
                            <div className="hidden lg:flex items-center gap-1">
                              {Object.entries(item.individual_votes).map(([mName, v]) => {
                                const mCfg = CLASS_CONFIG[v.predicted_class] || CLASS_CONFIG['Benign'];
                                return (
                                  <span
                                    key={mName}
                                    title={`${mName}: ${v.predicted_class} (${(v.confidence * 100).toFixed(0)}%)`}
                                    className={`w-2 h-2 rounded-full cursor-help ${mCfg.dot}`}
                                  />
                                );
                              })}
                            </div>
                          )}
                        </div>
                      </td>

                      {/* Certainty Percentage */}
                      <td className="py-2.5 px-4 font-mono font-bold text-slate-100">
                        {item.confidence_percentage}%
                      </td>

                      {/* 4-Class Probability Spectrum Bar */}
                      <td className="py-2.5 px-4 min-w-[140px]">
                        <div className="space-y-1">
                          <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden flex">
                            <div 
                              style={{ width: `${(item.probability_Normal_tissue || item.probabilities['Normal tissue'] || 0) * 100}%` }} 
                              className="bg-emerald-500 h-full" 
                              title={`Normal: ${((item.probability_Normal_tissue || item.probabilities['Normal tissue'] || 0) * 100).toFixed(1)}%`}
                            />
                            <div 
                              style={{ width: `${(item.probability_Benign || item.probabilities['Benign'] || 0) * 100}%` }} 
                              className="bg-sky-500 h-full" 
                              title={`Benign: ${((item.probability_Benign || item.probabilities['Benign'] || 0) * 100).toFixed(1)}%`}
                            />
                            <div 
                              style={{ width: `${(item.probability_In_situ_carcinoma || item.probabilities['In-situ carcinoma'] || 0) * 100}%` }} 
                              className="bg-amber-500 h-full" 
                              title={`In-situ: ${((item.probability_In_situ_carcinoma || item.probabilities['In-situ carcinoma'] || 0) * 100).toFixed(1)}%`}
                            />
                            <div 
                              style={{ width: `${(item.probability_Invasive_carcinoma || item.probabilities['Invasive carcinoma'] || 0) * 100}%` }} 
                              className="bg-rose-500 h-full" 
                              title={`Invasive: ${((item.probability_Invasive_carcinoma || item.probabilities['Invasive carcinoma'] || 0) * 100).toFixed(1)}%`}
                            />
                          </div>
                        </div>
                      </td>

                      {/* Inspect Action */}
                      <td className="py-2.5 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectItem(item);
                          }}
                          className="p-1.5 rounded-lg bg-slate-800 hover:bg-teal-600 text-slate-300 hover:text-white transition-colors"
                          title="Inspect Detailed Model Activations"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
};
