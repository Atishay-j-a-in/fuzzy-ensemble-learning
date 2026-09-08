import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileArchive, 
  Image as ImageIcon, 
  Play, 
  RefreshCw, 
  Microscope,
  Trash2,
  Sparkles,
  Layers,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
  ShieldAlert,
  Info
} from 'lucide-react';
import { BatchPredictionResponse } from '../types';

interface InferenceStudioProps {
  onPredictionsComplete: (response: BatchPredictionResponse) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  onLoadDemo: (limit: number) => void;
  inferenceMode: 'ensemble' | 'xception';
  setInferenceMode: (mode: 'ensemble' | 'xception') => void;
}

export const InferenceStudio: React.FC<InferenceStudioProps> = ({
  onPredictionsComplete,
  isLoading,
  setIsLoading,
  onLoadDemo,
  inferenceMode,
  setInferenceMode,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [statusStep, setStatusStep] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const handleFiles = (files: File[]) => {
    setErrorMsg(null);
    const validExtensions = ['tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp', 'zip'];
    const validFiles = files.filter((f) => {
      const ext = f.name.split('.').pop()?.toLowerCase();
      return validExtensions.includes(ext || '');
    });

    if (validFiles.length === 0) {
      setErrorMsg('Unsupported format. Please select histopathology images (.tif, .png, .jpg) or a .zip dataset archive.');
      return;
    }

    const zipFile = validFiles.find((f) => f.name.toLowerCase().endsWith('.zip'));
    if (zipFile) {
      setSelectedFiles([zipFile]);
    } else {
      setSelectedFiles((prev) => [...prev, ...validFiles]);
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const clearFiles = () => {
    setSelectedFiles([]);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const runUploadInference = async () => {
    if (selectedFiles.length === 0) {
      setErrorMsg('Please select at least one biopsy image or upload a dataset ZIP archive.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);
    setStatusStep(
      inferenceMode === 'ensemble'
        ? 'Extracting multi-scale deep features across 5 backbones & evaluating Choquet Fuzzy Integral...'
        : 'Harvesting Xception intermediate representations and computing 4-class classification...'
    );

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append('files', file));

      const response = await fetch(`/api/predict/batch?mode=${inferenceMode}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Inference request failed' }));
        throw new Error(err.detail || `Server error status: ${response.status}`);
      }

      const data: BatchPredictionResponse = await response.json();
      onPredictionsComplete(data);
    } catch (err: any) {
      console.error('Inference error:', err);
      setErrorMsg(err.message || 'Failed to process images with the deep learning model.');
    } finally {
      setIsLoading(false);
      setStatusStep('');
    }
  };

  return (
    <div className="py-8 sm:py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      
      {/* Studio Header Banner */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-teal-400 font-semibold mb-1.5">
            <Microscope className="w-4 h-4" />
            <span>ICIAR 2018 GRAND CHALLENGE BENCHMARK ENVIRONMENT</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Histopathology Biopsy Inference Studio
          </h1>
          <p className="text-slate-300 text-xs sm:text-sm mt-1 max-w-2xl">
            Evaluate breast microscopy patches across 4 histological diagnostic categories (Normal tissue, Benign lesion, In-situ carcinoma, and Invasive carcinoma) using multi-backbone feature harvesting and Choquet fuzzy consensus.
          </p>
        </div>

        {/* Model Execution Paradigm Selector */}
        <div className="bg-slate-900/90 p-1.5 rounded-xl border border-slate-800 shrink-0 self-start lg:self-center">
          <div className="text-[11px] font-mono text-slate-400 px-2 py-1 font-semibold uppercase tracking-wider">
            Classification Engine
          </div>
          <div className="flex items-center gap-1.5 mt-1">
            <button
              onClick={() => setInferenceMode('ensemble')}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                inferenceMode === 'ensemble'
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/80'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>5-Model Fuzzy Ensemble (95% Acc)</span>
            </button>
            <button
              onClick={() => setInferenceMode('xception')}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                inferenceMode === 'xception'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/80'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Single Xception (91% Acc)</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Workspace Layout: 2 Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: Interactive Upload Studio & Staging Box */}
        <div className="lg:col-span-8 space-y-6">
          
          {/* Dropzone Container */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative rounded-2xl border-2 border-dashed p-8 sm:p-10 text-center cursor-pointer transition-all ${
              dragActive
                ? 'border-teal-400 bg-teal-950/20'
                : 'border-slate-700 bg-slate-900/60 hover:border-slate-600 hover:bg-slate-900/90'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".tif,.tiff,.png,.jpg,.jpeg,.bmp,.zip"
              onChange={handleFileInput}
              className="hidden"
            />

            <div className="flex flex-col items-center max-w-md mx-auto">
              <div className="w-14 h-14 rounded-2xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400 mb-4">
                <UploadCloud className="w-7 h-7" />
              </div>

              <h2 className="text-base sm:text-lg font-bold text-white">
                Upload Microscopic Biopsy Patches or Dataset ZIP
              </h2>
              <p className="text-xs sm:text-sm text-slate-300 mt-1.5 leading-relaxed">
                Drag and drop whole-slide patch files here, or <span className="text-teal-400 font-semibold underline underline-offset-4">browse your local directory</span>
              </p>

              {/* Supported Format Pills */}
              <div className="mt-5 flex flex-wrap justify-center gap-2 font-mono text-[11px] text-slate-400">
                <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">.TIF / .TIFF (200×)</span>
                <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">.PNG (512×512)</span>
                <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">.JPG / .JPEG</span>
                <span className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">.ZIP Archive</span>
              </div>
            </div>
          </div>

          {/* Staged File Management Panel */}
          {selectedFiles.length > 0 && (
            <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 space-y-4 shadow-xl">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-white font-mono">
                    {selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''} staged for inference
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    ({(selectedFiles.reduce((acc, f) => acc + f.size, 0) / (1024 * 1024)).toFixed(2)} MB total)
                  </span>
                </div>
                <button
                  onClick={clearFiles}
                  className="text-xs text-rose-400 hover:text-rose-300 font-semibold flex items-center gap-1.5 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear queue</span>
                </button>
              </div>

              {/* Staged File List */}
              <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1 text-xs">
                {selectedFiles.map((file, idx) => (
                  <div 
                    key={idx} 
                    className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-850 hover:border-slate-750 transition-colors"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      {file.name.endsWith('.zip') ? (
                        <FileArchive className="w-4 h-4 text-amber-400 shrink-0" />
                      ) : (
                        <ImageIcon className="w-4 h-4 text-teal-400 shrink-0" />
                      )}
                      <span className="font-mono text-slate-200 truncate">{file.name}</span>
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-[11px] font-mono text-slate-400">
                        {(file.size / 1024).toFixed(1)} KB
                      </span>
                      <button
                        onClick={() => removeFile(idx)}
                        className="text-slate-500 hover:text-rose-400 transition-colors"
                        title="Remove file"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Primary Action Button */}
              <button
                onClick={runUploadInference}
                disabled={isLoading}
                className="w-full py-3.5 rounded-xl font-bold text-sm bg-teal-600 hover:bg-teal-500 text-white shadow-md shadow-teal-900/30 flex items-center justify-center gap-2 transition-all active:scale-[0.99] disabled:opacity-50"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-white" />
                    <span>Processing {selectedFiles.length} Biopsies...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>
                      Run {inferenceMode === 'ensemble' ? '5-Model Fuzzy Ensemble' : 'Xception'} Analysis ({selectedFiles.length} Samples)
                    </span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Loading Indicator with Step Description */}
          {isLoading && (
            <div className="p-4 rounded-xl bg-teal-950/30 border border-teal-500/30 text-xs text-teal-200 flex items-center gap-3">
              <RefreshCw className="w-4 h-4 animate-spin text-teal-400 shrink-0" />
              <span>{statusStep || 'Processing input histology images through neural backbones...'}</span>
            </div>
          )}

          {/* Error Alert Box */}
          {errorMsg && (
            <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/40 text-xs text-rose-200 flex items-start gap-3">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <strong className="font-semibold block text-rose-300">Inference Error</strong>
                <p className="mt-0.5 leading-relaxed">{errorMsg}</p>
              </div>
            </div>
          )}

        </div>

        {/* Right Column: Instant Benchmark Dataset Evaluator */}
        <div className="lg:col-span-4 space-y-6">
          
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-5 shadow-xl">
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-teal-400 uppercase tracking-wider">
              <Microscope className="w-4 h-4" />
              <span>1-Click Test Dataset Suite</span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Evaluate real histopathology biopsy images from the ICIAR 2018 BACH test dataset. Run end-to-end multi-backbone inference and Choquet fuzzy integral fusion instantly:
            </p>

            {/* Test Suite Action Buttons */}
            <div className="space-y-2.5">
              {[
                { 
                  count: 10, 
                  title: 'Run 10 Test Biopsies', 
                  desc: 'Fast verification (test0.tif — test9.tif)',
                  time: '~0.6s'
                },
                { 
                  count: 25, 
                  title: 'Run 25 Test Biopsies', 
                  desc: 'Standard cohort (test0.tif — test24.tif)',
                  time: '~1.4s'
                },
                { 
                  count: 50, 
                  title: 'Run 50 Test Biopsies', 
                  desc: 'Extended cohort (test0.tif — test49.tif)',
                  time: '~2.8s'
                },
                { 
                  count: 100, 
                  title: 'Run Complete 100 Biopsies', 
                  desc: 'Full test dataset benchmark',
                  time: '~5.5s'
                },
              ].map((batch) => (
                <button
                  key={batch.count}
                  onClick={() => onLoadDemo(batch.count)}
                  disabled={isLoading}
                  className="w-full text-left p-3.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-teal-500/50 hover:bg-slate-850 transition-all group disabled:opacity-50"
                >
                  <div className="flex items-center justify-between text-xs font-bold text-white group-hover:text-teal-300">
                    <span>{batch.title}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-slate-500 group-hover:text-teal-400">{batch.time}</span>
                      <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-teal-400 group-hover:translate-x-0.5 transition-transform" />
                    </div>
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1 font-mono">{batch.desc}</div>
                </button>
              ))}
            </div>

            {/* Technical Note Callout */}
            <div className="pt-4 border-t border-slate-800 text-[11px] text-slate-300 leading-relaxed flex items-start gap-2">
              <Info className="w-4 h-4 text-teal-400 shrink-0 mt-0.5" />
              <span>
                Each sample undergoes Optical Density Macenko stain normalization before parallel feature harvesting across Xception, IRV2, IV3, VGG19, and VGG16.
              </span>
            </div>
          </div>

          {/* 4 Pathology Classes Quick Legend */}
          <div className="rounded-2xl bg-slate-900/60 border border-slate-800 p-5 space-y-3">
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono">
              Diagnostic Target Classes
            </h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2.5 rounded-lg bg-slate-950 border border-emerald-500/30">
                <div className="font-bold text-emerald-400">Normal tissue</div>
                <div className="text-[10px] text-slate-400 mt-0.5">Physiological</div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950 border border-sky-500/30">
                <div className="font-bold text-sky-400">Benign lesion</div>
                <div className="text-[10px] text-slate-400 mt-0.5">Non-malignant</div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950 border border-amber-500/30">
                <div className="font-bold text-amber-400">In-situ carcinoma</div>
                <div className="text-[10px] text-slate-400 mt-0.5">DCIS pre-invasive</div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950 border border-rose-500/30">
                <div className="font-bold text-rose-400">Invasive carcinoma</div>
                <div className="text-[10px] text-slate-400 mt-0.5">IDC malignant</div>
              </div>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};
