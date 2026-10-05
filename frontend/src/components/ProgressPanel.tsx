import React from 'react';
import { RefreshCw, XCircle, CheckCircle2, AlertCircle } from 'lucide-react';
import type { JobStatus } from '../types';

interface ProgressPanelProps {
  job: JobStatus | null;
  pollError?: string | null;
  fallbackMessage?: string;
  onCancel?: () => void;
}

const MODEL_STAGES = ['Xception', 'VGG16', 'VGG19', 'InceptionV3', 'InceptionResNetV2', 'Choquet fusion'];

export const ProgressPanel: React.FC<ProgressPanelProps> = ({ job, pollError, fallbackMessage, onCancel }) => {
  if (!job) {
    return (
      <div className="p-4 rounded-xl bg-teal-950/30 border border-teal-500/30 text-xs text-teal-200 flex items-center gap-3">
        <RefreshCw className="w-4 h-4 animate-spin text-teal-400 shrink-0" />
        <span>{fallbackMessage || 'Contacting backend… (uploading / queueing job)'}</span>
      </div>
    );
  }

  const percent = Math.max(0, Math.min(100, job.percent ?? 0));
  const isDone = job.status === 'succeeded';
  const isFailed = job.status === 'failed';
  const isCancelled = job.status === 'cancelled';

  return (
    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-mono font-bold text-teal-300">
          {isDone ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          ) : isFailed ? (
            <AlertCircle className="w-4 h-4 text-rose-400" />
          ) : isCancelled ? (
            <XCircle className="w-4 h-4 text-slate-400" />
          ) : (
            <RefreshCw className="w-4 h-4 animate-spin text-teal-400" />
          )}
          <span>
            {isDone ? 'Inference complete' : isFailed ? 'Inference failed' : isCancelled ? 'Inference cancelled' : `Image ${job.current}/${job.total}`}
          </span>
        </div>
        {!isDone && !isFailed && !isCancelled && onCancel && (
          <button
            onClick={onCancel}
            className="text-[11px] font-mono px-2 py-1 rounded-lg bg-slate-800 hover:bg-rose-900/60 text-slate-300 hover:text-rose-200 border border-slate-700 transition-colors"
          >
            Cancel job
          </button>
        )}
      </div>

      <div className="h-2.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${isFailed ? 'bg-rose-500' : isDone ? 'bg-emerald-500' : 'bg-teal-500'}`}
          style={{ width: `${isDone ? 100 : percent}%` }}
        />
      </div>

      <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
        <span className="truncate">{job.message || job.stage}</span>
        <span className="font-bold text-slate-200 shrink-0 ml-2">{isDone ? '100%' : `${percent.toFixed(1)}%`}</span>
      </div>

      {job.current_model && !isDone && (
        <div className="flex flex-wrap gap-1.5">
          {MODEL_STAGES.map((m) => {
            const active = job.current_model === m;
            return (
              <span
                key={m}
                className={`px-2 py-0.5 rounded-full text-[10px] font-mono border ${
                  active
                    ? 'bg-teal-500/20 text-teal-200 border-teal-500/50'
                    : 'bg-slate-900 text-slate-500 border-slate-800'
                }`}
              >
                {m}
              </span>
            );
          })}
        </div>
      )}

      {job.current_image && (
        <div className="text-[11px] font-mono text-slate-400 truncate">
          Current file: <span className="text-slate-200">{job.current_image}</span>
        </div>
      )}

      {isFailed && job.error && (
        <div className="text-[11px] font-mono text-rose-300">Error: {job.error}</div>
      )}
      {pollError && <div className="text-[11px] font-mono text-amber-300">Progress polling: {pollError}</div>}
    </div>
  );
};
