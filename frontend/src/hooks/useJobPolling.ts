import { useEffect, useRef, useState } from 'react';
import type { JobStatus } from '../types';

interface UseJobPollingResult {
  job: JobStatus | null;
  error: string | null;
  cancel: () => Promise<void>;
  reset: () => void;
}

/** Poll GET /api/jobs/{jobId} every `intervalMs` until succeeded/failed/cancelled. */
export function useJobPolling(jobId: string | null, intervalMs = 1000): UseJobPollingResult {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const cancel = async () => {
    if (!jobId) return;
    try {
      await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
    } catch {
      /* best effort — polling will surface the final state */
    }
  };

  const reset = () => {
    setJob(null);
    setError(null);
  };

  useEffect(() => {
    if (!jobId) return;
    let stopped = false;

    const poll = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error(`Progress lookup failed (${res.status})`);
        const data: JobStatus = await res.json();
        if (stopped) return;
        setJob(data);
        if (data.status === 'succeeded' || data.status === 'failed' || data.status === 'cancelled') {
          if (timer.current) window.clearInterval(timer.current);
        }
      } catch (e: any) {
        if (stopped) return;
        setError(e?.message || 'Failed to poll job progress');
      }
    };

    setJob(null);
    setError(null);
    void poll();
    timer.current = window.setInterval(poll, intervalMs);

    return () => {
      stopped = true;
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [jobId, intervalMs]);

  return { job, error, cancel, reset };
}
