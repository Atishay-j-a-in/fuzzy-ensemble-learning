import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { InferenceStudio } from './components/InferenceStudio';
import { ResultsDashboard } from './components/ResultsDashboard';
import { ArchitectureSection } from './components/ArchitectureSection';
import { BenchmarksSection } from './components/BenchmarksSection';
import { CitationSection } from './components/CitationSection';
import { Footer } from './components/Footer';
import { ImageDetailModal } from './components/ImageDetailModal';
import { BatchPredictionResponse, PredictionItem, ModelInfo, StartupStatus } from './types';
import { useJobPolling } from './hooks/useJobPolling';

export function App() {
  const [backendHealthy, setBackendHealthy] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [predictionData, setPredictionData] = useState<BatchPredictionResponse | null>(null);
  const [selectedItem, setSelectedItem] = useState<PredictionItem | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'studio' | 'results' | 'architecture' | 'benchmarks' | 'citation'>('studio');
  const [inferenceMode, setInferenceMode] = useState<'ensemble' | 'xception'>('ensemble');
  const [demoJobId, setDemoJobId] = useState<string | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [startupStatus, setStartupStatus] = useState<StartupStatus | null>(null);
  const { job: demoJob, error: demoPollError, cancel: cancelDemoJob } = useJobPolling(demoJobId);

  useEffect(() => {
    checkHealth();
    fetchModelInfo();
    // Load initial 10 test samples
    loadDemoBatch(10);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll cold-start progress while the backend reports degraded/offline.
  useEffect(() => {
    if (backendHealthy) return;
    let stopped = false;
    const fetchStartup = async () => {
      try {
        const res = await fetch('/api/startup/progress');
        if (res.ok) {
          const data: StartupStatus = await res.json();
          if (!stopped) setStartupStatus(data);
        }
      } catch {
        /* backend may not be up yet — Navbar keeps showing Offline */
      }
    };
    void fetchStartup();
    const timer = window.setInterval(fetchStartup, 2000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [backendHealthy]);

  // Complete async demo/sample-dataset jobs.
  useEffect(() => {
    if (!demoJobId || !demoJob) return;
    if (demoJob.status === 'succeeded' && demoJob.result) {
      setPredictionData(demoJob.result);
      setDemoError(null);
      setActiveTab('results');
      setIsLoading(false);
      setDemoJobId(null);
    } else if (demoJob.status === 'failed') {
      console.error('Demo job failed:', demoJob.error);
      setDemoError(demoJob.error || 'Sample-dataset job failed on the backend.');
      setIsLoading(false);
      setDemoJobId(null);
    } else if (demoJob.status === 'cancelled') {
      setIsLoading(false);
      setDemoJobId(null);
    }
  }, [demoJob, demoJobId]);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setBackendHealthy(data.status === 'healthy' && data.model_loaded);
      }
    } catch {
      setBackendHealthy(false);
    }
  };

  const fetchModelInfo = async () => {
    try {
      const res = await fetch('/api/model/info');
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch (err) {
      console.error('Error fetching model info:', err);
    }
  };

  const loadDemoBatch = async (limit: number = 10) => {
    setIsLoading(true);
    setDemoJobId(null);
    setDemoError(null);
    try {
      // Preferred path: async job + live polling.
      try {
        const asyncRes = await fetch(
          `/api/predict/sample-dataset/async?limit=${limit}&offset=0&mode=${inferenceMode}`,
          { method: 'POST' }
        );
        if (asyncRes.ok) {
          const { job_id } = await asyncRes.json();
          setDemoJobId(job_id);
          return; // completion handled by the demo-job effect above
        }
        if (asyncRes.status === 404) {
          // Sample dataset missing on the server — surface it instead of failing silently.
          const err = await asyncRes.json().catch(() => ({ detail: 'Sample dataset not found on server.' }));
          throw new Error(err.detail || 'Sample dataset not found on server.');
        }
        // Other non-OK (old backend): fall through to legacy sync endpoint.
      } catch (e: any) {
        if (e?.message && !e.message.includes('Failed to fetch')) {
          // Backend answered with a real error (e.g. missing dataset): show it.
          setDemoError(`${e.message} Upload your own images instead, or add test*.tif files under ICIAR2018_BACH_Challenge_TestDataset/Photos/.`);
          setIsLoading(false);
          return;
        }
        // Network failure of /async: fall through to legacy sync endpoint.
      }

      const res = await fetch(`/api/predict/sample-dataset?limit=${limit}&offset=0&mode=${inferenceMode}`, {
        method: 'POST',
      });
      if (res.ok) {
        const data: BatchPredictionResponse = await res.json();
        setPredictionData(data);
        setActiveTab('results');
      } else {
        const err = await res.json().catch(() => ({ detail: 'Sample dataset request failed' }));
        throw new Error(err.detail || 'Sample dataset request failed.');
      }
      setIsLoading(false);
    } catch (err: any) {
      console.error('Error loading demo batch:', err);
      setDemoError(`${err?.message || 'Could not run sample dataset.'} Upload your own images instead, or add test*.tif files under ICIAR2018_BACH_Challenge_TestDataset/Photos/.`);
      setIsLoading(false);
    }
  };

  const handlePredictionsComplete = (response: BatchPredictionResponse) => {
    setPredictionData(response);
    setActiveTab('results');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-teal-500/30 selection:text-teal-200">
      
      {/* Sleek Top Navigation */}
      <Navbar
        backendHealthy={backendHealthy}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        resultCount={predictionData?.items.length || 0}
        inferenceMode={inferenceMode}
        setInferenceMode={setInferenceMode}
        startupStatus={startupStatus}
      />

      {/* Main Tab Content */}
      <main className="flex-grow">
        
        {/* TAB 1: STUDIO (Upload & Demo Runner) */}
        {activeTab === 'studio' && (
          <InferenceStudio
            onPredictionsComplete={handlePredictionsComplete}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            onLoadDemo={loadDemoBatch}
            inferenceMode={inferenceMode}
            setInferenceMode={setInferenceMode}
            demoJob={demoJob}
            demoPollError={demoPollError}
            onCancelDemoJob={cancelDemoJob}
            demoError={demoError}
          />
        )}

        {/* TAB 2: RESULTS & CSV EXPORT */}
        {activeTab === 'results' && (
          <div>
            {predictionData ? (
              <ResultsDashboard
                data={predictionData}
                onSelectItem={(item) => setSelectedItem(item)}
              />
            ) : (
              <div className="py-20 text-center text-slate-400">
                <p>No predictions available yet. Please upload a dataset or run demo samples.</p>
                <button
                  onClick={() => setActiveTab('studio')}
                  className="mt-4 px-4 py-2 rounded-xl bg-teal-600 text-white text-xs font-bold"
                >
                  Go to Inference Studio
                </button>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: ARCHITECTURE DIAGRAMS */}
        {activeTab === 'architecture' && <ArchitectureSection />}

        {/* TAB 4: 5-MODEL BENCHMARKS */}
        {activeTab === 'benchmarks' && <BenchmarksSection />}

        {/* TAB 5: CITATION */}
        {activeTab === 'citation' && <CitationSection />}

      </main>

      {/* Footer */}
      <Footer />

      {/* Sample Detail Modal */}
      <ImageDetailModal
        item={selectedItem}
        onClose={() => setSelectedItem(null)}
      />

    </div>
  );
}

export default App;
