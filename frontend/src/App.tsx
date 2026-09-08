import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { InferenceStudio } from './components/InferenceStudio';
import { ResultsDashboard } from './components/ResultsDashboard';
import { ArchitectureSection } from './components/ArchitectureSection';
import { BenchmarksSection } from './components/BenchmarksSection';
import { CitationSection } from './components/CitationSection';
import { Footer } from './components/Footer';
import { ImageDetailModal } from './components/ImageDetailModal';
import { BatchPredictionResponse, PredictionItem, ModelInfo } from './types';

export function App() {
  const [backendHealthy, setBackendHealthy] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [predictionData, setPredictionData] = useState<BatchPredictionResponse | null>(null);
  const [selectedItem, setSelectedItem] = useState<PredictionItem | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'studio' | 'results' | 'architecture' | 'benchmarks' | 'citation'>('studio');
  const [inferenceMode, setInferenceMode] = useState<'ensemble' | 'xception'>('ensemble');

  useEffect(() => {
    checkHealth();
    fetchModelInfo();
    // Load initial 10 test samples
    loadDemoBatch(10);
  }, []);

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
    try {
      const res = await fetch(`/api/predict/sample-dataset?limit=${limit}&offset=0&mode=${inferenceMode}`, {
        method: 'POST',
      });
      if (res.ok) {
        const data: BatchPredictionResponse = await res.json();
        setPredictionData(data);
        setActiveTab('results');
      }
    } catch (err) {
      console.error('Error loading demo batch:', err);
    } finally {
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
