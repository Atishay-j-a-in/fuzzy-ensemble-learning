export type DiagnosticClass = 'Benign' | 'In-situ carcinoma' | 'Invasive carcinoma' | 'Normal tissue';

export interface ClassProbabilityMap {
  'Benign': number;
  'In-situ carcinoma': number;
  'Invasive carcinoma': number;
  'Normal tissue': number;
  [key: string]: number;
}

export interface ModelVoteInfo {
  predicted_class: DiagnosticClass;
  predicted_class_id: number;
  confidence: number;
  probabilities: ClassProbabilityMap;
}

export interface PredictionItem {
  image: string;
  predicted_class_id: number;
  predicted_class: DiagnosticClass;
  confidence: number;
  confidence_percentage: number;
  probabilities: ClassProbabilityMap;
  probability_Benign: number;
  probability_In_situ_carcinoma: number;
  probability_Invasive_carcinoma: number;
  probability_Normal_tissue: number;
  severity: 'normal' | 'low' | 'moderate' | 'high';
  color: string;
  description: string;
  thumbnail: string;
  original_width?: number;
  original_height?: number;
  latency_ms: number;
  consensus?: string;
  agreeing_models_count?: number;
  individual_votes?: Record<string, ModelVoteInfo>;
  mode?: string;
  timestamp: string;
}

export interface BatchSummary {
  total: number;
  class_counts: Record<DiagnosticClass, number>;
  average_confidence_percentage: number;
  total_time_ms: number;
  avg_latency_ms: number;
  malignant_count: number;
  non_malignant_count: number;
  mode?: string;
  models_applied?: string[];
}

export interface BatchPredictionResponse {
  items: PredictionItem[];
  summary: BatchSummary;
}

export interface ModelInfo {
  architecture: string;
  backbones?: Array<{
    name: string;
    single_acc: number;
    weights: string;
    weight_dim: number;
  }>;
  fuzzy_densities?: Record<string, number>;
  input_resolution: string;
  preprocessing: string;
  classes: Array<{
    id: number;
    name: DiagnosticClass;
    severity: string;
    color: string;
    description: string;
  }>;
  benchmarks: {
    two_class: Array<{ model: string; val_acc: number | string; test_acc: number; highlight?: boolean; is_ensemble?: boolean }>;
    four_class: Array<{ model: string; val_acc: number | string; test_acc: number; highlight?: boolean; is_ensemble?: boolean }>;
  };
  citation: {
    title: string;
    authors: string;
    journal: string;
    year: number;
    publisher: string;
    doi_or_page: string;
  };
}
