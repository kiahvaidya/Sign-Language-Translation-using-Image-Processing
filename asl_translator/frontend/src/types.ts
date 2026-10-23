export interface DIPStages {
  acquisition_rgb?: string;
  high_boost_gray?: string;
  raw_mask?: string;
  cleaned_mask?: string;
  visualization?: string;
}

export interface DIPMetrics {
  "Area (pixels)"?: number;
  "Perimeter (pixels)"?: number;
  "Bounding Box (x,y,w,h)"?: [number, number, number, number];
  "Aspect Ratio (w/h)"?: number;
  "Extent (Area / BoxArea)"?: number;
  "Solidity (Area / HullArea)"?: number;
  "Compactness"?: number;
  "Centroid (cx, cy)"?: [number, number];
  "Convex Hull Area"?: number;
  "Convexity Defects Count"?: number;
  "Max Defect Depth"?: number;
  "Hu Moments (log10)"?: number[];
  "Equivalent Diameter"?: number;
}

export interface DIPMetadata {
  width: number;
  height: number;
  channels: number;
  dtype: string;
  min_intensity: number;
  max_intensity: number;
  mean_intensity: number;
}

export interface DIPEvidenceItem {
  index: number;
  predicted_sign: string;
  confidence: number;
  stages: DIPStages;
  metrics: DIPMetrics;
  metadata: DIPMetadata;
}

export interface VideoTimelineItem {
  frame: number;
  timestamp: string;
  sign: string;
  confidence: number;
}

export interface TranslationResponse {
  translated_sentence: string;
  raw_phrase?: string;
  characters?: string[];
  confidence: number;
  top_probabilities?: Record<string, number>;
  multilingual?: Record<string, string>;
  dip_evidence?: DIPEvidenceItem[];
  timeline?: VideoTimelineItem[];
  distinct_signs?: string[];
  stages?: DIPStages;
  metrics?: DIPMetrics;
  metadata?: DIPMetadata;
  viewfinder_overlay?: string;
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  mode: "image" | "webcam" | "video";
  summary: string;
  translatedText: string;
  confidence: number;
  characters: string[];
}

export interface SampleVideoItem {
  id: string;
  name: string;
  filename: string;
  description: string;
  url: string;
  size_kb: number;
}

