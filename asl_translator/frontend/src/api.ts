import { TranslationResponse, SampleVideoItem } from './types';

const API_BASE = import.meta.env.VITE_API_URL || '';

export async function fetchHealth(): Promise<{ status: string; model_loaded: boolean; vocabulary_size: number }> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSamples(): Promise<{ presets: string[]; has_sample_video: boolean; available_alphabet: string[] }> {
  const res = await fetch(`${API_BASE}/api/samples`);
  if (!res.ok) throw new Error(`Failed to fetch samples: ${res.statusText}`);
  return res.json();
}

export async function fetchSampleVideos(): Promise<{ sample_videos: SampleVideoItem[] }> {
  const res = await fetch(`${API_BASE}/api/sample-videos`);
  if (!res.ok) throw new Error(`Failed to fetch sample videos: ${res.statusText}`);
  return res.json();
}

export async function translateImages(files: File[]): Promise<TranslationResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }

  const res = await fetch(`${API_BASE}/api/translate/image`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || 'Image translation failed.');
  }

  return res.json();
}

export async function translatePreset(phrase: string): Promise<TranslationResponse> {
  const res = await fetch(`${API_BASE}/api/translate/preset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phrase }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || 'Preset translation failed.');
  }

  return res.json();
}

export async function translateVideo(
  file?: File,
  useSample: boolean = false,
  sampleFilename?: string
): Promise<TranslationResponse> {
  const formData = new FormData();
  if (useSample || sampleFilename) {
    formData.append('use_sample', 'true');
    if (sampleFilename) {
      formData.append('sample_filename', sampleFilename);
    }
  } else if (file) {
    formData.append('file', file);
  } else {
    throw new Error('Must provide a video file or choose sample.');
  }

  const res = await fetch(`${API_BASE}/api/translate/video`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || 'Video translation failed.');
  }

  return res.json();
}


export async function translateFrame(
  imageBase64: string,
  framingMode: string = 'target_box',
  boxPreset: string = 'right_hand',
  boxSize: number = 240
): Promise<TranslationResponse> {
  const res = await fetch(`${API_BASE}/api/translate/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: imageBase64,
      framing_mode: framingMode,
      box_preset: boxPreset,
      box_size: boxSize,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || 'Camera frame translation failed.');
  }

  return res.json();
}
