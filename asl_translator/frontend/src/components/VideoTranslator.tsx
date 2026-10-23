import React, { useState, useEffect, useRef } from 'react';
import { Film, ArrowRight, Play, CheckCircle2, X, RefreshCw, FileVideo } from 'lucide-react';
import { translateVideo, fetchSampleVideos } from '../api';
import { TranslationResponse, SampleVideoItem } from '../types';

interface VideoTranslatorProps {
  onTranslateStart: (message: string) => void;
  onTranslateSuccess: (response: TranslationResponse, summary: string, mode: 'video') => void;
  onTranslateError: (error: string) => void;
  disabled: boolean;
}

export const VideoTranslator: React.FC<VideoTranslatorProps> = ({
  onTranslateStart,
  onTranslateSuccess,
  onTranslateError,
  disabled,
}) => {
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleVideoItem | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [sampleVideos, setSampleVideos] = useState<SampleVideoItem[]>([]);
  const [dragOver, setDragOver] = useState<boolean>(false);
  const videoInputRef = useRef<HTMLInputElement | null>(null);

  // Fetch sample videos on mount
  useEffect(() => {
    let mounted = true;
    const loadSamples = async () => {
      try {
        const data = await fetchSampleVideos();
        if (mounted && data.sample_videos && data.sample_videos.length > 0) {
          setSampleVideos(data.sample_videos);
        }
      } catch (e) {
        console.warn('Could not load sample videos list', e);
      }
    };
    loadSamples();
    return () => {
      mounted = false;
    };
  }, []);

  const handleVideoSelect = (file: File) => {
    if (!file.type.startsWith('video/')) return;
    if (previewUrl && !selectedSample) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedVideoFile(file);
    setSelectedSample(null);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSelectSample = (sample: SampleVideoItem) => {
    if (previewUrl && !selectedSample) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedVideoFile(null);
    setSelectedSample(sample);
    // Use API streaming URL for browser video playback
    setPreviewUrl(sample.url);
  };

  const clearSelection = () => {
    if (previewUrl && !selectedSample) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedVideoFile(null);
    setSelectedSample(null);
    setPreviewUrl(null);
    if (videoInputRef.current) videoInputRef.current.value = '';
  };

  const handleTranslate = async () => {
    if (disabled || (!selectedVideoFile && !selectedSample)) return;

    if (selectedSample) {
      onTranslateStart(`Sampling frames & running DIP segmentation on ${selectedSample.name}...`);
      try {
        const response = await translateVideo(undefined, true, selectedSample.filename);
        onTranslateSuccess(response, selectedSample.name, 'video');
      } catch (err: any) {
        onTranslateError(err.message || 'Sample video translation failed.');
      }
    } else if (selectedVideoFile) {
      onTranslateStart(`Extracting keyframes & running DIP segmentation on ${selectedVideoFile.name}...`);
      try {
        const response = await translateVideo(selectedVideoFile, false);
        onTranslateSuccess(response, selectedVideoFile.name, 'video');
      } catch (err: any) {
        onTranslateError(err.message || 'Video translation failed.');
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Upload Dropzone */}
      <div
        className={`dropzone ${dragOver ? 'active' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleVideoSelect(e.dataTransfer.files[0]);
          }
        }}
        onClick={() => videoInputRef.current?.click()}
      >
        <input
          ref={videoInputRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleVideoSelect(e.target.files[0]);
            }
          }}
        />
        <div className="dropzone-icon" style={{ color: 'var(--accent-teal-dark)' }}>
          <Film size={38} />
        </div>
        <div className="dropzone-title">Upload sign language video</div>
        <div className="dropzone-desc">Drag and drop MP4, MOV, or WebM video file (up to 50MB)</div>
      </div>

      {/* Benchmark Sample Videos Selector */}
      {sampleVideos.length > 0 && (
        <div>
          <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 8 }}>
            Or Preview & Translate Benchmark Sample Videos:
          </span>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8 }}>
            {sampleVideos.map((item) => {
              const isCurrent = selectedSample?.id === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleSelectSample(item)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start',
                    padding: '8px 12px',
                    borderRadius: 8,
                    textAlign: 'left',
                    backgroundColor: isCurrent ? 'var(--card-bg)' : 'var(--bg-cream)',
                    border: isCurrent ? '2px solid var(--accent-pink)' : '1px solid var(--card-border)',
                    boxShadow: isCurrent ? '0 2px 8px rgba(255, 121, 172, 0.15)' : 'none',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', marginBottom: 3 }}>
                    <Play size={14} color={isCurrent ? 'var(--accent-pink)' : 'var(--text-muted)'} />
                    <span style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-dark)' }}>
                      {item.name}
                    </span>
                  </div>
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                    {item.description}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Video Preview Player with Native Controls */}
      {previewUrl && (
        <div
          style={{
            backgroundColor: 'var(--bg-cream)',
            border: '1px solid var(--card-border)',
            borderRadius: 'var(--radius-md)',
            padding: 12,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Play size={15} color="var(--accent-pink)" />
              <span style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-dark)' }}>
                {selectedSample ? `Sample Preview: ${selectedSample.name}` : `Uploaded Video: ${selectedVideoFile?.name}`}
              </span>
            </div>
            <button
              onClick={clearSelection}
              style={{ fontSize: '0.78rem', color: '#dc2626', fontWeight: 600 }}
            >
              Clear
            </button>
          </div>

          {/* HTML5 Native Video Player */}
          <div
            style={{
              width: '100%',
              maxHeight: 260,
              backgroundColor: '#1f1b1c',
              borderRadius: 8,
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <video
              key={previewUrl}
              src={previewUrl}
              controls
              playsInline
              style={{
                width: '100%',
                maxHeight: 260,
                objectFit: 'contain',
                display: 'block',
              }}
            />
          </div>

          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
            Press Play to preview the sign gestures, then click Translate Video to Text below.
          </p>
        </div>
      )}

      {/* Translate Action Button */}
      {(selectedVideoFile || selectedSample) && (
        <button
          className="btn-primary"
          onClick={handleTranslate}
          disabled={disabled}
          style={{ width: '100%' }}
        >
          <span>Translate Video to Text</span>
          <ArrowRight size={18} />
        </button>
      )}
    </div>
  );
};
