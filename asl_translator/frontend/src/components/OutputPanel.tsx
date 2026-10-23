import React, { useState } from 'react';
import { Copy, Check, RotateCcw, Eye, Sparkles, Globe, Clock, Layers } from 'lucide-react';
import { TranslationResponse } from '../types';

interface OutputPanelProps {
  response: TranslationResponse | null;
  loading: boolean;
  loadingMessage?: string;
  onClear: () => void;
  onOpenModal: () => void;
}

export const OutputPanel: React.FC<OutputPanelProps> = ({
  response,
  loading,
  loadingMessage = 'Processing Digital Image Pipeline...',
  onClear,
  onOpenModal,
}) => {
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopy = () => {
    if (!response?.translated_sentence) return;
    navigator.clipboard.writeText(response.translated_sentence);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const confidencePct = response?.confidence ? Math.round(response.confidence * 100) : 0;

  return (
    <div className="panel-card" style={{ height: '100%', minHeight: 460 }}>
      {/* Header */}
      <div className="panel-header">
        <div className="panel-title">
          <Globe size={18} color="var(--accent-pink)" />
          <span>English Translation</span>
        </div>
        {response && (
          <div style={{ display: 'flex', gap: 6 }}>
            <button
              className="btn-secondary"
              onClick={onOpenModal}
              title="Inspect DIP Tasks 1-8 pipeline stages and metrics"
              style={{ fontSize: '0.8rem', padding: '4px 10px' }}
            >
              <Layers size={14} /> Inspect DIP Pipeline
            </button>
            <button
              className="btn-secondary"
              onClick={onClear}
              title="Clear current translation"
              style={{ fontSize: '0.8rem', padding: '4px 10px' }}
            >
              <RotateCcw size={14} /> Clear
            </button>
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px 20px' }}>
            <div
              style={{
                width: 48,
                height: 48,
                border: '4px solid var(--card-border)',
                borderTopColor: 'var(--accent-pink)',
                borderRadius: '50%',
                margin: '0 auto 16px',
                animation: 'spin 0.8s linear infinite',
              }}
            />
            <p style={{ fontWeight: 600, color: 'var(--text-dark)', fontSize: '1rem' }}>Translating Sign Language</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem', marginTop: 4 }}>{loadingMessage}</p>
          </div>
        ) : response ? (
          <div className="translation-hero">
            {/* Translated Sentence */}
            <div className="translated-text">{response.translated_sentence}</div>

            {/* Recognized Sign Letter Chips */}
            {((response.characters && response.characters.length > 0) ||
              (response.distinct_signs && response.distinct_signs.length > 0)) && (
              <div style={{ marginBottom: 14 }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                  Recognized ASL Gestures:
                </span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {(response.characters || response.distinct_signs || []).map((ch, idx) => (
                    <span key={idx} className="char-chip">
                      {ch}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Confidence Badge */}
            {confidencePct > 0 && (
              <div className="confidence-badge">
                <Sparkles size={14} color="var(--accent-pink)" />
                <span>{confidencePct}% Confidence</span>
              </div>
            )}


            {/* Video Timeline if available */}
            {response.timeline && response.timeline.length > 0 && (
              <div style={{ marginTop: 18, paddingTop: 14, borderTop: '1px solid var(--card-border)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                  <Clock size={14} color="var(--text-muted)" />
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                    Temporal Frame Breakdown ({response.timeline.length} keyframes)
                  </span>
                </div>
                <div
                  style={{
                    maxHeight: 120,
                    overflowY: 'auto',
                    backgroundColor: 'var(--bg-cream)',
                    borderRadius: 6,
                    padding: 8,
                    fontSize: '0.78rem',
                  }}
                >
                  {response.timeline.map((item, idx) => (
                    <div
                      key={idx}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '3px 0',
                        borderBottom: idx < (response.timeline?.length || 0) - 1 ? '1px dashed var(--card-border)' : 'none',
                      }}
                    >
                      <span style={{ color: 'var(--text-muted)' }}>
                        Frame {item.frame} ({item.timestamp})
                      </span>
                      <span style={{ fontWeight: 600, color: 'var(--text-dark)' }}>
                        Sign: {item.sign} ({Math.round(item.confidence * 100)}%)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="output-actions">
              <button className="btn-primary" onClick={handleCopy} style={{ flex: 1 }}>
                {copied ? <Check size={18} /> : <Copy size={18} />}
                <span>{copied ? 'Copied to Clipboard' : 'Copy Translation'}</span>
              </button>
              <button className="btn-secondary" onClick={onOpenModal}>
                <Eye size={18} /> Inspect Stages
              </button>
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)' }}>
            <div
              style={{
                width: 56,
                height: 56,
                borderRadius: '50%',
                backgroundColor: 'var(--bg-cream)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 16px',
                border: '1px solid var(--card-border)',
              }}
            >
              <Globe size={26} color="var(--text-muted)" />
            </div>
            <p style={{ fontWeight: 600, color: 'var(--text-dark)', fontSize: '1.05rem', marginBottom: 4 }}>
              Ready to Translate
            </p>
            <p style={{ fontSize: '0.85rem', maxWidth: 320, margin: '0 auto' }}>
              Upload an image, activate your webcam, or choose a video to translate American Sign Language into English.
            </p>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};
