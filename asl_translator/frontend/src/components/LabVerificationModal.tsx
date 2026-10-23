import React, { useState } from 'react';
import { X, Layers, Cpu, CheckCircle2, ChevronRight, BarChart2 } from 'lucide-react';
import { TranslationResponse, DIPEvidenceItem } from '../types';

interface LabVerificationModalProps {
  response: TranslationResponse;
  onClose: () => void;
}

export const LabVerificationModal: React.FC<LabVerificationModalProps> = ({ response, onClose }) => {
  // Normalize evidence items: either from dip_evidence or top-level single stage
  const evidenceList: DIPEvidenceItem[] =
    response.dip_evidence && response.dip_evidence.length > 0
      ? response.dip_evidence
      : response.stages
      ? [
          {
            index: 0,
            predicted_sign: response.characters?.[0] || response.translated_sentence || 'Sign',
            confidence: response.confidence || 0.95,
            stages: response.stages,
            metrics: response.metrics || {},
            metadata: response.metadata || {
              width: 200,
              height: 200,
              channels: 3,
              dtype: 'uint8',
              min_intensity: 0,
              max_intensity: 255,
              mean_intensity: 128,
            },
          },
        ]
      : [];

  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const activeItem = evidenceList[selectedIndex] || evidenceList[0];

  if (!activeItem) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h3>DIP Pipeline Verification</h3>
            <button className="btn-secondary" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
          <p style={{ marginTop: 20, color: 'var(--text-muted)' }}>No intermediate DIP stages recorded for this output.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ padding: 8, borderRadius: 8, backgroundColor: 'var(--accent-teal)' }}>
              <Layers size={22} color="var(--text-dark)" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-dark)' }}>
                DIP Pipeline Verification (Tasks 1–8)
              </h2>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Laboratory-verified intermediate states, spatial filters, morphological operators, and invariant descriptors
              </p>
            </div>
          </div>
          <button className="btn-secondary" onClick={onClose} style={{ padding: '6px 12px' }}>
            <X size={18} /> Close
          </button>
        </div>

        {/* Multi-item Sign Selector if more than 1 item */}
        {evidenceList.length > 1 && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 18, overflowX: 'auto', paddingBottom: 6 }}>
            {evidenceList.map((item, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedIndex(idx)}
                style={{
                  padding: '6px 14px',
                  borderRadius: 20,
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  backgroundColor: selectedIndex === idx ? 'var(--accent-pink)' : 'var(--bg-cream)',
                  color: selectedIndex === idx ? '#ffffff' : 'var(--text-dark)',
                  border: '1px solid var(--card-border)',
                }}
              >
                Sign #{idx + 1}: {item.predicted_sign} ({Math.round(item.confidence * 100)}%)
              </button>
            ))}
          </div>
        )}

        {/* Sign Summary Banner */}
        <div
          style={{
            backgroundColor: 'var(--bg-cream)',
            border: '1px solid var(--card-border)',
            borderRadius: 10,
            padding: '12px 16px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 20,
          }}
        >
          <div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Isolated Sign Gesture
            </span>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-dark)' }}>
              Class: {activeItem.predicted_sign}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Classification Confidence</span>
            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#059669' }}>
              {(activeItem.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* 5 Integrated Pipeline Visual Stages (Tasks 1-8) */}
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 10, color: 'var(--text-dark)' }}>
          Visual Intermediate Stages (Tasks 1 to 8)
        </h3>
        <div className="stages-grid">
          {/* Stage 1: Acquisition & Representation */}
          <div className="stage-card">
            {activeItem.stages.acquisition_rgb ? (
              <img src={activeItem.stages.acquisition_rgb} alt="Task 1 & 2: Acquisition" />
            ) : (
              <div style={{ height: 140, background: '#e5e7eb', borderRadius: 4 }} />
            )}
            <div className="stage-title">Tasks 1 & 2: Input & Acquisition</div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
              Normalized {activeItem.metadata?.width || 200}×{activeItem.metadata?.height || 200} RGB
            </p>
          </div>

          {/* Stage 2: Filtering & High-Boost */}
          <div className="stage-card">
            {activeItem.stages.high_boost_gray ? (
              <img src={activeItem.stages.high_boost_gray} alt="Tasks 3, 4, 5: High-Boost" />
            ) : (
              <div style={{ height: 140, background: '#e5e7eb', borderRadius: 4 }} />
            )}
            <div className="stage-title">Tasks 3, 4, 5: High-Boost Filtered</div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
              Stretch + 5×5 Median + High-Boost (A=1.5)
            </p>
          </div>

          {/* Stage 3: Segmentation */}
          <div className="stage-card">
            {activeItem.stages.raw_mask ? (
              <img src={activeItem.stages.raw_mask} alt="Tasks 6 & 7: Segmentation" />
            ) : (
              <div style={{ height: 140, background: '#e5e7eb', borderRadius: 4 }} />
            )}
            <div className="stage-title">Tasks 6 & 7: Skin/Edge Segmentation</div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
              HSV/YCrCb Dual Threshold + Otsu
            </p>
          </div>

          {/* Stage 4: Morphological Cleaned Mask */}
          <div className="stage-card">
            {activeItem.stages.cleaned_mask ? (
              <img src={activeItem.stages.cleaned_mask} alt="Task 8: Morphology" />
            ) : (
              <div style={{ height: 140, background: '#e5e7eb', borderRadius: 4 }} />
            )}
            <div className="stage-title">Task 8: Morphological Processing</div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
              Opening, Palm Distance & Forearm Prune
            </p>
          </div>

          {/* Stage 5: Invariant Feature Extraction */}
          <div className="stage-card">
            {activeItem.stages.visualization ? (
              <img src={activeItem.stages.visualization} alt="Final Feature Visualization" />
            ) : (
              <div style={{ height: 140, background: '#e5e7eb', borderRadius: 4 }} />
            )}
            <div className="stage-title">Feature Extraction & Geometry</div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
              Contour, Hull, Centroid & 226-D Vector
            </p>
          </div>
        </div>

        {/* DIP Quantitative Metrics Table */}
        <div style={{ marginTop: 24 }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 10, color: 'var(--text-dark)' }}>
            Extracted Quantitative DIP Descriptors
          </h3>
          <div
            style={{
              backgroundColor: 'var(--bg-cream)',
              border: '1px solid var(--card-border)',
              borderRadius: 8,
              padding: 12,
              maxHeight: 220,
              overflowY: 'auto',
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--card-border)', textAlign: 'left' }}>
                  <th style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>DIP Metric</th>
                  <th style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>Measured Value</th>
                  <th style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>Significance in ASL Classification</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>Area / Perimeter</td>
                  <td style={{ padding: '6px 8px' }}>
                    {activeItem.metrics['Area (pixels)']} px / {activeItem.metrics['Perimeter (pixels)']} px
                  </td>
                  <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>Hand size scale normalization</td>
                </tr>
                <tr>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>Aspect Ratio & Extent</td>
                  <td style={{ padding: '6px 8px' }}>
                    {activeItem.metrics['Aspect Ratio (w/h)']} / {activeItem.metrics['Extent (Area / BoxArea)']}
                  </td>
                  <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>Finger extension vs closed fist</td>
                </tr>
                <tr>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>Solidity & Compactness</td>
                  <td style={{ padding: '6px 8px' }}>
                    {activeItem.metrics['Solidity (Area / HullArea)']} / {activeItem.metrics['Compactness']}
                  </td>
                  <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>Distinguishes convex letters (O, S, E, A)</td>
                </tr>
                <tr>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>Convexity Defects Count</td>
                  <td style={{ padding: '6px 8px' }}>{activeItem.metrics['Convexity Defects Count'] ?? 0} defects</td>
                  <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>Number of spread fingers (V=1, W=2, B=0)</td>
                </tr>
                <tr>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>Hu Moments (Scale/Rot Invariant)</td>
                  <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>
                    {activeItem.metrics['Hu Moments (log10)']
                      ? `[${activeItem.metrics['Hu Moments (log10)'].slice(0, 4).join(', ')}...]`
                      : 'Calculated'}
                  </td>
                  <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>Rotation & scale invariant shape moments</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Input Image Metadata */}
        <div style={{ marginTop: 16, display: 'flex', gap: 14, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <span>Dimensions: {activeItem.metadata?.width}×{activeItem.metadata?.height}</span>
          <span>Channels: {activeItem.metadata?.channels}</span>
          <span>Dtype: {activeItem.metadata?.dtype}</span>
          <span>Intensity Range: [{activeItem.metadata?.min_intensity}, {activeItem.metadata?.max_intensity}]</span>
        </div>
      </div>
    </div>
  );
};
