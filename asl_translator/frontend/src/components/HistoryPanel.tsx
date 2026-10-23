import React from 'react';
import { History, Trash2, Camera, Image as ImageIcon, Video, ArrowRight } from 'lucide-react';
import { HistoryItem } from '../types';

interface HistoryPanelProps {
  history: HistoryItem[];
  onClearHistory: () => void;
  onSelectHistoryItem: (item: HistoryItem) => void;
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({
  history,
  onClearHistory,
  onSelectHistoryItem,
}) => {
  if (history.length === 0) return null;

  const getModeIcon = (mode: string) => {
    switch (mode) {
      case 'webcam':
        return <Camera size={14} color="var(--accent-pink)" />;
      case 'video':
        return <Video size={14} color="var(--accent-teal-dark)" />;
      default:
        return <ImageIcon size={14} color="#6366f1" />;
    }
  };

  return (
    <div className="panel-card" style={{ marginTop: 20 }}>
      <div className="panel-header">
        <div className="panel-title">
          <History size={18} color="var(--text-muted)" />
          <span>Recent Translations ({history.length})</span>
        </div>
        <button
          onClick={onClearHistory}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            fontSize: '0.78rem',
            color: 'var(--text-muted)',
            fontWeight: 600,
          }}
          title="Clear all recent translation history"
        >
          <Trash2 size={13} />
          <span>Clear History</span>
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 10 }}>
        {history.slice(0, 8).map((item) => (
          <div
            key={item.id}
            onClick={() => onSelectHistoryItem(item)}
            style={{
              padding: '10px 14px',
              borderRadius: 8,
              border: '1px solid var(--card-border)',
              backgroundColor: 'var(--bg-cream)',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseOver={(e) => {
              (e.currentTarget as HTMLElement).style.borderColor = 'var(--accent-pink)';
            }}
            onMouseOut={(e) => {
              (e.currentTarget as HTMLElement).style.borderColor = 'var(--card-border)';
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.75rem', fontWeight: 600 }}>
                {getModeIcon(item.mode)}
                <span style={{ textTransform: 'capitalize' }}>{item.mode}</span>
              </div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{item.timestamp}</span>
            </div>

            <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-dark)', lineHeight: 1.25 }}>
              {item.translatedText}
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: 6,
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
              }}
            >
              <span>{item.summary}</span>
              <span style={{ fontWeight: 600, color: '#059669' }}>
                {Math.round(item.confidence * 100)}%
              </span>
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: 12,
          fontSize: '0.72rem',
          color: 'var(--text-muted)',
          textAlign: 'right',
        }}
      >
        History is preserved locally in your browser session for privacy.
      </div>
    </div>
  );
};
