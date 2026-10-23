import React from 'react';
import { Languages, CheckCircle2, AlertCircle } from 'lucide-react';

interface HeaderProps {
  isOnline: boolean;
  vocabSize: number;
}

export const Header: React.FC<HeaderProps> = ({ isOnline, vocabSize }) => {
  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon">
          <Languages size={26} strokeWidth={2.2} />
        </div>
        <div>
          <h1 className="brand-title">SignTranslate</h1>
          <p className="brand-tagline">
            ASL-to-English Sign Language Translator using Digital Image Processing
          </p>
        </div>
      </div>

      <div className="status-badge">
        <span className={`status-dot ${isOnline ? '' : 'offline'}`} />
        <span>{isOnline ? `Model Ready (${vocabSize} Signs)` : 'Connecting Engine...'}</span>
      </div>
    </header>
  );
};
