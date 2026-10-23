import React, { useState, useEffect } from 'react';
import { Image as ImageIcon, Camera, Video, AlertCircle } from 'lucide-react';
import { ImageTranslator } from './ImageTranslator';
import { WebcamTranslator } from './WebcamTranslator';
import { VideoTranslator } from './VideoTranslator';
import { OutputPanel } from './OutputPanel';
import { HistoryPanel } from './HistoryPanel';
import { LabVerificationModal } from './LabVerificationModal';
import { TranslationResponse, HistoryItem } from '../types';

type Mode = 'image' | 'webcam' | 'video';

const HISTORY_STORAGE_KEY = 'sign_translate_history_v1';

export const TranslationWorkspace: React.FC = () => {
  const [mode, setMode] = useState<Mode>('image');
  const [response, setResponse] = useState<TranslationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('Processing Digital Image Pipeline...');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // Load history from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (saved) {
        setHistory(JSON.parse(saved));
      }
    } catch (e) {
      console.warn('Could not read history from localStorage', e);
    }
  }, []);

  // Save history to localStorage
  const saveToHistory = (res: TranslationResponse, summary: string, inputMode: Mode) => {
    const newItem: HistoryItem = {
      id: `${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      mode: inputMode,
      summary,
      translatedText: res.translated_sentence,
      confidence: res.confidence,
      characters: res.characters || res.distinct_signs || [],
    };

    setHistory((prev) => {
      const updated = [newItem, ...prev.slice(0, 19)];
      try {
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(updated));
      } catch (e) {
        console.warn('Could not save history to localStorage', e);
      }
      return updated;
    });
  };

  const clearHistory = () => {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
    setHistory([]);
  };

  const handleTranslateStart = (message: string) => {
    setLoading(true);
    setLoadingMessage(message);
    setErrorMessage(null);
  };

  const handleTranslateSuccess = (res: TranslationResponse, summary: string, inputMode: Mode) => {
    setLoading(false);
    setResponse(res);
    setErrorMessage(null);
    saveToHistory(res, summary, inputMode);
  };

  const handleTranslateError = (error: string) => {
    setLoading(false);
    setErrorMessage(error);
  };

  const handleClearOutput = () => {
    setResponse(null);
    setErrorMessage(null);
  };

  const handleSelectHistoryItem = (item: HistoryItem) => {
    setResponse({
      translated_sentence: item.translatedText,
      confidence: item.confidence,
      characters: item.characters,
    });
  };

  return (
    <>
      {/* Error alert banner if any */}
      {errorMessage && (
        <div
          style={{
            backgroundColor: '#fee2e2',
            border: '1px solid #f87171',
            color: '#b91c1c',
            borderRadius: 'var(--radius-md)',
            padding: '12px 16px',
            marginBottom: 20,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            fontSize: '0.9rem',
          }}
        >
          <AlertCircle size={20} />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Main 2-Column Workspace */}
      <div className="workspace-grid">
        {/* Left Input Panel */}
        <div className="panel-card">
          {/* Segmented Mode Control */}
          <div className="mode-tabs">
            <button
              className={`tab-btn ${mode === 'image' ? 'active' : ''}`}
              onClick={() => {
                setMode('image');
                setErrorMessage(null);
              }}
            >
              <ImageIcon size={18} />
              <span>Image</span>
            </button>
            <button
              className={`tab-btn ${mode === 'webcam' ? 'active' : ''}`}
              onClick={() => {
                setMode('webcam');
                setErrorMessage(null);
              }}
            >
              <Camera size={18} />
              <span>Webcam</span>
            </button>
            <button
              className={`tab-btn ${mode === 'video' ? 'active' : ''}`}
              onClick={() => {
                setMode('video');
                setErrorMessage(null);
              }}
            >
              <Video size={18} />
              <span>Video</span>
            </button>
          </div>

          {/* Active Mode Input Area */}
          <div style={{ flex: 1 }}>
            {mode === 'image' && (
              <ImageTranslator
                onTranslateStart={handleTranslateStart}
                onTranslateSuccess={handleTranslateSuccess}
                onTranslateError={handleTranslateError}
                disabled={loading}
              />
            )}
            {mode === 'webcam' && (
              <WebcamTranslator
                onTranslateStart={handleTranslateStart}
                onTranslateSuccess={handleTranslateSuccess}
                onTranslateError={handleTranslateError}
                disabled={loading}
              />
            )}
            {mode === 'video' && (
              <VideoTranslator
                onTranslateStart={handleTranslateStart}
                onTranslateSuccess={handleTranslateSuccess}
                onTranslateError={handleTranslateError}
                disabled={loading}
              />
            )}
          </div>
        </div>

        {/* Right Output Panel */}
        <div>
          <OutputPanel
            response={response}
            loading={loading}
            loadingMessage={loadingMessage}
            onClear={handleClearOutput}
            onOpenModal={() => setIsModalOpen(true)}
          />
        </div>
      </div>

      {/* Recent History Panel */}
      <HistoryPanel
        history={history}
        onClearHistory={clearHistory}
        onSelectHistoryItem={handleSelectHistoryItem}
      />

      {/* Lab Verification Modal for DIP Tasks 1-8 inspection */}
      {isModalOpen && response && (
        <LabVerificationModal
          response={response}
          onClose={() => setIsModalOpen(false)}
        />
      )}
    </>
  );
};
