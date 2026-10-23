import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { TranslationWorkspace } from './components/TranslationWorkspace';
import { fetchHealth } from './api';

export const App: React.FC = () => {
  const [isOnline, setIsOnline] = useState<boolean>(false);
  const [vocabSize, setVocabSize] = useState<number>(29);

  useEffect(() => {
    let mounted = true;

    const checkEngine = async () => {
      try {
        const health = await fetchHealth();
        if (mounted) {
          setIsOnline(health.status === 'healthy' && health.model_loaded);
          if (health.vocabulary_size) setVocabSize(health.vocabulary_size);
        }
      } catch (e) {
        if (mounted) setIsOnline(false);
      }
    };

    checkEngine();
    const interval = setInterval(checkEngine, 8000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="app-container">
      <Header isOnline={isOnline} vocabSize={vocabSize} />

      <main style={{ flex: 1 }}>
        <TranslationWorkspace />
      </main>

      <footer
        style={{
          marginTop: 36,
          paddingTop: 16,
          borderTop: '1px solid var(--card-border)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 12,
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
        }}
      >
        <div>
          <strong>SignTranslate</strong> • Digital Image Processing & Computer Vision Lab Project
        </div>
        <div>
          Powered by DIP Pipeline: Tasks 1–8 (Acquisition, Enhancement, Filtering, Segmentation, Morphology & Invariant Geometry)
        </div>
      </footer>
    </div>
  );
};

export default App;
