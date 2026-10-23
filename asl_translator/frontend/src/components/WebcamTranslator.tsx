import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Video, Sparkles, RefreshCw, Sliders, Info } from 'lucide-react';
import { translateFrame } from '../api';
import { TranslationResponse } from '../types';

interface WebcamTranslatorProps {
  onTranslateStart: (message: string) => void;
  onTranslateSuccess: (response: TranslationResponse, summary: string, mode: 'webcam') => void;
  onTranslateError: (error: string) => void;
  disabled: boolean;
}

export const WebcamTranslator: React.FC<WebcamTranslatorProps> = ({
  onTranslateStart,
  onTranslateSuccess,
  onTranslateError,
  disabled,
}) => {
  const [isActive, setIsActive] = useState<boolean>(false);
  const [isContinuous, setIsContinuous] = useState<boolean>(false);
  const [framingPreset, setFramingPreset] = useState<string>('right_hand');
  const [boxSize, setBoxSize] = useState<number>(240);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const continuousTimerRef = useRef<number | null>(null);
  const isProcessingRef = useRef<boolean>(false);

  // Start webcam
  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsActive(true);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setCameraError('Unable to access webcam. Please allow camera permissions in your browser.');
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (continuousTimerRef.current) {
      clearInterval(continuousTimerRef.current);
      continuousTimerRef.current = null;
    }
    setIsContinuous(false);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Capture current frame from video to Base64
  const captureFrameBase64 = (): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Draw video to canvas (unmirrored for consistency)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  };

  // Capture and Translate single frame
  const handleCapture = useCallback(
    async (isAuto = false) => {
      if (isProcessingRef.current) return;
      const base64 = captureFrameBase64();
      if (!base64) return;

      isProcessingRef.current = true;
      if (!isAuto) {
        onTranslateStart('Isolating hand within viewfinder box & processing DIP pipeline...');
      }

      try {
        const response = await translateFrame(base64, 'target_box', framingPreset, boxSize);
        onTranslateSuccess(
          response,
          `Webcam Snapshot (${response.characters?.[0] || response.translated_sentence})`,
          'webcam'
        );
      } catch (err: any) {
        if (!isAuto) {
          onTranslateError(err.message || 'Camera translation failed.');
        }
      } finally {
        isProcessingRef.current = false;
      }
    },
    [framingPreset, boxSize, onTranslateStart, onTranslateSuccess, onTranslateError]
  );

  // Manage continuous auto-translation
  useEffect(() => {
    if (isContinuous && isActive) {
      continuousTimerRef.current = window.setInterval(() => {
        handleCapture(true);
      }, 1400);
    } else {
      if (continuousTimerRef.current) {
        clearInterval(continuousTimerRef.current);
        continuousTimerRef.current = null;
      }
    }

    return () => {
      if (continuousTimerRef.current) {
        clearInterval(continuousTimerRef.current);
        continuousTimerRef.current = null;
      }
    };
  }, [isContinuous, isActive, handleCapture]);

  // Viewfinder box styles depending on preset
  const getViewfinderStyle = () => {
    if (framingPreset === 'right_hand') {
      return { top: '15%', right: '8%', width: '38%', height: '52%' };
    } else if (framingPreset === 'left_hand') {
      return { top: '15%', left: '8%', width: '38%', height: '52%' };
    }
    // center
    return { top: '18%', left: '31%', width: '38%', height: '52%' };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Video Viewport with Viewfinder Overlay */}
      <div
        style={{
          position: 'relative',
          width: '100%',
          height: 300,
          backgroundColor: '#1f1b1c',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <video
          ref={videoRef}
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: isActive ? 'block' : 'none',
          }}
        />

        {!isActive && (
          <div style={{ textAlign: 'center', color: '#a8a29e', padding: 20 }}>
            <Camera size={44} style={{ margin: '0 auto 10px', color: 'var(--accent-teal)' }} />
            <p style={{ fontWeight: 600, fontSize: '0.95rem', color: '#f5f5f4' }}>Webcam is Off</p>
            <p style={{ fontSize: '0.8rem', marginTop: 4 }}>
              Click Start Camera to recognize hand signs in real-time.
            </p>
          </div>
        )}

        {/* Viewfinder Target Box Overlay (Active when camera is running) */}
        {isActive && (
          <div
            style={{
              position: 'absolute',
              ...getViewfinderStyle(),
              border: '2px solid var(--accent-pink)',
              borderRadius: 10,
              boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.45)',
              pointerEvents: 'none',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              padding: 6,
            }}
          >
            <span
              style={{
                fontSize: '0.68rem',
                fontWeight: 700,
                color: '#ffffff',
                backgroundColor: 'var(--accent-pink)',
                padding: '2px 6px',
                borderRadius: 4,
                width: 'fit-content',
                letterSpacing: '0.04em',
              }}
            >
              PLACE HAND HERE
            </span>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                color: 'var(--accent-pink)',
                fontSize: '0.7rem',
              }}
            >
              <span>+</span>
              <span>+</span>
            </div>
          </div>
        )}
      </div>

      {/* Error message */}
      {cameraError && (
        <div
          style={{
            backgroundColor: '#fee2e2',
            border: '1px solid #f87171',
            color: '#b91c1c',
            padding: '8px 12px',
            borderRadius: 8,
            fontSize: '0.82rem',
          }}
        >
          {cameraError}
        </div>
      )}

      {/* Camera Controls Bar */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {!isActive ? (
          <button className="btn-primary" onClick={startCamera} style={{ flex: 1 }}>
            <Camera size={18} />
            <span>Start Camera</span>
          </button>
        ) : (
          <>
            <button
              className="btn-primary"
              onClick={() => handleCapture(false)}
              disabled={disabled}
              style={{ flex: 1 }}
            >
              <Sparkles size={18} />
              <span>Capture & Translate</span>
            </button>
            <button
              className={`btn-secondary ${isContinuous ? 'active' : ''}`}
              onClick={() => setIsContinuous(!isContinuous)}
              style={{
                backgroundColor: isContinuous ? 'var(--accent-teal)' : 'var(--bg-cream)',
                borderColor: isContinuous ? 'var(--accent-teal-dark)' : 'var(--card-border)',
                fontWeight: 600,
              }}
            >
              <RefreshCw size={16} />
              <span>{isContinuous ? 'Live: ON' : 'Live: OFF'}</span>
            </button>
            <button className="btn-secondary" onClick={stopCamera}>
              <CameraOff size={18} />
              <span>Stop</span>
            </button>
          </>
        )}
      </div>

      {/* Framing Controls for Higher Accuracy */}
      {isActive && (
        <div
          style={{
            backgroundColor: 'var(--bg-cream)',
            border: '1px solid var(--card-border)',
            borderRadius: 8,
            padding: '10px 14px',
            fontSize: '0.82rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontWeight: 600, color: 'var(--text-dark)' }}>Hand Viewfinder Position:</span>
            <div style={{ display: 'flex', gap: 6 }}>
              {[
                { id: 'right_hand', label: 'Right Hand' },
                { id: 'center', label: 'Center' },
                { id: 'left_hand', label: 'Left Hand' },
              ].map((pos) => (
                <button
                  key={pos.id}
                  onClick={() => setFramingPreset(pos.id)}
                  style={{
                    padding: '3px 8px',
                    borderRadius: 4,
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    backgroundColor: framingPreset === pos.id ? 'var(--accent-pink)' : '#ffffff',
                    color: framingPreset === pos.id ? '#ffffff' : 'var(--text-dark)',
                    border: '1px solid var(--card-border)',
                  }}
                >
                  {pos.label}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', fontSize: '0.75rem' }}>
            <Info size={13} />
            <span>Positioning your hand inside the target box eliminates background noise for accurate DIP contouring.</span>
          </div>
        </div>
      )}
    </div>
  );
};
