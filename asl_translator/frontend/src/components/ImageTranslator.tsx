import React, { useState, useRef } from 'react';
import { Upload, X, ArrowRight, Info, ChevronLeft, ChevronRight, ArrowLeftRight } from 'lucide-react';
import { translateImages } from '../api';
import { TranslationResponse } from '../types';

interface ImageTranslatorProps {
  onTranslateStart: (message: string) => void;
  onTranslateSuccess: (response: TranslationResponse, summary: string, mode: 'image') => void;
  onTranslateError: (error: string) => void;
  disabled: boolean;
}

export const ImageTranslator: React.FC<ImageTranslatorProps> = ({
  onTranslateStart,
  onTranslateSuccess,
  onTranslateError,
  disabled,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState<boolean>(false);
  const [draggedCardIndex, setDraggedCardIndex] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFiles = (files: FileList | File[]) => {
    const validFiles: File[] = [];
    const newPreviews: string[] = [];

    Array.from(files).forEach((file) => {
      if (file.type.startsWith('image/')) {
        validFiles.push(file);
        newPreviews.push(URL.createObjectURL(file));
      }
    });

    if (validFiles.length === 0) return;

    setSelectedFiles((prev) => [...prev, ...validFiles]);
    setPreviewUrls((prev) => [...prev, ...newPreviews]);
  };

  const removeFile = (idx: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx));
    setPreviewUrls((prev) => {
      URL.revokeObjectURL(prev[idx]);
      return prev.filter((_, i) => i !== idx);
    });
  };

  const clearAllFiles = () => {
    previewUrls.forEach((url) => URL.revokeObjectURL(url));
    setSelectedFiles([]);
    setPreviewUrls([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Reorder: Move photo left or right
  const moveFile = (fromIdx: number, toIdx: number) => {
    if (toIdx < 0 || toIdx >= selectedFiles.length) return;

    setSelectedFiles((prevFiles) => {
      const updated = [...prevFiles];
      const [item] = updated.splice(fromIdx, 1);
      updated.splice(toIdx, 0, item);
      return updated;
    });

    setPreviewUrls((prevPreviews) => {
      const updated = [...prevPreviews];
      const [item] = updated.splice(fromIdx, 1);
      updated.splice(toIdx, 0, item);
      return updated;
    });
  };

  // Reverse sequence
  const reverseOrder = () => {
    setSelectedFiles((prev) => [...prev].reverse());
    setPreviewUrls((prev) => [...prev].reverse());
  };

  // Drag and drop reordering between cards
  const handleCardDragStart = (e: React.DragEvent, idx: number) => {
    setDraggedCardIndex(idx);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleCardDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleCardDrop = (e: React.DragEvent, targetIdx: number) => {
    e.preventDefault();
    if (draggedCardIndex === null || draggedCardIndex === targetIdx) return;
    moveFile(draggedCardIndex, targetIdx);
    setDraggedCardIndex(null);
  };

  const handleTranslateFiles = async () => {
    if (selectedFiles.length === 0 || disabled) return;

    onTranslateStart(`Processing ${selectedFiles.length} ordered image(s) through DIP enhancement and classification...`);
    try {
      const response = await translateImages(selectedFiles);
      const summary = selectedFiles.length === 1 ? selectedFiles[0].name : `${selectedFiles.length} ordered photos`;
      onTranslateSuccess(response, summary, 'image');
    } catch (err: any) {
      onTranslateError(err.message || 'Image translation failed.');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Dropzone */}
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
          if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files);
          }}
        />
        <div className="dropzone-icon">
          <Upload size={38} />
        </div>
        <div className="dropzone-title">Upload sign image(s)</div>
        <div className="dropzone-desc">Drag and drop photos or click to browse (PNG, JPG, BMP)</div>
      </div>

      {/* Selected & Ordered Image Previews */}
      {selectedFiles.length > 0 ? (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div>
              <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-dark)' }}>
                Arranged Photo Sequence ({selectedFiles.length})
              </span>
              <p style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                Arrange in the order you want the word/phrase to be spelled (Left to Right).
              </p>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {selectedFiles.length > 1 && (
                <button
                  onClick={reverseOrder}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    fontSize: '0.76rem',
                    color: 'var(--text-dark)',
                    backgroundColor: 'var(--bg-cream)',
                    border: '1px solid var(--card-border)',
                    padding: '3px 8px',
                    borderRadius: 4,
                    fontWeight: 600,
                  }}
                  title="Reverse sequence order"
                >
                  <ArrowLeftRight size={12} />
                  <span>Reverse</span>
                </button>
              )}
              <button
                onClick={clearAllFiles}
                style={{ fontSize: '0.78rem', color: '#dc2626', fontWeight: 600 }}
              >
                Clear all
              </button>
            </div>
          </div>

          {/* Cards container with drag-to-reorder and arrow controls */}
          <div
            style={{
              display: 'flex',
              gap: 12,
              overflowX: 'auto',
              padding: '8px 2px 12px',
            }}
          >
            {previewUrls.map((url, idx) => {
              const isFirst = idx === 0;
              const isLast = idx === previewUrls.length - 1;
              const isBeingDragged = draggedCardIndex === idx;

              return (
                <div
                  key={`${url}_${idx}`}
                  draggable
                  onDragStart={(e) => handleCardDragStart(e, idx)}
                  onDragOver={handleCardDragOver}
                  onDrop={(e) => handleCardDrop(e, idx)}
                  style={{
                    position: 'relative',
                    width: 96,
                    flexShrink: 0,
                    borderRadius: 8,
                    overflow: 'hidden',
                    border: isBeingDragged ? '2px dashed var(--accent-pink)' : '1px solid var(--card-border)',
                    backgroundColor: 'var(--card-bg)',
                    boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
                    cursor: 'grab',
                    opacity: isBeingDragged ? 0.4 : 1.0,
                    transition: 'border-color 0.15s ease',
                  }}
                >
                  {/* Sequence Position Badge (1, 2, 3...) */}
                  <div
                    style={{
                      position: 'absolute',
                      top: 4,
                      left: 4,
                      zIndex: 2,
                      backgroundColor: 'var(--accent-pink)',
                      color: '#ffffff',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      width: 20,
                      height: 20,
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                    }}
                  >
                    {idx + 1}
                  </div>

                  {/* Remove Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(idx);
                    }}
                    style={{
                      position: 'absolute',
                      top: 4,
                      right: 4,
                      zIndex: 2,
                      background: 'rgba(0,0,0,0.65)',
                      color: '#fff',
                      borderRadius: '50%',
                      width: 18,
                      height: 18,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                    title="Remove this photo"
                  >
                    <X size={12} />
                  </button>

                  {/* Image Thumbnail */}
                  <div style={{ width: '100%', height: 96, backgroundColor: '#f0ede8' }}>
                    <img
                      src={url}
                      alt={`ordered-sign-${idx}`}
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                  </div>

                  {/* Reorder Arrows Bar */}
                  {selectedFiles.length > 1 && (
                    <div
                      style={{
                        display: 'flex',
                        borderTop: '1px solid var(--card-border)',
                        backgroundColor: 'var(--bg-cream)',
                      }}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          moveFile(idx, idx - 1);
                        }}
                        disabled={isFirst}
                        style={{
                          flex: 1,
                          padding: '4px 0',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: isFirst ? '#cbd5e1' : 'var(--text-dark)',
                          cursor: isFirst ? 'not-allowed' : 'pointer',
                        }}
                        title="Move Left (Earlier in word)"
                      >
                        <ChevronLeft size={14} />
                      </button>

                      <div style={{ width: 1, backgroundColor: 'var(--card-border)' }} />

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          moveFile(idx, idx + 1);
                        }}
                        disabled={isLast}
                        style={{
                          flex: 1,
                          padding: '4px 0',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: isLast ? '#cbd5e1' : 'var(--text-dark)',
                          cursor: isLast ? 'not-allowed' : 'pointer',
                        }}
                        title="Move Right (Later in word)"
                      >
                        <ChevronRight size={14} />
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Sequence Flow Helper */}
          {selectedFiles.length > 1 && (
            <div
              style={{
                fontSize: '0.74rem',
                color: 'var(--text-muted)',
                backgroundColor: 'var(--bg-cream)',
                borderRadius: 6,
                padding: '6px 10px',
                marginBottom: 12,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <Info size={13} color="var(--accent-teal-dark)" />
              <span>
                Tip: Drag photos or click <strong>←</strong> / <strong>→</strong> to arrange their order before translating.
              </span>
            </div>
          )}

          <button
            className="btn-primary"
            onClick={handleTranslateFiles}
            disabled={disabled}
            style={{ width: '100%', marginTop: selectedFiles.length > 1 ? 0 : 10 }}
          >
            <span>Translate {selectedFiles.length} Ordered Image(s)</span>
            <ArrowRight size={18} />
          </button>
        </div>
      ) : (
        <div
          style={{
            backgroundColor: 'var(--bg-cream)',
            border: '1px solid var(--card-border)',
            borderRadius: 8,
            padding: '12px 14px',
            fontSize: '0.82rem',
            color: 'var(--text-muted)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
          }}
        >
          <Info size={16} color="var(--accent-pink)" style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <strong>How to translate:</strong>
            <p style={{ marginTop: 2 }}>
              Upload one photo of an ASL hand gesture, or upload multiple photos to spell a word. You can drag and arrange the uploaded photos to ensure the letters are in the correct sequence before translating.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
