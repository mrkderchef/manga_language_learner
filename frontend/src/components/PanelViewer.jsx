import React, { useState } from 'react';
import { panelAPI, getPanelImageUrl } from '../api/client';
import '../styles/PanelViewer.css';

export default function PanelViewer({ filename, panelPath }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('translate'); // 'translate' or 'ocr'

  const handleExtractAndTranslate = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await panelAPI.extractAndTranslate(filename);
      if (response.data.success) {
        setResult(response.data);
      } else {
        setError("OCR service not available. Try uploading sample text instead.");
      }
    } catch (err) {
      setError("OCR service not available. Try uploading sample text instead.");
    } finally {
      setLoading(false);
    }
  };

  const handleExtractOnly = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await panelAPI.extractText(filename);
      if (response.data.success) {
        setResult(response.data);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to extract text');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel-viewer">
      <div className="panel-header">
        <h3>📖 {filename}</h3>
        <div className="mode-selector">
          <button 
            className={mode === 'ocr' ? 'active' : ''} 
            onClick={() => setMode('ocr')}
          >
            🔤 Extract Text
          </button>
          <button 
            className={mode === 'translate' ? 'active' : ''} 
            onClick={() => setMode('translate')}
          >
            🌐 Translate
          </button>
        </div>
      </div>

      <div className="panel-image">
        <img src={getPanelImageUrl(panelPath)} alt={filename} />
      </div>

      <div className="panel-actions">
        {mode === 'ocr' ? (
          <button 
            onClick={handleExtractOnly} 
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? '⏳ Extracting...' : '🔤 Extract Text'}
          </button>
        ) : (
          <button 
            onClick={handleExtractAndTranslate} 
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? '⏳ Processing...' : '🌐 Extract & Translate'}
          </button>
        )}
      </div>

      {error && <div className="error-message">❌ {error}</div>}

      {result && (
        <div className="result-panel">
          <div className="result-tabs">
            <div className="tab">
              <h4>📝 Original Text</h4>
              <div className="text-content">
                {result.text || result.original_text || 'No text extracted'}
              </div>
            </div>

            {result.translated_text && (
              <div className="tab">
                <h4>🌐 Translated Text</h4>
                <div className="text-content">
                  {result.translated_text}
                </div>
              </div>
            )}

            {result.annotations && result.annotations.length > 0 && (
              <div className="tab">
                <h4>📊 Word-by-Word</h4>
                <div className="annotations-list">
                  {result.annotations.map((ann, idx) => (
                    <div key={idx} className="annotation-item">
                      <span className="japanese">{ann.text}</span>
                      {ann.translation && (
                        <span className="english">{ann.translation}</span>
                      )}
                      {ann.confidence && (
                        <span className="confidence">{(ann.confidence * 100).toFixed(0)}%</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
