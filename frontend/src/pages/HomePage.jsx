import React, { useState, useEffect } from 'react';
import { panelAPI, getPanelImageUrl } from '../api/client';
import PanelUpload from '../components/PanelUpload';
import PanelViewer from '../components/PanelViewer';
import '../styles/HomePage.css';

export default function HomePage() {
  const [panels, setPanels] = useState([]);
  const [selectedPanel, setSelectedPanel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [serviceStatus, setServiceStatus] = useState(null);

  useEffect(() => {
    loadPanels();
    loadServiceStatus();
  }, []);

  const loadPanels = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await panelAPI.listPanels();
      if (response.data.success) {
        setPanels(response.data.panels || []);
      }
    } catch (err) {
      setError('Failed to load panels');
    } finally {
      setLoading(false);
    }
  };

  const loadServiceStatus = async () => {
    try {
      const response = await panelAPI.getStatus();
      setServiceStatus(response.data);
    } catch (err) {
      console.error('Failed to load service status');
    }
  };

  const handleDeletePanel = async (filename) => {
    if (!confirm(`Delete ${filename}?`)) return;

    try {
      await panelAPI.deletePanel(filename);
      alert('Panel deleted');
      loadPanels();
      setSelectedPanel(null);
    } catch (err) {
      alert('Failed to delete panel');
    }
  };

  return (
    <div className="home-page">
      {/* Header */}
      <header className="app-header">
        <div className="container">
          <h1>📚 Manga Language Learner</h1>
          <p>Learn Japanese through manga panels</p>
        </div>
      </header>

      {/* Service Status */}
      {serviceStatus && (
        <div className="service-status">
          <div className="status-item">
            <span className={serviceStatus.ocr.available ? 'status-ok' : 'status-warn'}>
              🔤 OCR: {serviceStatus.ocr.ocr_service}
            </span>
          </div>
          <div className="status-item">
            <span className={serviceStatus.translation.available ? 'status-ok' : 'status-warn'}>
              🌐 Translation: {serviceStatus.translation.translation_service}
            </span>
          </div>
        </div>
      )}

      <div className="container">
        <div className="main-content">
          {/* Left: Panel List */}
          <aside className="panel-list-sidebar">
            <PanelUpload onUploadSuccess={loadPanels} />

            <div className="panels-section">
              <h2>📖 Panels ({panels.length})</h2>
              
              {loading && <p className="loading">Loading panels...</p>}
              {error && <p className="error">{error}</p>}
              
              {panels.length === 0 && !loading && (
                <p className="no-panels">No panels yet. Upload one to start!</p>
              )}

              <div className="panels-grid">
                {panels.map((panel) => (
                  <div 
                    key={panel.filename}
                    className={`panel-card ${selectedPanel?.filename === panel.filename ? 'active' : ''}`}
                    onClick={() => setSelectedPanel(panel)}
                  >
                    <div className="panel-card-thumb">
                      <img src={getPanelImageUrl(panel.path)} alt={panel.filename} />
                    </div>
                    <div className="panel-card-info">
                      <p className="panel-name">{panel.filename}</p>
                      <p className="panel-size">{(panel.size / 1024).toFixed(0)} KB</p>
                    </div>
                    <button 
                      className="delete-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeletePanel(panel.filename);
                      }}
                    >
                      🗑️
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </aside>

          {/* Right: Panel Viewer */}
          <main className="viewer-section">
            {selectedPanel ? (
              <PanelViewer 
                filename={selectedPanel.filename}
                panelPath={selectedPanel.path}
              />
            ) : (
              <div className="empty-state">
                <div className="empty-icon">📖</div>
                <h2>Select a panel to start</h2>
                <p>Choose a manga panel from the left to extract text and get translations</p>
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
