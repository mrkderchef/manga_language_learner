import React, { useState, useRef } from 'react';
import { panelAPI } from '../api/client';
import '../styles/PanelUpload.css';

export default function PanelUpload({ onUploadSuccess }) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    
    // Validate file type
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setError('Please upload a JPG or PNG image');
      return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      setError('File is too large (max 10MB)');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const response = await panelAPI.uploadPanel(file);
      if (response.data.success) {
        alert('Panel uploaded successfully!');
        onUploadSuccess();
        fileInputRef.current.value = '';
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="panel-upload">
      <h2>📤 Upload Manga Panel</h2>
      
      <div className="upload-area">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleFileSelect}
          disabled={uploading}
          className="file-input"
        />
        
        <label htmlFor="file-input" className="upload-label">
          {uploading ? 'Uploading...' : 'Click to select or drag and drop'}
        </label>
      </div>

      {error && <div className="error-message">{error}</div>}
    </div>
  );
}
