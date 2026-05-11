import axios from 'axios';

const API_BASE_URL = '/api';
const BACKEND_URL = typeof window !== 'undefined' && window.location.hostname === 'localhost' 
  ? 'http://localhost:8000'
  : '';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Helper to convert panel paths to absolute URLs
export const getPanelImageUrl = (panelPath) => {
  if (!panelPath) return '';
  if (panelPath.startsWith('http')) return panelPath;
  if (panelPath.startsWith('/')) return `${BACKEND_URL}${panelPath}`;
  return panelPath;
};

// Panel APIs
export const panelAPI = {
  listPanels: () => apiClient.get('/panels/list'),
  uploadPanel: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/panels/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  deletePanel: (filename) => apiClient.delete(`/panels/${filename}`),
  extractText: (filename) => apiClient.post(`/panels/${filename}/ocr`),
  extractAndTranslate: (filename) => apiClient.post(`/panels/${filename}/extract-and-translate`),
  translateText: (text, targetLanguage = 'en') => 
    apiClient.post('/panels/translate', null, { params: { text, target_language: targetLanguage } }),
  getStatus: () => apiClient.get('/panels/status'),
};

export default apiClient;
