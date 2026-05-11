/**
 * API Client - Kommunikation mit dem Backend
 */
const API = (() => {
    const BASE_URL = 'http://localhost:8000';

    async function request(endpoint, options = {}) {
        try {
            const response = await fetch(`${BASE_URL}${endpoint}`, {
                headers: { 'Content-Type': 'application/json' },
                ...options,
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || 'Request failed');
            }
            return await response.json();
        } catch (err) {
            console.error(`API Error [${endpoint}]:`, err);
            throw err;
        }
    }

    return {
        // === Scanner Endpoints ===
        listPanels() {
            return request('/api/scanner/panels');
        },

        uploadPanel(file) {
            const formData = new FormData();
            formData.append('file', file);
            return fetch(`${BASE_URL}/api/scanner/upload`, {
                method: 'POST',
                body: formData,
            }).then(r => r.json());
        },

        scanPanel(filename) {
            return request(`/api/scanner/${filename}/ocr`, { method: 'POST' });
        },

        scanAndTranslate(filename) {
            return request(`/api/scanner/${filename}/scan-translate`, { method: 'POST' });
        },

        translateText(text) {
            return request('/api/scanner/translate', {
                method: 'POST',
                body: JSON.stringify({ text }),
            });
        },

        // === Learning Endpoints ===
        getLearningPanels() {
            return request('/api/learning/panels');
        },

        getPanelVocab(filename) {
            return request(`/api/learning/${filename}/vocab`);
        },

        submitAnswer(filename, word, knew) {
            return request(`/api/learning/${filename}/answer`, {
                method: 'POST',
                body: JSON.stringify({ word, knew }),
            });
        },

        getProgress() {
            return request('/api/learning/progress');
        },

        // === Panel image URL helper ===
        panelImageUrl(filename) {
            return `${BASE_URL}/panels/${filename}`;
        }
    };
})();
