/**
 * API Client - Kommunikation mit dem Backend (mit Response-Cache)
 */
const API = (() => {
    const BASE_URL = window.location.origin;

    // In-memory cache for GET requests
    const _cache = new Map();
    const CACHE_TTL = 60_000; // 60 seconds

    function _cacheKey(endpoint) { return endpoint; }

    function _getCached(key) {
        const entry = _cache.get(key);
        if (entry && Date.now() - entry.ts < CACHE_TTL) return entry.data;
        _cache.delete(key);
        return null;
    }

    function _setCache(key, data) {
        _cache.set(key, { data, ts: Date.now() });
    }

    function invalidateCache(prefix) {
        for (const key of _cache.keys()) {
            if (!prefix || key.startsWith(prefix)) _cache.delete(key);
        }
    }

    async function request(endpoint, options = {}) {
        const method = (options.method || 'GET').toUpperCase();

        // Use cache for GET requests
        if (method === 'GET') {
            const cached = _getCached(endpoint);
            if (cached) return cached;
        }

        try {
            const response = await fetch(`${BASE_URL}${endpoint}`, {
                headers: { 'Content-Type': 'application/json' },
                ...options,
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || 'Request failed');
            }
            const data = await response.json();

            if (method === 'GET') _setCache(endpoint, data);
            return data;
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
            }).then(r => r.json()).then(data => {
                invalidateCache('/api/scanner/panels');
                invalidateCache('/api/learning/panels');
                return data;
            });
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

        // === Image URL helpers ===
        panelImageUrl(pathOrFilename) {
            if (pathOrFilename.startsWith('/')) {
                return `${BASE_URL}${pathOrFilename}`;
            }
            return `${BASE_URL}/panels/${pathOrFilename}`;
        },

        /** Thumbnail URL — small cached version for grid views */
        thumbUrl(filename, size = 160) {
            const name = filename.includes('/') ? filename.split('/').pop() : filename;
            return `${BASE_URL}/api/thumb/${name}?size=${size}`;
        },

        invalidateCache,
    };
})();
