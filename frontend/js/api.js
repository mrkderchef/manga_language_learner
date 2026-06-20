/**
 * API Client - Kommunikation mit dem Backend (mit Response-Cache)
 */
const API = (() => {
    const BASE_URL = window.location.origin;

    // In-memory cache for GET requests
    const _cache = new Map();
    const CACHE_TTL = 60_000; // 60 seconds
    const RABBITHOLE_POLL_INTERVAL_MS = 1000;
    const RABBITHOLE_POLL_TIMEOUT_MS = 180_000;

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

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async function pollRabbitholeJob(filename, jobId, options = {}) {
        const intervalMs = options.intervalMs || RABBITHOLE_POLL_INTERVAL_MS;
        const timeoutMs = options.timeoutMs || RABBITHOLE_POLL_TIMEOUT_MS;
        const started = Date.now();

        while (Date.now() - started < timeoutMs) {
            const job = await request(`/api/scanner/${filename}/rabbithole/jobs/${jobId}`, { responseCache: false });
            if (job.status === 'done') {
                if (job.result?.success) {
                    invalidateCache(`/api/scanner/${filename}/cache-status`);
                    return job.result;
                }
                throw new Error(job.result?.error || 'Rabbithole processing failed');
            }
            if (job.status === 'error') {
                throw new Error(job.error || job.result?.error || 'Rabbithole processing failed');
            }
            await delay(intervalMs);
        }

        throw new Error('Rabbithole processing timed out');
    }

    async function request(endpoint, options = {}) {
        const method = (options.method || 'GET').toUpperCase();
        const responseCache = options.responseCache !== false;
        const fetchOptions = { ...options };
        delete fetchOptions.responseCache;

        // Use cache for GET requests
        if (method === 'GET' && responseCache) {
            const cached = _getCached(endpoint);
            if (cached) return cached;
        }

        try {
            const response = await fetch(`${BASE_URL}${endpoint}`, {
                headers: { 'Content-Type': 'application/json' },
                ...fetchOptions,
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || 'Request failed');
            }
            const data = await response.json();

            if (method === 'GET' && responseCache) _setCache(endpoint, data);
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
                return data;
            });
        },

        scanPanel(filename, options = {}) {
            return request(`/api/scanner/${filename}/ocr`, {
                method: 'POST',
                body: JSON.stringify(options),
            }).then(data => {
                invalidateCache(`/api/scanner/${filename}/cache-status`);
                invalidateCache(`/api/scanner/${filename}/regions`);
                return data;
            });
        },

        buildRabbithole(filename, options = {}) {
            return request(`/api/scanner/${filename}/rabbithole`, {
                method: 'POST',
                body: JSON.stringify(options),
            }).then(data => {
                invalidateCache(`/api/scanner/${filename}/cache-status`);
                if (data?.rabbithole_job && data.job_id) {
                    return pollRabbitholeJob(filename, data.job_id);
                }
                return data;
            });
        },

        getRabbitholeJob(filename, jobId) {
            return request(`/api/scanner/${filename}/rabbithole/jobs/${jobId}`, { responseCache: false });
        },

        getCachedRabbithole(filename, options = {}) {
            return request(`/api/scanner/${filename}/rabbithole`, {
                method: 'POST',
                body: JSON.stringify({ ...options, cache_only: true }),
            });
        },

        translatePanel(filename, options = {}) {
            return request(`/api/scanner/${filename}/translate`, {
                method: 'POST',
                body: JSON.stringify(options),
            }).then(data => {
                invalidateCache(`/api/scanner/${filename}/cache-status`);
                return data;
            });
        },

        getCachedTranslation(filename, options = {}) {
            return request(`/api/scanner/${filename}/translate`, {
                method: 'POST',
                body: JSON.stringify({ ...options, cache_only: true }),
            });
        },

        getCacheStatus(filename) {
            return request(`/api/scanner/${filename}/cache-status`);
        },

        getRegions(filename) {
            return request(`/api/scanner/${filename}/regions`);
        },

        deletePanelCache(filename, kind = null) {
            const query = kind ? `?kind=${encodeURIComponent(kind)}` : '';
            return request(`/api/scanner/${filename}/cache${query}`, { method: 'DELETE' }).then(data => {
                invalidateCache(`/api/scanner/${filename}/cache-status`);
                invalidateCache(`/api/scanner/${filename}/regions`);
                return data;
            });
        },

        getTranslationEngines() {
            return request('/api/scanner/translation-engines');
        },

        getOllamaModels() {
            return request('/api/scanner/ollama/models');
        },

        getRuntimeStatus(force = false) {
            const suffix = force ? `?t=${Date.now()}` : '';
            return request(`/api/runtime/status${suffix}`);
        },

        downloadOcrAssets() {
            return request('/api/runtime/ocr-assets/download', { method: 'POST' }).then(data => {
                invalidateCache('/api/runtime/status');
                return data;
            });
        },

        overrideRegion(filename, regionId, data) {
            return request(`/api/scanner/${filename}/regions/${regionId}/override`, {
                method: 'POST',
                body: JSON.stringify(data),
            }).then(result => {
                invalidateCache(`/api/scanner/${filename}/cache-status`);
                invalidateCache(`/api/scanner/${filename}/regions`);
                return result;
            });
        },

        recomputeRegion(filename, regionId, options = {}) {
            return request(`/api/scanner/${filename}/regions/${regionId}/recompute`, {
                method: 'POST',
                body: JSON.stringify(options),
            }).then(result => {
                invalidateCache(`/api/scanner/${filename}/cache-status`);
                invalidateCache(`/api/scanner/${filename}/regions`);
                return result;
            });
        },

        addRegion(filename, data) {
            return request(`/api/scanner/${filename}/regions`, {
                method: 'POST',
                body: JSON.stringify(data),
            }).then(result => {
                invalidateCache(`/api/scanner/${filename}/cache-status`);
                invalidateCache(`/api/scanner/${filename}/regions`);
                return result;
            });
        },

        deleteRegion(filename, regionId) {
            return request(`/api/scanner/${filename}/regions/${regionId}`, {
                method: 'DELETE',
            }).then(result => {
                invalidateCache(`/api/scanner/${filename}/cache-status`);
                invalidateCache(`/api/scanner/${filename}/regions`);
                return result;
            });
        },

        lookupText(text) {
            return request(`/api/rabbithole/lookup?text=${encodeURIComponent(text)}`);
        },

        lookupKanji(character) {
            return request(`/api/rabbithole/kanji/${encodeURIComponent(character)}`);
        },

        lookupWord(text) {
            return request(`/api/rabbithole/word?text=${encodeURIComponent(text)}`);
        },

        lookupReading(reading) {
            return request(`/api/rabbithole/reading/${encodeURIComponent(reading)}`);
        },

        // === Image URL helpers ===
        panelImageUrl(pathOrFilename) {
            if (pathOrFilename.startsWith('/')) {
                return `${BASE_URL}${pathOrFilename}`;
            }
            return `${BASE_URL}/api/media/panel/${pathOrFilename}`;
        },

        /** Thumbnail URL — small cached version for grid views */
        thumbUrl(filename, size = 160) {
            const name = filename.includes('/') ? filename.split('/').pop() : filename;
            return `${BASE_URL}/api/thumb/${name}?size=${size}`;
        },

        invalidateCache,
    };
})();
