/**
 * Scanner Mode - OCR + Übersetzung von Manga Panels
 * Hover über erkannten Text zeigt Übersetzung direkt auf dem Panel
 */
const Scanner = (() => {
    let selectedPanel = null;
    let ocrText = '';
    let latestScan = null;
    let _panelsLoaded = false;
    let controlsHideTimer = null;
    let pinnedRegionId = null;
    let editMode = false;
    let addBoxMode = false;
    let draftBox = null;
    let engineControlsLoaded = false;
    let latestCacheStatus = null;

    function init() {
        bindEvents();
    }

    function bindEvents() {
        document.getElementById('upload-area').addEventListener('click', () => {
            document.getElementById('panel-upload').click();
        });

        document.getElementById('panel-upload').addEventListener('change', handleUpload);
        document.getElementById('btn-scan').addEventListener('click', handleScanTranslate);
        document.getElementById('btn-scan-options').addEventListener('click', toggleScanOptions);
        document.getElementById('btn-edit-boxes').addEventListener('click', toggleEditMode);
        document.getElementById('btn-add-box').addEventListener('click', toggleAddBoxMode);
        document.getElementById('btn-debug-toggle').addEventListener('click', () => toggleDebugPanel());
        document.getElementById('btn-debug-close').addEventListener('click', () => toggleDebugPanel(false));
        document.getElementById('btn-reader-fullscreen').addEventListener('click', openReaderFullscreen);
        document.getElementById('btn-fullscreen-close').addEventListener('click', closeReaderFullscreen);
        document.getElementById('btn-clear-panel-cache').addEventListener('click', () => handleCacheClear());
        document.querySelectorAll('[data-cache-kind]').forEach(btn => {
            btn.addEventListener('click', () => handleCacheClear(btn.dataset.cacheKind));
        });
        document.getElementById('home-ollama-model')?.addEventListener('change', persistHomeModelSelection);
        document.getElementById('home-custom-model')?.addEventListener('input', persistHomeModelSelection);
        document.getElementById('translation-slider').addEventListener('input', (e) => {
            setTranslationReveal(Number(e.target.value));
        });
        bindTranslationHandle();
        bindReaderPanelControls();
        bindBoxCreation();
        document.querySelector('.scan-combo')?.appendChild(document.getElementById('scan-options-panel'));
        loadHomeOllamaModels();
        document.getElementById('btn-translation-toggle').addEventListener('click', () => {
            const slider = document.getElementById('translation-slider');
            const next = Number(slider.value) >= 50 ? 0 : 100;
            setTranslationReveal(next);
        });
        document.getElementById('scanner-panel-img').addEventListener('load', rerenderLatestOverlays);
        window.addEventListener('resize', rerenderLatestOverlays);
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeReaderFullscreen();
            }
        });
    }

    function toggleEditMode() {
        editMode = !editMode;
        if (!editMode) addBoxMode = false;
        const wrapper = document.getElementById('reader-panel-wrapper');
        const editBtn = document.getElementById('btn-edit-boxes');
        const addBtn = document.getElementById('btn-add-box');
        wrapper.classList.toggle('box-edit-mode', editMode);
        editBtn.setAttribute('aria-pressed', String(editMode));
        editBtn.classList.toggle('active', editMode);
        addBtn.disabled = !editMode || !selectedPanel;
        addBtn.setAttribute('aria-pressed', String(addBoxMode));
        addBtn.classList.toggle('active', addBoxMode);
        rerenderLatestOverlays();
    }

    function toggleAddBoxMode() {
        if (!editMode || !selectedPanel) return;
        addBoxMode = !addBoxMode;
        const btn = document.getElementById('btn-add-box');
        btn.setAttribute('aria-pressed', String(addBoxMode));
        btn.classList.toggle('active', addBoxMode);
        document.getElementById('reader-panel-wrapper').classList.toggle('box-add-mode', addBoxMode);
    }

    async function loadEngineControls() {
        if (engineControlsLoaded) return;
        try {
            const [ocr, translation, ollama] = await Promise.all([
                API.getOcrEngines(),
                API.getTranslationEngines(),
                API.getOllamaModels(),
            ]);
            fillSelect('scan-ocr-engine', ocr.engines || [], 'mangaocr');
            fillSelect('scan-translation-engine', translation.engines || [], 'ollama');
            fillOllamaModelSelect('scan-translation-model', ollama);
            fillOllamaModelSelect('home-ollama-model', ollama);
            updateHomeOllamaStatus(ollama);
            engineControlsLoaded = true;
        } catch (err) {
            console.warn('Could not load engine controls:', err);
            updateHomeOllamaStatus({ discovery_error: err.message, models: [] });
        }
    }

    async function loadHomeOllamaModels() {
        try {
            const ollama = await API.getOllamaModels();
            fillOllamaModelSelect('home-ollama-model', ollama);
            fillOllamaModelSelect('scan-translation-model', ollama);
            updateHomeOllamaStatus(ollama);
        } catch (err) {
            updateHomeOllamaStatus({ discovery_error: err.message, models: [] });
        }
    }

    function fillSelect(id, engines, fallback) {
        const select = document.getElementById(id);
        if (!select) return;
        select.innerHTML = '';
        engines.forEach(engine => {
            const option = document.createElement('option');
            option.value = engine.id;
            const suffix = engine.available ? '' : ' (unavailable)';
            option.textContent = `${engine.label || engine.id}${suffix}`;
            option.disabled = !engine.available;
            option.selected = engine.id === fallback;
            select.appendChild(option);
        });
    }

    function fillOllamaModelSelect(id, ollama) {
        const select = document.getElementById(id);
        if (!select) return;
        const saved = localStorage.getItem('preferredOllamaModel') || '';
        const models = ollama.models || [];
        select.innerHTML = '<option value="">Auto</option>';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            option.selected = saved ? model === saved : model === ollama.preferred_model;
            select.appendChild(option);
        });
        if (saved && !models.includes(saved)) {
            const option = document.createElement('option');
            option.value = saved;
            option.textContent = `${saved} (saved)`;
            option.selected = true;
            select.appendChild(option);
        }
    }

    function updateHomeOllamaStatus(ollama) {
        const status = document.getElementById('home-ollama-status');
        if (!status) return;
        const models = ollama.models || [];
        if (models.length) {
            status.textContent = `${models.length} model(s) discovered. Preferred: ${ollama.preferred_model || 'Auto'}.`;
        } else if (ollama.discovery_error) {
            status.textContent = `${ollama.discovery_error}. A saved/custom model can still be attempted during translation.`;
        } else {
            status.textContent = 'No Ollama models discovered yet. A saved/custom model can still be attempted.';
        }
    }

    function persistHomeModelSelection() {
        const selected = document.getElementById('home-ollama-model')?.value || '';
        const custom = document.getElementById('home-custom-model')?.value.trim() || '';
        const value = custom || selected;
        if (value) {
            localStorage.setItem('preferredOllamaModel', value);
        } else {
            localStorage.removeItem('preferredOllamaModel');
        }
        const scanSelect = document.getElementById('scan-translation-model');
        if (scanSelect && value) {
            if (![...scanSelect.options].some(option => option.value === value)) {
                scanSelect.appendChild(new Option(`${value} (saved)`, value, true, true));
            }
            scanSelect.value = value;
        }
    }

    async function toggleScanOptions() {
        const panel = document.getElementById('scan-options-panel');
        const btn = document.getElementById('btn-scan-options');
        const open = panel.classList.contains('hidden');
        panel.classList.toggle('hidden', !open);
        btn.setAttribute('aria-expanded', String(open));
        if (open) await loadEngineControls();
    }

    function getScanOptions() {
        const customModel = document.getElementById('scan-custom-model').value.trim();
        const savedHomeModel = localStorage.getItem('preferredOllamaModel') || '';
        return {
            use_cache: document.getElementById('scan-use-cache').checked,
            fresh: document.getElementById('scan-fresh').checked,
            ocr_engine: document.getElementById('scan-ocr-engine').value,
            ocr_quality_mode: document.getElementById('scan-ocr-quality').value,
            semantic_rerank: 'close',
            vertical_preference: document.getElementById('scan-vertical-preference').value,
            rotation_win_margin: Number(document.getElementById('scan-rotation-margin').value) || 15,
            preprocessing_set: 'standard',
            detection_sensitivity: 'normal',
            translation_engine: document.getElementById('scan-translation-engine').value,
            translation_model: customModel || document.getElementById('scan-translation-model').value || savedHomeModel || null,
            target_lang: document.getElementById('scan-target-lang').value || 'en',
            translation_style: document.getElementById('scan-translation-style').value,
            temperature: Number(document.getElementById('scan-temperature').value) || 0.1,
            reset_manual_edits: document.getElementById('scan-reset-edits').checked,
        };
    }

    async function refreshCacheStatus() {
        if (!selectedPanel) {
            latestCacheStatus = null;
            renderCacheSummary(null);
            return;
        }
        try {
            const status = await API.getCacheStatus(selectedPanel.filename);
            latestCacheStatus = status;
            renderCacheSummary(status);
        } catch {
            latestCacheStatus = null;
            renderCacheSummary(null);
        }
    }

    function renderCacheSummary(status) {
        const summary = document.getElementById('panel-cache-summary');
        if (!summary) return;
        const buckets = status?.buckets || {};
        summary.innerHTML = '';
        [
            ['ocr', 'OCR'],
            ['translation', 'Translation'],
            ['learning', 'Learning'],
        ].forEach(([key, label]) => {
            const hasCache = Boolean(buckets[key]?.has_cache);
            const pill = document.createElement('div');
            pill.className = `cache-pill ${hasCache ? 'cache-pill-hit' : 'cache-pill-miss'}`;
            pill.textContent = `${label} ${hasCache ? '✓' : '✕'}`;
            pill.title = `${label} cache entries: ${buckets[key]?.entries || 0}`;
            summary.appendChild(pill);
        });
        const scanCombo = document.querySelector('.scan-combo');
        scanCombo?.classList.toggle('has-panel-cache', Boolean(status?.has_cache));
    }

    async function handleCacheClear(kind = null) {
        if (!selectedPanel) return;
        try {
            await API.deletePanelCache(selectedPanel.filename, kind);
            latestScan = null;
            document.getElementById('ocr-overlay').innerHTML = '';
            resetTranslationView();
            setDebugStatus(`${kind ? kind.toUpperCase() : 'Panel'} cache cleared. Manual edits were preserved.`);
            await refreshCacheStatus();
        } catch (err) {
            setDebugStatus('Could not clear cache: ' + err.message);
        }
    }

    function bindReaderPanelControls() {
        const wrapper = document.getElementById('reader-panel-wrapper');
        if (!wrapper) return;

        wrapper.addEventListener('mousemove', showReaderPanelControls);
        wrapper.addEventListener('pointerdown', showReaderPanelControls);
        wrapper.addEventListener('mouseleave', () => {
            window.clearTimeout(controlsHideTimer);
            wrapper.classList.remove('reader-controls-visible');
        });
    }

    function showReaderPanelControls() {
        const wrapper = document.getElementById('reader-panel-wrapper');
        wrapper.classList.add('reader-controls-visible');
        window.clearTimeout(controlsHideTimer);
        controlsHideTimer = window.setTimeout(() => {
            wrapper.classList.remove('reader-controls-visible');
        }, 1500);
    }

    function bindTranslationHandle() {
        const handle = document.getElementById('translation-handle');
        if (!handle) return;

        handle.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            handle.setPointerCapture?.(e.pointerId);
            updateRevealFromPointer(e);

            const move = (moveEvent) => updateRevealFromPointer(moveEvent);
            const up = (upEvent) => {
                handle.releasePointerCapture?.(upEvent.pointerId);
                window.removeEventListener('pointermove', move);
                window.removeEventListener('pointerup', up);
            };

            window.addEventListener('pointermove', move);
            window.addEventListener('pointerup', up);
        });

        handle.addEventListener('keydown', (e) => {
            const slider = document.getElementById('translation-slider');
            const current = Number(slider.value) || 0;
            let next = current;
            if (e.key === 'ArrowLeft') next = current - 5;
            if (e.key === 'ArrowRight') next = current + 5;
            if (e.key === 'Home') next = 0;
            if (e.key === 'End') next = 100;
            if (next !== current) {
                e.preventDefault();
                setTranslationReveal(next);
            }
        });
    }

    function updateRevealFromPointer(e) {
        const wrapper = document.querySelector('.panel-img-wrapper');
        const rect = wrapper.getBoundingClientRect();
        if (!rect.width) return;
        const reveal = ((e.clientX - rect.left) / rect.width) * 100;
        setTranslationReveal(reveal);
    }

    async function loadPanels(force = false) {
        if (_panelsLoaded && !force) return;
        const grid = document.getElementById('panel-list');

        // Show skeleton placeholders
        grid.innerHTML = Array(4).fill(
            '<div class="panel-thumb skeleton"></div>'
        ).join('');

        try {
            const data = await API.listPanels();
            const panels = data.panels || [];
            _panelsLoaded = true;
            renderPanelGrid(panels);
        } catch (err) {
            console.error('Failed to load panels:', err);
            grid.innerHTML = '<p style="color:var(--text-secondary)">Fehler beim Laden</p>';
        }
    }

    function renderPanelGrid(panels) {
        const grid = document.getElementById('panel-list');
        grid.innerHTML = '';

        panels.forEach(panel => {
            const img = document.createElement('img');
            img.src = API.thumbUrl(panel.filename);
            img.alt = panel.filename;
            img.className = 'panel-thumb';
            img.loading = 'lazy';
            img.decoding = 'async';
            img.addEventListener('click', () => selectPanel(panel, img));
            grid.appendChild(img);
        });
    }

    function selectPanel(panel, imgEl) {
        document.querySelectorAll('.panel-thumb.selected').forEach(el => el.classList.remove('selected'));
        imgEl.classList.add('selected');

        selectedPanel = panel;
        const panelImg = document.getElementById('scanner-panel-img');
        panelImg.src = API.panelImageUrl(panel.path || panel.filename);
        panelImg.classList.remove('hidden');
        document.getElementById('btn-scan').disabled = false;
        document.getElementById('ocr-overlay').innerHTML = '';
        latestScan = null;
        resetTranslationView();
        setDebugStatus('Ready to scan.');
        ocrText = '';
        pinnedRegionId = null;
        document.getElementById('btn-add-box').disabled = !editMode;
        refreshCacheStatus();
    }

    async function handleUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const result = await API.uploadPanel(file);
            if (result.success) {
                _panelsLoaded = false;
                loadPanels(true);
            }
        } catch (err) {
            alert('Upload fehlgeschlagen: ' + err.message);
        }
    }

    async function handleScanTranslate() {
        if (!selectedPanel) return;

        const btn = document.getElementById('btn-scan');

        btn.disabled = true;
        btn.classList.add('loading');
        btn.innerHTML = '<span class="spinner"></span>Running...';
        document.getElementById('ocr-overlay').innerHTML = '';
        latestScan = null;
        resetTranslationView();
        setDebugStatus('Scanning...');

        try {
            const result = await API.scanAndTranslate(selectedPanel.filename, getScanOptions());
            latestScan = result;
            ocrText = result.text || '';
            renderDebugPanel(result);
            renderOverlays(result.annotations || [], result.image_width, result.image_height);
            showTranslatedPanel(result.translated_image_url);
            if (result.translation_error) {
                setDebugStatus(`OCR completed, but translation failed: ${result.translation_error}`);
                renderDebugPanel(result);
            }
            if (result.render_warnings && result.render_warnings.length) {
                console.warn('Panel render warnings:', result.render_warnings);
            }
            refreshCacheStatus();
        } catch (err) {
            setDebugStatus('Error: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.classList.remove('loading');
            btn.textContent = 'Scan & Translate';
        }
    }

    function showTranslatedPanel(url) {
        const translatedImg = document.getElementById('translated-panel-img');
        const controls = document.getElementById('translation-controls');
        if (!url) {
            resetTranslationView();
            return;
        }

        translatedImg.src = new URL(url, window.location.origin).toString();
        translatedImg.classList.remove('hidden');
        controls.classList.remove('hidden');
        setTranslationReveal(0);
    }

    function resetTranslationView() {
        const translatedImg = document.getElementById('translated-panel-img');
        const controls = document.getElementById('translation-controls');
        translatedImg.removeAttribute('src');
        translatedImg.classList.add('hidden');
        controls.classList.add('hidden');
        setTranslationReveal(0);
    }

    function setTranslationReveal(value) {
        const reveal = Math.max(0, Math.min(100, Number(value) || 0));
        const wrapper = document.querySelector('.panel-img-wrapper');
        const translatedImg = document.getElementById('translated-panel-img');
        const slider = document.getElementById('translation-slider');
        const toggle = document.getElementById('btn-translation-toggle');
        const handle = document.getElementById('translation-handle');
        wrapper.style.setProperty('--translation-reveal', `${reveal}%`);
        translatedImg.style.setProperty('--translation-reveal', `${reveal}%`);
        slider.value = String(reveal);
        toggle.textContent = reveal >= 50 ? 'EN' : 'JP';
        toggle.setAttribute('aria-label', reveal >= 50 ? 'English translation visible' : 'Japanese original visible');
        handle.setAttribute('aria-valuenow', String(Math.round(reveal)));
        handle.setAttribute('aria-valuetext', `${Math.round(reveal)} percent English`);
        wrapper.classList.toggle('translation-active', reveal > 8 && !translatedImg.classList.contains('hidden'));
    }

    function toggleDebugPanel(force) {
        const panel = document.getElementById('reader-debug-panel');
        const btn = document.getElementById('btn-debug-toggle');
        const nextOpen = typeof force === 'boolean' ? force : panel.classList.contains('hidden');
        panel.classList.toggle('hidden', !nextOpen);
        btn.setAttribute('aria-expanded', String(nextOpen));
    }

    function setDebugStatus(message) {
        const content = document.getElementById('reader-debug-content');
        content.innerHTML = '';
        const empty = document.createElement('div');
        empty.className = 'reader-debug-empty';
        empty.textContent = message;
        content.appendChild(empty);
    }

    function renderDebugPanel(result) {
        const content = document.getElementById('reader-debug-content');
        const annotations = result.annotations || [];
        content.innerHTML = '';

        if (result.scan_trace && result.scan_trace.length) {
            content.appendChild(renderScanTrace(result.scan_trace, result));
        }

        if (!annotations.length) {
            if (!result.scan_trace?.length) setDebugStatus('No text recognized.');
            return;
        }

        const frag = document.createDocumentFragment();
        annotations.forEach((ann, index) => {
            const debug = ann.ocr_debug || {};
            const quality = debug.quality || qualityFromConfidence(ann.confidence);
            const entry = document.createElement('div');
            entry.className = `reader-debug-entry ocr-quality-${quality}`;

            const title = document.createElement('div');
            title.className = 'reader-debug-title';
            title.textContent = `Box ${ann.reading_order || index + 1}`;

            const meta = document.createElement('div');
            meta.className = 'reader-debug-meta';
            [
                `quality: ${quality}`,
                `confidence: ${formatConfidence(ann.confidence)}`,
                `variant: ${debug.selected_variant || ann.ocr_variant || 'unknown'}`,
                `score: ${debug.score ?? 'n/a'}`,
                `ocr: ${result.ocr_engine_used || result.method || 'unknown'}`,
                `translation: ${result.translation_engine_used || 'unknown'}${result.translation_model ? `/${result.translation_model}` : ''}`,
                `orientation: ${ann.recognized_orientation || 'unknown'}`,
                `vertical: ${debug.detector?.vertical ? 'yes' : 'no'}`,
                `angle: ${debug.detector?.angle ?? ann.angle ?? 0}`,
                `font: ${debug.detector?.font_size ?? ann.font_size ?? 0}`,
            ].forEach(item => {
                const pill = document.createElement('span');
                pill.textContent = item;
                meta.appendChild(pill);
            });

            const jp = document.createElement('div');
            jp.className = 'reader-debug-jp';
            jp.textContent = ann.text || '';

            const en = document.createElement('div');
            en.className = 'reader-debug-en';
            en.textContent = ann.translated || 'No translation available';
            const reading = document.createElement('div');
            reading.className = 'reader-debug-reading';
            reading.textContent = [ann.reading_kana, ann.reading_romaji].filter(Boolean).join(' | ');

            entry.appendChild(title);
            entry.appendChild(meta);
            entry.appendChild(jp);
            if (reading.textContent) entry.appendChild(reading);
            entry.appendChild(en);

            if (debug.score_breakdown) {
                const breakdown = document.createElement('div');
                breakdown.className = 'reader-debug-boxes';
                const parts = Object.entries(debug.score_breakdown)
                    .filter(([key, value]) => typeof value !== 'object' && value !== null)
                    .map(([key, value]) => `${key}: ${value}`);
                breakdown.textContent = `Score: ${parts.join(' | ')}`;
                entry.appendChild(breakdown);
            }

            if (debug.warnings && debug.warnings.length) {
                const warnings = document.createElement('div');
                warnings.className = 'reader-debug-warnings';
                warnings.textContent = `Warnings: ${debug.warnings.join(', ')}`;
                entry.appendChild(warnings);
            }

            if (debug.crop_box || debug.detected_box) {
                const boxes = document.createElement('div');
                boxes.className = 'reader-debug-boxes';
                boxes.textContent = [
                    debug.detected_box ? `detected [${debug.detected_box.join(', ')}]` : '',
                    debug.crop_box ? `crop [${debug.crop_box.join(', ')}]` : '',
                ].filter(Boolean).join(' | ');
                entry.appendChild(boxes);
            }

            if (debug.previews?.crop || debug.selected_preview_url) {
                const previews = document.createElement('div');
                previews.className = 'reader-debug-previews';
                [
                    ['Crop', debug.previews?.crop],
                    ['Selected', debug.selected_preview_url],
                ].forEach(([label, url]) => {
                    if (!url) return;
                    previews.appendChild(createDebugPreview(label, url));
                });
                entry.appendChild(previews);
            }

            if (debug.candidates && debug.candidates.length) {
                const details = document.createElement('details');
                details.className = 'reader-debug-candidates';
                const summary = document.createElement('summary');
                summary.textContent = `OCR candidates (${debug.candidates.length})`;
                details.appendChild(summary);

                debug.candidates.forEach(candidate => {
                    const row = document.createElement('div');
                    row.className = candidate.variant === debug.selected_variant
                        ? 'reader-debug-candidate selected'
                        : 'reader-debug-candidate';

                    const head = document.createElement('div');
                    head.className = 'reader-debug-candidate-head';
                    head.textContent = `${candidate.variant || 'unknown'} | score ${candidate.score ?? 'n/a'}`;

                    const text = document.createElement('div');
                    text.className = 'reader-debug-candidate-text';
                    text.textContent = candidate.error ? `Error: ${candidate.error}` : (candidate.text || '(empty)');

                    row.appendChild(head);
                    if (candidate.preview_url) {
                        row.appendChild(createDebugPreview('Preview', candidate.preview_url));
                    }
                    row.appendChild(text);
                    details.appendChild(row);
                });
                entry.appendChild(details);
            }

            frag.appendChild(entry);
        });

        content.appendChild(frag);
    }

    function renderScanTrace(trace, result) {
        const section = document.createElement('section');
        section.className = 'reader-debug-trace';

        const title = document.createElement('div');
        title.className = 'reader-debug-title';
        title.textContent = 'Scan trace';
        section.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'reader-debug-meta';
        [
            `ocr: ${result.ocr_engine_used || 'unknown'}`,
            `translation: ${result.translation_engine_used || 'none'}${result.translation_model ? `/${result.translation_model}` : ''}`,
            `lookup hits: ${result.global_lookup_hits ?? 0}`,
            `lookup misses: ${result.global_lookup_misses ?? 0}`,
        ].forEach(item => {
            const pill = document.createElement('span');
            pill.textContent = item;
            meta.appendChild(pill);
        });
        section.appendChild(meta);

        trace.forEach(event => {
            const row = document.createElement('div');
            row.className = `reader-debug-trace-row trace-${event.status || 'info'}`;
            const elapsed = Number.isFinite(event.elapsed_ms) ? ` · ${event.elapsed_ms}ms` : '';
            row.textContent = `${event.stage}: ${event.status} · ${event.message}${elapsed}`;
            section.appendChild(row);
        });

        if (result.translation_error) {
            const warning = document.createElement('div');
            warning.className = 'reader-debug-warnings';
            warning.textContent = `Translation error: ${result.translation_error}`;
            section.appendChild(warning);
        }
        if (result.learning_error) {
            const warning = document.createElement('div');
            warning.className = 'reader-debug-warnings';
            warning.textContent = `Learning error: ${result.learning_error}`;
            section.appendChild(warning);
        }
        return section;
    }

    function openReaderFullscreen() {
        const src = getCurrentReaderImageSrc();
        if (!src) return;

        const overlay = document.getElementById('reader-fullscreen');
        const img = document.getElementById('reader-fullscreen-img');
        img.src = src;
        overlay.classList.remove('hidden');
        document.body.classList.add('reader-fullscreen-open');
    }

    function closeReaderFullscreen() {
        const overlay = document.getElementById('reader-fullscreen');
        if (overlay.classList.contains('hidden')) return;

        overlay.classList.add('hidden');
        document.getElementById('reader-fullscreen-img').removeAttribute('src');
        document.body.classList.remove('reader-fullscreen-open');
    }

    function getCurrentReaderImageSrc() {
        const slider = document.getElementById('translation-slider');
        const translatedImg = document.getElementById('translated-panel-img');
        const originalImg = document.getElementById('scanner-panel-img');
        const showEnglish = Number(slider.value) >= 50;
        if (showEnglish && !translatedImg.classList.contains('hidden') && translatedImg.src) {
            return translatedImg.src;
        }
        return originalImg.src || '';
    }

    function rerenderLatestOverlays() {
        if (!latestScan) return;
        renderOverlays(latestScan.annotations || [], latestScan.image_width, latestScan.image_height);
    }

    function bindBoxCreation() {
        const overlay = document.getElementById('ocr-overlay');
        overlay.addEventListener('pointerdown', (e) => {
            if (!editMode || !addBoxMode || !selectedPanel || !latestScan) return;
            if (e.target !== overlay) return;
            e.preventDefault();
            const rect = overlay.getBoundingClientRect();
            draftBox = {
                startX: e.clientX - rect.left,
                startY: e.clientY - rect.top,
                currentX: e.clientX - rect.left,
                currentY: e.clientY - rect.top,
            };
            overlay.setPointerCapture?.(e.pointerId);
            drawDraftBox();
        });

        overlay.addEventListener('pointermove', (e) => {
            if (!draftBox) return;
            const rect = overlay.getBoundingClientRect();
            draftBox.currentX = e.clientX - rect.left;
            draftBox.currentY = e.clientY - rect.top;
            drawDraftBox();
        });

        overlay.addEventListener('pointerup', async (e) => {
            if (!draftBox) return;
            overlay.releasePointerCapture?.(e.pointerId);
            const region = draftToImageRegion();
            clearDraftBox();
            if (!region || region.width < 8 || region.height < 8) return;
            try {
                await API.addRegion(selectedPanel.filename, region);
                await rescanAfterBoxEdit('Box added. Rescanning...');
            } catch (err) {
                setDebugStatus('Could not add box: ' + err.message);
            }
        });
    }

    function drawDraftBox() {
        const overlay = document.getElementById('ocr-overlay');
        let box = overlay.querySelector('.ocr-draft-box');
        if (!box) {
            box = document.createElement('div');
            box.className = 'ocr-draft-box';
            overlay.appendChild(box);
        }
        const left = Math.min(draftBox.startX, draftBox.currentX);
        const top = Math.min(draftBox.startY, draftBox.currentY);
        const width = Math.abs(draftBox.currentX - draftBox.startX);
        const height = Math.abs(draftBox.currentY - draftBox.startY);
        box.style.cssText = `left:${left}px;top:${top}px;width:${width}px;height:${height}px`;
    }

    function clearDraftBox() {
        document.querySelector('.ocr-draft-box')?.remove();
        draftBox = null;
    }

    function draftToImageRegion() {
        const geom = getOverlayGeometry();
        if (!geom || !draftBox) return null;
        const left = Math.min(draftBox.startX, draftBox.currentX);
        const top = Math.min(draftBox.startY, draftBox.currentY);
        const width = Math.abs(draftBox.currentX - draftBox.startX);
        const height = Math.abs(draftBox.currentY - draftBox.startY);
        return displayRectToImageRegion(left, top, width, height, geom);
    }

    function getOverlayGeometry() {
        const panelImg = document.getElementById('scanner-panel-img');
        if (!latestScan || !latestScan.image_width || !latestScan.image_height) return null;
        const displayW = panelImg.clientWidth;
        const displayH = panelImg.clientHeight;
        if (!displayW || !displayH) return null;
        return {
            displayW,
            displayH,
            imageW: latestScan.image_width,
            imageH: latestScan.image_height,
            scaleX: displayW / latestScan.image_width,
            scaleY: displayH / latestScan.image_height,
        };
    }

    function displayRectToImageRegion(left, top, width, height, geom) {
        const x = Math.max(0, Math.round(left / geom.scaleX));
        const y = Math.max(0, Math.round(top / geom.scaleY));
        const w = Math.max(1, Math.round(width / geom.scaleX));
        const h = Math.max(1, Math.round(height / geom.scaleY));
        return {
            x,
            y,
            width: Math.min(w, geom.imageW - x),
            height: Math.min(h, geom.imageH - y),
            orientation: h >= w ? 'vertical' : 'horizontal',
        };
    }

    async function rescanAfterBoxEdit(message) {
        setDebugStatus(message);
        const options = { ...getScanOptions(), fresh: true };
        const result = await API.scanAndTranslate(selectedPanel.filename, options);
        latestScan = result;
        renderDebugPanel(result);
        renderOverlays(result.annotations || [], result.image_width, result.image_height);
        showTranslatedPanel(result.translated_image_url);
        refreshCacheStatus();
    }

    function renderOverlays(annotations, imgNatW, imgNatH) {
        const overlay = document.getElementById('ocr-overlay');
        const panelImg = document.getElementById('scanner-panel-img');
        overlay.innerHTML = '';

        const withBbox = annotations.filter(a => a.bbox && a.bbox.length >= 4);
        if (!withBbox.length || !imgNatW || !imgNatH) return;

        const displayW = panelImg.clientWidth;
        const displayH = panelImg.clientHeight;
        const scaleX = displayW / imgNatW;
        const scaleY = displayH / imgNatH;

        // Build all boxes in a document fragment (single reflow)
        const frag = document.createDocumentFragment();

        withBbox.forEach(ann => {
            if (!ann.bbox || ann.bbox.length < 4) return;

            const [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = ann.bbox;
            const left = Math.min(x1, x4) * scaleX;
            const top = Math.min(y1, y2) * scaleY;
            const width = (Math.max(x2, x3) - Math.min(x1, x4)) * scaleX;
            const height = (Math.max(y3, y4) - Math.min(y1, y2)) * scaleY;

            const box = document.createElement('div');
            const quality = ann.ocr_debug?.quality || qualityFromConfidence(ann.confidence);
            const regionId = ann.region_id || ann.id;
            box.className = `ocr-box ocr-quality-${quality}${pinnedRegionId === regionId ? ' pinned' : ''}`;
            box.style.cssText = `left:${left}px;top:${top}px;width:${width}px;height:${height}px`;
            box.title = `OCR ${quality} | ${ann.ocr_variant || 'unknown'} | ${formatConfidence(ann.confidence)}`;
            box.dataset.regionId = regionId;
            if (editMode) {
                box.tabIndex = 0;
                box.appendChild(createResizeHandle());
                box.addEventListener('pointerdown', (e) => startBoxDrag(e, box, ann));
                box.addEventListener('keydown', (e) => handleBoxKeydown(e, box, ann));
            }
            box.addEventListener('click', (e) => {
                if (editMode && e.target.classList.contains('ocr-resize-handle')) return;
                e.stopPropagation();
                pinnedRegionId = pinnedRegionId === regionId ? null : regionId;
                rerenderLatestOverlays();
            });

            const tooltip = document.createElement('div');
            tooltip.className = 'ocr-tooltip';
            const jpDiv = document.createElement('div');
            jpDiv.className = 'ocr-tooltip-jp';
            jpDiv.appendChild(renderInteractiveJapanese(ann));
            const readingDiv = document.createElement('div');
            readingDiv.className = 'ocr-tooltip-reading';
            readingDiv.textContent = [ann.reading_kana, ann.reading_romaji].filter(Boolean).join(' | ');
            const enDiv = document.createElement('div');
            enDiv.className = 'ocr-tooltip-en';
            enDiv.textContent = ann.translated || '...';
            const orientationBtn = document.createElement('button');
            orientationBtn.className = 'ocr-orientation-toggle';
            const orientation = ann.recognized_orientation || (ann.vertical ? 'vertical' : 'horizontal');
            orientationBtn.textContent = `recognized orientation: ${orientation} ${orientation === 'vertical' ? '↓' : '→'}`;
            orientationBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                await handleOrientationToggle(ann);
            });
            const removeBtn = document.createElement('button');
            removeBtn.className = 'ocr-remove-box';
            removeBtn.textContent = 'Remove box';
            removeBtn.hidden = !editMode;
            removeBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                await handleRemoveBox(ann);
            });
            const metaDiv = document.createElement('div');
            metaDiv.className = 'ocr-tooltip-meta';
            metaDiv.textContent = `${quality.toUpperCase()} | ${ann.ocr_variant || 'unknown'} | ${formatConfidence(ann.confidence)}`;
            tooltip.appendChild(jpDiv);
            if (readingDiv.textContent) tooltip.appendChild(readingDiv);
            tooltip.appendChild(enDiv);
            tooltip.appendChild(orientationBtn);
            tooltip.appendChild(removeBtn);
            tooltip.appendChild(metaDiv);
            box.appendChild(tooltip);

            frag.appendChild(box);
        });

        overlay.appendChild(frag);
    }

    function renderInteractiveJapanese(ann) {
        const wrapper = document.createElement('span');
        const text = ann.text || '';
        const tokens = ann.tokens || [];
        if (!tokens.length) {
            wrapper.textContent = text;
            return wrapper;
        }
        let cursor = 0;
        tokens.forEach(token => {
            if (token.start > cursor) {
                wrapper.append(document.createTextNode(text.slice(cursor, token.start)));
            }
            const span = document.createElement('span');
            span.className = 'jp-token';
            span.textContent = text.slice(token.start, token.end);
            span.title = [token.reading_kana, token.reading_romaji, token.lemma].filter(Boolean).join(' | ');
            span.addEventListener('click', async (e) => {
                e.stopPropagation();
                span.classList.add('loading-token');
                try {
                    const data = await API.lookupWord(span.textContent);
                    span.title = `${span.title}\nlookup: ${data.source || data.type}`;
                } catch (err) {
                    span.title = `Lookup failed: ${err.message}`;
                } finally {
                    span.classList.remove('loading-token');
                }
            });
            wrapper.appendChild(span);
            cursor = token.end;
        });
        if (cursor < text.length) wrapper.append(document.createTextNode(text.slice(cursor)));
        return wrapper;
    }

    function createResizeHandle() {
        const handle = document.createElement('span');
        handle.className = 'ocr-resize-handle';
        handle.setAttribute('aria-hidden', 'true');
        return handle;
    }

    function startBoxDrag(e, box, ann) {
        if (!editMode || e.button !== 0) return;
        if (e.target.closest('.ocr-tooltip')) return;
        e.preventDefault();
        e.stopPropagation();
        const geom = getOverlayGeometry();
        if (!geom) return;
        const resizing = e.target.classList.contains('ocr-resize-handle');
        const start = {
            x: e.clientX,
            y: e.clientY,
            left: parseFloat(box.style.left) || 0,
            top: parseFloat(box.style.top) || 0,
            width: parseFloat(box.style.width) || 0,
            height: parseFloat(box.style.height) || 0,
        };
        box.setPointerCapture?.(e.pointerId);

        const move = (moveEvent) => {
            const dx = moveEvent.clientX - start.x;
            const dy = moveEvent.clientY - start.y;
            if (resizing) {
                box.style.width = `${Math.max(12, start.width + dx)}px`;
                box.style.height = `${Math.max(12, start.height + dy)}px`;
            } else {
                box.style.left = `${Math.max(0, Math.min(geom.displayW - start.width, start.left + dx))}px`;
                box.style.top = `${Math.max(0, Math.min(geom.displayH - start.height, start.top + dy))}px`;
            }
        };

        const up = async (upEvent) => {
            box.releasePointerCapture?.(upEvent.pointerId);
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
            await persistBoxGeometry(box, ann, true);
        };

        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', up);
    }

    async function handleBoxKeydown(e, box, ann) {
        if (!editMode) return;
        const step = e.shiftKey ? 10 : 2;
        const geom = getOverlayGeometry();
        if (!geom) return;
        const left = parseFloat(box.style.left) || 0;
        const top = parseFloat(box.style.top) || 0;
        let handled = true;
        if (e.key === 'ArrowLeft') box.style.left = `${Math.max(0, left - step)}px`;
        else if (e.key === 'ArrowRight') box.style.left = `${Math.min(geom.displayW - box.offsetWidth, left + step)}px`;
        else if (e.key === 'ArrowUp') box.style.top = `${Math.max(0, top - step)}px`;
        else if (e.key === 'ArrowDown') box.style.top = `${Math.min(geom.displayH - box.offsetHeight, top + step)}px`;
        else if (e.key === 'Delete' || e.key === 'Backspace') await handleRemoveBox(ann);
        else handled = false;
        if (handled) {
            e.preventDefault();
            if (!['Delete', 'Backspace'].includes(e.key)) await persistBoxGeometry(box, ann, false);
        }
    }

    async function persistBoxGeometry(box, ann, rescan = false) {
        if (!selectedPanel || !ann.region_id) return;
        const geom = getOverlayGeometry();
        if (!geom) return;
        const region = displayRectToImageRegion(
            parseFloat(box.style.left) || 0,
            parseFloat(box.style.top) || 0,
            parseFloat(box.style.width) || 0,
            parseFloat(box.style.height) || 0,
            geom,
        );
        try {
            await API.overrideRegion(selectedPanel.filename, ann.region_id, {
                x: region.x,
                y: region.y,
                width: region.width,
                height: region.height,
                vertical: region.orientation === 'vertical',
            });
            if (rescan) {
                await rescanAfterBoxEdit('Box edited. Rescanning...');
            } else {
                await refreshCacheStatus();
            }
        } catch (err) {
            setDebugStatus('Could not save box edit: ' + err.message);
        }
    }

    async function handleRemoveBox(ann) {
        if (!selectedPanel || !ann.region_id) return;
        const ok = window.confirm('Remove this OCR box and rescan?');
        if (!ok) return;
        try {
            await API.deleteRegion(selectedPanel.filename, ann.region_id);
            pinnedRegionId = null;
            await rescanAfterBoxEdit('Box removed. Rescanning...');
        } catch (err) {
            setDebugStatus('Could not remove box: ' + err.message);
        }
    }

    async function handleOrientationToggle(ann) {
        if (!selectedPanel || !ann.region_id) return;
        const current = ann.recognized_orientation || (ann.vertical ? 'vertical' : 'horizontal');
        const next = current === 'vertical' ? 'horizontal' : 'vertical';
        const ok = window.confirm(`Recompute this OCR box as ${next}?`);
        if (!ok) return;
        try {
            await API.overrideRegion(selectedPanel.filename, ann.region_id, { orientation: next });
            const data = await API.recomputeRegion(selectedPanel.filename, ann.region_id, getScanOptions());
            latestScan = data.result || latestScan;
            if (latestScan) {
                renderDebugPanel(latestScan);
                renderOverlays(latestScan.annotations || [], latestScan.image_width, latestScan.image_height);
                showTranslatedPanel(latestScan.translated_image_url);
            }
            refreshCacheStatus();
        } catch (err) {
            setDebugStatus('Orientation recompute failed: ' + err.message);
        }
    }

    function qualityFromConfidence(confidence) {
        const value = Number(confidence) || 0;
        if (value >= 0.78) return 'good';
        if (value >= 0.52) return 'warn';
        return 'bad';
    }

    function formatConfidence(confidence) {
        const value = Number(confidence);
        if (!Number.isFinite(value)) return 'n/a';
        return `${Math.round(value * 100)}%`;
    }

    function createDebugPreview(label, url) {
        const figure = document.createElement('figure');
        figure.className = 'reader-debug-preview';

        const img = document.createElement('img');
        img.src = new URL(url, window.location.origin).toString();
        img.alt = label;
        img.loading = 'lazy';

        const caption = document.createElement('figcaption');
        caption.textContent = label;

        figure.appendChild(img);
        figure.appendChild(caption);
        return figure;
    }

    return { init, loadPanels };
})();
