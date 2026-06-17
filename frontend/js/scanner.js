/**
 * Scanner Mode - OCR + Übersetzung von Manga Panels
 * Hover über erkannten Text zeigt Übersetzung direkt auf dem Panel
 */
const Scanner = (() => {
    const TRANSLATION_PENDING_TEXT = 'Translation not available yet.';
    const UNCOMPUTED_TEXT = 'Please run scan to see contents.';
    const RABBITHOLE_LAYER_DEFS = [
        { id: 'words', label: 'Lexical Break', upcoming: false },
        { id: 'morphemes', label: 'Morpheme Break', upcoming: false },
        { id: 'phrases', label: 'Phrases', upcoming: true },
        { id: 'stems', label: 'Stems', upcoming: true },
        { id: 'grammar', label: 'Grammar', upcoming: true },
    ];
    const DETECTOR_DOC = {
        source: 'comic-text-detector ONNX pipeline via backend/services/detection/region_detector.py',
        stages: 'letterbox resize -> YOLOv5 blocks + UNet mask + DBNet lines -> grouping -> mask refinement -> reading-order sort',
        mask: 'The raw mask is a coarse text-region prediction, not a speech-bubble boundary.',
        refined_mask: 'The refined mask is a per-text-block cleanup pass around detected text, useful as a candidate text area but still not a true bubble extraction.',
        bubble: 'Bubble allocation is currently heuristic: bright connected regions around the text evidence are estimated as likely balloon interiors and exposed as placement hints.',
        next_step: 'A cleaner pipeline would keep text detection, OCR reading, and bubble allocation separate, with a seeded wand/flood-fill style bubble pass using OCR text geometry as support evidence.',
    };

    let selectedPanel = null;
    let ocrText = '';
    let latestScan = null;
    let _panelsLoaded = false;
    let controlsHideTimer = null;
    const MAX_OPEN_POPUPS = 6;
    let openPopupRegionIds = new Set();
    let popupOpenOrder = [];
    const popupPositionsByRegion = new Map();
    const rabbitholeCardStates = new Map();

    function elementRectWithinBounds(element, boundsRect) {
        const rect = element.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        return {
            left: rect.left - boundsRect.left,
            top: rect.top - boundsRect.top,
            right: rect.right - boundsRect.left,
            bottom: rect.bottom - boundsRect.top,
        };
    }

    let editMode = false;
    let addBoxMode = false;
    let draftBox = null;
    let engineControlsLoaded = false;
    let latestCacheStatus = null;
    let lastScannedOcrFingerprint = null;
    let runtimeDownloadRunning = false;
    let activeDebugTab = 'rabbithole';
    let activeVisionOverlay = 'none';
    const DEBUG_PANEL_DEFAULT_WIDTH = 620;
    const DEBUG_PANEL_MIN_WIDTH = 280;
    const DEBUG_PANEL_MAX_SCALE = 2;
    function init() {
        ScannerSettings.restore();
        bindEvents();
        initializeDebugPanelWidth();
        loadEngineControls().catch(err => {
            console.warn('Could not initialize engine controls:', err);
        });
    }

    function bindEvents() {
        document.getElementById('upload-area').addEventListener('click', () => {
            document.getElementById('panel-upload').click();
        });

        document.getElementById('panel-upload').addEventListener('change', handleUpload);
        document.getElementById('btn-scan').addEventListener('click', handleScan);
        document.getElementById('btn-translate-panel').addEventListener('click', handleTranslate);
        document.getElementById('btn-scan-options').addEventListener('click', toggleScanOptions);
        document.getElementById('scan-translation-engine')?.addEventListener('change', handleTranslationSettingsChanged);
        document.getElementById('scan-translation-model')?.addEventListener('change', handleTranslationSettingsChanged);
        document.getElementById('btn-edit-boxes').addEventListener('click', toggleEditMode);
        document.getElementById('btn-add-box').addEventListener('click', toggleAddBoxMode);
        document.getElementById('btn-debug-toggle').addEventListener('click', () => toggleDebugPanel());
        document.getElementById('btn-debug-close').addEventListener('click', () => toggleDebugPanel(false));
        document.querySelectorAll('[data-debug-tab]').forEach(btn => {
            btn.addEventListener('click', () => setDebugTab(btn.dataset.debugTab));
        });
        document.getElementById('btn-reader-fullscreen').addEventListener('click', openReaderFullscreen);
        document.getElementById('btn-fullscreen-close').addEventListener('click', closeReaderFullscreen);
        document.getElementById('btn-clear-panel-cache').addEventListener('click', () => handleCacheClear());
        document.querySelectorAll('[data-cache-kind]').forEach(btn => {
            btn.addEventListener('click', () => handleCacheClear(btn.dataset.cacheKind));
        });
        document.getElementById('btn-reset-scan-settings')?.addEventListener('click', resetScanSettings);
        document.getElementById('btn-refresh-runtime')?.addEventListener('click', () => refreshRuntimeStatus());
        document.getElementById('btn-download-ocr-assets')?.addEventListener('click', downloadOcrAssets);
        // Home Ollama controls removed
        document.getElementById('translation-slider').addEventListener('input', (e) => {
            setTranslationReveal(Number(e.target.value));
        });
        bindTranslationHandle();
        bindReaderPanelControls();
        bindBoxCreation();
        document.querySelector('.scan-combo')?.appendChild(document.getElementById('scan-options-panel'));
        // Home Ollama controls removed
        document.getElementById('btn-translation-toggle').addEventListener('click', () => {
            const slider = document.getElementById('translation-slider');
            const next = Number(slider.value) >= 50 ? 0 : 100;
            setTranslationReveal(next);
        });
        [
            'scan-detection-confidence',
            'scan-detection-nms',
            'scan-detection-mask',
            'scan-detection-box',
            'scan-detection-max-regions',
            'scan-ocr-quality',
            'scan-preprocessing-set',
            'scan-vertical-preference',
            'scan-rotation-margin',
            'scan-crop-upscale',
            'scan-crop-padding-ratio',
            'scan-semantic-rerank',
            'scan-rotated-variants',
            'scan-bubble-search-scale',
            'scan-bubble-wand',
            'scan-bubble-overlap',
            'scan-target-lang',
            'scan-translation-style',
            'scan-temperature',
        ].forEach(id => {
            document.getElementById(id)?.addEventListener('change', () => {
                ScannerSettings.persist();
                markScanSettingsChanged();
            });
        });
        document.getElementById('scanner-panel-img').addEventListener('load', rerenderLatestOverlays);
        window.addEventListener('resize', rerenderLatestOverlays);
        window.addEventListener('resize', constrainDebugPanelWidth);
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeReaderFullscreen();
            }
        });
        bindDebugPanelResize();
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
            const [translation, ollama] = await Promise.all([
                API.getTranslationEngines(),
                API.getOllamaModels(),
            ]);
            fillSelect('scan-translation-engine', translation.engines || [], 'ollama');
            fillOllamaModelSelect('scan-translation-model', ollama);
            ScannerSettings.restore();
            updateTranslationModelVisibility();
            engineControlsLoaded = true;
        } catch (err) {
            console.warn('Could not load engine controls:', err);
            // Home Ollama status removed
        }
    }

    // Home Ollama model loader removed

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
        const models = ollama.models || [];
        select.innerHTML = '<option value="">Auto</option>';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            option.selected = model === ollama.preferred_model;
            select.appendChild(option);
        });
    }

    // Home Ollama status updater removed

    // Home Ollama persistence removed

    function updateTranslationModelVisibility() {
        const engine = document.getElementById('scan-translation-engine')?.value || 'ollama';
        const field = document.getElementById('scan-ollama-model-field');
        const select = document.getElementById('scan-translation-model');
        const useOllama = engine === 'ollama';
        if (field) field.hidden = !useOllama;
        if (select) {
            select.disabled = !useOllama;
            if (!useOllama) select.value = '';
        }
    }

    async function handleTranslationSettingsChanged() {
        updateTranslationModelVisibility();
        ScannerSettings.persist();
        await syncTranslatedPanelForCurrentSelection();
    }

    async function toggleScanOptions(force) {
        const panel = document.getElementById('scan-options-panel');
        const btn = document.getElementById('btn-scan-options');
        const nextOpen = typeof force === 'boolean' ? force : panel.classList.contains('hidden');
        if (nextOpen) {
            toggleDebugPanel(false);
        }
        panel.classList.toggle('hidden', !nextOpen);
        btn.setAttribute('aria-expanded', String(nextOpen));
        if (nextOpen) {
            await loadEngineControls();
            await refreshRuntimeStatus();
        }
    }

    function getScanOptions() {
        return ScannerSettings.getOptions();
    }

    function resetScanSettings() {
        ScannerSettings.reset();
        updateTranslationModelVisibility();
        markScanSettingsChanged();
    }

    function setRuntimeRow(id, available, value, detail = '') {
        const row = document.getElementById(id);
        if (!row) return;
        row.classList.toggle('runtime-ready', Boolean(available));
        row.classList.toggle('runtime-missing', !available);
        row.querySelector('.runtime-status-value').textContent = value;
        row.title = detail || value;
    }

    function renderRuntimeStatus(status, error = null) {
        const note = document.getElementById('runtime-status-note');
        const downloadBtn = document.getElementById('btn-download-ocr-assets');
        if (error || !status) {
            setRuntimeRow('runtime-mangaocr-package', false, 'Unknown');
            setRuntimeRow('runtime-mangaocr-cache', false, 'Unknown');
            setRuntimeRow('runtime-detector', false, 'Unknown');
            setRuntimeRow('runtime-ollama', false, 'Unknown');
            if (note) note.textContent = error ? `Runtime check failed: ${error.message}` : 'Runtime status unavailable.';
            if (downloadBtn) downloadBtn.disabled = runtimeDownloadRunning;
            return;
        }

        const ocr = status.ocr || {};
        const pkg = ocr.package || {};
        const cache = ocr.mangaocr_cache || {};
        const detector = ocr.detector || {};
        const ollama = status.ollama || {};

        setRuntimeRow('runtime-mangaocr-package', pkg.available, pkg.available ? 'Ready' : 'Missing', pkg.error);
        setRuntimeRow('runtime-mangaocr-cache', cache.available, cache.available ? 'Ready' : 'Missing', cache.error);
        setRuntimeRow('runtime-detector', detector.available, detector.available ? 'Ready' : 'Missing', detector.error);
        setRuntimeRow(
            'runtime-ollama',
            ollama.available,
            ollama.available ? 'Ready' : (ollama.reachable ? 'Model missing' : 'Offline'),
            ollama.error || ollama.configured_model || ''
        );

        if (downloadBtn) {
            downloadBtn.disabled = runtimeDownloadRunning || !pkg.available;
        }
        if (note) {
            const warnings = status.warnings || [];
            note.textContent = warnings.length ? warnings.join(' · ') : 'Runtime ready.';
        }
    }

    async function refreshRuntimeStatus() {
        const note = document.getElementById('runtime-status-note');
        if (note && !runtimeDownloadRunning) note.textContent = 'Checking runtime...';
        try {
            const status = await API.getRuntimeStatus(true);
            renderRuntimeStatus(status);
        } catch (err) {
            renderRuntimeStatus(null, err);
        }
    }

    async function downloadOcrAssets() {
        const btn = document.getElementById('btn-download-ocr-assets');
        const note = document.getElementById('runtime-status-note');
        runtimeDownloadRunning = true;
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Downloading...';
        }
        if (note) note.textContent = 'Downloading OCR assets...';
        try {
            const status = await API.downloadOcrAssets();
            renderRuntimeStatus(status);
        } catch (err) {
            renderRuntimeStatus(null, err);
        } finally {
            runtimeDownloadRunning = false;
            if (btn) {
                btn.textContent = 'Download OCR assets';
                btn.disabled = false;
            }
            await refreshRuntimeStatus();
        }
    }

    function getOcrFingerprint() {
        return ScannerSettings.getOcrFingerprint();
    }

    function setTranslateEnabled(enabled) {
        const btn = document.getElementById('btn-translate-panel');
        if (btn) btn.disabled = !enabled;
    }

    function markScanSettingsChanged() {
        if (!lastScannedOcrFingerprint) return;
        const stillMatchesScan = getOcrFingerprint() === lastScannedOcrFingerprint;
        const annotations = latestScan?.annotations || [];
        setTranslateEnabled(Boolean(latestScan?.success && stillMatchesScan && hasComputedText(annotations) && !annotations.some(isUncomputedAnnotation)));
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
            ['rabbithole', 'Rabbithole'],
            ['translation', 'Translation'],
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
            if (kind === 'ocr') {
                setDebugStatus('OCR cache cleared. Current Reader view stays available until reload or the next edit.');
                await refreshCacheStatus();
                return;
            }
            if (!kind) {
                latestScan = null;
                lastScannedOcrFingerprint = null;
                openPopupRegionIds = new Set();
                popupOpenOrder = [];
                popupPositionsByRegion.clear();
                setTranslateEnabled(false);
                document.getElementById('ocr-overlay').innerHTML = '';
                document.getElementById('ocr-popup-layer').innerHTML = '';
                resetTranslationView();
                setDebugStatus('Panel data cleared.');
                await loadPanelBoxes(selectedPanel.filename);
            } else {
                if (kind === 'translation') {
                    resetTranslationView();
                    latestScan?.annotations?.forEach(ann => delete ann.translated);
                }
                setDebugStatus(`${kind.toUpperCase()} cache cleared. Box edits were preserved.`);
            }
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

    async function selectPanel(panel, imgEl) {
        document.querySelectorAll('.panel-thumb.selected').forEach(el => el.classList.remove('selected'));
        imgEl.classList.add('selected');

        selectedPanel = panel;
        const panelImg = document.getElementById('scanner-panel-img');
        panelImg.src = API.panelImageUrl(panel.path || panel.filename);
        panelImg.classList.remove('hidden');
        document.getElementById('btn-scan').disabled = false;
        setTranslateEnabled(false);
        document.getElementById('ocr-overlay').innerHTML = '';
        latestScan = null;
        lastScannedOcrFingerprint = null;
        resetTranslationView();
        setDebugStatus('Ready to scan.');
        ocrText = '';
        openPopupRegionIds = new Set();
        popupOpenOrder = [];
        popupPositionsByRegion.clear();
        rabbitholeCardStates.clear();
        document.getElementById('ocr-popup-layer').innerHTML = '';
        document.getElementById('btn-add-box').disabled = !editMode;
        refreshCacheStatus();
        await loadPanelBoxes(panel.filename);
    }

    async function loadPanelBoxes(filename) {
        try {
            await loadEngineControls();
            const data = await API.getRegions(filename);
            if (!selectedPanel || selectedPanel.filename !== filename) return;
            applyRegionPayload(data);
            const hydrated = await hydrateCachedPanelStages(filename, data);
            if (hydrated) {
                applyRegionPayload(hydrated);
                showTranslatedPanel(hydrated.translated_image_url);
                await refreshCacheStatus();
                return;
            }
            if ((data.annotations || []).length) {
                setDebugStatus('Box edits loaded. Run Scan to compute dashed boxes.');
            }
        } catch (err) {
            console.warn('Could not load panel boxes:', err);
        }
    }

    async function hydrateCachedPanelStages(filename, baseData) {
        const annotations = baseData.annotations || [];
        const buckets = baseData.panel_cache?.buckets || {};
        const hasCachedOcr = Boolean(buckets.ocr?.has_cache) && hasComputedText(annotations) && !annotations.some(isUncomputedAnnotation);
        if (!hasCachedOcr) return null;

        const requests = [];
        if (buckets.rabbithole?.has_cache) {
            requests.push(API.getCachedRabbithole(filename, getScanOptions()).then(result => ({ kind: 'rabbithole', result })));
        }
        if (buckets.translation?.has_cache) {
            requests.push(API.getCachedTranslation(filename, getScanOptions()).then(result => ({ kind: 'translation', result })));
        }
        if (!requests.length) return null;

        const settled = await Promise.allSettled(requests);
        let hydrated = null;
        let rabbitholeResult = null;

        settled.forEach(entry => {
            if (entry.status !== 'fulfilled') return;
            if (entry.value.result?.cache_miss) return;
            if (entry.value.kind === 'rabbithole') {
                rabbitholeResult = entry.value.result;
                hydrated = mergeHydratedScanResult(hydrated || baseData, entry.value.result);
                return;
            }
            if (entry.value.kind === 'translation') {
                hydrated = mergeHydratedScanResult(hydrated || baseData, entry.value.result);
            }
        });

        if (!hydrated) return null;
        if (rabbitholeResult) {
            hydrated = mergeHydratedScanResult(hydrated, rabbitholeResult);
        }
        return hydrated;
    }

    async function syncTranslatedPanelForCurrentSelection() {
        if (!selectedPanel || !latestScan) {
            markScanSettingsChanged();
            return;
        }
        const annotations = latestScan.annotations || [];
        if (!hasComputedText(annotations) || annotations.some(isUncomputedAnnotation)) {
            markScanSettingsChanged();
            return;
        }
        if ((document.getElementById('scan-translation-engine')?.value || 'ollama') !== 'ollama') {
            markScanSettingsChanged();
            return;
        }

        try {
            const result = await API.getCachedTranslation(selectedPanel.filename, getScanOptions());
            if (!result || result.cache_miss) {
                markScanSettingsChanged();
                return;
            }
            latestScan = mergeHydratedScanResult(latestScan, result);
            ocrText = latestScan.text || ocrText;
            renderDebugPanel(latestScan);
            renderOverlays(latestScan.annotations || [], latestScan.image_width, latestScan.image_height);
            showTranslatedPanel(latestScan.translated_image_url);
            await refreshCacheStatus();
        } catch (err) {
            console.warn('Could not hydrate cached translation for current model:', err);
        } finally {
            markScanSettingsChanged();
        }
    }

    function mergeHydratedScanResult(baseResult, stageResult) {
        const merged = {
            ...baseResult,
            ...stageResult,
        };
        const baseAnnotations = baseResult.annotations || [];
        const stageAnnotations = stageResult.annotations || [];
        merged.annotations = baseAnnotations.map((ann, index) => {
            const stageAnn = stageAnnotations[index] || {};
            return {
                ...ann,
                ...stageAnn,
                ocr_debug: stageAnn.ocr_debug || ann.ocr_debug,
                rabbithole: stageAnn.rabbithole || ann.rabbithole,
            };
        });
        merged.image_width = stageResult.image_width || baseResult.image_width;
        merged.image_height = stageResult.image_height || baseResult.image_height;
        merged.text = stageResult.text || baseResult.text || '';
        return merged;
    }

    function annotationHydrationSignature(ann) {
        return JSON.stringify({
            text: ann?.text || '',
            bbox: ann?.bbox || null,
            vertical: Boolean(ann?.vertical),
            recognized_orientation: ann?.recognized_orientation || '',
            computed: !(ann?.uncomputed || ann?.computed === false),
        });
    }

    function shouldPreserveAnnotationPayload(previousAnn, nextAnn) {
        if (!previousAnn || !nextAnn) return false;
        if (isUncomputedAnnotation(previousAnn) || isUncomputedAnnotation(nextAnn)) return false;
        return annotationHydrationSignature(previousAnn) === annotationHydrationSignature(nextAnn);
    }

    function mergeRegionPayloadAnnotations(previousAnnotations, nextAnnotations) {
        const previousById = new Map(
            (previousAnnotations || []).map((ann, index) => [getAnnotationRegionId(ann, index), ann])
        );
        return (nextAnnotations || []).map((ann, index) => {
            const previousAnn = previousById.get(getAnnotationRegionId(ann, index));
            if (!shouldPreserveAnnotationPayload(previousAnn, ann)) {
                return { ...ann };
            }
            return {
                ...previousAnn,
                ...ann,
                ocr_debug: ann.ocr_debug || previousAnn.ocr_debug,
            };
        });
    }

    function applyRegionPayload(data) {
        const previousScan = latestScan || {};
        latestScan = {
            ...previousScan,
            success: true,
            text: data.text || '',
            annotations: mergeRegionPayloadAnnotations(previousScan.annotations || [], data.annotations || []),
            image_width: data.image_width || previousScan.image_width,
            image_height: data.image_height || previousScan.image_height,
            panel_cache: data.panel_cache,
        };
        ocrText = latestScan.text || '';
        renderOverlays(latestScan.annotations, latestScan.image_width, latestScan.image_height);
        renderDebugPanel(latestScan);
        setTranslateEnabled(hasComputedText(latestScan.annotations) && !latestScan.annotations.some(isUncomputedAnnotation));
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

    async function handleScan() {
        if (!selectedPanel) return;

        const btn = document.getElementById('btn-scan');
        const hadExistingScan = Boolean(latestScan?.annotations?.length);
        let scanSucceeded = false;

        btn.disabled = true;
        setTranslateEnabled(false);
        btn.classList.add('loading');
        btn.innerHTML = '<span class="spinner"></span>Scanning...';
        if (!hadExistingScan) {
            setDebugStatus('Running OCR...');
        }

        try {
            const ocrResult = await API.scanPanel(selectedPanel.filename, getScanOptions());
            resetTranslationView();
            rabbitholeCardStates.clear();
            latestScan = ocrResult;
            lastScannedOcrFingerprint = getOcrFingerprint();
            ocrText = ocrResult.text || '';
            renderDebugPanel(ocrResult);
            renderOverlays(ocrResult.annotations || [], ocrResult.image_width, ocrResult.image_height);
            setTranslateEnabled(hasComputedText(ocrResult.annotations || []) && !(ocrResult.annotations || []).some(isUncomputedAnnotation));
            try {
                const rabbitholeResult = await API.buildRabbithole(selectedPanel.filename, getScanOptions());
                latestScan = rabbitholeResult;
                ocrText = rabbitholeResult.text || ocrText;
                renderDebugPanel(rabbitholeResult);
                renderOverlays(rabbitholeResult.annotations || [], rabbitholeResult.image_width, rabbitholeResult.image_height);
                setTranslateEnabled(hasComputedText(rabbitholeResult.annotations || []));
            } catch (rabbitErr) {
                renderDebugPanel(ocrResult);
                renderOverlays(ocrResult.annotations || [], ocrResult.image_width, ocrResult.image_height);
                setTranslateEnabled(hasComputedText(ocrResult.annotations || []));
                setPaneMessage('reader-rabbithole-content', `Rabbithole error: ${rabbitErr.message}`);
            }
            scanSucceeded = true;
            refreshCacheStatus();
        } catch (err) {
            setDebugStatus('Scan error: ' + err.message);
        } finally {
            if (!scanSucceeded) {
                markScanSettingsChanged();
            }
            btn.disabled = !selectedPanel;
            btn.classList.remove('loading');
            btn.textContent = 'Scan';
        }
    }

    function hasComputedText(annotations) {
        return annotations.some(ann => ann.computed !== false && (ann.text || '').trim());
    }

    function isUncomputedAnnotation(ann) {
        return ann.uncomputed || ann.computed === false;
    }

    async function handleTranslate() {
        if (!selectedPanel || !latestScan) return;

        const btn = document.getElementById('btn-translate-panel');

        btn.disabled = true;
        btn.classList.add('loading');
        btn.innerHTML = '<span class="spinner"></span>Translating...';
        setDebugStatus('Running translation...');

        try {
            const translatedResult = await API.translatePanel(selectedPanel.filename, getScanOptions());
            latestScan = translatedResult;
            lastScannedOcrFingerprint = getOcrFingerprint();
            ocrText = translatedResult.text || ocrText;
            renderDebugPanel(translatedResult);
            renderOverlays(translatedResult.annotations || [], translatedResult.image_width, translatedResult.image_height);
            showTranslatedPanel(translatedResult.translated_image_url);
            if (translatedResult.render_warnings && translatedResult.render_warnings.length) {
                console.warn('Panel render warnings:', translatedResult.render_warnings);
            }
            refreshCacheStatus();
        } catch (err) {
            setDebugStatus('Translation error: ' + err.message);
        } finally {
            markScanSettingsChanged();
            btn.classList.remove('loading');
            btn.textContent = 'Translate';
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
        if (nextOpen) {
            toggleScanOptions(false);
        }
        panel.classList.toggle('hidden', !nextOpen);
        btn.setAttribute('aria-expanded', String(nextOpen));
        if (nextOpen) {
            constrainDebugPanelWidth();
        }
    }

    function setDebugTab(tabName) {
        activeDebugTab = tabName === 'diagnose' ? 'diagnose' : 'rabbithole';
        document.querySelectorAll('[data-debug-tab]').forEach(btn => {
            const isActive = btn.dataset.debugTab === activeDebugTab;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-selected', String(isActive));
        });
        document.getElementById('reader-rabbithole-content').classList.toggle('hidden', activeDebugTab !== 'rabbithole');
        document.getElementById('reader-diagnose-content').classList.toggle('hidden', activeDebugTab !== 'diagnose');
    }

    function setPaneMessage(elementId, message) {
        const content = document.getElementById(elementId);
        if (!content) return;
        content.innerHTML = '';
        const empty = document.createElement('div');
        empty.className = 'reader-debug-empty';
        empty.textContent = message;
        content.appendChild(empty);
    }

    function setDebugStatus(message) {
        setPaneMessage('reader-rabbithole-content', message);
        setPaneMessage('reader-diagnose-content', message);
    }

    function renderDebugPanel(result) {
        renderRabbitholePanel(result);
        renderDiagnosePanel(result);
        setDebugTab(activeDebugTab);
    }

    function renderRabbitholePanel(result) {
        const content = document.getElementById('reader-rabbithole-content');
        const annotations = result.annotations || [];
        content.innerHTML = '';

        if (!annotations.length) {
            if (!result.scan_trace?.length) setDebugStatus('No text recognized.');
            return;
        }

        const frag = document.createDocumentFragment();
        annotations.forEach((ann, index) => {
            frag.appendChild(createRabbitholeCard(ann, index, false));
        });

        content.appendChild(frag);
    }

    function createVisionOverlayControls(result) {
        const section = document.createElement('section');
        section.className = 'reader-debug-entry reader-vision-controls';
        section.appendChild(createDebugBlockTitle('Vision overlay'));

        const controls = document.createElement('div');
        controls.className = 'reader-overlay-toggle-group';
        [
            { id: 'none', label: 'Off' },
            { id: 'text', label: 'Text geometry' },
            { id: 'bubble', label: 'Bubble estimate' },
            { id: 'all', label: 'Combined' },
        ].forEach(option => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = `reader-overlay-toggle${activeVisionOverlay === option.id ? ' active' : ''}`;
            button.textContent = option.label;
            button.setAttribute('aria-pressed', String(activeVisionOverlay === option.id));
            button.addEventListener('click', () => {
                activeVisionOverlay = option.id;
                if (latestScan) {
                    rerenderLatestOverlays();
                    renderDebugPanel(latestScan);
                }
            });
            controls.appendChild(button);
        });
        section.appendChild(controls);

        const summary = document.createElement('div');
        summary.className = 'reader-debug-meta';
        const annotations = result.annotations || [];
        const lineCount = annotations.reduce((count, ann) => count + ((ann.lines || []).length || 0), 0);
        const bubbleCount = annotations.filter(ann => normalizeRectBox(getVisionDebug(ann).bubble_box)).length;
        const placementCount = annotations.filter(ann => normalizeRectBox(getVisionDebug(ann).placement_box)).length;
        [
            `regions: ${annotations.length}`,
            `line polygons: ${lineCount}`,
            `candidate areas: ${annotations.filter(ann => computeCandidateTextArea(ann)).length}`,
            `bubble estimates: ${bubbleCount}`,
            `placement boxes: ${placementCount}`,
        ].forEach(item => {
            const pill = document.createElement('span');
            pill.textContent = item;
            summary.appendChild(pill);
        });
        section.appendChild(summary);
        return section;
    }

    function createDetectorDocumentationBlock(result) {
        const section = document.createElement('section');
        section.className = 'reader-debug-entry';
        section.appendChild(createDebugBlockTitle('Recognition stack'));

        const body = document.createElement('div');
        body.className = 'reader-debug-detector-copy';
        body.textContent = DETECTOR_DOC.stages;
        section.appendChild(body);

        section.appendChild(createKeyValueGrid({
            source: DETECTOR_DOC.source,
            text_geometry: 'Detector block, line polygons, OCR crop, raw line candidate, and ink-expanded candidate text area are all shown separately now.',
            raw_mask: DETECTOR_DOC.mask,
            refined_mask: DETECTOR_DOC.refined_mask,
            bubble_estimate: DETECTOR_DOC.bubble,
            next_step: DETECTOR_DOC.next_step,
            overlay: activeVisionOverlay === 'none'
                ? 'Enable a vision overlay to inspect text geometry, estimated bubble interiors, and derived placement boxes on top of the panel.'
                : 'Bounding boxes remain the source of truth. The extra overlays are recognition diagnostics and placement hints.',
        }));

        if (!(result.annotations || []).some(ann => (ann.lines || []).length)) {
            const warning = document.createElement('div');
            warning.className = 'reader-debug-warnings';
            warning.textContent = 'This panel does not expose detector line polygons yet, so the candidate area estimate will be coarse.';
            section.appendChild(warning);
        }
        return section;
    }

    function renderDiagnosePanel(result) {
        const content = document.getElementById('reader-diagnose-content');
        const annotations = result.annotations || [];
        content.innerHTML = '';

        content.appendChild(createVisionOverlayControls(result));
        content.appendChild(createDetectorDocumentationBlock(result));

        if (result.scan_trace && result.scan_trace.length) {
            content.appendChild(renderScanTrace(result.scan_trace, result));
        }

        if (result.translation_prompt_payload) {
            const promptBlock = document.createElement('div');
            promptBlock.className = 'reader-debug-entry';
            promptBlock.appendChild(createDebugBlockTitle('Prompt payload'));
            promptBlock.appendChild(createKeyValueGrid(result.translation_prompt_payload));
            content.appendChild(promptBlock);
        }

        if (!annotations.length) {
            if (!result.scan_trace?.length) setPaneMessage('reader-diagnose-content', 'No text recognized.');
            return;
        }

        const frag = document.createDocumentFragment();
        annotations.forEach((ann, index) => {
            const debug = ann.ocr_debug || {};
            const vision = getVisionDebug(ann);
            const candidateBox = computeCandidateTextArea(ann);
            const bubbleBox = normalizeRectBox(vision.bubble_box);
            const placementBox = normalizeRectBox(vision.placement_box);
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
                `box source: ${ann.box_source || 'detected'}`,
                `vertical: ${debug.detector?.vertical ? 'yes' : 'no'}`,
                `angle: ${debug.detector?.angle ?? ann.angle ?? 0}`,
                `font: ${debug.detector?.font_size ?? ann.font_size ?? 0}`,
                `bubble: ${bubbleBox ? `${Math.round((Number(vision.bubble_confidence) || 0) * 100)}%` : 'n/a'}`,
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
            en.textContent = ann.translated?.trim() || TRANSLATION_PENDING_TEXT;

            entry.appendChild(title);
            entry.appendChild(meta);
            const actions = createRegionActionRow(ann);
            if (actions) entry.appendChild(actions);
            entry.appendChild(jp);
            entry.appendChild(en);

            const geometry = document.createElement('div');
            geometry.className = 'reader-debug-boxes';
            geometry.appendChild(createDebugBlockTitle('Geometry + placement'));
            geometry.appendChild(createKeyValueGrid({
                region_id: ann.region_id,
                reading_order: ann.reading_order,
                computed: ann.computed !== false,
                recognized_orientation: ann.recognized_orientation,
                orientation_source: ann.orientation_source,
                detected_box: boxToDebugText(debug.detected_box),
                crop_box: boxToDebugText(debug.crop_box),
                line_candidate_box: boxToDebugText(vision.line_candidate_box),
                candidate_box: boxToDebugText(candidateBox),
                bubble_box: boxToDebugText(bubbleBox),
                placement_box: boxToDebugText(placementBox),
                search_box: boxToDebugText(vision.search_box),
            }));
            entry.appendChild(geometry);

            const signals = document.createElement('div');
            signals.className = 'reader-debug-boxes';
            signals.appendChild(createDebugBlockTitle('Vision signals'));
            signals.appendChild(createKeyValueGrid({
                line_count: vision.line_count ?? (ann.lines || []).length,
                threshold: vision.threshold,
                bubble_source: vision.source,
                bubble_confidence: vision.bubble_confidence,
                bubble_area_ratio: vision.bubble_area_ratio,
                bubble_fill_ratio: vision.bubble_fill_ratio,
                bubble_overlap_ratio: vision.bubble_overlap_ratio,
                detector_vertical: debug.detector?.vertical,
                vertical_candidate: debug.vertical_candidate,
            }));
            entry.appendChild(signals);

            if (debug.score_breakdown) {
                const { core, bias } = splitScoreBreakdown(debug.score_breakdown);
                const breakdown = document.createElement('div');
                breakdown.className = 'reader-debug-boxes';
                breakdown.appendChild(createDebugBlockTitle('Score breakdown'));
                breakdown.appendChild(createKeyValueGrid(core));
                entry.appendChild(breakdown);

                if (!isEmptyDebugValue(bias.orientation_bias)) {
                    const biasBlock = document.createElement('div');
                    biasBlock.className = 'reader-debug-boxes';
                    biasBlock.appendChild(createDebugBlockTitle('Biases'));
                    biasBlock.appendChild(createKeyValueGrid(bias));
                    entry.appendChild(biasBlock);
                }
            }

            if (debug.selection_note) {
                const note = document.createElement('div');
                note.className = 'reader-debug-boxes';
                note.textContent = `Selection: ${debug.selection_note}`;
                entry.appendChild(note);
            }

            if (debug.warnings && debug.warnings.length) {
                const warnings = document.createElement('div');
                warnings.className = 'reader-debug-warnings';
                warnings.textContent = `Warnings: ${debug.warnings.join(', ')}`;
                entry.appendChild(warnings);
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
                    const candidateScoreBreakdown = splitScoreBreakdown(candidate.score_breakdown);
                    row.appendChild(createKeyValueGrid({
                        recognized_orientation: candidate.recognized_orientation,
                        width: candidate.width,
                        height: candidate.height,
                        baseline_score: candidate.baseline_score ?? candidate.legacy_score,
                        score_breakdown: candidateScoreBreakdown.core,
                    }));
                    if (!isEmptyDebugValue(candidateScoreBreakdown.bias.orientation_bias)) {
                        row.appendChild(createKeyValueGrid({
                            biases: candidateScoreBreakdown.bias,
                        }));
                    }
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
            `lookup hits: ${result.rabbithole_lookup_hits ?? 0}`,
            `lookup misses: ${result.rabbithole_lookup_misses ?? 0}`,
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
            const extras = Object.fromEntries(Object.entries(event).filter(([key]) => !['scan_id', 'stage', 'status', 'message', 'ts', 'elapsed_ms'].includes(key)));
            if (Object.keys(extras).length) {
                row.appendChild(createKeyValueGrid(extras));
            }
            section.appendChild(row);
        });

        if (result.translation_error) {
            const warning = document.createElement('div');
            warning.className = 'reader-debug-warnings';
            warning.textContent = `Translation error: ${result.translation_error}`;
            section.appendChild(warning);
        }
        if (result.rabbithole_error) {
            const warning = document.createElement('div');
            warning.className = 'reader-debug-warnings';
            warning.textContent = `Rabbithole error: ${result.rabbithole_error}`;
            section.appendChild(warning);
        }
        return section;
    }

    function getAnnotationRegionId(ann, fallback = 0) {
        return String(ann.region_id ?? ann.id ?? fallback);
    }

    function getRabbitholeData(ann) {
        return ann?.rabbithole || null;
    }

    function getRabbitholeCardState(instanceKey) {
        const key = String(instanceKey || 'side:default');
        if (!rabbitholeCardStates.has(key)) {
            rabbitholeCardStates.set(key, {
                selection: { regionId: null, unitId: null },
                layer: 'words',
            });
        }
        return rabbitholeCardStates.get(key);
    }

    function getCardLayer(instanceKey) {
        return getRabbitholeCardState(instanceKey).layer || 'words';
    }

    function getSelectedRabbitholeUnit(ann, fallback = 0, instanceKey = 'side:default') {
        const rabbit = getRabbitholeData(ann);
        const regionId = getAnnotationRegionId(ann, fallback);
        const selection = getRabbitholeCardState(instanceKey).selection;
        if (!rabbit || selection.regionId !== regionId) return null;
        return rabbit.units_by_id?.[selection.unitId] || null;
    }

    function setActiveRabbitholeUnit(regionId, unitId, instanceKey = 'side:default') {
        const nextRegion = String(regionId);
        getRabbitholeCardState(instanceKey).selection = { regionId: nextRegion, unitId: String(unitId) };
        if (latestScan) {
            renderDebugPanel(latestScan);
            renderOpenPopups();
        }
    }

    function isRabbitholeItemActive(ann, targetUnitId, instanceKey = 'side:default') {
        if (!targetUnitId) return false;
        const regionId = getAnnotationRegionId(ann);
        const selection = getRabbitholeCardState(instanceKey).selection;
        if (selection.regionId !== regionId) return false;
        return selection.unitId === targetUnitId;
    }

    function createRabbitholeTextRow(label, value, compact = false, className = '') {
        const row = document.createElement('div');
        row.className = `rabbithole-row rabbithole-text-row${compact ? ' compact' : ''}`;

        const labelEl = document.createElement('div');
        labelEl.className = 'rabbithole-row-label';
        labelEl.textContent = label;
        row.appendChild(labelEl);

        const valueEl = document.createElement('div');
        valueEl.className = `rabbithole-row-value rabbithole-text-value${className ? ` ${className}` : ''}`;
        valueEl.textContent = value || '—';
        row.appendChild(valueEl);
        return row;
    }

    function setActiveRabbitholeLayer(layerId, instanceKey = 'side:default') {
        const target = RABBITHOLE_LAYER_DEFS.find(layer => layer.id === layerId && !layer.upcoming);
        if (!target || getCardLayer(instanceKey) === target.id) return;
        getRabbitholeCardState(instanceKey).layer = target.id;
        if (latestScan) {
            renderDebugPanel(latestScan);
            renderOpenPopups();
        }
    }

    function createRabbitholeLayerTabs(compact = false, instanceKey = 'side:default') {
        const tabs = document.createElement('div');
        tabs.className = `rabbithole-layer-tabs${compact ? ' compact' : ''}`;
        const activeLayer = getCardLayer(instanceKey);

        RABBITHOLE_LAYER_DEFS.forEach(layer => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = `rabbithole-layer-tab${activeLayer === layer.id ? ' active' : ''}${layer.upcoming ? ' upcoming' : ''}`;
            button.textContent = layer.label;
            button.disabled = Boolean(layer.upcoming);
            button.setAttribute('aria-pressed', String(activeLayer === layer.id));
            if (layer.upcoming) {
                button.title = `${layer.label} is planned for the next rabbithole phase.`;
            } else {
                button.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setActiveRabbitholeLayer(layer.id, instanceKey);
                });
            }
            tabs.appendChild(button);
        });

        return tabs;
    }

    function createRabbitholeUnitRow(label, ann, items, compact = false, fallback = '', className = '', instanceKey = 'side:default') {
        const row = document.createElement('div');
        row.className = `rabbithole-row${compact ? ' compact' : ''}`;

        const labelEl = document.createElement('div');
        labelEl.className = 'rabbithole-row-label';
        labelEl.textContent = label;
        row.appendChild(labelEl);

        const valueEl = document.createElement('div');
        valueEl.className = `rabbithole-row-value${className ? ` ${className}` : ''}`;
        const regionId = getAnnotationRegionId(ann);

        (items || []).forEach(item => {
            const text = String(item?.text || '').trim();
            if (!text) return;
            const button = document.createElement('button');
            button.type = 'button';
            button.className = item.className || 'rabbithole-segment';
            const targetUnitId = String(item.unitId || '');
            // Kanji items may include context (rest of unit) shown at lower opacity
            if (item.className === 'rabbithole-kanji' && (item.prefix || item.suffix || item.context)) {
                const prefix = item.prefix ? `<span class="kanji-context">${escapeHtml(item.prefix)}</span>` : '';
                const suffix = item.suffix || item.context ? `<span class="kanji-context">${escapeHtml(item.suffix || item.context)}</span>` : '';
                button.innerHTML = `${prefix}<span class="kanji-main">${escapeHtml(text)}</span>${suffix}`;
            } else {
                button.textContent = text;
            }
            button.setAttribute('data-text', text);
            button.classList.toggle('active', Boolean(targetUnitId) && isRabbitholeItemActive(ann, targetUnitId, instanceKey));
            if (targetUnitId) {
                button.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setActiveRabbitholeUnit(regionId, targetUnitId, instanceKey);
                });
            } else {
                button.disabled = true;
            }
            valueEl.appendChild(button);
        });

        if (!valueEl.childNodes.length) {
            valueEl.textContent = fallback || '—';
        }
        row.appendChild(valueEl);
        return row;
    }

    function uniqueRabbitholeItems(items) {
        const seen = new Set();
        return (items || []).filter(item => {
            const key = `${item.unitId || ''}:${item.text || ''}:${item.className || ''}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, char => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
        }[char]));
    }

    function getUnitForSegment(rabbit, segment) {
        return rabbit?.units_by_id?.[segment?.unit_id] || null;
    }

    function getRabbitHiragana(rabbit) {
        return rabbit?.reading_hiragana || '';
    }

    function getUnitHiragana(unit) {
        return unit?.reading_hiragana || '';
    }

    function getSegmentHiragana(segment) {
        return segment?.hiragana || segment?.text || '';
    }

    function isMeaningfulRabbitholeUnit(unit) {
        if (!unit) return false;
        const text = String(unit.text || '').trim();
        if (!text) return false;
        if (unit.kind === 'kanji') return true;
        const pos = Array.isArray(unit.pos) ? unit.pos.map(String) : [];
        if (pos.some(part => part.includes('補助記号'))) return false;
        const hasJapaneseOrAlnum = Array.from(text).some(char => /[\u3040-\u30ff\u3400-\u9fff\p{L}\p{N}]/u.test(char));
        if (!hasJapaneseOrAlnum) return false;
        return unit.kind !== 'token'
            || Boolean(unit.reading_hiragana)
            || Boolean(unit.reading_romanji)
            || Boolean(unit.dictionary_entries?.length);
    }

    function isMeaningfulRabbitholeSegment(ann, segment) {
        const unit = getUnitForSegment(getRabbitholeData(ann), segment);
        return isMeaningfulRabbitholeUnit(unit);
    }

    function buildKanjiItems(ann, instanceKey = 'side:default') {
        const rabbit = getRabbitholeData(ann);
        const units = rabbit?.units_by_id || {};
        const kanjiSpans = getBreakdownSegments(ann, getCardLayer(instanceKey))
            .map(segment => getUnitForSegment(rabbit, segment))
            .filter(unit => unit && unit.kind !== 'kanji')
            .flatMap(unit => (unit.children || [])
                .map(childId => units[childId])
                .filter(child => child?.kind === 'kanji')
                .map(child => ({
                    char: child.text,
                    start: Number(child.start),
                    end: Number(child.end),
                    container: unit,
                    child,
                })));
        const seen = new Set();

        return kanjiSpans
            .filter(({ char, start, end, child }) => {
                const key = `${char}:${start}:${end}`;
                if (!child?.id || seen.has(key)) return false;
                seen.add(key);
                return true;
            })
            .map(({ char, start, container, child }) => {
                const index = start - Number(container.start);
                const containerText = String(container.text || '');
                return {
                    unitId: child.id,
                    text: char,
                    prefix: index > 0 ? containerText.slice(0, index) : '',
                    suffix: index >= 0 ? containerText.slice(index + char.length) : '',
                    className: 'rabbithole-kanji',
                };
            });
    }

    function getRabbitholeBreakdown(ann, layerId = 'words') {
        const rabbit = getRabbitholeData(ann);
        const breakdowns = rabbit?.breakdowns || {};
        if (breakdowns[layerId]) return breakdowns[layerId];
        return null;
    }

    function getBreakdownSegments(ann, layerId = 'words') {
        return (getRabbitholeBreakdown(ann, layerId)?.segments || [])
            .filter(segment => isMeaningfulRabbitholeSegment(ann, segment));
    }

    function buildSegmentItems(ann, layerId, valueSelector = segment => segment.text || '') {
        return uniqueRabbitholeItems(getBreakdownSegments(ann, layerId).map(segment => ({
            unitId: segment.unit_id,
            text: valueSelector(segment),
        })));
    }

    function buildRabbitholeItems(ann, layerId, instanceKey = 'side:default') {
        const rabbit = getRabbitholeData(ann);
        const activeLayer = getCardLayer(instanceKey);
        if (layerId === 'hiragana') {
            return buildSegmentItems(ann, activeLayer, getSegmentHiragana);
        }
        if (layerId === 'romaji') {
            return buildSegmentItems(ann, activeLayer, segment => segment.romanji || getUnitForSegment(rabbit, segment)?.reading_romanji || '');
        }
        if (layerId === 'glossary') {
            return buildSegmentItems(ann, activeLayer, segment => segment.gloss || getUnitForSegment(rabbit, segment)?.primary_meaning || '');
        }
        if (layerId === 'words' || layerId === 'morphemes') return buildSegmentItems(ann, layerId);
        return [];
    }

    function getRabbitholeUnitKindLabel(unit) {
        const labels = {
            whole: 'full text',
            word: 'segment',
            particle: 'particle',
            aux: 'auxiliary',
            suffix: 'suffix',
            token: 'token',
            kanji: 'kanji',
        };
        return labels[unit?.kind] || unit?.kind || 'unit';
    }

    function createRabbitholeBaseRows(ann, compact = false) {
        const rabbit = getRabbitholeData(ann);
        return [
            createRabbitholeTextRow('Recognized Text', ann.text || '', compact),
            createRabbitholeTextRow('Hiragana', getRabbitHiragana(rabbit), compact),
            createRabbitholeTextRow('EN', ann.translated?.trim() || TRANSLATION_PENDING_TEXT, compact, 'rabbithole-translation'),
        ];
    }

    function createRabbitholeLayerRows(ann, compact = false, instanceKey = 'side:default') {
        const layer = getCardLayer(instanceKey);
        const rows = createRabbitholeBaseRows(ann, compact);
        const label = RABBITHOLE_LAYER_DEFS.find(item => item.id === layer)?.label || 'Breakdown';

        rows.push(createRabbitholeUnitRow(label, ann, buildRabbitholeItems(ann, layer, instanceKey), compact, ann.text || '', '', instanceKey));
        rows.push(createRabbitholeUnitRow('Hiragana', ann, buildRabbitholeItems(ann, 'hiragana', instanceKey), compact, getRabbitHiragana(getRabbitholeData(ann)), '', instanceKey));

        const kanjiItems = buildKanjiItems(ann, instanceKey);
        if (kanjiItems.length) {
            rows.push(createRabbitholeUnitRow('Kanji', ann, kanjiItems, compact, 'No kanji units available.', '', instanceKey));
        }

        rows.push(createRabbitholeUnitRow('Romaji', ann, buildRabbitholeItems(ann, 'romaji', instanceKey), compact, getRabbitholeData(ann)?.reading_romanji || '', 'rabbithole-romanji-row', instanceKey));
        return rows;
    }

    function createDictionaryEntriesBlock(unit) {
        const entries = Array.isArray(unit?.dictionary_entries) ? unit.dictionary_entries : [];
        if (!entries.length) return null;

        const block = document.createElement('div');
        block.className = 'rabbithole-dictionary';
        block.appendChild(createDebugBlockTitle('Dictionary'));

        entries.slice(0, 3).forEach((entry, index) => {
            const article = document.createElement('article');
            article.className = 'rabbithole-dictionary-entry';

            const header = document.createElement('div');
            header.className = 'rabbithole-dictionary-head';
            const variants = Array.isArray(entry.variants) ? entry.variants : [];
            const primaryVariant = variants[0] || {};
            const term = document.createElement('strong');
            term.textContent = primaryVariant.written || unit.text || `Entry ${index + 1}`;
            header.appendChild(term);

            const reading = primaryVariant.reading_hiragana || '';
            if (reading && reading !== term.textContent) {
                const readingEl = document.createElement('span');
                readingEl.textContent = reading;
                header.appendChild(readingEl);
            }

            if (entry.source) {
                const sourceEl = document.createElement('span');
                sourceEl.className = 'rabbithole-dictionary-source';
                sourceEl.textContent = entry.source;
                header.appendChild(sourceEl);
            }
            article.appendChild(header);

            const tags = [
                ...(entry.priority_labels || []),
                ...(entry.priority_tags || []),
            ].filter(Boolean);
            if (tags.length) {
                const tagRow = document.createElement('div');
                tagRow.className = 'rabbithole-dictionary-tags';
                uniqueRabbitholeItems(tags.map(tag => ({ text: tag }))).slice(0, 5).forEach(tag => {
                    const chip = document.createElement('span');
                    chip.textContent = tag.text;
                    tagRow.appendChild(chip);
                });
                article.appendChild(tagRow);
            }

            const list = document.createElement('ol');
            list.className = 'rabbithole-dictionary-senses';
            (entry.senses || []).slice(0, 4).forEach(sense => {
                const glosses = Array.isArray(sense.glosses) ? sense.glosses.filter(Boolean) : [];
                if (!glosses.length) return;
                const item = document.createElement('li');
                item.textContent = glosses.join('; ');
                list.appendChild(item);
            });
            if (list.children.length) article.appendChild(list);
            block.appendChild(article);
        });

        return block;
    }

    function getUnitFeatureGroups(unit) {
        return unit?.feature_groups || {
            reading: {
                hiragana: getUnitHiragana(unit),
                romaji: unit?.reading_romanji,
            },
            meaning: {
                glossary: unit?.primary_meaning,
                alternate_glossary: unit?.alternate_meanings,
            },
            grammar: {
                segment_type: unit?.kind,
                part_of_speech: unit?.part_of_speech_labels,
                analysis_tags: unit?.pos,
                function: unit?.grammar_detail,
            },
            kanji: unit?.kanji_details,
            dictionary: {
                source: unit?.dictionary_source,
                sources: unit?.dictionary_sources,
                candidate_count: unit?.dictionary_candidate_count,
            },
            translation_context: {},
        };
    }

    function getUnitInformation(unit) {
        const features = getUnitFeatureGroups(unit);
        if (unit?.kind === 'kanji') {
            return {
                ...features.kanji,
                meanings: features.kanji?.meanings || unit.alternate_meanings || [],
                primary_meaning: unit.primary_meaning,
            };
        }
        return {
            text: unit?.text,
            hiragana: features.reading?.hiragana,
            romaji: features.reading?.romaji,
            meaning: features.meaning?.glossary,
            alternate_meaning: features.meaning?.alternate_glossary,
            unit_type: unit?.unit_label || getRabbitholeUnitKindLabel(unit),
            part_of_speech: features.grammar?.part_of_speech,
            grammar_notes: features.grammar?.function,
            analysis_tags: features.grammar?.analysis_tags,
        };
    }

    function createRabbitholeFeatureSection(title, data) {
        if (isEmptyDebugValue(data)) return null;
        const section = document.createElement('section');
        section.className = 'rabbithole-feature-section';
        section.appendChild(createDebugBlockTitle(title));
        section.appendChild(createKeyValueGrid(data));
        return section;
    }

    function createNestedUnitsBlock(ann, rabbit, unit, instanceKey = 'side:default') {
        if (!Array.isArray(unit.children) || !unit.children.length) return null;
        const block = document.createElement('section');
        block.className = 'rabbithole-feature-section rabbithole-children';
        block.appendChild(createDebugBlockTitle('Nested units'));
        unit.children.forEach(childId => {
            const child = rabbit.units_by_id?.[childId];
            if (!child) return;
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'rabbithole-child';
            chip.textContent = `${child.text} · ${child.unit_label || getRabbitholeUnitKindLabel(child)}`;
            chip.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                setActiveRabbitholeUnit(getAnnotationRegionId(ann), child.id, instanceKey);
            });
            block.appendChild(chip);
        });
        return block.children.length > 1 ? block : null;
    }

    function createRabbitholeUnitInspector(ann, compact = false, instanceKey = 'side:default') {
        const rabbit = getRabbitholeData(ann);
        const unit = getSelectedRabbitholeUnit(ann, 0, instanceKey);
        if (!rabbit || !unit) return null;

        const inspector = document.createElement('div');
        inspector.className = `rabbithole-inspector${compact ? ' compact' : ''}`;

        const title = document.createElement('div');
        title.className = 'rabbithole-inspector-title';
        title.textContent = `${unit.text} · ${getRabbitholeUnitKindLabel(unit)}`;
        inspector.appendChild(title);

        const infoTitle = unit.kind === 'kanji' ? 'Kanji information' : 'Segment information';
        const infoBlock = createRabbitholeFeatureSection(infoTitle, getUnitInformation(unit));
        if (infoBlock) inspector.appendChild(infoBlock);

        const dictionaryBlock = createDictionaryEntriesBlock(unit);
        if (dictionaryBlock) inspector.appendChild(dictionaryBlock);

        const nestedBlock = createNestedUnitsBlock(ann, rabbit, unit, instanceKey);
        if (nestedBlock) inspector.appendChild(nestedBlock);

        return inspector;
    }

    function createRegionActionRow(ann, compact = false) {
        if (!ann?.region_id) return null;
        const actions = document.createElement('div');
        actions.className = `reader-box-actions${compact ? ' compact' : ''}`;

        const orientationBtn = document.createElement('button');
        orientationBtn.type = 'button';
        orientationBtn.className = 'ocr-orientation-toggle details-action';
        const orientation = ann.recognized_orientation || (ann.vertical ? 'vertical' : 'horizontal');
        orientationBtn.textContent = `orientation: ${orientation} ${orientation === 'vertical' ? '↓' : '→'}`;
        orientationBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            await handleOrientationToggle(ann);
        });

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'ocr-remove-box details-action';
        removeBtn.textContent = 'Remove box';
        removeBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            await handleRemoveBox(ann);
        });

        actions.appendChild(orientationBtn);
        actions.appendChild(removeBtn);
        return actions;
    }

    function createRabbitholeCard(ann, index, compact = false, instanceKey = `side:${getAnnotationRegionId(ann, index)}`) {
        const entry = document.createElement('section');
        entry.className = `rabbithole-card${compact ? ' compact' : ''}`;
        const rabbit = getRabbitholeData(ann);
        const summary = rabbit?.summary || {};
        const regionId = getAnnotationRegionId(ann, index);
        const header = document.createElement('div');
        header.className = 'rabbithole-card-header';

        const identity = document.createElement('div');
        identity.className = 'rabbithole-card-identity';

        const title = document.createElement('div');
        title.className = 'reader-debug-title';
        title.textContent = `Box ${ann.reading_order || index + 1}`;
        identity.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'reader-debug-meta rabbithole-card-meta';
        [
            rabbit ? `tokens: ${summary.token_count ?? 0}` : null,
            rabbit ? `kanji: ${summary.kanji_count ?? 0}` : null,
            ann.translated ? 'translation: yes' : 'translation: pending',
        ].filter(Boolean).forEach(item => {
            const pill = document.createElement('span');
            pill.textContent = item;
            meta.appendChild(pill);
        });
        identity.appendChild(meta);
        if (!compact) {
            const actions = createRegionActionRow(ann);
            if (actions) identity.appendChild(actions);
        }
        header.appendChild(identity);

        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'rabbithole-close';
        closeBtn.setAttribute('aria-label', `Close Box ${ann.reading_order || index + 1} rabbithole`);
        closeBtn.textContent = '×';
        closeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            closePopupRegion(regionId);
        });
        if (compact) header.appendChild(closeBtn);

        entry.appendChild(header);

        entry.appendChild(createRabbitholeLayerTabs(compact, instanceKey));

        const grid = document.createElement('div');
        grid.className = 'rabbithole-table';
        createRabbitholeLayerRows(ann, compact, instanceKey).forEach(row => grid.appendChild(row));
        entry.appendChild(grid);

        const inspector = createRabbitholeUnitInspector(ann, compact, instanceKey);
        if (inspector) entry.appendChild(inspector);

        return entry;
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
            if (!editMode || !addBoxMode || !selectedPanel) return;
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
                const data = await API.addRegion(selectedPanel.filename, region);
                applyRegionPayload(data);
                resetTranslationView();
                setDebugStatus('Box added. Run Scan to compute dashed boxes.');
                refreshCacheStatus();
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
        const imageW = latestScan?.image_width || panelImg.naturalWidth;
        const imageH = latestScan?.image_height || panelImg.naturalHeight;
        if (!imageW || !imageH) return null;
        const displayW = panelImg.clientWidth;
        const displayH = panelImg.clientHeight;
        if (!displayW || !displayH) return null;
        return {
            displayW,
            displayH,
            imageW,
            imageH,
            scaleX: displayW / imageW,
            scaleY: displayH / imageH,
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

    function extractLinePoints(lines) {
        return (lines || [])
            .flatMap(line => Array.isArray(line) ? line : [])
            .filter(point => Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]));
    }

    function getVisionDebug(ann) {
        return ann?.ocr_debug?.vision || {};
    }

    function normalizeRectBox(box) {
        if (!Array.isArray(box) || box.length < 4) return null;
        const [x, y, width, height] = box.map(Number);
        if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;
        return [x, y, width, height];
    }

    function boxToDebugText(box) {
        const normalized = normalizeRectBox(box);
        if (!normalized) return '';
        const [x, y, width, height] = normalized;
        return `${Math.round(x)}, ${Math.round(y)}, ${Math.round(width)}, ${Math.round(height)}`;
    }

    function scalePolygonPoints(points, scaleX, scaleY) {
        return (points || [])
            .filter(point => Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]))
            .map(point => `${Number(point[0]) * scaleX},${Number(point[1]) * scaleY}`)
            .join(' ');
    }

    function computeCandidateTextArea(ann) {
        const visionCandidate = normalizeRectBox(getVisionDebug(ann).candidate_box);
        if (visionCandidate) {
            return { x: visionCandidate[0], y: visionCandidate[1], width: visionCandidate[2], height: visionCandidate[3] };
        }
        const bbox = ann?.bbox;
        if (!bbox || bbox.length < 4) return null;
        const points = extractLinePoints(ann.lines);
        if (!points.length) return null;
        const xs = points.map(point => Number(point[0]));
        const ys = points.map(point => Number(point[1]));
        const bboxXs = bbox.map(point => Number(point[0]));
        const bboxYs = bbox.map(point => Number(point[1]));
        const pad = Math.max(Number(ann.font_size) || 0, 10);
        const minX = clamp(Math.min(...xs) - pad, Math.min(...bboxXs), Math.max(...bboxXs));
        const minY = clamp(Math.min(...ys) - pad, Math.min(...bboxYs), Math.max(...bboxYs));
        const maxX = clamp(Math.max(...xs) + pad, Math.min(...bboxXs), Math.max(...bboxXs));
        const maxY = clamp(Math.max(...ys) + pad, Math.min(...bboxYs), Math.max(...bboxYs));
        if (maxX <= minX || maxY <= minY) return null;
        return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
    }

    function createDetectorOverlay(annotations, scaleX, scaleY, displayW, displayH) {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'ocr-vision-overlay');
        svg.setAttribute('viewBox', `0 0 ${Math.max(1, displayW)} ${Math.max(1, displayH)}`);
        svg.setAttribute('preserveAspectRatio', 'none');
        const showText = activeVisionOverlay === 'text' || activeVisionOverlay === 'all';
        const showBubble = activeVisionOverlay === 'bubble' || activeVisionOverlay === 'all';

        annotations.forEach(ann => {
            const points = extractLinePoints(ann.lines);
            const vision = getVisionDebug(ann);

            if (showText) {
                (ann.lines || []).forEach(line => {
                    const polygonPoints = (line || [])
                        .filter(point => Array.isArray(point) && point.length >= 2)
                        .map(point => `${Number(point[0]) * scaleX},${Number(point[1]) * scaleY}`)
                        .join(' ');
                    if (!polygonPoints) return;
                    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    polygon.setAttribute('class', 'ocr-vision-line');
                    polygon.setAttribute('points', polygonPoints);
                    svg.appendChild(polygon);
                });

                const candidate = computeCandidateTextArea(ann);
                if (candidate) {
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('class', 'ocr-vision-candidate');
                    rect.setAttribute('x', String(candidate.x * scaleX));
                    rect.setAttribute('y', String(candidate.y * scaleY));
                    rect.setAttribute('width', String(candidate.width * scaleX));
                    rect.setAttribute('height', String(candidate.height * scaleY));
                    rect.setAttribute('rx', '10');
                    rect.setAttribute('ry', '10');
                    svg.appendChild(rect);
                } else if (points.length) {
                    const outline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
                    outline.setAttribute('class', 'ocr-vision-hull');
                    outline.setAttribute(
                        'points',
                        points.map(point => `${Number(point[0]) * scaleX},${Number(point[1]) * scaleY}`).join(' ')
                    );
                    svg.appendChild(outline);
                }
            }

            if (showBubble) {
                const searchBox = normalizeRectBox(vision.search_box);
                if (searchBox && activeVisionOverlay === 'all') {
                    const [x, y, width, height] = searchBox;
                    const searchRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    searchRect.setAttribute('class', 'ocr-vision-search');
                    searchRect.setAttribute('x', String(x * scaleX));
                    searchRect.setAttribute('y', String(y * scaleY));
                    searchRect.setAttribute('width', String(width * scaleX));
                    searchRect.setAttribute('height', String(height * scaleY));
                    svg.appendChild(searchRect);
                }

                const bubblePoints = scalePolygonPoints(vision.bubble_points, scaleX, scaleY);
                if (bubblePoints) {
                    const bubble = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    bubble.setAttribute('class', 'ocr-vision-bubble');
                    bubble.setAttribute('points', bubblePoints);
                    svg.appendChild(bubble);
                } else {
                    const bubbleBox = normalizeRectBox(vision.bubble_box);
                    if (bubbleBox) {
                        const [x, y, width, height] = bubbleBox;
                        const bubbleRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                        bubbleRect.setAttribute('class', 'ocr-vision-bubble');
                        bubbleRect.setAttribute('x', String(x * scaleX));
                        bubbleRect.setAttribute('y', String(y * scaleY));
                        bubbleRect.setAttribute('width', String(width * scaleX));
                        bubbleRect.setAttribute('height', String(height * scaleY));
                        bubbleRect.setAttribute('rx', '10');
                        bubbleRect.setAttribute('ry', '10');
                        svg.appendChild(bubbleRect);
                    }
                }

                const placementBox = normalizeRectBox(vision.placement_box);
                if (placementBox) {
                    const [x, y, width, height] = placementBox;
                    const placementRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    placementRect.setAttribute('class', 'ocr-vision-placement');
                    placementRect.setAttribute('x', String(x * scaleX));
                    placementRect.setAttribute('y', String(y * scaleY));
                    placementRect.setAttribute('width', String(width * scaleX));
                    placementRect.setAttribute('height', String(height * scaleY));
                    placementRect.setAttribute('rx', '8');
                    placementRect.setAttribute('ry', '8');
                    svg.appendChild(placementRect);
                }
            }
        });

        return svg;
    }

    function createBubbleShape(ann, scaleX, scaleY, className) {
        const vision = getVisionDebug(ann);
        const bubblePoints = scalePolygonPoints(vision.bubble_points, scaleX, scaleY);
        if (bubblePoints) {
            const bubble = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            bubble.setAttribute('class', className);
            bubble.setAttribute('points', bubblePoints);
            return bubble;
        }

        const bubbleBox = normalizeRectBox(vision.bubble_box);
        if (!bubbleBox) return null;

        const [x, y, width, height] = bubbleBox;
        const bubbleRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bubbleRect.setAttribute('class', className);
        bubbleRect.setAttribute('x', String(x * scaleX));
        bubbleRect.setAttribute('y', String(y * scaleY));
        bubbleRect.setAttribute('width', String(width * scaleX));
        bubbleRect.setAttribute('height', String(height * scaleY));
        bubbleRect.setAttribute('rx', '10');
        bubbleRect.setAttribute('ry', '10');
        return bubbleRect;
    }

    function createHoverBubbleOverlay(annotations, scaleX, scaleY, displayW, displayH) {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'ocr-hover-bubble-overlay');
        svg.setAttribute('viewBox', `0 0 ${Math.max(1, displayW)} ${Math.max(1, displayH)}`);
        svg.setAttribute('preserveAspectRatio', 'none');

        annotations.forEach((ann, index) => {
            const regionId = String(ann.region_id ?? ann.id ?? index);
            const bubble = createBubbleShape(ann, scaleX, scaleY, 'ocr-hover-bubble');
            if (!bubble) return;
            bubble.dataset.regionId = regionId;
            svg.appendChild(bubble);
        });

        return svg;
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

        if (activeVisionOverlay !== 'none') {
            frag.appendChild(createDetectorOverlay(withBbox, scaleX, scaleY, displayW, displayH));
        }
        frag.appendChild(createHoverBubbleOverlay(withBbox, scaleX, scaleY, displayW, displayH));

        withBbox.forEach((ann, index) => {
            if (!ann.bbox || ann.bbox.length < 4) return;

            const [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = ann.bbox;
            const imageLeft = clamp(Math.min(x1, x4), 0, imgNatW);
            const imageTop = clamp(Math.min(y1, y2), 0, imgNatH);
            const imageRight = clamp(Math.max(x2, x3), 0, imgNatW);
            const imageBottom = clamp(Math.max(y3, y4), 0, imgNatH);
            if (imageRight <= imageLeft || imageBottom <= imageTop) return;

            const left = imageLeft * scaleX;
            const top = imageTop * scaleY;
            const width = (imageRight - imageLeft) * scaleX;
            const height = (imageBottom - imageTop) * scaleY;

            const box = document.createElement('div');
            const uncomputed = isUncomputedAnnotation(ann);
            const quality = uncomputed ? 'warn' : (ann.ocr_debug?.quality || qualityFromConfidence(ann.confidence));
            const regionId = String(ann.region_id ?? ann.id ?? index);
            const isOpen = openPopupRegionIds.has(regionId);
            box.className = `ocr-box ocr-quality-${quality}${uncomputed ? ' ocr-uncomputed' : ''}${isOpen ? ' tooltip-open' : ''}`;
            box.style.cssText = `left:${left}px;top:${top}px;width:${width}px;height:${height}px`;
            box.title = uncomputed
                ? 'Uncomputed OCR box'
                : `OCR ${quality} | ${ann.ocr_variant || 'unknown'} | ${formatConfidence(ann.confidence)}`;
            box.dataset.regionId = regionId;
            if (editMode) {
                box.tabIndex = 0;
                box.appendChild(createResizeHandle());
                box.addEventListener('pointerdown', (e) => startBoxDrag(e, box, ann));
                box.addEventListener('keydown', (e) => handleBoxKeydown(e, box, ann));
            }

            box.addEventListener('click', (e) => {
                if (editMode && e.target.classList.contains('ocr-resize-handle')) return;
                if (e.target !== box) return;
                e.stopPropagation();
                togglePopupRegion(regionId);
            });
            box.addEventListener('pointerenter', () => setActiveHoverBubble(regionId));
            box.addEventListener('pointerleave', () => {
                setActiveHoverBubble(null, regionId);
            });

            frag.appendChild(box);
        });

        overlay.appendChild(frag);
        requestAnimationFrame(renderOpenPopups);
    }

    function togglePopupRegion(regionId) {
        const key = String(regionId);
        if (openPopupRegionIds.has(key)) {
            closePopupRegion(key);
            return;
        }
        openPopupRegionIds.add(key);
        popupOpenOrder = popupOpenOrder.filter(item => item !== key);
        popupOpenOrder.push(key);
        while (popupOpenOrder.length > MAX_OPEN_POPUPS) {
            closePopupRegion(popupOpenOrder[0], { render: false });
        }
        setActiveHoverBubble(key);
        renderOpenPopups();
    }

    function closePopupRegion(regionId, options = {}) {
        const key = String(regionId);
        openPopupRegionIds.delete(key);
        popupOpenOrder = popupOpenOrder.filter(item => item !== key);
        popupPositionsByRegion.delete(key);
        setActiveHoverBubble(null, key);
        document
            .querySelector(`#ocr-overlay .ocr-box[data-region-id="${CSS.escape(key)}"]`)
            ?.classList.remove('tooltip-open');
        if (options.render !== false) renderOpenPopups();
    }

    function setActiveHoverBubble(regionId, closingRegionId = null) {
        const activeKey = regionId == null ? null : String(regionId);
        const closingKey = closingRegionId == null ? null : String(closingRegionId);
        document.querySelectorAll('#ocr-overlay .ocr-hover-bubble').forEach(bubble => {
            const currentRegionId = String(bubble.dataset.regionId);
            if (activeKey != null) {
                bubble.classList.toggle('active', currentRegionId === activeKey);
            } else if (closingKey == null || currentRegionId === closingKey) {
                bubble.classList.remove('active');
            }
        });
    }

    function renderOpenPopups() {
        const layer = document.getElementById('ocr-popup-layer');
        const wrapper = document.getElementById('reader-panel-wrapper');
        const overlay = document.getElementById('ocr-overlay');
        if (!layer || !wrapper || !overlay || !latestScan) return;
        layer.innerHTML = '';
        const annotations = latestScan.annotations || [];
        const byRegion = new Map(
            annotations.map((ann, index) => [getAnnotationRegionId(ann, index), { ann, index }])
        );
        openPopupRegionIds = new Set([...openPopupRegionIds].filter(regionId => byRegion.has(regionId)));
        popupOpenOrder = popupOpenOrder.filter(regionId => openPopupRegionIds.has(regionId));
        document.querySelectorAll('#ocr-overlay .ocr-box').forEach(box => {
            box.classList.toggle('tooltip-open', openPopupRegionIds.has(String(box.dataset.regionId || '')));
        });
        popupOpenOrder.forEach(regionId => {
            const item = byRegion.get(regionId);
            const box = document.querySelector(`#ocr-overlay .ocr-box[data-region-id="${CSS.escape(regionId)}"]`);
            if (!item || !box) return;
            const popup = document.createElement('div');
            popup.className = 'ocr-popup';
            popup.dataset.regionId = regionId;
            popup.addEventListener('click', e => e.stopPropagation());
            popup.addEventListener('pointerdown', e => e.stopPropagation());
            if (isUncomputedAnnotation(item.ann)) {
                const closeBtn = document.createElement('button');
                closeBtn.type = 'button';
                closeBtn.className = 'rabbithole-close';
                closeBtn.setAttribute('aria-label', `Close Box ${item.ann.reading_order || item.index + 1} popup`);
                closeBtn.textContent = '×';
                closeBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    closePopupRegion(regionId);
                });
                popup.appendChild(closeBtn);
                const message = document.createElement('div');
                message.className = 'ocr-tooltip-message';
                message.textContent = UNCOMPUTED_TEXT;
                popup.appendChild(message);
            } else {
                popup.appendChild(createRabbitholeCard(item.ann, item.index, true, `popup:${regionId}`));
                if (!item.ann.rabbithole) {
                    const message = document.createElement('div');
                    message.className = 'ocr-tooltip-message';
                    message.textContent = 'Rabbithole analysis loading...';
                    popup.appendChild(message);
                }
            }
            layer.appendChild(popup);
            positionPopupNearBox(box, popup, wrapper);
        });
    }

    function rectsOverlap(a, b) {
        return !(
            a.right <= b.left ||
            a.left >= b.right ||
            a.bottom <= b.top ||
            a.top >= b.bottom
        );
    }

    function overlapArea(a, b) {
        if (!rectsOverlap(a, b)) return 0;
        return Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left))
            * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
    }

    function clampTooltipCandidate(left, top, tooltipWidth, tooltipHeight, boundsRect, margin) {
        const maxLeft = Math.max(margin, boundsRect.width - tooltipWidth - margin);
        const maxTop = Math.max(margin, boundsRect.height - tooltipHeight - margin);
        return {
            left: clamp(left, margin, maxLeft),
            top: clamp(top, margin, maxTop),
        };
    }

    function positionPopupNearBox(box, popup, wrapper) {
        const boundsRect = wrapper.getBoundingClientRect();
        const boxRect = box.getBoundingClientRect();
        if (!boundsRect.width || !boundsRect.height) return;
        const margin = 12;
        const gap = 10;
        popup.style.visibility = 'hidden';
        popup.style.maxWidth = `${Math.max(260, boundsRect.width - margin * 2)}px`;
        popup.style.maxHeight = `${Math.max(220, boundsRect.height - margin * 2)}px`;
        const popupRect = popup.getBoundingClientRect();
        const popupWidth = popupRect.width || 360;
        const popupHeight = popupRect.height || 260;
        const boxBounds = {
            left: boxRect.left - boundsRect.left,
            top: boxRect.top - boundsRect.top,
            right: boxRect.right - boundsRect.left,
            bottom: boxRect.bottom - boundsRect.top,
        };
        const seeds = [
            { left: boxBounds.left + (boxBounds.right - boxBounds.left - popupWidth) / 2, top: boxBounds.top - popupHeight - gap, rank: 0 },
            { left: boxBounds.left + (boxBounds.right - boxBounds.left - popupWidth) / 2, top: boxBounds.bottom + gap, rank: 1 },
            { left: boxBounds.right + gap, top: boxBounds.top, rank: 2 },
            { left: boxBounds.left - popupWidth - gap, top: boxBounds.top, rank: 3 },
        ];
        const obstacles = Array.from(document.querySelectorAll('#ocr-popup-layer .ocr-popup'))
            .filter(item => item !== popup)
            .map(item => elementRectWithinBounds(item, boundsRect))
            .filter(Boolean);
        const scored = seeds.map(seed => {
            const position = clampTooltipCandidate(seed.left, seed.top, popupWidth, popupHeight, boundsRect, margin);
            const rect = {
                left: position.left,
                top: position.top,
                right: position.left + popupWidth,
                bottom: position.top + popupHeight,
            };
            const overlap = obstacles.reduce((sum, obstacle) => sum + overlapArea(rect, obstacle), 0);
            const drift = Math.abs(position.left - seed.left) + Math.abs(position.top - seed.top);
            return { ...position, overlap, drift, rank: seed.rank };
        });
        scored.sort((a, b) => {
            if (a.overlap !== b.overlap) return a.overlap - b.overlap;
            if (a.drift !== b.drift) return a.drift - b.drift;
            return a.rank - b.rank;
        });
        const best = scored[0] || { left: margin, top: margin };
        popup.style.left = `${Math.round(best.left)}px`;
        popup.style.top = `${Math.round(best.top)}px`;
        popup.style.visibility = '';
        popupPositionsByRegion.set(String(box.dataset.regionId || ''), { left: best.left, top: best.top });
    }

    function createResizeHandle() {
        const handle = document.createElement('span');
        handle.className = 'ocr-resize-handle';
        handle.setAttribute('aria-hidden', 'true');
        return handle;
    }

    const BOX_EDIT_DRAG_THRESHOLD_PX = 8;

    function startBoxDrag(e, box, ann) {
        if (!editMode || e.button !== 0) return;
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
        let dragArmed = false;
        box.setPointerCapture?.(e.pointerId);

        const move = (moveEvent) => {
            const dx = moveEvent.clientX - start.x;
            const dy = moveEvent.clientY - start.y;
            if (!dragArmed && Math.hypot(dx, dy) < BOX_EDIT_DRAG_THRESHOLD_PX) return;
            dragArmed = true;
            if (resizing) {
                box.style.width = `${clamp(start.width + dx, 12, Math.max(12, geom.displayW - start.left))}px`;
                box.style.height = `${clamp(start.height + dy, 12, Math.max(12, geom.displayH - start.top))}px`;
            } else {
                box.style.left = `${Math.max(0, Math.min(geom.displayW - start.width, start.left + dx))}px`;
                box.style.top = `${Math.max(0, Math.min(geom.displayH - start.height, start.top + dy))}px`;
            }
        };

        const up = async (upEvent) => {
            box.releasePointerCapture?.(upEvent.pointerId);
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
            if (!dragArmed) return;
            await persistBoxGeometry(box, ann);
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
            if (!['Delete', 'Backspace'].includes(e.key)) await persistBoxGeometry(box, ann);
        }
    }

    async function persistBoxGeometry(box, ann) {
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
            const data = await API.overrideRegion(selectedPanel.filename, ann.region_id, {
                x: region.x,
                y: region.y,
                width: region.width,
                height: region.height,
                vertical: region.orientation === 'vertical',
            });
            applyRegionPayload(data);
            resetTranslationView();
            setDebugStatus('Box edited. Run Scan to compute dashed boxes.');
            await refreshCacheStatus();
        } catch (err) {
            setDebugStatus('Could not save box edit: ' + err.message);
        }
    }

    async function handleRemoveBox(ann) {
        if (!selectedPanel || !ann.region_id) return;
        const ok = window.confirm('Remove this OCR box?');
        if (!ok) return;
        try {
            const data = await API.deleteRegion(selectedPanel.filename, ann.region_id);
            closePopupRegion(ann.region_id, { render: false });
            rabbitholeCardStates.delete(`popup:${ann.region_id}`);
            rabbitholeCardStates.delete(`side:${ann.region_id}`);
            applyRegionPayload(data);
            resetTranslationView();
            setDebugStatus('Box removed.');
            await refreshCacheStatus();
        } catch (err) {
            setDebugStatus('Could not remove box: ' + err.message);
        }
    }

    async function handleOrientationToggle(ann) {
        if (!selectedPanel || !ann.region_id) return;
        const current = ann.recognized_orientation || (ann.vertical ? 'vertical' : 'horizontal');
        const next = current === 'vertical' ? 'horizontal' : 'vertical';
        try {
            const data = await API.overrideRegion(selectedPanel.filename, ann.region_id, { orientation: next });
            applyRegionPayload(data);
            resetTranslationView();
            setDebugStatus(`Orientation set to ${next}. Run Scan to compute dashed boxes.`);
            await refreshCacheStatus();
        } catch (err) {
            setDebugStatus('Orientation update failed: ' + err.message);
        }
    }

    function qualityFromConfidence(confidence) {
        const value = Number(confidence) || 0;
        if (value >= 0.78) return 'good';
        if (value >= 0.52) return 'warn';
        return 'bad';
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function getTargetLanguageLabel() {
        const target = String(latestScan?.translation_target_lang || document.getElementById('scan-target-lang')?.value || 'en').trim();
        return (target || 'en').toUpperCase();
    }

    function initializeDebugPanelWidth() {
        const panel = document.getElementById('reader-debug-panel');
        if (!panel) return;
        const stored = Number(window.localStorage.getItem('readerDebugWidth') || '');
        const width = Number.isFinite(stored) && stored > 0 ? stored : DEBUG_PANEL_DEFAULT_WIDTH;
        panel.style.width = `${width}px`;
        constrainDebugPanelWidth();
    }

    function constrainDebugPanelWidth() {
        const panel = document.getElementById('reader-debug-panel');
        if (!panel) return;
        const current = parseFloat(panel.style.width) || DEBUG_PANEL_DEFAULT_WIDTH;
        const max = Math.min(window.innerWidth - 32, DEBUG_PANEL_DEFAULT_WIDTH * DEBUG_PANEL_MAX_SCALE);
        const next = clamp(current, DEBUG_PANEL_MIN_WIDTH, Math.max(DEBUG_PANEL_MIN_WIDTH, max));
        panel.style.width = `${next}px`;
        window.localStorage.setItem('readerDebugWidth', String(Math.round(next)));
    }

    function bindDebugPanelResize() {
        const handle = document.getElementById('reader-debug-resize');
        const panel = document.getElementById('reader-debug-panel');
        if (!handle || !panel) return;

        handle.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startWidth = panel.getBoundingClientRect().width;
            const max = () => Math.min(window.innerWidth - 32, DEBUG_PANEL_DEFAULT_WIDTH * DEBUG_PANEL_MAX_SCALE);
            handle.setPointerCapture?.(e.pointerId);

            const move = (moveEvent) => {
                const dx = moveEvent.clientX - startX;
                const nextWidth = clamp(startWidth - dx, DEBUG_PANEL_MIN_WIDTH, Math.max(DEBUG_PANEL_MIN_WIDTH, max()));
                panel.style.width = `${Math.round(nextWidth)}px`;
            };

            const up = (upEvent) => {
                handle.releasePointerCapture?.(upEvent.pointerId);
                window.removeEventListener('pointermove', move);
                window.removeEventListener('pointerup', up);
                constrainDebugPanelWidth();
            };

            window.addEventListener('pointermove', move);
            window.addEventListener('pointerup', up);
        });
    }

    function formatConfidence(confidence) {
        const value = Number(confidence);
        if (!Number.isFinite(value)) return 'n/a';
        return `${Math.round(value * 100)}%`;
    }

    function createDebugBlockTitle(title) {
        const el = document.createElement('div');
        el.className = 'reader-debug-box-title';
        el.textContent = title;
        return el;
    }

    function createKeyValueGrid(data) {
        const grid = document.createElement('div');
        grid.className = 'reader-debug-kv';
        const entries = data && typeof data === 'object' && !Array.isArray(data)
            ? Object.entries(data)
            : [['value', data]];

        entries.forEach(([key, value]) => {
            if (isEmptyDebugValue(value)) return;
            const row = document.createElement('div');
            row.className = 'reader-debug-kv-row';

            const keyEl = document.createElement('div');
            keyEl.className = 'reader-debug-kv-key';
            keyEl.textContent = humanizeDebugKey(key);

            const valueEl = document.createElement('div');
            valueEl.className = 'reader-debug-kv-value';
            appendDebugValue(valueEl, value);

            row.appendChild(keyEl);
            row.appendChild(valueEl);
            grid.appendChild(row);
        });

        if (!grid.children.length) {
            const empty = document.createElement('div');
            empty.className = 'reader-debug-kv-empty';
            empty.textContent = 'No extra metadata';
            grid.appendChild(empty);
        }
        return grid;
    }

    function splitScoreBreakdown(scoreBreakdown) {
        const breakdown = scoreBreakdown && typeof scoreBreakdown === 'object' ? { ...scoreBreakdown } : {};
        const bias = {};
        if (Object.prototype.hasOwnProperty.call(breakdown, 'orientation_bias')) {
            bias.orientation_bias = breakdown.orientation_bias;
            delete breakdown.orientation_bias;
        }
        return {
            core: breakdown,
            bias,
        };
    }

    function appendDebugValue(container, value) {
        if (Array.isArray(value)) {
            if (!value.length) {
                container.textContent = 'n/a';
                return;
            }
            if (value.every(item => item === null || typeof item !== 'object')) {
                container.textContent = value.map(formatDebugValue).join(', ');
                return;
            }
            const nested = document.createElement('div');
            nested.className = 'reader-debug-kv-nested';
            value.slice(0, 8).forEach((item, index) => {
                nested.appendChild(createKeyValueGrid({ [`#${index + 1}`]: item }));
            });
            if (value.length > 8) {
                const more = document.createElement('div');
                more.className = 'reader-debug-kv-more';
                more.textContent = `+${value.length - 8} more`;
                nested.appendChild(more);
            }
            container.appendChild(nested);
            return;
        }

        if (value && typeof value === 'object') {
            container.appendChild(createKeyValueGrid(value));
            return;
        }

        container.textContent = formatDebugValue(value);
    }

    function formatDebugValue(value) {
        if (value === null || typeof value === 'undefined') return 'n/a';
        if (typeof value === 'number') return Number.isFinite(value) ? String(Math.round(value * 1000) / 1000) : 'n/a';
        if (typeof value === 'boolean') return value ? 'yes' : 'no';
        return String(value);
    }

    function humanizeDebugKey(key) {
        return String(key).replace(/_/g, ' ');
    }

    function isEmptyDebugValue(value) {
        if (value === null || typeof value === 'undefined') return true;
        if (typeof value === 'string' && value.trim() === '') return true;
        if (Array.isArray(value)) return value.length === 0;
        return typeof value === 'object' && Object.keys(value).length === 0;
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
