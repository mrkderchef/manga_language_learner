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

    function init() {
        bindEvents();
    }

    function bindEvents() {
        document.getElementById('upload-area').addEventListener('click', () => {
            document.getElementById('panel-upload').click();
        });

        document.getElementById('panel-upload').addEventListener('change', handleUpload);
        document.getElementById('btn-scan').addEventListener('click', handleScanTranslate);
        document.getElementById('btn-debug-toggle').addEventListener('click', () => toggleDebugPanel());
        document.getElementById('btn-debug-close').addEventListener('click', () => toggleDebugPanel(false));
        document.getElementById('btn-reader-fullscreen').addEventListener('click', openReaderFullscreen);
        document.getElementById('btn-fullscreen-close').addEventListener('click', closeReaderFullscreen);
        document.getElementById('translation-slider').addEventListener('input', (e) => {
            setTranslationReveal(Number(e.target.value));
        });
        bindTranslationHandle();
        bindReaderPanelControls();
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
        btn.innerHTML = '<span class="spinner"></span>Scanning...';
        document.getElementById('ocr-overlay').innerHTML = '';
        latestScan = null;
        resetTranslationView();
        setDebugStatus('Scanning...');

        try {
            const result = await API.scanAndTranslate(selectedPanel.filename);
            latestScan = result;
            ocrText = result.text || '';
            renderDebugPanel(result);
            renderOverlays(result.annotations || [], result.image_width, result.image_height);
            showTranslatedPanel(result.translated_image_url);
            if (result.render_warnings && result.render_warnings.length) {
                console.warn('Panel render warnings:', result.render_warnings);
            }
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

        if (!annotations.length) {
            setDebugStatus('No text recognized.');
            return;
        }

        const frag = document.createDocumentFragment();
        annotations.forEach((ann, index) => {
            const entry = document.createElement('div');
            entry.className = 'reader-debug-entry';

            const title = document.createElement('div');
            title.className = 'reader-debug-title';
            title.textContent = `Box ${ann.reading_order || index + 1}`;

            const jp = document.createElement('div');
            jp.className = 'reader-debug-jp';
            jp.textContent = ann.text || '';

            const en = document.createElement('div');
            en.className = 'reader-debug-en';
            en.textContent = ann.translated || 'No translation available';

            entry.appendChild(title);
            entry.appendChild(jp);
            entry.appendChild(en);
            frag.appendChild(entry);
        });

        content.appendChild(frag);
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
            box.className = 'ocr-box';
            box.style.cssText = `left:${left}px;top:${top}px;width:${width}px;height:${height}px`;

            const tooltip = document.createElement('div');
            tooltip.className = 'ocr-tooltip';
            const jpDiv = document.createElement('div');
            jpDiv.className = 'ocr-tooltip-jp';
            jpDiv.textContent = ann.text;
            const enDiv = document.createElement('div');
            enDiv.className = 'ocr-tooltip-en';
            enDiv.textContent = ann.translated || '...';
            tooltip.appendChild(jpDiv);
            tooltip.appendChild(enDiv);
            box.appendChild(tooltip);

            frag.appendChild(box);
        });

        overlay.appendChild(frag);
    }

    return { init, loadPanels };
})();
