/**
 * Scanner Mode - OCR + Übersetzung von Manga Panels
 * Hover über erkannten Text zeigt Übersetzung direkt auf dem Panel
 */
const Scanner = (() => {
    let selectedPanel = null;
    let ocrText = '';
    let _panelsLoaded = false;

    function init() {
        bindEvents();
    }

    function bindEvents() {
        document.getElementById('upload-area').addEventListener('click', () => {
            document.getElementById('panel-upload').click();
        });

        document.getElementById('panel-upload').addEventListener('change', handleUpload);
        document.getElementById('btn-scan').addEventListener('click', handleScanTranslate);
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
        document.getElementById('ocr-result').textContent = 'Bereit zum Scannen...';
        document.getElementById('translation-result').textContent = '—';
        document.getElementById('ocr-overlay').innerHTML = '';
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
        const ocrResult = document.getElementById('ocr-result');
        const transResult = document.getElementById('translation-result');

        btn.disabled = true;
        btn.classList.add('loading');
        btn.innerHTML = '<span class="spinner"></span>Scanne...';
        ocrResult.innerHTML = '<div class="skeleton-text"></div><div class="skeleton-text short"></div>';
        ocrResult.classList.add('loading');
        transResult.innerHTML = '<div class="skeleton-text"></div>';
        transResult.classList.add('loading');
        document.getElementById('ocr-overlay').innerHTML = '';

        try {
            const result = await API.scanAndTranslate(selectedPanel.filename);
            ocrText = result.text || '';
            ocrResult.textContent = ocrText || 'Kein Text erkannt';
            ocrResult.classList.remove('loading');

            const translations = (result.annotations || [])
                .map(a => a.translated)
                .filter(t => t)
                .join('\n');
            transResult.textContent = translations || '—';
            transResult.classList.remove('loading');

            renderOverlays(result.annotations || [], result.image_width, result.image_height);
        } catch (err) {
            ocrResult.textContent = 'Fehler: ' + err.message;
            ocrResult.classList.remove('loading');
            transResult.textContent = '—';
            transResult.classList.remove('loading');
        } finally {
            btn.disabled = false;
            btn.classList.remove('loading');
            btn.textContent = 'Scannen & Übersetzen';
        }
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

        const container = panelImg.parentElement;
        const offsetX = (container.clientWidth - displayW) / 2;
        const offsetY = (container.clientHeight - displayH) / 2;

        // Build all boxes in a document fragment (single reflow)
        const frag = document.createDocumentFragment();

        withBbox.forEach(ann => {
            if (!ann.bbox || ann.bbox.length < 4) return;

            const [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = ann.bbox;
            const left = Math.min(x1, x4) * scaleX + offsetX;
            const top = Math.min(y1, y2) * scaleY + offsetY;
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
