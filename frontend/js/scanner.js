/**
 * Scanner Mode - OCR + Übersetzung von Manga Panels
 * Hover über erkannten Text zeigt Übersetzung direkt auf dem Panel
 */
const Scanner = (() => {
    let selectedPanel = null;
    let ocrText = '';

    function init() {
        loadPanels();
        bindEvents();
    }

    function bindEvents() {
        document.getElementById('upload-area').addEventListener('click', () => {
            document.getElementById('panel-upload').click();
        });

        document.getElementById('panel-upload').addEventListener('change', handleUpload);
        document.getElementById('btn-scan').addEventListener('click', handleScanTranslate);
    }

    async function loadPanels() {
        try {
            const data = await API.listPanels();
            renderPanelGrid(data.panels || []);
        } catch (err) {
            console.error('Failed to load panels:', err);
        }
    }

    function renderPanelGrid(panels) {
        const grid = document.getElementById('panel-list');
        grid.innerHTML = '';

        panels.forEach(panel => {
            const img = document.createElement('img');
            img.src = API.panelImageUrl(panel.filename);
            img.alt = panel.filename;
            img.className = 'panel-thumb';
            img.addEventListener('click', () => selectPanel(panel, img));
            grid.appendChild(img);
        });
    }

    function selectPanel(panel, imgEl) {
        document.querySelectorAll('.panel-thumb.selected').forEach(el => el.classList.remove('selected'));
        imgEl.classList.add('selected');

        selectedPanel = panel;
        const panelImg = document.getElementById('scanner-panel-img');
        panelImg.src = API.panelImageUrl(panel.filename);
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
                loadPanels();
            }
        } catch (err) {
            alert('Upload fehlgeschlagen: ' + err.message);
        }
    }

    async function handleScanTranslate() {
        if (!selectedPanel) return;

        const btn = document.getElementById('btn-scan');
        btn.disabled = true;
        btn.textContent = 'Scanne & Übersetze...';
        document.getElementById('ocr-result').textContent = 'OCR + Übersetzung läuft...';
        document.getElementById('translation-result').textContent = '...';
        document.getElementById('ocr-overlay').innerHTML = '';

        try {
            const result = await API.scanAndTranslate(selectedPanel.filename);
            ocrText = result.text || '';
            document.getElementById('ocr-result').textContent = ocrText || 'Kein Text erkannt';

            // Collect all translations
            const translations = (result.annotations || [])
                .map(a => a.translated)
                .filter(t => t)
                .join('\n');
            document.getElementById('translation-result').textContent = translations || '—';

            // Render hover overlays on the panel
            renderOverlays(result.annotations || [], result.image_width, result.image_height);
        } catch (err) {
            document.getElementById('ocr-result').textContent = 'Fehler: ' + err.message;
        } finally {
            btn.disabled = false;
            btn.textContent = 'Scannen & Übersetzen';
        }
    }

    function renderOverlays(annotations, imgNatW, imgNatH) {
        const overlay = document.getElementById('ocr-overlay');
        const panelImg = document.getElementById('scanner-panel-img');
        overlay.innerHTML = '';

        if (!annotations.length || !imgNatW || !imgNatH) return;

        // Use displayed image size for scaling
        const displayW = panelImg.clientWidth;
        const displayH = panelImg.clientHeight;
        const scaleX = displayW / imgNatW;
        const scaleY = displayH / imgNatH;

        // Calculate image offset within panel-display (centered)
        const container = panelImg.parentElement;
        const offsetX = (container.clientWidth - displayW) / 2;
        const offsetY = (container.clientHeight - displayH) / 2;

        annotations.forEach(ann => {
            if (!ann.bbox || ann.bbox.length < 4) return;

            const [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = ann.bbox;
            const left = Math.min(x1, x4) * scaleX + offsetX;
            const top = Math.min(y1, y2) * scaleY + offsetY;
            const width = (Math.max(x2, x3) - Math.min(x1, x4)) * scaleX;
            const height = (Math.max(y3, y4) - Math.min(y1, y2)) * scaleY;

            const box = document.createElement('div');
            box.className = 'ocr-box';
            box.style.left = `${left}px`;
            box.style.top = `${top}px`;
            box.style.width = `${width}px`;
            box.style.height = `${height}px`;

            // Tooltip with translation
            const tooltip = document.createElement('div');
            tooltip.className = 'ocr-tooltip';
            tooltip.innerHTML = `
                <div class="ocr-tooltip-jp">${ann.text}</div>
                <div class="ocr-tooltip-en">${ann.translated || '...'}</div>
            `;
            box.appendChild(tooltip);

            overlay.appendChild(box);
        });
    }

    return { init, loadPanels };
})();
