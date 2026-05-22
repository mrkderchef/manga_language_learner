/**
 * Rabbithole Mode - Spielerisch Vokabeln aus Manga Panels lernen
 */
const Rabbithole = (() => {
    let panels = [];
    let currentPanelIndex = 0;
    let vocab = [];
    let currentVocabIndex = 0;
    let knownCount = 0;
    let _panelsLoaded = false;

    function init() {
        bindEvents();
    }

    function bindEvents() {
        document.getElementById('btn-reveal').addEventListener('click', revealMeaning);
        document.getElementById('btn-knew').addEventListener('click', () => submitAnswer(true));
        document.getElementById('btn-didnt-know').addEventListener('click', () => submitAnswer(false));
        document.getElementById('btn-next-panel').addEventListener('click', nextPanel);
    }

    async function loadPanels(force = false) {
        if (_panelsLoaded && !force) return;
        try {
            const data = await API.getRabbitholePanels();
            panels = data.panels || [];
            _panelsLoaded = true;
            if (panels.length > 0) {
                loadPanel(0);
            }
        } catch (err) {
            console.error('Failed to load Rabbithole panels:', err);
        }
    }

    async function loadPanel(index) {
        if (index >= panels.length) return;
        currentPanelIndex = index;
        const panel = panels[index];

        document.getElementById('rabbithole-panel-img').src = API.panelImageUrl(panel.path || panel.filename);

        document.getElementById('vocab-japanese').innerHTML = '<div class="skeleton-text"></div>';
        document.getElementById('vocab-reading').textContent = '';
        document.getElementById('vocab-meaning').textContent = '';

        try {
            const data = await API.getPanelVocab(panel.filename);
            vocab = data.vocab || [];
            currentVocabIndex = 0;
            knownCount = 0;
            updateProgress();
            showCurrentVocab();
        } catch (err) {
            console.error('Failed to load vocab:', err);
        }
    }

    function showCurrentVocab() {
        if (currentVocabIndex >= vocab.length) {
            showPanelComplete();
            return;
        }

        const word = vocab[currentVocabIndex];
        document.getElementById('vocab-japanese').textContent = word.japanese;
        document.getElementById('vocab-reading').textContent = word.reading || '';
        document.getElementById('vocab-meaning').textContent = word.meaning || '';
        document.getElementById('vocab-meaning').classList.add('hidden');
        document.getElementById('vocab-actions').classList.add('hidden');
        document.getElementById('btn-reveal').classList.remove('hidden');
    }

    function revealMeaning() {
        document.getElementById('vocab-meaning').classList.remove('hidden');
        document.getElementById('vocab-actions').classList.remove('hidden');
        document.getElementById('btn-reveal').classList.add('hidden');
    }

    async function submitAnswer(knew) {
        const word = vocab[currentVocabIndex];
        if (knew) knownCount++;

        try {
            await API.submitAnswer(
                panels[currentPanelIndex].filename,
                word.japanese,
                knew
            );
        } catch (err) {
            console.warn('Could not save answer:', err);
        }

        currentVocabIndex++;
        updateProgress();
        showCurrentVocab();
    }

    function updateProgress() {
        const total = vocab.length;
        const done = currentVocabIndex;
        const percent = total > 0 ? Math.round((done / total) * 100) : 0;

        document.getElementById('progress-fill').style.width = `${percent}%`;
        document.getElementById('progress-text').textContent = `${done} / ${total} Wörter`;
    }

    function showPanelComplete() {
        document.getElementById('vocab-japanese').textContent = '🎉';
        document.getElementById('vocab-reading').textContent = 'Panel abgeschlossen!';
        document.getElementById('vocab-meaning').textContent = `${knownCount}/${vocab.length} gewusst`;
        document.getElementById('vocab-meaning').classList.remove('hidden');
        document.getElementById('vocab-actions').classList.add('hidden');
        document.getElementById('btn-reveal').classList.add('hidden');
    }

    function nextPanel() {
        if (currentPanelIndex + 1 < panels.length) {
            loadPanel(currentPanelIndex + 1);
        } else {
            loadPanel(0);
        }
    }

    return { init, loadPanels };
})();
