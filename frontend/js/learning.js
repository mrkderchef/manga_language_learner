/**
 * Learning Mode - Spielerisch Vokabeln aus Manga Panels lernen
 */
const Learning = (() => {
    let panels = [];
    let currentPanelIndex = 0;
    let vocab = [];
    let currentVocabIndex = 0;
    let knownCount = 0;

    function init() {
        bindEvents();
        loadPanels();
    }

    function bindEvents() {
        document.getElementById('btn-reveal').addEventListener('click', revealMeaning);
        document.getElementById('btn-knew').addEventListener('click', () => submitAnswer(true));
        document.getElementById('btn-didnt-know').addEventListener('click', () => submitAnswer(false));
        document.getElementById('btn-next-panel').addEventListener('click', nextPanel);
    }

    async function loadPanels() {
        try {
            const data = await API.getLearningPanels();
            panels = data.panels || [];
            if (panels.length > 0) {
                loadPanel(0);
            }
        } catch (err) {
            console.error('Failed to load learning panels:', err);
        }
    }

    async function loadPanel(index) {
        if (index >= panels.length) return;
        currentPanelIndex = index;
        const panel = panels[index];

        document.getElementById('learning-panel-img').src = API.panelImageUrl(panel.filename);

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
            // Continue even if server unreachable
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
            loadPanel(0); // Loop back
        }
    }

    return { init, loadPanels };
})();
