/**
 * Learning page - practice vocabulary collected from manga panels.
 */
const Learning = (() => {
    let panels = [];
    let currentPanelIndex = 0;
    let vocab = [];
    let currentVocabIndex = 0;
    let knownCount = 0;
    let panelsLoaded = false;

    function element(id) {
        return document.getElementById(id);
    }

    function init() {
        element('btn-learning-reveal').addEventListener('click', revealMeaning);
        element('btn-learning-knew').addEventListener('click', () => submitAnswer(true));
        element('btn-learning-didnt-know').addEventListener('click', () => submitAnswer(false));
        element('btn-learning-next-panel').addEventListener('click', nextPanel);
    }

    async function loadPanels(force = false) {
        if (panelsLoaded && !force) return;

        try {
            const data = await API.getLearningPanels();
            panels = data.panels || [];
            panelsLoaded = true;
            if (panels.length) {
                await loadPanel(0);
            } else {
                showStatus('No manga panels available yet.');
            }
        } catch (error) {
            console.error('Failed to load Learning panels:', error);
            showStatus(`Could not load panels: ${error.message}`);
        }
    }

    async function loadPanel(index) {
        if (index < 0 || index >= panels.length) return;

        currentPanelIndex = index;
        const panel = panels[index];
        element('learning-panel-img').src = API.panelImageUrl(panel.path || panel.filename);
        element('learning-vocab-japanese').innerHTML = '<div class="skeleton-text"></div>';
        element('learning-vocab-reading').textContent = '';
        element('learning-vocab-meaning').textContent = '';

        try {
            const data = await API.getLearningPanelVocab(panel.filename);
            vocab = data.vocab || [];
            currentVocabIndex = 0;
            knownCount = 0;
            updateProgress();
            showCurrentVocab();
        } catch (error) {
            console.error('Failed to load Learning vocabulary:', error);
            showStatus(`Could not load vocabulary: ${error.message}`);
        }
    }

    function showCurrentVocab() {
        if (!vocab.length) {
            showStatus('No vocabulary found in this panel.');
            return;
        }
        if (currentVocabIndex >= vocab.length) {
            showPanelComplete();
            return;
        }

        const word = vocab[currentVocabIndex];
        element('learning-vocab-japanese').textContent = word.japanese;
        element('learning-vocab-reading').textContent = word.reading || '';
        element('learning-vocab-meaning').textContent = word.meaning || '';
        element('learning-vocab-meaning').classList.add('hidden');
        element('learning-vocab-actions').classList.add('hidden');
        element('btn-learning-reveal').classList.remove('hidden');
    }

    function revealMeaning() {
        element('learning-vocab-meaning').classList.remove('hidden');
        element('learning-vocab-actions').classList.remove('hidden');
        element('btn-learning-reveal').classList.add('hidden');
    }

    async function submitAnswer(knew) {
        const word = vocab[currentVocabIndex];
        if (!word) return;
        if (knew) knownCount++;

        try {
            const panelComplete = currentVocabIndex === vocab.length - 1;
            await API.submitLearningAnswer(panels[currentPanelIndex].filename, word.japanese, knew, panelComplete);
        } catch (error) {
            console.warn('Could not save Learning answer:', error);
        }

        currentVocabIndex++;
        updateProgress();
        showCurrentVocab();
    }

    function updateProgress() {
        const total = vocab.length;
        const done = Math.min(currentVocabIndex, total);
        const percent = total ? Math.round((done / total) * 100) : 0;
        element('learning-progress-fill').style.width = `${percent}%`;
        element('learning-progress-text').textContent = `${done} / ${total} Wörter`;
    }

    function showPanelComplete() {
        element('learning-vocab-japanese').textContent = '🎉';
        element('learning-vocab-reading').textContent = 'Panel abgeschlossen!';
        element('learning-vocab-meaning').textContent = `${knownCount}/${vocab.length} gewusst`;
        element('learning-vocab-meaning').classList.remove('hidden');
        element('learning-vocab-actions').classList.add('hidden');
        element('btn-learning-reveal').classList.add('hidden');
    }

    function showStatus(message) {
        vocab = [];
        currentVocabIndex = 0;
        element('learning-vocab-japanese').textContent = message;
        element('learning-vocab-reading').textContent = '';
        element('learning-vocab-meaning').textContent = '';
        element('learning-vocab-meaning').classList.add('hidden');
        element('learning-vocab-actions').classList.add('hidden');
        element('btn-learning-reveal').classList.add('hidden');
        updateProgress();
    }

    function nextPanel() {
        if (!panels.length) return;
        loadPanel((currentPanelIndex + 1) % panels.length);
    }

    return { init, loadPanels };
})();
