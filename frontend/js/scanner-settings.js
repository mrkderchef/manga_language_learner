const ScannerSettings = (() => {
    const STORAGE_KEY = 'readerScanSettingsV2';
    const DEFAULTS = {
        detection: {
            confidence_threshold: 0.4,
            nms_threshold: 0.35,
            mask_threshold: 0.3,
            box_threshold: 0.6,
            max_regions: null,
        },
        ocr: {
            quality_mode: 'balanced',
            semantic_rerank: 'close',
            vertical_preference: 'normal',
            rotation_win_margin: 15,
            preprocessing_set: 'standard',
            crop_upscale: 3,
            crop_padding_ratio: 0.05,
            crop_padding_min: 4,
            enable_rotated_variants: true,
        },
        bubble: {
            mode: 'hybrid',
            model_confidence: 0.25,
            model_iou: 0.7,
            search_scale: 1.0,
            wand_enabled: true,
            overlap_reconciliation: true,
        },
        translation: {
            engine: 'ollama',
            model: null,
            target_lang: 'en',
            style: 'natural',
            temperature: 0.1,
        },
    };

    function readNumberInput(id, fallback) {
        const value = Number(document.getElementById(id)?.value);
        return Number.isFinite(value) ? value : fallback;
    }

    function setElementValue(id, value) {
        const element = document.getElementById(id);
        if (!element) return;
        if (element.type === 'checkbox') {
            element.checked = Boolean(value);
            return;
        }
        element.value = value == null ? '' : String(value);
    }

    function getOptions() {
        const maxRegionsValue = document.getElementById('scan-detection-max-regions')?.value;
        const translationEngine = document.getElementById('scan-translation-engine')?.value || 'ollama';
        const translationModel = translationEngine === 'ollama'
            ? document.getElementById('scan-translation-model')?.value || null
            : null;
        const targetLang = document.getElementById('scan-target-lang')?.value || 'en';
        const style = document.getElementById('scan-translation-style')?.value || 'natural';
        const temperature = readNumberInput('scan-temperature', 0.1);
        return {
            ocr_engine: 'mangaocr',
            detection: {
                confidence_threshold: readNumberInput('scan-detection-confidence', 0.4),
                nms_threshold: readNumberInput('scan-detection-nms', 0.35),
                mask_threshold: readNumberInput('scan-detection-mask', 0.3),
                box_threshold: readNumberInput('scan-detection-box', 0.6),
                max_regions: maxRegionsValue ? Number(maxRegionsValue) : null,
            },
            ocr: {
                quality_mode: document.getElementById('scan-ocr-quality')?.value || 'balanced',
                semantic_rerank: document.getElementById('scan-semantic-rerank')?.checked ? 'close' : 'off',
                vertical_preference: document.getElementById('scan-vertical-preference')?.value || 'normal',
                rotation_win_margin: readNumberInput('scan-rotation-margin', 15),
                preprocessing_set: document.getElementById('scan-preprocessing-set')?.value || 'standard',
                crop_upscale: readNumberInput('scan-crop-upscale', 3),
                crop_padding_ratio: readNumberInput('scan-crop-padding-ratio', 0.05),
                crop_padding_min: 4,
                enable_rotated_variants: Boolean(document.getElementById('scan-rotated-variants')?.checked),
            },
            bubble: {
                mode: document.getElementById('scan-bubble-mode')?.value || 'hybrid',
                model_confidence: readNumberInput('scan-bubble-confidence', 0.25),
                model_iou: readNumberInput('scan-bubble-iou', 0.7),
                search_scale: readNumberInput('scan-bubble-search-scale', 1.0),
                wand_enabled: Boolean(document.getElementById('scan-bubble-wand')?.checked),
                overlap_reconciliation: Boolean(document.getElementById('scan-bubble-overlap')?.checked),
            },
            translation: {
                engine: translationEngine,
                model: translationModel,
                target_lang: targetLang,
                style,
                temperature,
            },
            translation_engine: translationEngine,
            translation_model: translationModel,
            target_lang: targetLang,
            translation_style: style,
            temperature,
            reset_manual_edits: false,
        };
    }

    function apply(settings) {
        setElementValue('scan-detection-confidence', settings.detection.confidence_threshold);
        setElementValue('scan-detection-nms', settings.detection.nms_threshold);
        setElementValue('scan-detection-mask', settings.detection.mask_threshold);
        setElementValue('scan-detection-box', settings.detection.box_threshold);
        setElementValue('scan-detection-max-regions', settings.detection.max_regions);
        setElementValue('scan-ocr-quality', settings.ocr.quality_mode);
        setElementValue('scan-preprocessing-set', settings.ocr.preprocessing_set);
        setElementValue('scan-vertical-preference', settings.ocr.vertical_preference);
        setElementValue('scan-rotation-margin', settings.ocr.rotation_win_margin);
        setElementValue('scan-crop-upscale', settings.ocr.crop_upscale);
        setElementValue('scan-crop-padding-ratio', settings.ocr.crop_padding_ratio);
        setElementValue('scan-semantic-rerank', settings.ocr.semantic_rerank !== 'off');
        setElementValue('scan-rotated-variants', settings.ocr.enable_rotated_variants);
        setElementValue('scan-bubble-mode', settings.bubble.mode);
        setElementValue('scan-bubble-confidence', settings.bubble.model_confidence);
        setElementValue('scan-bubble-iou', settings.bubble.model_iou);
        setElementValue('scan-bubble-search-scale', settings.bubble.search_scale);
        setElementValue('scan-bubble-wand', settings.bubble.wand_enabled);
        setElementValue('scan-bubble-overlap', settings.bubble.overlap_reconciliation);
        setElementValue('scan-translation-engine', settings.translation.engine);
        setElementValue('scan-translation-model', settings.translation.model);
        setElementValue('scan-target-lang', settings.translation.target_lang);
        setElementValue('scan-translation-style', settings.translation.style);
        setElementValue('scan-temperature', settings.translation.temperature);
    }

    function load() {
        try {
            const stored = window.localStorage.getItem(STORAGE_KEY);
            if (!stored) return DEFAULTS;
            const parsed = JSON.parse(stored);
            return {
                detection: { ...DEFAULTS.detection, ...(parsed.detection || {}) },
                ocr: { ...DEFAULTS.ocr, ...(parsed.ocr || {}) },
                bubble: { ...DEFAULTS.bubble, ...(parsed.bubble || {}) },
                translation: { ...DEFAULTS.translation, ...(parsed.translation || {}) },
            };
        } catch {
            return DEFAULTS;
        }
    }

    function restore() {
        apply(load());
    }

    function persist() {
        try {
            window.localStorage.setItem(STORAGE_KEY, JSON.stringify(getOptions()));
        } catch {
            // Local persistence is optional; scans should still work without it.
        }
    }

    function reset() {
        apply(DEFAULTS);
        persist();
    }

    function getOcrFingerprint() {
        const options = getOptions();
        return JSON.stringify({
            detection: options.detection,
            ocr: options.ocr,
            bubble: options.bubble,
        });
    }

    return {
        apply,
        getOcrFingerprint,
        getOptions,
        persist,
        reset,
        restore,
    };
})();
