from typing import List, Dict
import os
from config import GOOGLE_PROJECT_ID, TRANSLATION_SOURCE_LANGUAGE, TRANSLATION_TARGET_LANGUAGE
import logging

logger = logging.getLogger(__name__)

try:
    from google.cloud import translate_v2
    HAS_GOOGLE_TRANSLATE = True
except ImportError:
    HAS_GOOGLE_TRANSLATE = False


class TranslationService:
    def __init__(self):
        """Initialize Google Translate client"""
        self.use_google_translate = HAS_GOOGLE_TRANSLATE and bool(GOOGLE_PROJECT_ID)
        
        if self.use_google_translate:
            try:
                self.client = translate_v2.Client()
                logger.info("Google Translate Service initialized")
            except Exception as e:
                logger.warning(f"Google Translate not available: {e}")
                self.use_google_translate = False
    
    def translate_text(
        self,
        text: str,
        source_language: str = TRANSLATION_SOURCE_LANGUAGE,
        target_language: str = TRANSLATION_TARGET_LANGUAGE
    ) -> Dict:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_language: Source language code (e.g., 'ja')
            target_language: Target language code (e.g., 'en')
            
        Returns:
            Dict with translated text and metadata
        """
        if not text:
            return {
                "success": True,
                "original": "",
                "translated": "",
                "source_language": source_language,
                "target_language": target_language
            }
        
        if not self.use_google_translate:
            return self._fallback_translation(text, source_language, target_language)
        
        try:
            result = self.client.translate_text(
                text,
                source_language_code=source_language,
                target_language_code=target_language
            )
            
            return {
                "success": True,
                "original": text,
                "translated": result["translatedText"],
                "source_language": source_language,
                "target_language": target_language,
                "method": "google_translate"
            }
        
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original": text,
                "translated": "",
                "source_language": source_language,
                "target_language": target_language
            }
    
    def translate_annotations(
        self,
        annotations: List[Dict],
        source_language: str = TRANSLATION_SOURCE_LANGUAGE,
        target_language: str = TRANSLATION_TARGET_LANGUAGE
    ) -> List[Dict]:
        """
        Translate multiple text annotations
        
        Args:
            annotations: List of annotation dicts with 'text' field
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            List of annotations with translations added
        """
        translated_annotations = []
        
        for annotation in annotations:
            original_text = annotation.get("text", "")
            
            translation = self.translate_text(
                original_text,
                source_language,
                target_language
            )
            
            translated_annotation = annotation.copy()
            translated_annotation["translation"] = translation.get("translated", "")
            translated_annotation["translation_success"] = translation.get("success", False)
            
            translated_annotations.append(translated_annotation)
        
        return translated_annotations
    
    def _fallback_translation(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """
        Fallback translation using free services
        Note: This is a placeholder. In production, consider using:
        - LibreTranslate (self-hosted)
        - MyMemory API (free, limited)
        """
        try:
            import requests
            
            # Using MyMemory API (free, no authentication)
            url = "https://api.mymemory.translated.net/get"
            params = {
                "q": text,
                "langpair": f"{source_lang}|{target_lang}"
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data["responseStatus"] == 200:
                return {
                    "success": True,
                    "original": text,
                    "translated": data["responseData"]["translatedText"],
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "method": "mymemory_api"
                }
            else:
                return {
                    "success": False,
                    "error": "Translation API error",
                    "original": text,
                    "translated": "",
                    "source_language": source_lang,
                    "target_language": target_lang
                }
        
        except Exception as e:
            logger.error(f"Fallback translation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original": text,
                "translated": "",
                "source_language": source_lang,
                "target_language": target_lang
            }
    
    def get_service_status(self) -> Dict:
        """Get status of translation service"""
        return {
            "translation_service": "google_translate" if self.use_google_translate else "fallback",
            "available": True
        }
