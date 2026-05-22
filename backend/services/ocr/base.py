"""Base pipeline primitives and OCR adapter interface.

This module defines a high-level, engine-independent pipeline description
(`PipelineConfig`) and a minimal adapter interface (`BaseOCRAdapter`).
Individual engine adapters should subclass `BaseOCRAdapter` and implement
the `extract_and_translate` method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineConfig:
    name: str
    enabled: bool = True
    steps: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)


class BaseOCRAdapter:
    """Minimal OCR adapter interface.

    Subclasses must implement `is_available()` and
    `extract_and_translate(image_path, **options)`.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config

    def is_available(self) -> bool:  # pragma: no cover - adapter must override
        return True

    def extract_and_translate(self, image_path: str, **options) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError()
