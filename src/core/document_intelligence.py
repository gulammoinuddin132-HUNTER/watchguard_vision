"""
Document Intelligence Module (Modular Interface)
------------------------------------------------
Extensible interface for document detection, OCR text extraction,
and sensitive keyword identification (e.g. CONFIDENTIAL, RESTRICTED).
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.core.base_engine import BaseVisionEngine


class DocumentIntelligenceEngine(BaseVisionEngine):
    """
    Document Intelligence & OCR Engine interface.
    Ready for integration with Tesseract / EasyOCR / PaddleOCR.
    """

    SENSITIVE_KEYWORDS = [
        "CONFIDENTIAL",
        "SECRET",
        "RESTRICTED",
        "CLASSIFIED",
        "INTERNAL ONLY",
        "PROPRIETARY",
        "PRIVILEGED",
        "DO NOT DISTRIBUTE",
    ]

    def __init__(self):
        self._is_ready = False

    def initialize(self) -> bool:
        """Initializes OCR engine and word dictionaries."""
        self._is_ready = True
        return True

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects document in view and performs OCR scanning.
        """
        return {
            "document_present": False,
            "raw_text": "",
            "sensitive_keywords": [],
            "status": "NORMAL",
        }

    def scan_document(self, frame: np.ndarray) -> Dict[str, Any]:
        """Explicitly scans and extracts text from a target frame."""
        return self.process_frame(frame)

    def release(self) -> None:
        self._is_ready = False
