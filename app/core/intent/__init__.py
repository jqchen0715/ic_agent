# -*- coding: utf-8 -*-
"""意图识别模块。"""

from app.core.intent.domain_classifier import DomainClassification, ICDomainClassifier
from app.core.intent.recognizer import IntentRecognizer, IntentResult

__all__ = [
    "DomainClassification",
    "ICDomainClassifier",
    "IntentRecognizer",
    "IntentResult",
]
