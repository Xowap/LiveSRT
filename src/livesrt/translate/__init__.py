"""
Translation module: interfaces and implementations
"""

from .base import TranslatedTurn, Translator
from .mock import MockTranslator

__all__ = [
    "MockTranslator",
    "TranslatedTurn",
    "Translator",
]
