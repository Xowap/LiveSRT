"""
Translation module: interfaces and implementations
"""

from .base import TranslatedTurn, TranslationReceiver, Translator
from .local_llm import LocalLLM
from .remote_llm import RemoteLLM

__all__ = [
    "LocalLLM",
    "RemoteLLM",
    "TranslatedTurn",
    "TranslationReceiver",
    "Translator",
]
