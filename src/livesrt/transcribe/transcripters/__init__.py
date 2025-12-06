"""
This module gives access to all the transcripter implementations
"""

from .aai import AssemblyAITranscripter
from .elevenlabs import ElevenLabsTranscripter

__all__ = [
    "AssemblyAITranscripter",
    "ElevenLabsTranscripter",
]
