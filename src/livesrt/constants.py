"""Constants used throughout the LiveSRT application."""

# Original content of constants.py follows
# (Assuming original content is not lost and will be appended or preserved)
from enum import Enum


class ProviderType(str, Enum):
    """Enum for different transcription and translation providers."""

    ASSEMBLY_AI = "assembly_ai"
    ELEVENLABS = "elevenlabs"
    SPEECHMATICS = "speechmatics"
    OPENROUTER = "openrouter"
    MISTRAL = "mistral"
    GROQ = "groq"
    GOOGLE = "google"
    DEEPINFRA = "deepinfra"
    OLLAMA = "ollama"
