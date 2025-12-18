"""Default configuration file template for LiveSRT."""

DEFAULT_CONFIG_CONTENT = """# LiveSRT Configuration

# Audio settings
audio:
  # Source type for audio input: 'mic' for microphone, 'file' for an audio file.
  source_type: mic # or 'file'
  # Device index for microphone input. Run 'livesrt list-microphones' to find
  # available devices.
  # If null, the default microphone will be used.
  device_index: null # or an integer like 0, 1, etc.
  # Path to an audio file if source_type is 'file'.
  file_path: null # e.g., 'path/to/your/audio.wav'

# Transcription settings
transcription:
  # Backend service for transcription: 'assembly_ai', 'elevenlabs', or 'speechmatics'.
  backend: assembly_ai # Recommended: assembly_ai
  assembly_ai:
    # Region for AssemblyAI: 'us' or 'eu'.
    region: us
  speechmatics:
    # Language for Speechmatics transcription.
    language: en # e.g., 'en', 'fr', 'es'
  # No ElevenLabs specific settings needed in config for now

# Translation settings
translation:
  # Enable or disable translation (true/false).
  enabled: false
  # Translation engine: 'local-llm' or 'remote-llm'.
  engine: remote-llm # Recommended: remote-llm
  local_llm:
    # Language to translate to (e.g., 'fr', 'es').
    lang_to: en
    # Language to translate from (e.g., 'en', 'fr', 'es'). If null, auto-detects.
    lang_from: null
    # Local LLM model to use.
    model: ministral:8b:q4-k-m # Check documentation for supported local models
  remote_llm:
    # Language to translate to (e.g., 'fr', 'es').
    lang_to: en
    # Language to translate from (e.g., 'en', 'fr', 'es'). If null, auto-detects.
    lang_from: null
    # Remote LLM model to use (provider/model-name).
    # Recommended: mistral/ministral-8b-latest
    model: mistral/ministral-8b-latest

# API Keys namespace: used for storing API keys securely with 'livesrt set-token'.
# This helps categorize keys if you use multiple instances of LiveSRT.
api_keys:
  namespace: default
"""
