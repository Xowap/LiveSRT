"""Dependency injection container for LiveSRT."""

from dependency_injector import containers, providers

from .constants import ProviderType
from .services import ApiKeyStore
from .transcribe.audio_sources.mic import MicSourceFactory
from .transcribe.audio_sources.replay_file import FileSourceFactory
from .transcribe.transcripters.aai import AssemblyAITranscripter
from .transcribe.transcripters.elevenlabs import ElevenLabsTranscripter
from .transcribe.transcripters.speechmatics import SpeechmaticsTranscripter
from .translate.local_llm import LocalLLM
from .translate.remote_llm import RemoteLLM
from .tui import LiveSrtApp


def create_remote_llm(lang_to, lang_from, model, api_key_store):
    """Factory function to create a RemoteLLM instance."""
    provider, _, _ = model.partition("/")
    key = api_key_store.get(provider)
    if not key:
        error_msg = f"No key stored for {provider}"
        raise ValueError(error_msg)

    return RemoteLLM(
        lang_to=lang_to,
        lang_from=lang_from,
        model=model,
        api_key=key,
    )


def create_audio_source(source_type, device_index, file_path, transcription_backend):
    """Factory function to create an AudioSource instance."""
    sample_rate = (
        48_000 if transcription_backend == ProviderType.SPEECHMATICS.value else 16_000
    )

    if source_type == "file":
        if not file_path:
            error_msg = "source_type is 'file' but no file_path provided in config."
            raise ValueError(error_msg)
        file_factory = FileSourceFactory(sample_rate=sample_rate, realtime=True)
        return file_factory.create_source(file_path)
    else:
        # Default to mic
        mic_factory = MicSourceFactory(sample_rate=sample_rate)
        if device_index is not None and not mic_factory.is_device_valid(device_index):
            error_msg = f"Device #{device_index} is not a valid device."
            raise ValueError(error_msg)
        return mic_factory.create_source(device_index)


def create_app(source, transcripter, translation_enabled, translator):
    """Factory function to create the LiveSrtApp instance."""
    return LiveSrtApp(
        source=source,
        transcripter=transcripter,
        translator=translator if translation_enabled else None,
    )


class Container(containers.DeclarativeContainer):
    """Dependency injection container for the LiveSRT application."""

    config = providers.Configuration()

    api_key_store = providers.Singleton(
        ApiKeyStore,
        namespace=config.api_keys.namespace,
    )

    # Transcripters
    assembly_ai_transcripter = providers.Factory(
        AssemblyAITranscripter,
        api_key=api_key_store.provided.get.call(ProviderType.ASSEMBLY_AI.value),
        region=config.transcription.assembly_ai.region,
    )

    elevenlabs_transcripter = providers.Factory(
        ElevenLabsTranscripter,
        api_key=api_key_store.provided.get.call(ProviderType.ELEVENLABS.value),
    )

    speechmatics_transcripter = providers.Factory(
        SpeechmaticsTranscripter,
        api_key=api_key_store.provided.get.call(ProviderType.SPEECHMATICS.value),
        language=config.transcription.speechmatics.language,
    )

    transcripter = providers.Selector(
        config.transcription.backend,
        assembly_ai=assembly_ai_transcripter,
        elevenlabs=elevenlabs_transcripter,
        speechmatics=speechmatics_transcripter,
    )

    # Audio Source
    audio_source = providers.Factory(
        create_audio_source,
        source_type=config.audio.source_type,
        device_index=config.audio.device_index,
        file_path=config.audio.file_path,
        transcription_backend=config.transcription.backend,
    )

    # Translators
    local_llm_translator = providers.Factory(
        LocalLLM,
        lang_to=config.translation.local_llm.lang_to,
        lang_from=config.translation.local_llm.lang_from,
        model=config.translation.local_llm.model,
    )
    remote_llm_translator = providers.Factory(
        create_remote_llm,
        lang_to=config.translation.remote_llm.lang_to,
        lang_from=config.translation.remote_llm.lang_from,
        model=config.translation.remote_llm.model,
        api_key_store=api_key_store,
    )

    # Mock translator can be added if needed, for now focusing on these two
    translator = providers.Selector(
        config.translation.engine,
        **{"local-llm": local_llm_translator},
        **{"remote-llm": remote_llm_translator},
    )

    # Application
    app = providers.Factory(
        create_app,
        source=audio_source,
        transcripter=transcripter,
        translation_enabled=config.translation.enabled,
        translator=translator,
    )
