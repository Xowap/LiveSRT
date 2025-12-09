# LiveSRT: Live Speech-to-Text & Translation

LiveSRT is a modular command-line interface (CLI) tool for real-time
speech-to-text transcription and translation. It captures audio from your
microphone (or a file), streams it to state-of-the-art AI transcription
providers, and uses Large Language Models (LLMs) to correct and translate the
output on the fly.

## ✨ Features

- **Live Transcription:** Real-time speech-to-text using top-tier providers.
- **Live Translation:** Translate speech instantly using LLMs (Local or Remote).
- **Intelligent Post-processing:** Uses LLMs to clean up stutters, fix ASR
  errors, and separate speakers.
- **Audio Sources:** Support for microphones and audio file replay (via ffmpeg).

## 🔌 Supported Providers

### Transcription (ASR)

- **AssemblyAI** (Streaming API) - _Default_
- **ElevenLabs** (Realtime Speech-to-Text)
- **Speechmatics** (Realtime API)

### Translation (LLMs)

- **Local LLMs:** Runs locally via `llama.cpp` (e.g., Ministral, Qwen).
- **Remote LLMs:** Support for Groq, Mistral, Google Gemini, DeepInfra, and
  OpenRouter.

## 🚀 Quick Start

As this is a PyPI package, you can run it directly without any installation
using `uvx` (or install via pip).

### 1. Basic Transcription

1.  **Set your Transcription API key (AssemblyAI is default):**
    ```bash
    uvx livesrt set-token assembly_ai
    ```
2.  **Start transcribing:**
    ```bash
    uvx livesrt transcribe
    ```

### 2. Live Translation (Local LLM)

This runs the translation model on your machine. It requires no API key for
translation but needs a decent CPU/GPU. It will download the model automatically
on first run.

```bash
# Transcribe (using AssemblyAI) and translate to French using a local model
uvx livesrt translate "French"
```

### 3. Live Translation (Remote LLM)

For faster performance without local hardware load, use a remote provider (e.g.,
Groq).

1.  **Set your LLM API key:**
    ```bash
    uvx livesrt set-token groq
    ```
2.  **Run translation:**
    ```bash
    uvx livesrt translate "Spanish" --translation-engine remote-llm --model "groq/openai/gpt-oss-120b"
    ```

## 📝 Command Reference

All commands start with `livesrt`.

### `livesrt set-token <provider>`

Sets the API token for a specific provider.

- `<provider>` choices:
    - ASR: `assembly_ai`, `elevenlabs`, `speechmatics`
    - LLM: `groq`, `mistral`, `google`, `deepinfra`, `openrouter`
- `--api-key <key>`, `-k <key>`: (Optional) Your secret API key. If omitted, you
  are prompted securely.

### `livesrt list-microphones`

Lists all available input microphone devices and their IDs.

### `livesrt transcribe [OPTIONS]`

Starts live transcription without translation.

- `--provider`, `-p`: The transcription provider (default: `assembly_ai`).
- `--device`, `-d <index>`: Microphone device index.
- `--replay-file`, `-f <path>`: Use a file as audio source.
- `--language`, `-l <code >`: Language code (Mandatory for Speechmatics).
- `--region`, `-r`: API region (AssemblyAI only).

### `livesrt translate <lang_to> [OPTIONS]`

Starts live transcription and translates it to the target language.

- **`<lang_to>`**: (Required) The target language (e.g., "French", "Japanese").
- `--lang-from`: Source language (optional, LLM can usually infer it).
- `--translation-engine`:
    - `local-llm` (default): Runs a model locally (e.g., Ministral 8B).
    - `remote-llm`: Uses an external API.
- `--model`: Specific model string.
    - For remote: `provider/model-name` (e.g., `mistral/mistral-large-latest`).
- _Plus all options available in `transcribe` (provider, device, file, etc.)._

## 💡 Usage Scenarios

### Using a specific microphone

1.  List devices: `uvx livesrt list-microphones`
2.  Run: `uvx livesrt transcribe --device 2`

### Debugging with a file

Simulate a live stream using an audio file (requires `ffmpeg`):

```bash
uvx livesrt translate "German" --replay-file ./interview.wav
```

### Using ElevenLabs for ASR

```bash
uvx livesrt set-token elevenlabs
uvx livesrt transcribe --provider elevenlabs
```

## 🛠 Development

To set up a local development environment:

```bash
uv sync
```

### Development Commands

The `Makefile` contains helpers for common tasks:

- **`make format`**: Formats the code using `ruff format`.
- **`make lint`**: Lints the code using `ruff check --fix`.
- **`make types`**: Performs static type checking using `mypy`.
- **`make prettier`**: Formats Markdown and source files using `prettier`.
- **`make clean`**: Runs all formatters, linters, and type checkers.

## 🏗 Code Structure

- **`src/livesrt/cli.py`**: Entry point and UI.
- **`src/livesrt/transcribe/`**: Audio capture and ASR logic.
    - **`transcripters/`**: Implementations for AssemblyAI, ElevenLabs,
      Speechmatics.
    - **`audio_sources/`**: Mic (`pyaudio`) and File (`ffmpeg`) sources.
- **`src/livesrt/translate/`**: Translation logic.
    - **`local_llm.py`**: Wraps `llama_cpp` for local inference.
    - **`remote_llm.py`**: Wraps `httpx` for OpenAI-compatible APIs.
    - **`base.py`**: Handles conversation context and tool-use for accurate
      translations.

## 📜 License

This project is licensed under the WTFPL (Do What The Fuck You Want To Public
License).
