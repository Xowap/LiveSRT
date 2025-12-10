# LiveSRT: Live Speech-to-Text & Translation

LiveSRT is a modular tool for real-time speech-to-text transcription and
translation. It captures audio from your microphone (or a file), streams it to
state-of-the-art AI transcription providers, and uses Large Language Models
(LLMs) to correct and translate the output on the fly, displaying the results in
a rich Terminal User Interface (TUI).

## 📺 Demo

Here's a quick demonstration of LiveSRT in action:

[![asciicast](https://asciinema.org/a/Y52mPjSCMlHsl6dCVzR60Gy2p.svg)](https://asciinema.org/a/Y52mPjSCMlHsl6dCVzR60Gy2p)

## ✨ Features

- **Live Transcription:** Real-time speech-to-text using top-tier providers.
- **Live Translation:** Translate speech instantly using LLMs (Local or Remote).
- **Rich TUI:** A dedicated terminal interface to view live transcripts and
  translations side-by-side.
- **Intelligent Post-processing:** Uses LLMs to clean up stutters, fix ASR
  errors, and separate speakers.
- **Audio Sources:** Support for microphones and audio file replay (via ffmpeg).
- **Configurable:** Uses a YAML configuration file for reproducible setups.

## 🔌 Supported Providers

### Transcription (ASR)

- **AssemblyAI** (Streaming API) - _Default_
- **ElevenLabs** (Realtime Speech-to-Text)
- **Speechmatics** (Realtime API)

### Translation (LLMs)

- **Local LLMs:** Runs locally via `llama.cpp` (e.g., Ministral, Qwen).
- **Remote LLMs:** Support for Groq, Mistral, Google Gemini, DeepInfra, Ollama
  and OpenRouter.

## 🚀 Quick Start

As this is a PyPI package, you can run it directly without any installation
using `uvx` (or install via pip).

### 1. Initialization

First, create a default configuration file. This allows you to select your audio
source, transcription backend, and translation settings.

```bash
uvx livesrt init-config
# Created config.yml
```

### 2. Authentication

Set the API key for your chosen provider (default is AssemblyAI). Keys are
stored securely in your system keyring.

```bash
uvx livesrt set-token assembly_ai
```

### 3. Run

Start the application using the configuration from `config.yml`.

```bash
uvx livesrt run
```

To enable translation (if disabled in config), you can use the flag:

```bash
uvx livesrt run --translate
```

## ⚙ Configuration

LiveSRT relies on a `config.yml` file. You can generate a template using
`livesrt init-config`.

### Key Configuration Sections:

- **Audio:** Select `mic` or `file`. If using a microphone, find your device
  index using `livesrt list-microphones`.
- **Transcription:** Choose between `assembly_ai`, `elevenlabs`, or
  `speechmatics`.
- **Translation:** Toggle enabled/disabled, choose `local-llm` or `remote-llm`,
  and set source/target languages.
- **API Keys:** Manage namespaces for multiple environments.

## 📝 Command Reference

All commands start with `livesrt`. Use `--help` on any command for more details.

### `livesrt init-config`

Creates a default `config.yml` in the current directory.

- `--output`, `-o`: Path to the output file (default: `config.yml`).

### `livesrt run [OPTIONS]`

Runs the main application using the loaded configuration.

- `--config`, `-c`: Path to the configuration file (default: `config.yml`).
- `--translate / --no-translate`: Override the translation setting in the
  config.

### `livesrt set-token <provider> [OPTIONS]`

Sets the API token for a specific provider securely.

- `<provider>` choices:
    - ASR: `assembly_ai`, `elevenlabs`, `speechmatics`
    - LLM: `groq`, `mistral`, `google`, `deepinfra`, `openrouter`, `ollama`
- `--api-key`, `-k`: (Optional) Your secret API key. If omitted, you are
  prompted securely.

### `livesrt list-microphones`

Lists all available input microphone devices and their IDs. Use the resulting ID
to update the `device_index` in your `config.yml`.

## 💡 Usage Scenarios

### Using a specific microphone

1.  List devices: `uvx livesrt list-microphones`.
2.  Edit `config.yml`: Set `audio.device_index` to the desired ID.
3.  Run: `uvx livesrt run`.

### Debugging with a file

Simulate a live stream using an audio file (requires `ffmpeg`):

1.  Edit `config.yml`:
    ```yaml
    audio:
        source_type: file
        file_path: "./interview.wav"
    ```
2.  Run: `uvx livesrt run`.

### Live Translation with Remote LLM

To offload processing to a fast remote API (e.g., Groq):

1.  Set the key: `uvx livesrt set-token groq`.
2.  Edit `config.yml`:
    ```yaml
    translation:
        enabled: true
        engine: remote-llm
        remote_llm:
            lang_to: Spanish
            model: groq/llama-3.3-70b-versatile
    ```
3.  Run: `uvx livesrt run`.

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

- **`src/livesrt/cli.py`**: Entry point and CLI logic using `click`.
- **`src/livesrt/containers.py`**: Dependency Injection container used to wire
  components based on configuration.
- **`src/livesrt/tui.py`**: The Textual-based UI implementation.
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
