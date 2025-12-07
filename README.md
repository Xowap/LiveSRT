# LiveSRT: Live Speech-to-Text Transcription

LiveSRT is a modular command-line interface (CLI) tool for real-time
speech-to-text transcription. It captures audio from your microphone (or a file)
and streams it to state-of-the-art AI transcription providers.

Currently supported providers:

- **AssemblyAI** (Streaming API) - _Default_
- **ElevenLabs** (Realtime Speech-to-Text)
- **Speechmatics** (Realtime API)

## 🚀 Quick Start

As this is a PyPI package, you can run it directly without any installation
using `uvx`.

### Using AssemblyAI (Default)

1.  **Set your API key:**

    ```bash
    uvx livesrt set-token assembly_ai
    # You will be prompted to enter your API key securely.
    ```

2.  **Start transcribing:**
    ```bash
    uvx livesrt transcribe
    ```

### Using ElevenLabs

1.  **Set your API key:**

    ```bash
    uvx livesrt set-token elevenlabs
    ```

2.  **Start transcribing:**
    ```bash
    uvx livesrt transcribe --provider elevenlabs
    ```

### Using Speechmatics

1.  **Set your API key:**

    ```bash
    uvx livesrt set-token speechmatics
    ```

2.  **Start transcribing:** _Note: You must specify the language code (e.g.,
    'en', 'fr') for Speechmatics._

    ```bash
    uvx livesrt transcribe --provider speechmatics --language en
    ```

Press `Ctrl+C` to stop the transcription session.

## 💡 Scenarios

Here are some common ways you might use `livesrt`:

### Using a specific microphone

If you have multiple microphones and want to use one other than the system
default:

1.  **List available microphones:**

    ```bash
    uvx livesrt list-microphones
    ```

    This outputs a table with the `Index` and `Name` of available devices.

2.  **Transcribe using that device:**
    ```bash
    uvx livesrt transcribe --device <device_index>
    ```

### Simulating with an Audio File (Replay)

For debugging or testing without a microphone, you can stream an audio file as
if it were live input. _Note: This requires `ffmpeg` to be installed on your
system._

```bash
uvx livesrt transcribe --replay-file /path/to/audio.wav
```

### Changing Regions (AssemblyAI only)

By default, `livesrt` connects to the EU AssemblyAI endpoint. To use the US
endpoint:

```bash
uvx livesrt transcribe --provider assembly_ai --region us
```

## 📝 Command Reference

All commands start with `livesrt`.

### `livesrt set-token <provider> [--api-key <key>]`

Sets the API token for a specific transcription provider.

- `<provider>`: The name of the provider. Choices: `assembly_ai`, `elevenlabs`,
  `speechmatics`.
- `--api-key <key>`, `-k <key>`: (Optional) Your secret API key. If not
  provided, you will be securely prompted to enter it.

### `livesrt list-microphones`

Lists all available input microphone devices and their corresponding device IDs.

### `livesrt transcribe [OPTIONS]`

Starts live transcription.

- `--provider`, `-p`: The transcription provider to use (default:
  `assembly_ai`).
- `--device`, `-d <index>`: The index of the microphone device to use.
- `--replay-file`, `-f <path>`: Use a file as the audio source instead of the
  microphone.
- `--language`, `-l <code >`: Language code (e.g., 'en', 'fr'). **Mandatory**
  when using Speechmatics.
- `--region`, `-r`: The API region (only applies to AssemblyAI). Choices: `eu`
  (default), `us`.

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

The project uses a modular architecture defined in `src/livesrt/transcribe/`:

- **`src/livesrt/cli.py`**: The main entry point. Handles arguments,
  configuration, and the UI (using `rich`). It instantiates the appropriate
  Source and Transcripter based on user input.
- **`src/livesrt/transcribe/base.py`**: Defines the abstract interfaces:
    - `AudioSource`: Interface for yielding audio frames (from mic or file).
    - `Transcripter`: Interface for processing audio and emitting events.
    - `TranscriptReceiver`: Interface for handling transcription results (e.g.,
      printing to console).
- **`src/livesrt/transcribe/transcripters/`**:
    - **`aai.py`**: AssemblyAI Streaming API implementation.
    - **`elevenlabs.py`**: ElevenLabs Realtime API implementation.
    - **`speechmatics.py`**: Speechmatics Realtime API implementation.
- **`src/livesrt/transcribe/audio_sources/`**:
    - **`mic.py`**: Captures live audio using `pyaudio`.
    - **`replay_file.py`**: Streams audio from files using `ffmpeg`.

## 📜 License

This project is licensed under the WTFPL (Do What The Fuck You Want To Public
License). For more details, see [http://www.wtfpl.net/](http://www.wtfpl.net/).

## ⚠ Warranty Waiver

This software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability, fitness
for a particular purpose and noninfringement. In no event shall the authors or
copyright holders be liable for any claim, damages or other liability, whether
in an action of contract, tort or otherwise, arising from, out of or in
connection with the software or the use or other dealings in the software.
