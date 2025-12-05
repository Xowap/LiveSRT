# LiveSRT: Live Speech-to-Text Transcription

LiveSRT is a command-line interface (CLI) tool for real-time speech-to-text
transcription directly from your microphone. It leverages the AssemblyAI
streaming API to provide live transcriptions with speaker turns, confidence
scores, and language detection.

## 🚀 Quick Start

As this is a PyPI package, you can run it directly without any installation
using `uvx`:

1.  **Set your AssemblyAI API key:**

    ```bash
    uvx livesrt set-token assembly_ai
    # You will be prompted to enter your API key securely.
    ```

    (Alternatively, you can provide the key directly:
    `uvx livesrt set-token assembly_ai --api-key sk-YOUR_API_KEY`)

2.  **Start transcribing!**
    ```bash
    uvx livesrt transcribe
    ```
    Press `Ctrl+C` to stop the transcription.

## 💡 Scenarios

Here are some common ways you might use `livesrt`:

### Using a specific microphone

If you have multiple microphones and want to use one other than the default, you
first need to find its device ID:

1.  **List available microphones:**

    ```bash
    uvx livesrt list-microphones
    ```

    This will output a table of microphones with their `Index` and `Name`.

2.  **Transcribe using the desired microphone:** Replace `<device_index>` with
    the index from the previous step.
    ```bash
    uvx livesrt transcribe --device <device_index>
    ```

### Using a different AssemblyAI region

By default, `livesrt` connects to the EU AssemblyAI API. If you are in the US
(or prefer the US endpoint), you can specify the region:

```bash
uvx livesrt transcribe --region us
```

## 📝 Command Reference

All commands start with `livesrt`.

- **`livesrt --help`**: Show general help message and available commands.
- **`livesrt <command> --help`**: Show help for a specific command.

### `livesrt set-token <provider> [--api-key <key>]`

Sets the API token for a specific transcription provider.

- `<provider>`: The name of the provider. Currently, only `assembly_ai` is
  supported.
- `--api-key <key>`, `-k <key>`: (Optional) Your secret API key. If not
  provided, you will be securely prompted to enter it.

**Example:**

```bash
uvx livesrt set-token assembly_ai
```

### `livesrt list-microphones`

Lists all available input microphone devices and their corresponding device IDs.
This is useful for selecting a specific microphone for transcription.

**Example:**

```bash
uvx livesrt list-microphones
```

### `livesrt transcribe [--device <index>] [--region <eu|us>]`

Starts live transcription from your microphone.

- `--device <index>`, `-d <index>`: (Optional) The index of the microphone
  device to use. Obtain this from `livesrt list-microphones`. If not specified,
  the default microphone is used.
- `--region <eu|us>`, `-r <eu|us>`: (Optional) The AssemblyAI API region to
  connect to. Defaults to `eu`.

**Examples:**

```bash
uvx livesrt transcribe
uvx livesrt transcribe --device 2
uvx livesrt transcribe --region us
```

## 🛠️ Development

To set up a local development environment and install all dependencies:

```bash
uv sync
```

This command creates a virtual environment if one doesn't exist and installs all
project and development dependencies defined in `pyproject.toml`.

### Development Commands

The `Makefile` contains helpers for common development tasks, which internally
use `uv run` to execute commands within the managed virtual environment:

- **`make format`**: Formats the code using `ruff format`.
- **`make lint`**: Lints the code using `ruff check --fix`.
- **`make types`**: Performs static type checking using `mypy`.
- **`make prettier`**: Formats Markdown and source files using `prettier`.
- **`make clean`**: Runs all formatters, linters, and type checkers.

## 🏗️ Code Structure

The project is structured as a Python package `livesrt` within the `src/`
directory.

- **`src/livesrt/__main__.py`**: The main entry point for the CLI application,
  allowing execution via `python -m livesrt`.
- **`src/livesrt/cli.py`**: Contains the core `click` CLI application
  definition.
    - Defines commands like `set-token`, `list-microphones`, and `transcribe`.
    - Handles API key storage using `keyring`.
    - Includes the `Receiver` class, which implements `StreamReceiver` to
      process and display real-time transcription updates.
    - Provides error handling and rich console output.
- **`src/livesrt/aai.py`**: Encapsulates logic for interacting with the
  AssemblyAI streaming API.
    - `AAI` class: Manages API key, region-specific endpoints, obtaining
      streaming tokens, and establishing WebSocket connections for
      transcription.
    - `StreamReceiver` (abstract base class): Defines the interface for
      receiving transcription events (`session_begins`, `turn`, `termination`).
    - `Turn` and `Word` dataclasses: Represent transcription segments and
      individual words, respectively.
- **`src/livesrt/mic.py`**: Manages audio input from the microphone using
  `pyaudio`.
    - `MicManager` class: Handles listing available microphones, checking device
      validity, and providing an asynchronous context manager (`stream_mic`) to
      capture audio chunks from the microphone.
- **`src/livesrt/async_tools.py`**: Contains utility functions for bridging
  synchronous and asynchronous code. (e.g. `run_sync`, `sync_to_async`).

## 📜 License

This project is licensed under the WTFPL (Do What The Fuck You Want To Public
License). For more details, see [http://www.wtfpl.net/](http://www.wtfpl.net/).

## ⚠️ Warranty Waiver

This software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability, fitness
for a particular purpose and noninfringement. In no event shall the authors or
copyright holders be liable for any claim, damages or other liability, whether
in an action of contract, tort or otherwise, arising from, out of or in
connection with the software or the use or other dealings in the software.
