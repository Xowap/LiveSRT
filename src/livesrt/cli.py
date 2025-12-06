"""
The logic specific to the CLI interface
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import httpx
import keyring
import pwinput
import rich_click as click
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from livesrt.async_tools import run_sync
from livesrt.transcribe.audio_sources.mic import MicSourceFactory
from livesrt.transcribe.audio_sources.replay_file import FileSourceFactory
from livesrt.transcribe.base import AudioSource, TranscriptReceiver, Turn
from livesrt.transcribe.transcripters.aai import AssemblyAITranscripter

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "magenta",
        "danger": "bold red",
    }
)
console = Console(theme=custom_theme)


def validate_no_colon(ctx, param, value):
    """
    Callback to ensure the namespace does not contain a colon.
    """

    if ":" in value:
        msg = "The character ':' is not allowed in the namespace."
        raise click.BadParameter(msg)

    return value


@dataclass
class ApiKeyStore:
    """
    Utility class that serves to store API keys
    """

    namespace: str
    system: str = "livesrt"

    def key(self, provider: str) -> str:
        """Generates the key name used for storage of this provider"""

        return f"{self.namespace}:{provider}"

    def get(self, provider: str) -> str | None:
        """Gets the API key for a provider, or None if it doesn't exist."""

        return keyring.get_password(self.system, self.key(provider))

    def set(self, provider: str, value: str) -> None:
        """Sets the API key for a provider"""

        keyring.set_password(self.system, self.key(provider), value)


@dataclass
class Context:
    """
    Internal context of the CLI app
    """

    namespace: str
    store: ApiKeyStore


@click.group()
@click.option(
    *["--namespace", "-n"],
    default="default",
    help="The namespace into which to store this key.",
    show_default=True,
    callback=validate_no_colon,
)
@click.pass_context
def cli(ctx, namespace: str):
    """
    Main entrypoint of the whole thing
    """

    ctx.obj = Context(
        namespace=namespace,
        store=ApiKeyStore(namespace),
    )


class ProviderType(Enum):
    """This is a provider for which we might want to register an API token"""

    ASSEMBLY_AI = "assembly_ai"


@cli.command()
@click.argument(
    "provider",
    type=click.Choice([p.value for p in ProviderType], case_sensitive=False),
)
@click.option(
    *["--api-key", "-k"],
    required=False,
    help="Your secret API key.",
)
@click.pass_obj
def set_token(obj: Context, provider, api_key):
    """
    Sets the API token for a specific provider.
    """

    if not api_key:
        console.print("🔐 API key: ", style="bold", end="")
        api_key = pwinput.pwinput(prompt="", mask="*")

    if not api_key:
        console.print("\n[warning]🫡 Not setting anything")
        return

    obj.store.set(provider, api_key)

    console.print(
        f"\n[green]✔[/green] Configuration started for "
        f"[bold cyan]{provider}[/bold cyan]"
    )


@cli.command()
def list_microphones():
    """
    Utility to list microphones, and beyond that, obtain their device ID so that
    it can be used in the `transcribe` command.
    """
    factory = MicSourceFactory()
    devices = factory.list_devices()

    table = Table(title="Microphones", title_justify="left", title_style="bold")
    table.add_column("Index", justify="right", style="bold cyan")
    table.add_column("Name", justify="left", style="magenta")

    for i, info in devices.items():
        table.add_row(str(i), info.name)

    console.print(table)


def display_http_error(error: httpx.HTTPError) -> None:
    """Display a formatted HTTP error with request and response details."""

    # Build error content
    content_parts = []

    # Request information
    if hasattr(error, "request") and error.request:
        request = error.request
        request_table = Table(
            show_header=False, box=None, padding=(0, 1), pad_edge=False
        )
        request_table.add_column(style="bold")
        request_table.add_column(style="cyan")
        request_table.add_row("Method:", request.method)
        request_table.add_row("URL:", str(request.url))

        content_parts.append(
            Panel(
                request_table,
                title="[bold]Request",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    # Response information
    if hasattr(error, "response") and error.response:
        response = error.response

        # Status code with color based on severity
        status_color = "red" if response.status_code >= 500 else "yellow"

        response_table = Table(
            show_header=False, box=None, padding=(0, 1), pad_edge=False
        )
        response_table.add_column(style="bold")
        response_table.add_column()
        response_table.add_row(
            "Status:",
            f"[{status_color}]{response.status_code}[/{status_color}] "
            f"[dim]{response.reason_phrase}[/dim]",
        )

        # Try to parse and display JSON response
        try:
            json_data = response.json()
            response_table.add_row("Body:", "")

            response_panel = Panel(
                response_table,
                title="[bold]Response",
                title_align="left",
                border_style="red",
                padding=(0, 1),
            )
            content_parts.append(response_panel)
            content_parts.append(
                Panel(
                    JSON.from_data(json_data),
                    title="[bold]Response Body (JSON)",
                    title_align="left",
                    border_style="red",
                    padding=(0, 1),
                )
            )
        except Exception:
            # If not JSON or can't parse, show raw text
            text = response.text[:500]  # Limit to first 500 chars
            body_text = text if text else "[dim](empty)[/dim]"
            if len(response.text) > 500:
                body_text += "\n[dim]... (truncated)[/dim]"

            response_table.add_row("Body:", body_text)
            content_parts.append(
                Panel(
                    response_table,
                    title="[bold]Response",
                    title_align="left",
                    border_style="red",
                    padding=(0, 1),
                )
            )
    else:
        # Connection errors, etc. without a response
        error_table = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
        error_table.add_column(style="bold")
        error_table.add_column()
        error_table.add_row("Type:", f"[red]{type(error).__name__}[/red]")
        error_table.add_row("Message:", str(error))

        content_parts.append(
            Panel(
                error_table,
                title="[bold]Error",
                title_align="left",
                border_style="red",
                padding=(0, 1),
            )
        )

    # Display all parts
    console.print()
    console.print(
        Panel(
            "\n\n".join([""]) * (len(content_parts) - 1),  # Spacer
            title="[bold red]❌ HTTP ERROR",
            title_align="center",
            border_style="bold red",
            padding=(0, 1),
        )
    )

    for part in content_parts:
        console.print(part)

    console.print()


@dataclass
class Receiver(TranscriptReceiver):
    """
    Implementation of the TranscriptReceiver, which will print to the
    console the things currently being said.
    """

    turn_count: int = field(init=False, default=0)
    last_transcript: str = field(init=False, default="")
    start_time: datetime | None = field(init=False, default=None)

    async def start(self) -> None:
        """
        Prints meta information about the session being started.
        """
        self.start_time = datetime.now().astimezone()

        # Create session info table
        info_table = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
        info_table.add_column(style="dim")
        info_table.add_column()
        info_table.add_row(
            "Started at:", f"[cyan]{self.start_time.strftime('%H:%M:%S')}[/cyan]"
        )
        info_table.add_row("", "[dim]Listening... (Press CTRL+C to stop)[/dim]")

        console.print()
        console.print(
            Panel(
                info_table,
                title="[bold green]SESSION STARTED",
                title_align="center",
                border_style="bold green",
                padding=(0, 1),
            )
        )
        console.print()

    async def receive_turn(self, turn: Turn) -> None:
        """
        Receives updates about every turn and prints them to console.
        """
        # Skip empty transcripts
        if not turn.text.strip():
            return

        if turn.final:
            self.turn_count += 1
            # Final transcript - print with formatting
            console.print(
                f"[bold cyan]►[/bold cyan] {turn.text}",
            )
            console.print()  # Add spacing after final turns
            self.last_transcript = ""
        else:
            # Interim transcript - show with dim styling and overwrite previous
            if turn.text != self.last_transcript:
                console.print(
                    f"[dim cyan]…[/dim cyan] [dim]{turn.text}[/dim]", end="\r"
                )
                self.last_transcript = turn.text

    async def stop(self) -> None:
        """
        When the stream ends, a recap table is created.
        """
        if not self.start_time:
            return

        duration = datetime.now().astimezone() - self.start_time

        summary_table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            pad_edge=False,
        )
        summary_table.add_column(style="dim")
        summary_table.add_column(style="cyan")

        summary_table.add_row("Total turns:", str(self.turn_count))
        summary_table.add_row("Session duration:", str(duration).split(".")[0])

        console.print()
        console.print(
            Panel(
                summary_table,
                title="[bold yellow]SESSION ENDED",
                title_align="center",
                border_style="bold yellow",
                padding=(0, 1),
            )
        )
        console.print()


@cli.command()
@click.option(
    *["--device", "-d"],
    type=int,
    default=None,
    help="The index of the device to use.",
    required=False,
)
@click.option(
    *["--region", "-r"],
    type=click.Choice(["eu", "us"]),
    default="eu",
)
@click.option(
    *["--replay-file", "-f"],
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "If specified, the content of this file will be used instead of the "
        "actual microphone. It still goes through the same API, this is mostly "
        "useful for debugging purposes. Requires `ffmpeg`."
    ),
)
@click.pass_obj
@run_sync
async def transcribe(
    obj: Context,
    region: Literal["eu", "us"],
    replay_file: str,
    device: int | None = None,
):
    """Transcribes live the audio from the microphone"""

    if not (key := obj.store.get("assembly_ai")):
        msg = "Assembly AI key not found, set it with the set-token command."
        raise click.ClickException(msg)

    # Initialize Audio Source
    source: AudioSource
    if replay_file:
        file_factory = FileSourceFactory(sample_rate=16_000, realtime=True)
        source = file_factory.create_source(replay_file)
    else:
        mic_factory = MicSourceFactory(sample_rate=16_000)
        if device is not None and not mic_factory.is_device_valid(device):
            msg = f"Device #{device} is not a valid device."
            raise click.BadParameter(msg, param_hint="--device")
        source = mic_factory.create_source(device)

    # Initialize Transcripter
    transcripter = AssemblyAITranscripter(api_key=key, region=region)
    receiver = Receiver()

    try:
        await transcripter.process(source, receiver)
    except httpx.HTTPError as e:
        display_http_error(e)
        msg = "HTTP request failed"
        raise click.ClickException(msg) from e
