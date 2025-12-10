"""
The logic specific to the CLI interface
"""

import importlib.metadata
import sys
from pathlib import Path

import httpx
import pwinput
import rich_click as click
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from livesrt.async_tools import run_sync
from livesrt.config_template import DEFAULT_CONFIG_CONTENT
from livesrt.constants import ProviderType
from livesrt.containers import Container
from livesrt.transcribe.audio_sources.mic import MicSourceFactory

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


def print_version(ctx, _, value):
    """Prints the version and exits."""
    if not value or ctx.resilient_parsing:
        return

    version_str = importlib.metadata.version("livesrt")
    console.print(f"[bold green]livesrt version: [bold yellow]{version_str}[/]")

    ctx.exit()


@click.group()
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show the version and exit.",
)
@click.option(
    "--config",
    "-c",
    default="config.yml",
    help="Path to the configuration file.",
    type=click.Path(dir_okay=False),
)
@click.pass_context
def cli(ctx, config):
    """
    Main entrypoint of the whole thing
    """
    container = Container()

    # Load configuration
    config_path = Path(config)
    if config_path.exists():
        container.config.from_yaml(str(config_path))
    else:
        # If the user didn't provide a config and the default doesn't exist,
        # we might want to warn or just use defaults if possible.
        # But for now, let's assume if it's explicitly passed it must exist,
        # but if it's default and missing, we might proceed with defaults
        # if defined in container.
        # However, our container relies on config for many things.
        if config != "config.yml":
            console.print(f"[warning] Configuration file {config} not found.[/warning]")

    ctx.obj = container


@cli.command()
@click.option(
    "--output",
    "-o",
    default="config.yml",
    help="Path to the output configuration file.",
    type=click.Path(dir_okay=False, writable=True, resolve_path=True),
)
def init_config(output):
    """
    Creates a default configuration file.
    """
    output_path = Path(output)

    if output_path.exists():
        if not click.confirm(
            f"File {output_path} already exists. Overwrite?",
            abort=False,
        ):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

    try:
        output_path.write_text(DEFAULT_CONFIG_CONTENT)
        console.print(f"[green]Default configuration written to {output_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error writing config file: {e}[/bold red]")
        sys.exit(1)


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
def set_token(container: Container, provider, api_key):
    """
    Sets the API token for a specific provider.
    """

    if not api_key:
        console.print("🔐 API key: ", style="bold", end="")
        api_key = pwinput.pwinput(prompt="", mask="*")

    if not api_key:
        console.print("\n[warning]🫡 Not setting anything")
        return

    # access the singleton ApiKeyStore
    store = container.api_key_store()
    store.set(provider, api_key)

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


@cli.command()
@click.option(
    "--translate/--no-translate",
    default=False,
    help="Enable translation.",
)
@click.pass_obj
@run_sync
async def run(container: Container, translate: bool):
    """
    Run the LiveSRT application.
    """

    # Override translation enabled status
    if translate:
        container.config.translation.enabled.from_value(True)

    try:
        app = container.app()

        with console.status("[bold green]Performing health checks..."):
            await app.health_check()

        result = await app.run_async()

        if isinstance(result, Exception):
            if (
                isinstance(result, httpx.HTTPStatusError)
                and result.response.status_code == 401
            ):
                console.print(f"[bold red]Authentication Error:[/bold red] {result}")
                console.print(
                    "[yellow]Please check your API key configuration in config.yaml "
                    "or use 'livesrt set-token'.[/yellow]"
                )
            else:
                console.print(f"[bold red]Application Error:[/bold red] {result}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
