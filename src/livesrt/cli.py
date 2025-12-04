from dataclasses import dataclass
from enum import Enum

import keyring
import pwinput
import rich_click as click
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "danger": "bold red"})
console = Console(theme=custom_theme)


def validate_no_colon(ctx, param, value):
    """
    Callback to ensure the namespace does not contain a colon.
    """

    if ":" in value:
        raise click.BadParameter("The character ':' is not allowed in the namespace.")

    return value


@dataclass
class ApiKeyStore:
    namespace: str
    system: str = "livesrt"

    def key(self, provider: str) -> str:
        return f"{self.namespace}:{provider}"

    def get(self, provider: str) -> str:
        return keyring.get_password(self.system, self.key(provider))

    def set(self, provider: str, value: str) -> None:
        keyring.set_password(self.system, self.key(provider), value)


@dataclass
class Context:
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
    ctx.obj = Context(
        namespace=namespace,
        store=ApiKeyStore(namespace),
    )


class ProviderType(Enum):
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
        f"\n[green]✔[/green] Configuration started for [bold cyan]{provider}[/bold cyan]"
    )


@cli.command()
@click.pass_obj
def transcribe(obj: Context):
    """Transcribes live the audio from the microphone"""

    if not (key := obj.store.get("assembly_ai2")):
        msg = "No API key for assembly_ai, please set it with the `transcribe` command."
        raise click.ClickException(msg)

    console.print(key)
