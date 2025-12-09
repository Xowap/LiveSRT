"""
TUI implementation for LiveSRT
"""

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Static

from livesrt.transcribe.base import (
    AudioSource,
    Transcripter,
    TranscriptReceiver,
    Turn,
)
from livesrt.translate import TranslatedTurn, TranslationReceiver, Translator


class TurnWidget(Static):
    """Display a single source turn."""

    def __init__(self, turn: Turn, **kwargs):
        self.turn_id = turn.id
        self.text_content = turn.text
        super().__init__(self._get_renderable(), **kwargs)

    def _get_renderable(self) -> str:
        return f"[bold cyan]@{self.turn_id}[/bold cyan] [dim]{self.text_content}[/dim]"

    def update_text(self, text: str):
        """Update the text of the turn."""
        self.text_content = text
        self.update(self._get_renderable())


class TranslatedWidget(Static):
    """Display a translated turn."""

    def __init__(self, turn: TranslatedTurn, **kwargs):
        self.turn_id = turn.id
        self.speaker = turn.speaker
        self.text_content = turn.text
        super().__init__(self._get_renderable(), **kwargs)
        # Indent visually
        self.styles.margin = (0, 0, 0, 4)

    def _get_renderable(self) -> str:
        return (
            f"[bold yellow]#{self.turn_id}[/bold yellow] "
            f"[bold green]{self.speaker}[/bold green]: "
            f"[white]{self.text_content}[/white]"
        )

    def update_content(self, speaker: str, text: str):
        """Update the content of the translated turn."""
        self.speaker = speaker
        self.text_content = text
        self.update(self._get_renderable())


class AppReceiver(TranscriptReceiver, TranslationReceiver):
    """Adapter to route receiver calls to the App."""

    def __init__(self, app: "LiveSrtApp"):
        self.app = app

    async def receive_turn(self, turn: Turn) -> None:
        """Receive turn and forward to app."""
        await self.app.receive_turn(turn)

    async def receive_translations(self, turns: list[TranslatedTurn]) -> None:
        """Receive translations and forward to app."""
        await self.app.receive_translations(turns)

    async def stop(self) -> None:
        """Receive stop and forward to app."""
        await self.app.stop()


class LiveSrtApp(App):
    """
    Main Textual Application for LiveSRT.
    """

    CSS = """
    TurnWidget {
        padding: 0 1;
        height: auto;
        width: 100%;
    }
    TranslatedWidget {
        padding: 0 1;
        height: auto;
        width: 100%;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [  # type: ignore
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        source: AudioSource,
        transcripter: Transcripter,
        translator: Translator | None = None,
    ):
        super().__init__()
        self.source = source
        self.transcripter = transcripter
        self.translator = translator

        self.source_widgets: dict[int, TurnWidget] = {}
        self.translated_widgets: dict[int, TranslatedWidget] = {}
        self._source_turns: dict[int, Turn] = {}
        self.auto_scroll = True
        self.receiver = AppReceiver(self)

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        yield VerticalScroll(id="content")
        yield Footer()

    async def on_mount(self):
        """Handle app mount event."""
        self.title = "LiveSRT"
        if self.translator:
            self.sub_title = "Translation Mode"
            await self.translator.init()
            self.run_worker(
                self.translator.process(self.receiver),
                exclusive=False,
                group="services",
            )
        else:
            self.sub_title = "Transcription Mode"

        self.run_worker(
            self.transcripter.process(self.source, self.receiver),
            exclusive=False,
            group="services",
        )

    async def receive_turn(self, turn: Turn) -> None:
        """Receive a source turn and update UI."""
        if not turn.text.strip():
            return

        self._source_turns[turn.id] = turn

        container = self.query_one("#content", VerticalScroll)

        if turn.id in self.source_widgets:
            self.source_widgets[turn.id].update_text(turn.text)
        else:
            widget = TurnWidget(turn)
            self.source_widgets[turn.id] = widget
            await container.mount(widget)
            if self.auto_scroll:
                widget.scroll_visible()

        if self.translator:
            await self.translator.update_turns(list(self._source_turns.values()))

    async def receive_translations(self, turns: list[TranslatedTurn]) -> None:
        """Receive translated turns and update UI."""
        container = self.query_one("#content", VerticalScroll)

        for turn in turns:
            if turn.id in self.translated_widgets:
                self.translated_widgets[turn.id].update_content(turn.speaker, turn.text)
            else:
                widget = TranslatedWidget(turn)
                self.translated_widgets[turn.id] = widget
                await container.mount(widget)
                if self.auto_scroll:
                    widget.scroll_visible()

    async def stop(self) -> None:
        """Called by transcripter/translator if they were to call it."""
        pass
