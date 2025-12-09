"""
TUI implementation for LiveSRT
"""

import json
from typing import ClassVar

from rich.syntax import Syntax
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Static

from livesrt.transcribe.base import (
    AudioSource,
    Transcripter,
    TranscriptReceiver,
    Turn,
)
from livesrt.translate import TranslatedTurn, TranslationReceiver, Translator


class DebugDetailsScreen(ModalScreen):
    """Screen to show debug details."""

    CSS = """
    DebugDetailsScreen {
        align: center middle;
    }

    DebugDetailsScreen > Vertical {
        width: 80%;
        height: 80%;
        border: solid green;
        background: $surface;
    }

    DebugDetailsScreen .json-scroll {
        height: 1fr;
        padding: 1;
    }

    DebugDetailsScreen Button {
        dock: bottom;
        width: 100%;
    }
    """

    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        with Vertical():
            # Use Syntax with word_wrap=True to ensure text wrapping
            json_str = json.dumps(self.data, indent=2, ensure_ascii=False)
            with VerticalScroll(classes="json-scroll"):
                yield Static(Syntax(json_str, "json", word_wrap=True))
            yield Button("Close", variant="primary", id="close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press to close screen."""
        self.dismiss()


class DebugEntry(Static):
    """A single debug entry."""

    def __init__(self, summary: str, details: dict):
        super().__init__(summary)
        self.details = details

    def on_click(self) -> None:
        """Handle click to show details."""
        self.app.push_screen(DebugDetailsScreen(self.details))


class DebugGroup(Vertical):
    """A group of debug entries for a specific turn."""

    def __init__(self, turn_id: int):
        self.turn_id = turn_id
        self.current_debug_data: list[dict] = []
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the group layout."""
        yield Static(f"Turn #{self.turn_id}", classes="debug-group-header")


class DebugPanel(VerticalScroll):
    """Panel to show debug entries."""


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
        self.original_id = turn.original_id
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

    def update_content(self, speaker: str, text: str, original_id: int):
        """Update the content of the translated turn."""
        self.speaker = speaker
        self.text_content = text
        self.original_id = original_id
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
    DebugPanel {
        width: 30%;
        dock: right;
        background: $surface;
        border-left: solid $primary;
        display: none;
    }
    DebugPanel.-open {
        display: block;
    }
    DebugGroup {
        height: auto;
        margin-bottom: 1;
    }
    .debug-group-header {
        background: $accent;
        color: $text;
        padding: 0 1;
        text-style: bold;
    }
    DebugEntry {
        padding: 0 1;
        height: auto;
        border-bottom: none;
    }
    DebugEntry:hover {
        background: $primary 30%;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [  # type: ignore
        Binding("q", "quit", "Quit"),
        Binding("s", "toggle_autoscroll", "Toggle Auto-Scroll"),
        Binding("d", "toggle_debug", "Debug"),
    ]

    auto_scroll: bool = True

    def action_toggle_autoscroll(self) -> None:
        """Toggle auto-scroll."""
        self.auto_scroll = not self.auto_scroll
        status = "enabled" if self.auto_scroll else "disabled"
        self.notify(f"Auto-scroll {status}")

    def action_toggle_debug(self) -> None:
        """Toggle debug panel."""
        panel = self.query_one(DebugPanel)
        panel.toggle_class("-open")

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
        self._debug_groups: dict[int, DebugGroup] = {}
        self.auto_scroll = True
        self.receiver = AppReceiver(self)

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        yield VerticalScroll(id="content")
        yield DebugPanel(id="debug-panel")
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
            container.scroll_end()
        if self.translator:
            await self.translator.update_turns(list(self._source_turns.values()))

    async def _update_debug_panel(self, turns: list[TranslatedTurn]) -> None:
        debug_panel = self.query_one(DebugPanel)
        source_ids = {t.original_id for t in turns}

        for source_id in source_ids:
            if source_id not in self._source_turns:
                continue

            turn = self._source_turns[source_id]

            # Get or create group for this source turn
            if source_id not in self._debug_groups:
                group = DebugGroup(source_id)
                self._debug_groups[source_id] = group
                await debug_panel.mount(group)
            else:
                group = self._debug_groups[source_id]

            if group.current_debug_data == turn.debug:
                continue

            group.current_debug_data = turn.debug

            # Clear existing debug entries
            # Note: We await the removal of all DebugEntry widgets
            await group.query(DebugEntry).remove()

            for entry in turn.debug:
                await group.mount(DebugEntry(entry["summary"], entry["details"]))

    async def receive_translations(self, turns: list[TranslatedTurn]) -> None:
        """Receive translated turns and update UI."""
        container = self.query_one("#content", VerticalScroll)
        debug_panel = self.query_one(DebugPanel)

        # 1. Update main content (filter out hidden turns)
        visible_turns = [t for t in turns if not t.hidden]
        incoming_ids = {t.id for t in visible_turns}
        to_remove = [tid for tid in self.translated_widgets if tid not in incoming_ids]

        for tid in to_remove:
            widget = self.translated_widgets.pop(tid)
            await widget.remove()

        anchors: dict[int, Static] = dict(self.source_widgets)

        for turn in visible_turns:
            if turn.id in self.translated_widgets:
                widget = self.translated_widgets[turn.id]
                widget.update_content(turn.speaker, turn.text, turn.original_id)
            else:
                widget = TranslatedWidget(turn)
                self.translated_widgets[turn.id] = widget

            anchor = anchors.get(turn.original_id)

            if anchor:
                if widget.parent is None:
                    await container.mount(widget, after=anchor)
                else:
                    container.move_child(widget, after=anchor)
                anchors[turn.original_id] = widget
            elif widget.parent is None:
                await container.mount(widget)

        # 2. Update debug panel (process ALL turns)
        await self._update_debug_panel(turns)

        if self.auto_scroll:
            container.scroll_end()
            debug_panel.scroll_end()

    async def stop(self) -> None:
        """Called by transcripter/translator if they were to call it."""
        pass
