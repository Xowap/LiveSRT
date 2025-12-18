"""
Base interface for translation systems
"""

import abc
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import groupby
from typing import Literal

from livesrt.transcribe.base import Turn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslatedTurn:
    """
    Represents a turn of speech, after translation and post-processing.

    These turns are never final, as further context might change the meaning
    of previously said things.

    The `id` give you a unique ID to follow up updates in the future, and the
    `original_id` maps back to the original turn (might not be a 1:1 mapping).
    """

    id: int
    original_id: int
    speaker: str
    text: str
    start: timedelta | None = None
    end: timedelta | None = None
    hidden: bool = False
    tone: str | None = None


class TranslationReceiver(abc.ABC):
    """
    Abstract base class for receiving translated turns.
    """

    @abc.abstractmethod
    async def receive_translations(self, turns: list[TranslatedTurn]) -> None:
        """
        Implement this function and receive turns that have been translated.
        Same as the translator gets all the turns for update at once, this will
        receive all translations, whether they changed or not.
        """


class Translator(abc.ABC):
    """
    Implement this interface in order to have a working translation system
    """

    async def init(self):
        """
        One-time init function that gets called before going into business. You
        don't have to implement it, but you can.
        """
        return

    @abc.abstractmethod
    async def update_turns(self, turns: list[Turn]) -> None:
        """
        This gets called when there is a change in the conversation turns. The
        idea is that those changes will trigger a (re-)translation of the
        relevant parts.
        """

    @abc.abstractmethod
    async def process(self, receiver: TranslationReceiver) -> None:
        """
        Run this function forever/in an independent task in order to process
        the turn changes and receive updates whenever something is new. If
        anything goes wrong in the process, it's up to the implementer to figure
        a way to have retries and other strategies.
        """

    async def health_check(self) -> None:
        """
        Checks if the translation service is available.
        Should raise an exception if not.
        """
        return

    def get_settings(self) -> dict[str, str]:
        """
        Returns a dictionary of relevant settings for display.
        """
        return {}


@dataclass
class LlmTranslationEntry:
    """
    Represents an entry in the LLM translation process, holding the original
    turn, the LLM's completion, and the final translated turns.
    """

    turn: Turn
    completion: dict | None = None
    translated: list[TranslatedTurn] | None = None
    tool_outputs: list[str] | None = None


@dataclass
class LlmTranslator(Translator, abc.ABC):
    """
    A translator that can be used as a base for LLM-based translation (namely
    for local and remote models).
    """

    lang_to: str
    lang_from: str = ""
    has_new_turns: asyncio.Event = field(default_factory=asyncio.Event)
    turns: dict[int, LlmTranslationEntry] = field(default_factory=dict)
    _queued_turns: list[Turn] = field(default_factory=list)

    def get_settings(self) -> dict[str, str]:
        """Returns a dictionary of relevant settings for display."""
        return {
            "From": self.lang_from or "Auto",
            "To": self.lang_to,
        }

    async def update_turns(self, turns: list[Turn]) -> None:
        """
        We store the next batch of turns to update and mark the new turns flag.
        This way the processing loop can pick the latest version and
        intermediate versions of turns that appeared during the processing of
        the translation will get discarded.
        """

        self._queued_turns = turns
        self.has_new_turns.set()

    def _update_turns(self) -> None:
        """
        We are getting the new turns list, and then we make the diff with the
        existing list. The idea is that we'll detect the earliest change and
        then blank out the subsequent turn's translation, if any. The idea is
        that given a past translation might affect future translations, we
        want to start back form there. In practice the translation system only
        changes the last turn anyway.
        """

        min_diff = float("inf")

        for turn in self._queued_turns:
            if not turn.words:
                continue

            old_turn = self.turns.get(turn.id)

            if not old_turn:
                self.turns[turn.id] = LlmTranslationEntry(turn=turn)
                min_diff = min(turn.id, min_diff)
            elif old_turn.turn.text != turn.text:
                self.turns[turn.id].turn = turn
                min_diff = min(turn.id, min_diff)

        for entry in self.turns.values():
            if entry.turn.id >= min_diff:
                entry.completion = None
                entry.translated = None

    def _build_system_prompt(self):
        return (
            f"You are a professional interpreter translating to {self.lang_to}. "
            "You will receive live ASR transcripts that may contain errors, "
            "wrong speaker assignments, and fragmentation.\n\n"
            "Your task is to:\n"
            "1. Reconstruct the dialogue line by line.\n"
            "2. Assign the correct speaker (correcting the ASR if needed).\n"
            "3. Translate the dialogue into natural, grammatically correct "
            f"{self.lang_to}.\n"
            "   - PRESERVE the original tone, style, and register of the speaker.\n"
            "   - SPLIT the output so that EVERY sentence is on its own separate "
            "line.\n"
            "4. Emit EXACTLY ONE tool call containing the list of dialogue "
            "lines.\n\n"
            "Use the `status`, `comment` and `tone` fields to express uncertainty, "
            "impossibility or specific tone, ensuring the `text` field remains "
            "clean and reader-friendly. NEVER include the tone in the `text`."
        )

    def _build_user_message(self, turn: Turn) -> str:
        """
        Transforming one turn into a simplified JSON structure that will be
        translated by our LLM. The idea is to group the words by whom uttered
        them (which is usually all of them in a single turn).
        """

        sentences = []

        for speaker, words in groupby(turn.words, lambda w: w.speaker):
            sentences.append(
                dict(
                    speaker=speaker,
                    asr_words=[w.text for w in words],
                )
            )

        return json.dumps(sentences, ensure_ascii=False)

    def _build_conversation(self) -> tuple[LlmTranslationEntry | None, int, list[dict]]:
        """
        Building up the conversation. The idea is that for each turn there is
        a message from the user, a bunch of tool calls with the translation, and
        so forth. Which allows to keep in cache most of the conversation
        (including input and output) while only having to translate the final
        turn, effectively making it very fast to do even while keeping the
        whole context.

        If there is an entry returned then it means that this is the entry that
        has to be translated right now. If not it means that no entry needs to
        be translated. Essentially to catch up you need to build the
        conversation and translate it until no entry is returned.
        """

        turn_id = 0
        conversation = []
        to_translate: LlmTranslationEntry | None = None

        all_turns = sorted(self.turns.values(), key=lambda t: t.turn.id)
        keep_turns = 10 + len(all_turns) % 10
        start_index = max(0, len(all_turns) - keep_turns)

        for entry in all_turns[:start_index]:
            if entry.translated:
                turn_id += len(entry.translated)

        for entry in all_turns[start_index:]:
            conversation.append(
                dict(
                    role="user",
                    content=self._build_user_message(entry.turn),
                )
            )

            if entry.completion:
                turn_id += len(entry.translated or [])
                conversation.append(entry.completion)

                tool_outputs = iter(entry.tool_outputs or [])

                for tool_call in entry.completion.get("tool_calls") or []:
                    conversation.append(
                        dict(
                            role="tool",
                            tool_call_id=tool_call["id"],
                            content=next(tool_outputs, "Recorded"),
                        )
                    )

                conversation.append(
                    dict(
                        role="assistant",
                        content="ok",
                    )
                )
            else:
                to_translate = entry
                break

        if to_translate:
            return to_translate, turn_id, conversation
        else:
            return None, turn_id, conversation

    def _build_tools(self):
        """
        Builds the tools that explain the LLM what to do. That's how we make
        sure that the LLM does what it's asked.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": "translate",
                    "description": (
                        "Emit the translated dialogue. Break down the input "
                        "into individual dialogue lines."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lines": {
                                "type": "array",
                                "description": "List of translated dialogue lines.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "speaker": {
                                            "type": "string",
                                            "description": "Speaker name (corrected).",
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": (
                                                "The translated text. Must be "
                                                "clean, grammatical, and "
                                                "readable. Include sounds in "
                                                "parentheses."
                                            ),
                                        },
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "success",
                                                "uncertain",
                                                "impossible",
                                            ],
                                            "description": (
                                                "The confidence status of the "
                                                "translation. Use 'impossible' "
                                                "for gibberish."
                                            ),
                                        },
                                        "comment": {
                                            "type": "string",
                                            "description": (
                                                "Any notes about the "
                                                "translation, uncertainty, or "
                                                "issues."
                                            ),
                                        },
                                        "tone": {
                                            "type": "string",
                                            "description": "Translation tone.",
                                        },
                                    },
                                    "required": ["speaker", "text", "status"],
                                },
                            }
                        },
                        "required": ["lines"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_turn",
                    "description": (
                        "Call this function to delete a previously emitted "
                        "turn. This is useful when the context changes and a "
                        "previous translation is no longer valid (e.g. because "
                        "a sentence was cut in half). You should then emit a "
                        "new replacement turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "turn_id": {
                                "type": "integer",
                                "description": "The ID of the turn to delete.",
                            },
                        },
                        "required": ["turn_id"],
                    },
                },
            },
        ]

    def _decode_completion(
        self,
        turn: Turn,
        next_id: int,
        completion: dict,
    ) -> tuple[dict, list[TranslatedTurn], list[int], list[str]]:
        out: list[TranslatedTurn] = []
        deleted_ids: list[int] = []
        tool_outputs: list[str] = []
        debug_entries: list[dict] = []

        input_data = json.loads(self._build_user_message(turn))

        message: dict
        match message := completion["choices"][0]["message"]:
            case {"role": "assistant", "tool_calls": [*tool_calls]}:
                for call in tool_calls:
                    match call:
                        case {
                            "function": {
                                "name": "translate",
                                "arguments": arguments,
                            }
                        }:
                            parsed = json.loads(arguments)
                            lines = parsed.get("lines", [])

                            for line in lines:
                                match line:
                                    case {
                                        "speaker": speaker,
                                        "text": text,
                                        "status": status,
                                    }:
                                        out.append(
                                            TranslatedTurn(
                                                id=next_id,
                                                original_id=turn.id,
                                                speaker=speaker,
                                                text=text,
                                                hidden=(status == "impossible"),
                                                tone=line.get("tone"),
                                            )
                                        )
                                        debug_entries.append(
                                            {
                                                "summary": (
                                                    f"[{status.upper()}] {speaker}: "
                                                    f"{text[:20]}..."
                                                ),
                                                "details": {
                                                    "input": input_data,
                                                    "output": {
                                                        "function": "translate",
                                                        "parameters": line,
                                                    },
                                                    "comment": line.get("comment"),
                                                    "tone": line.get("tone"),
                                                },
                                            }
                                        )
                                        next_id += 1
                                    case _:
                                        logger.warning(
                                            "Unexpected line format in completion: %s",
                                            line,
                                        )
                                        continue

                            tool_outputs.append(str(len(lines)))

                        case {
                            "function": {
                                "name": "delete_turn",
                                "arguments": arguments,
                            }
                        }:
                            parsed = json.loads(arguments)
                            deleted_ids.append(parsed["turn_id"])
                            out.append(
                                TranslatedTurn(
                                    id=next_id,
                                    original_id=turn.id,
                                    speaker="",
                                    text="",
                                    hidden=True,
                                )
                            )
                            debug_entries.append(
                                {
                                    "summary": f"Delete {parsed['turn_id']}",
                                    "details": {
                                        "input": input_data,
                                        "output": {
                                            "function": "delete_turn",
                                            "parameters": parsed,
                                        },
                                    },
                                }
                            )
                            tool_outputs.append("Deleted")
                            next_id += 1

                        case _:
                            tool_outputs.append("Recorded")

        turn.debug = debug_entries
        return message, out, deleted_ids, tool_outputs

    async def _translate_next_turn(self) -> bool:
        to_translate, next_id, conversation = self._build_conversation()

        if not to_translate:
            return False

        messages = [
            dict(
                role="system",
                content=self._build_system_prompt(),
            ),
            *conversation,
        ]

        completion = await self.completion(
            messages=messages,
            tools=self._build_tools(),
            tool_choice="required",
        )

        response, turns, deleted_ids, tool_outputs = self._decode_completion(
            to_translate.turn, next_id, completion
        )
        to_translate.completion = response
        to_translate.translated = turns
        to_translate.tool_outputs = tool_outputs

        if deleted_ids:
            for entry in self.turns.values():
                if entry.translated:
                    entry.translated = [
                        t for t in entry.translated if t.id not in deleted_ids
                    ]

        return True

    @abc.abstractmethod
    async def completion(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: Literal["auto", "required", "none"] | dict = "auto",
    ) -> dict:
        """
        Abstract method to be implemented by concrete LLM translation classes
        to perform the actual completion call to the LLM API.
        """
        raise NotImplementedError

    async def process(self, receiver: TranslationReceiver):
        """
        As soon as there is new
        """

        while await self.has_new_turns.wait():
            self.has_new_turns.clear()

            try:
                self._update_turns()

                while await self._translate_next_turn():
                    await receiver.receive_translations(
                        sorted(
                            [
                                t
                                for e in self.turns.values()
                                if e.translated
                                for t in e.translated
                            ],
                            key=lambda t: t.id,
                        )
                    )
            except (asyncio.CancelledError, SystemExit):
                raise
            except Exception:
                logger.exception("Unexpected exception occurred")
