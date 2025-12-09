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


@dataclass
class LlmTranslationEntry:
    """
    Represents an entry in the LLM translation process, holding the original
    turn, the LLM's completion, and the final translated turns.
    """

    turn: Turn
    completion: dict | None = None
    translated: list[TranslatedTurn] | None = None


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
    _queued_turns: list[Turn] = field(default_factory=dict)

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
            "You are a translator. The user provides the output of an ASR "
            "service. Your job is to interpret who said what (keep in mind "
            "that the ASR makes mistakes) and report properly formatted and "
            "constructed sentences using the available tool. Call the tool "
            f"once or several times at each turn. The target language is: "
            f"{self.lang_to}"
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

        for entry in sorted(self.turns.values(), key=lambda t: t.turn.id):
            conversation.append(
                dict(
                    role="user",
                    content=self._build_user_message(entry.turn),
                )
            )

            if entry.completion:
                turn_id += len(entry.translated or [])
                conversation.append(entry.completion)

                for tool_call in entry.completion.get("tool_calls") or []:
                    conversation.append(
                        dict(
                            role="tool",
                            tool_call_id=tool_call["id"],
                            content="Recorded",
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
                        "⚠ CALL THIS FUNCTION TO SUBMIT YOUR ANSWER ⚠\n\n"
                        "You receive messy ASR transcription with errors, "
                        "overlaps, and incomplete words. DO NOT ask for help "
                        "or tools. YOU must:\n1. Fix ASR errors and typos\n2. "
                        "Separate overlapping speech\n3. Remove stutters and "
                        "filler words\n4. Create grammatically correct "
                        "sentences\n5. Translate to target language\n6. CALL "
                        "THIS FUNCTION with the result\n\nThe function "
                        "parameters are where you write your cleaned, "
                        "translated output."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "speaker": {
                                "type": "string",
                                "description": "Speaker name/ID from the input",
                            },
                            "text": {
                                "type": "string",
                                "description": (
                                    "⚠ PUT YOUR CLEANED & TRANSLATED TEXT "
                                    "HERE ⚠ - This is your final answer: "
                                    "properly formatted, error-free, "
                                    f"translated into {self.lang_to} sentences"
                                ),
                            },
                            "comment": {
                                "type": "string",
                                "description": (
                                    "Translation comments. Leave blank unless "
                                    "there is something really important to "
                                    "say."
                                ),
                            },
                        },
                        "required": ["speaker", "text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pass",
                    "description": (
                        "The input might be gibberish, incomplete our too "
                        "out-of-context to be translated. In this case, call "
                        "that function."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "A potential question you might have "
                                    "regarding the input. Only fill if "
                                    "necessary."
                                ),
                            }
                        },
                    },
                },
            },
        ]

    def _decode_completion(
        self,
        turn: Turn,
        next_id: int,
        completion: dict,
    ) -> tuple[dict, list[TranslatedTurn]]:
        out: list[TranslatedTurn] = []

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
                            out.append(
                                TranslatedTurn(
                                    id=next_id,
                                    original_id=turn.id,
                                    speaker=parsed["speaker"],
                                    text=parsed["text"],
                                )
                            )
                            next_id += 1

        return message, out

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

        response, turns = self._decode_completion(
            to_translate.turn, next_id, completion
        )
        to_translate.completion = response
        to_translate.translated = turns

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
