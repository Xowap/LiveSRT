"""A translator that uses a local LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import groupby
from tenacity import retry, stop_after_attempt, retry_if_exception
from typing import TYPE_CHECKING, Any, cast

import httpx
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from .base import TranslatedTurn, Translator

if TYPE_CHECKING:
    from ..transcribe.base import Turn


console = Console()
logger = logging.getLogger(__name__)


@dataclass
class TurnEntry:
    """An entry for a turn."""

    turn: Turn
    completion: dict
    translated: list[TranslatedTurn]


def is_missing_tool_call(exception: BaseException) -> bool:
    """If the model didn't call the tool, let's try again"""

    if not isinstance(exception, httpx.HTTPStatusError):
        return False

    try:
        data = exception.response.json()
    except json.decoder.JSONDecodeError:
        return False

    match data:
        case {"error": {"code": "tool_use_failed"}}:
            return True

    return False


@retry(
    retry=retry_if_exception(is_missing_tool_call),
    stop=stop_after_attempt(5),
)
async def call_completion(
    model: str,
    api_key: str,
    messages: list[dict],
    tools: list[dict],
    tool_choice: dict | str = 'auto',
) -> dict:
    provider, _, model_id = model.partition("/")

    base_url = {
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "mistral": "https://api.mistral.ai/v1/chat/completions",
    }[provider]

    client: httpx.AsyncClient
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=600,
    ) as client:
        req_body = dict(
            model=model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        print('req start')
        resp = await client.post(
            base_url,
            json=req_body,
        )
        print('req done')

        if 400 <= resp.status_code < 500:
            p_req = Panel(
                JSON.from_data(req_body),
                title="Request",
                border_style="red",
            )
            console.print(p_req)
            p_resp = Panel(
                JSON(resp.text),
                title="Response (with error)",
                border_style="red",
            )
            console.print(p_resp)

        resp.raise_for_status()

        return resp.json()


@dataclass(kw_only=True)
class RemoteLLM(Translator):
    """A translator that uses a local LLM."""

    model: str = "groq/openai/gpt-oss-120b"
    api_key: str
    lang_to: str
    lang_from: str = ""
    known_turns: dict[int, TurnEntry] = field(default_factory=dict, init=False)

    @classmethod
    async def create_translator(
        cls, lang_to: str, lang_from: str = "", **extra: Any
    ) -> Translator:
        """Create a new translator."""
        return RemoteLLM(
            lang_to=lang_to,
            lang_from=lang_from,
            **extra,
        )

    async def _next_turn(self, turns: list[Turn]) -> bool:
        full_turns = [t for t in turns if t.words]

        if not full_turns:
            return False

        system = (
            "You are a translator. The user provides the output of an ASR "
            "service. Your job is to interpret who said what (keep in mind "
            "that the ASR makes mistakes) and report properly formatted and "
            "constructed sentences using the available tool. Call the tool "
            f"once or several times at each turn. The target language is: "
            f"{self.lang_to}"
        )

        conversation = [
            dict(
                role="system",
                content=system,
            )
        ]

        missing_turns = False
        translated_turn = None

        for turn in full_turns:
            sentences = []

            for speaker, words in groupby(turn.words, lambda w: w.speaker):
                sentences.append(
                    dict(
                        speaker=(speaker or "Someone"),
                        asr_words=[w.text for w in words],
                    )
                )

            conversation.append(
                {
                    "role": "user",
                    "content": json.dumps(sentences, ensure_ascii=False),
                }
            )

            entry = self.known_turns.get(turn.id)

            if entry and entry.turn.words == turn.words:
                conversation.append(self.known_turns[turn.id].completion)
            else:
                missing_turns = True
                translated_turn = turn
                break

        if not missing_turns:
            return False

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "translate",
                    "description": (
                        "⚠️ CALL THIS FUNCTION TO SUBMIT YOUR ANSWER ⚠️\n\n"
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
                                    "⚠️ PUT YOUR CLEANED & TRANSLATED TEXT "
                                    "HERE ⚠️ - This is your final answer: "
                                    "properly formatted, error-free, "
                                    f"translated into {self.lang_to} sentences"
                                ),
                            },
                        },
                        "required": ["speaker", "text"],
                    },
                },
            }
        ]

        completion = await call_completion(
            model=self.model,
            api_key=self.api_key,
            messages=conversation,
            tools=tools,
        )

        # panel = Panel(JSON.from_data(completion), title="Completion")
        # console.print(panel)

        translated = []
        turn_id = 0

        if not translated_turn:
            return False

        assert isinstance(completion, dict)  # noqa: S101

        match completion["choices"][0]:
            case {"message": {"role": "assistant", "tool_calls": tool_calls}}:
                for call in tool_calls or []:
                    match call:
                        case {
                            "function": {"name": "translate", "arguments": arguments}
                        }:
                            parsed = json.loads(arguments)
                            translated.append(
                                TranslatedTurn(
                                    id=turn_id,
                                    original_id=translated_turn.id,
                                    speaker=parsed["speaker"],
                                    text=parsed["text"],
                                )
                            )
                            turn_id += 1

        entry = TurnEntry(
            turn=translated_turn,
            translated=translated,
            completion=cast("dict", completion["choices"][0]["message"]),
        )
        self.known_turns[translated_turn.id] = entry

        return missing_turns

    async def translate(self, turns: list[Turn]) -> list[TranslatedTurn]:
        """Translate a list of turns."""

        try:
            while await self._next_turn(turns):
                pass

            return [
                tt for entry in self.known_turns.values() for tt in entry.translated
            ]
        except Exception:
            logger.exception("Translation failed")
