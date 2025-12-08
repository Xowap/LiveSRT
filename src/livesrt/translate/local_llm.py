"""A translator that uses a local LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING, Any, cast

from rich.console import Console

from livesrt.utils import ignore_stderr

from ..async_tools import sync_to_async
from .base import TranslatedTurn, Translator

if TYPE_CHECKING:
    from llama_cpp import Llama

    from ..transcribe.base import Turn


console = Console()


MODELS = {
    "qwen-3:14b:q4-k-m": ("unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf"),
    "ministral:8b:q4-k-m": ("bartowski/Ministral-8B-Instruct-2410-GGUF", "Ministral-8B-Instruct-2410-Q4_K_M.gguf"),
    "ministral:3b:q4-k-m": ("mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
}


@sync_to_async
def download_model(model: str) -> str:
    """Download a model from Hugging Face."""
    from huggingface_hub import hf_hub_download

    repo, filename = MODELS[model]
    return hf_hub_download(repo_id=repo, filename=filename)


@sync_to_async
def init_model(model_path: str, context_size: int = 10_000) -> Llama:
    """Initialize a model from a local path."""
    from llama_cpp import Llama

    with ignore_stderr():
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=context_size,
        )


@dataclass
class TurnEntry:
    """An entry for a turn."""

    turn: Turn
    completion: dict
    translated: list[TranslatedTurn]


@dataclass
class LocalLLM(Translator):
    """A translator that uses a local LLM."""

    llm: Llama
    lang_to: str
    lang_from: str = ""
    known_turns: dict[int, TurnEntry] = field(default_factory=dict, init=False)

    @classmethod
    async def create_translator(
        cls, lang_to: str, lang_from: str = "", **extra: Any
    ) -> Translator:
        """Create a new translator."""
        model = extra.get("model", "ministral:3b:q4-k-m")
        model_path = await download_model(model)
        llm = await init_model(model_path)

        return LocalLLM(
            llm=llm,
            lang_to=lang_to,
            lang_from=lang_from,
        )

    @sync_to_async
    def _next_turn(self, turns: list[Turn]) -> bool:
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
                    "content": json.dumps(sentences),
                }
            )

            entry = self.known_turns.get(turn.id)

            if entry and entry.turn.words == turn.words:
                # Append the assistant's previous response
                completion = self.known_turns[turn.id].completion
                conversation.append(completion)

                # --- FIX START ---
                # If the previous response was a tool call, we MUST append a
                # corresponding 'tool' role message to satisfy the chat template
                # structure (User -> Assistant(Tool) -> Tool -> User).
                if completion.get("tool_calls"):
                    for tool_call in completion["tool_calls"]:
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": '{"status": "logged"}',  # Dummy content to satisfy history
                        })
                # --- FIX END ---
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

        completion = self.llm.create_chat_completion(
            messages=conversation,  # type: ignore
            tools=tools,  # type: ignore
        )

        translated = []
        turn_id = 0

        if not translated_turn:
            return False

        assert isinstance(completion, dict)  # noqa: S101

        match completion["choices"][0]:
            case {"message": {"role": "assistant", "tool_calls": tool_calls}}:
                for call in tool_calls:
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

            return [tt for entry in self.known_turns.values() for tt in entry.translated]
        except Exception:
            console.print_exception(show_locals=True)
