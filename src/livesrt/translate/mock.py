"""
Mock translator
"""

from dataclasses import dataclass
from itertools import groupby
from typing import Any

from ..transcribe.base import Turn
from .base import TranslatedTurn, Translator


@dataclass
class MockTranslator(Translator):
    """Mock translator class for UI testing"""

    lang_to: str

    @classmethod
    async def create_translator(
        cls, lang_to: str, lang_from: str = "", **extra: Any
    ) -> "Translator":
        """Long and expensive initialization"""

        return MockTranslator(lang_to=lang_to)

    async def translate(self, turns: list[Turn]) -> list[TranslatedTurn]:
        """
        This is just a mock translation method which will pretend that it's
        doing something, but it's doing nothing. It's mainly here to test out
        the UI.

        Parameters
        ----------
        turns
            A list of turns to translate
        """

        out: list[TranslatedTurn] = []

        for turn in turns:
            for speaker, words in groupby(turn.words, lambda w: w.speaker):
                if not (words_list := list(words)):
                    continue

                start = words_list[0].start
                end = words_list[-1].end

                out.append(
                    TranslatedTurn(
                        id=len(out),
                        original_id=turn.id,
                        speaker=(speaker or "Someone"),
                        text="".join(w.text for w in words_list),
                        start=start,
                        end=end,
                    )
                )

        return out
