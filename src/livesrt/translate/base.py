"""
Base interface for translation systems
"""

import abc
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from livesrt.transcribe.base import Turn


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


class Translator(abc.ABC):
    """
    Implement this interface in order to have a working translation system
    """

    @classmethod
    @abc.abstractmethod
    async def create_translator(
        cls,
        lang_to: str,
        lang_from: str = "",
        **extra: Any,
    ) -> "Translator":
        """
        Creates and initializes an instance of the translator.

        Parameters
        ----------
        lang_to
            Language to translate to
        lang_from
            Language to translate from (leave blank if auto-detect)
        extra
            Extra init parameters for when building the class
        """

    @abc.abstractmethod
    async def translate(self, turns: list[Turn]) -> list[TranslatedTurn]:
        """
        Implement this method to allow for translation of all the turns into
        the target language. This will get called repeatedly with usually only
        the last turn changing, so feel free to cache stuff for better
        performance.

        Parameters
        ----------
        turns
            Translate those turns

        Returns
        -------
        The returned list of turns will be a new list of turns, possibly of a
        different length, that are re-formatted, translated and cleaned versions
        of the initial turns.
        """
        raise NotImplementedError
