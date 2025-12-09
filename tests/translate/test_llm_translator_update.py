import asyncio
from dataclasses import dataclass

from livesrt.transcribe.base import Turn, Word
from livesrt.translate.base import LlmTranslator


@dataclass
class MockLlmTranslator(LlmTranslator):
    async def completion(self, messages, tools, tool_choice="auto"):
        return {}


def test_update_turns_propagates_changes():
    async def _test():
        translator = MockLlmTranslator(lang_to="fr")

        # Initial turn 1
        turn1_v1 = Turn(
            id=1, text="It", final=False, words=[Word(type="word", text="It")]
        )

        await translator.update_turns([turn1_v1])

        assert translator.turns[1].turn.text == "It"

        # Updated turn 1
        turn1_v2 = Turn(
            id=1,
            text="It works",
            final=False,
            words=[Word(type="word", text="It"), Word(type="word", text="works")],
        )

        await translator.update_turns([turn1_v2])

        # This assertion is expected to fail with the current bug
        assert translator.turns[1].turn.text == "It works"

    asyncio.run(_test())


def test_update_turns_invalidation():
    async def _test():
        translator = MockLlmTranslator(lang_to="fr")

        turn1 = Turn(
            id=1, text="One", final=True, words=[Word(type="word", text="One")]
        )
        await translator.update_turns([turn1])

        # Simulate that it was translated
        translator.turns[1].completion = {"some": "completion"}
        translator.turns[1].translated = ["some translation"]

        # Update turn 1 with new text
        turn1_new = Turn(
            id=1,
            text="One updated",
            final=True,
            words=[Word(type="word", text="One"), Word(type="word", text="updated")],
        )
        await translator.update_turns([turn1_new])

        # Check that translation was invalidated
        assert translator.turns[1].completion is None
        assert translator.turns[1].translated is None
        # And check text is updated
        assert translator.turns[1].turn.text == "One updated"

    asyncio.run(_test())
