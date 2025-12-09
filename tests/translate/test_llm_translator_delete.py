import json
from dataclasses import dataclass, field
from typing import Literal

import pytest

from livesrt.transcribe.base import Turn, Word
from livesrt.translate.base import LlmTranslator


@dataclass
class MockCompletionLlmTranslator(LlmTranslator):
    mock_responses: list[dict] = field(default_factory=list)

    async def completion(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: Literal["auto", "required", "none"] | dict = "auto",
    ) -> dict:
        if self.mock_responses:
            return self.mock_responses.pop(0)
        return {
            "choices": [{"message": {"role": "assistant", "content": "No response"}}]
        }


@pytest.mark.asyncio
async def test_translate_returns_id():
    translator = MockCompletionLlmTranslator(lang_to="fr")

    # Input turn
    turn = Turn(id=1, text="Hello", final=True, words=[Word(type="word", text="Hello")])
    await translator.update_turns([turn])
    # Manually trigger update since we don't run process loop
    translator._update_turns()

    # Mock response for translate
    # Tool call that translates "Hello" -> "Bonjour"
    translator.mock_responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "translate",
                                    "arguments": json.dumps(
                                        {"speaker": "me", "text": "Bonjour"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    ]

    # Run translation
    await translator._translate_next_turn()

    # Check results
    entry = translator.turns[1]
    assert len(entry.translated) == 1
    assert entry.translated[0].text == "Bonjour"
    assert entry.translated[0].id == 0  # First ID is 0

    # Check that tool output recorded the ID
    # The conversation history logic uses entry.tool_outputs
    assert entry.tool_outputs == ["0"]

    # Verify conversation build includes the ID
    _, _, conversation = translator._build_conversation()
    # Check the last tool message
    tool_msg = next(
        m
        for m in conversation
        if m.get("role") == "tool" and m.get("tool_call_id") == "call_1"
    )
    assert tool_msg["content"] == "0"


@pytest.mark.asyncio
async def test_delete_turn():
    translator = MockCompletionLlmTranslator(lang_to="fr")

    # Input turn 1
    turn1 = Turn(
        id=1, text="Hello", final=True, words=[Word(type="word", text="Hello")]
    )
    await translator.update_turns([turn1])
    translator._update_turns()

    # 1. Translate turn 1 -> ID 0
    translator.mock_responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "translate",
                                    "arguments": json.dumps(
                                        {"speaker": "me", "text": "Bonjour"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    ]
    await translator._translate_next_turn()
    assert len(translator.turns[1].translated) == 1
    assert translator.turns[1].translated[0].id == 0

    # 2. Add turn 2, which triggers a delete of ID 0 (simulating context change)
    turn2 = Turn(
        id=2, text=" world", final=True, words=[Word(type="word", text=" world")]
    )
    await translator.update_turns([turn1, turn2])
    translator._update_turns()

    # Mock response for turn 2: Delete ID 0, and emit new translation for
    # full sentence "Hello world"
    translator.mock_responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "delete_turn",
                                    "arguments": json.dumps({"turn_id": 0}),
                                },
                            },
                            {
                                "id": "call_3",
                                "type": "function",
                                "function": {
                                    "name": "translate",
                                    "arguments": json.dumps(
                                        {"speaker": "me", "text": "Bonjour le monde"}
                                    ),
                                },
                            },
                        ],
                    }
                }
            ]
        }
    ]

    await translator._translate_next_turn()

    # Check results

    # Turn 1's translated list should now be empty (since ID 0 was deleted)
    assert len(translator.turns[1].translated) == 0

    # Turn 2 should have the new translation (ID 1)
    visible_translated = [t for t in translator.turns[2].translated if not t.hidden]
    assert len(visible_translated) == 1
    assert visible_translated[0].text == "Bonjour le monde"

    # The ID of the visible turn will now be 2, because the delete_turn consumed ID 1
    assert visible_translated[0].id == 2

    # Check tool outputs for turn 2
    assert translator.turns[2].tool_outputs == ["Deleted", "2"]
