
from dataclasses import dataclass
import pytest
from livesrt.translate.base import LlmTranslator, LlmTranslationEntry
from livesrt.transcribe.base import Turn, Word

@dataclass
class MockLlmTranslator(LlmTranslator):
    async def completion(self, messages, tools, tool_choice="auto"):
        return {}

def create_turn(turn_id, text):
    return Turn(
        id=turn_id,
        text=text,
        final=True,
        words=[Word(type="word", text=w, speaker="S1") for w in text.split()]
    )

def test_build_conversation_with_tools():
    translator = MockLlmTranslator(lang_to="fr")
    
    turn = create_turn(1, "Hello world")
    
    # Simulate a completion with tool calls
    completion = {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "function": {"name": "translate", "arguments": "{}"}}
        ]
    }
    
    entry = LlmTranslationEntry(
        turn=turn,
        completion=completion,
        translated=[], # Doesn't matter for this test
        tool_outputs=["Recorded"]
    )
    
    translator.turns[1] = entry
    
    # We need another turn to trigger building history for turn 1
    # _build_conversation builds history up to the first non-completed turn.
    # So we add a second turn that is not completed.
    turn2 = create_turn(2, "Next turn")
    translator.turns[2] = LlmTranslationEntry(turn=turn2)
    
    to_translate, turn_id, conversation = translator._build_conversation()
    
    # Check conversation history
    # Expected: User(1), Assistant(Completion), Tool(Result), Assistant("ok"), User(2)
    
    assert len(conversation) == 5
    assert conversation[0]["role"] == "user"
    assert conversation[1] == completion
    assert conversation[2]["role"] == "tool"
    assert conversation[3]["role"] == "assistant"
    assert conversation[3]["content"] == "ok"
    assert conversation[4]["role"] == "user"

def test_build_conversation_without_tools():
    translator = MockLlmTranslator(lang_to="fr")
    
    turn = create_turn(1, "Hello world")
    
    # Simulate a completion WITHOUT tool calls (e.g. model chatty refusal)
    completion = {
        "role": "assistant",
        "content": "I cannot translate this."
        # No tool_calls
    }
    
    entry = LlmTranslationEntry(
        turn=turn,
        completion=completion,
        translated=[], 
        tool_outputs=[] 
    )
    
    translator.turns[1] = entry
    
    # Add next turn to trigger history build
    turn2 = create_turn(2, "Next turn")
    translator.turns[2] = LlmTranslationEntry(turn=turn2)
    
    to_translate, turn_id, conversation = translator._build_conversation()
    
    # Expected: User(1), Assistant(Completion), User(2)
    # NOT followed by Tool or Assistant("ok")
    
    assert len(conversation) == 3
    assert conversation[0]["role"] == "user"
    assert conversation[1] == completion
    assert conversation[2]["role"] == "user"
    
    # Ensure "ok" is NOT present (implied by length 3, as User(2) is at index 2)

