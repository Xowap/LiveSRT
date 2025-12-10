from dataclasses import dataclass
import json
import pytest
from livesrt.transcribe.base import Turn, Word
from livesrt.translate.base import LlmTranslator, TranslatedTurn

@dataclass
class MockLlmTranslator(LlmTranslator):
    async def completion(self, messages, tools, tool_choice="auto"):
        return {}

@pytest.mark.asyncio
async def test_pruning_keeps_turn_id_monotonic():
    translator = MockLlmTranslator(lang_to="fr")
    
    # Create 25 turns
    turns = []
    for i in range(1, 26):
        turn = Turn(
            id=i, 
            text=f"Turn {i}", 
            final=True, 
            words=[Word(type="word", text=f"Turn"), Word(type="word", text=f"{i}")]
        )
        turns.append(turn)
        
    await translator.update_turns(turns)
    translator._update_turns()
    
    # Simulate that turns 1-24 are already translated
    # Each turn produced 1 translated item
    current_translated_id = 0
    for i in range(1, 25):
        entry = translator.turns[i]
        # Simulate completion and translation
        entry.completion = {
            "choices": [{"message": {"role": "assistant", "tool_calls": []}}]
        }
        # Create a dummy translated turn
        entry.translated = [
            TranslatedTurn(
                id=current_translated_id,
                original_id=i,
                speaker="Speaker",
                text=f"Translated {i}"
            )
        ]
        # Mock tool outputs to match expected format in _build_conversation
        entry.tool_outputs = ["1"] # 1 line translated
        
        current_translated_id += 1
        
    # Now trigger _build_conversation for turn 25 (which is the only one not translated)
    # Total turns = 25.
    # keep_turns = 10 + 25 % 10 = 15.
    # We keep last 15 turns: 11 to 25.
    # Turns 1 to 10 are pruned.
    # Each of 1..10 had 1 translation. Total 10 items pruned.
    
    to_translate, next_id, conversation = translator._build_conversation()
    
    assert to_translate is not None
    assert to_translate.turn.id == 25
    
    # The expected next_id should be 24 (since we have IDs 0..23 for turns 1..24)
    assert next_id == 24
    
    # Also verify that the conversation contains the expected number of turns (keep_turns)
    # Turn 25 is user message only (no completion yet) -> 1 message
    # Turns 11..24 are user + tool_call + assistant -> 3 messages each
    # Total messages = 14 * 3 + 1 = 43?
    # Let's check conversation length
    # Loop over all_turns[-keep_turns:] (15 turns: 11..25)
    # 11..24 (14 turns): have completion.
    # _build_conversation adds:
    #   User
    #   if completion:
    #       Completion (Assistant with tool calls) -- wait, in _build_conversation:
    #       conversation.append(entry.completion) (This is the assistant message)
    #       Tool output (Role tool)
    #       Assistant "ok"
    # So 4 messages per completed turn?
    
    # Let's look at code:
    # conversation.append(user) (1)
    # if completion:
    #    conversation.append(completion) (2)
    #    for tool_call... conversation.append(tool) (3) -> This loop is empty in our mock!
    #    conversation.append(assistant "ok") (4) -> Becomes (3)
    
    # So 3 messages per completed turn in this mock (since tool_calls is empty).
    # Turn 25: only User message. (1)
    
    # Turns 11..24 (14 turns). 14 * 3 = 42 messages.
    # Turn 25: 1 message.
    # Total 43 messages.
    
    assert len(conversation) == 43
    
    # Verify the first message in conversation corresponds to turn 11
    first_msg = conversation[0]
    content = json.loads(first_msg["content"])
    # Turn 11 content
    assert content[0]["asr_words"] == ["Turn", "11"]
