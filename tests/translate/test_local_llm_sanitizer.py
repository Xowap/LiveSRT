
import pytest
from livesrt.translate.local_llm import LocalLLM

class MockLocalLLM(LocalLLM):
    def __init__(self):
        # Bypass init
        pass

def test_sanitize_messages_basic():
    llm = MockLocalLLM()
    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    sanitized = llm._sanitize_messages(messages)
    assert sanitized == messages

def test_sanitize_messages_merges_assistant():
    llm = MockLocalLLM()
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": "ok"}
    ]
    sanitized = llm._sanitize_messages(messages)
    
    assert len(sanitized) == 2
    assert sanitized[0] == {"role": "user", "content": "Hi"}
    assert sanitized[1]["role"] == "assistant"
    assert "Hello" in sanitized[1]["content"]
    assert "ok" in sanitized[1]["content"]

def test_sanitize_messages_converts_tool_and_merges_user():
    llm = MockLocalLLM()
    messages = [
        {"role": "user", "content": "Do X"},
        {"role": "assistant", "content": "Calling X"},
        {"role": "tool", "content": "Result X"},
        {"role": "tool", "content": "Result Y"}
    ]
    sanitized = llm._sanitize_messages(messages)
    
    # Expected: User, Assistant, User (Tool X + Tool Y merged)
    assert len(sanitized) == 3
    assert sanitized[0]["role"] == "user"
    assert sanitized[1]["role"] == "assistant"
    assert sanitized[2]["role"] == "user"
    assert "Result X" in sanitized[2]["content"]
    assert "Result Y" in sanitized[2]["content"]

def test_sanitize_messages_mixed_merge():
    llm = MockLocalLLM()
    messages = [
        {"role": "user", "content": "Start"},
        {"role": "user", "content": "More"},
        {"role": "assistant", "content": "R1"},
        {"role": "assistant", "content": "R2"},
        {"role": "tool", "content": "T1"},
        {"role": "user", "content": "U3"}
    ]
    sanitized = llm._sanitize_messages(messages)
    
    # User(Start+More) -> Assistant(R1+R2) -> User(Tool T1 + U3)
    assert len(sanitized) == 3
    assert sanitized[0]["role"] == "user"
    assert "Start" in sanitized[0]["content"]
    assert "More" in sanitized[0]["content"]
    
    assert sanitized[1]["role"] == "assistant"
    assert "R1" in sanitized[1]["content"]
    assert "R2" in sanitized[1]["content"]
    
    assert sanitized[2]["role"] == "user"
    assert "Tool output: T1" in sanitized[2]["content"]
    assert "U3" in sanitized[2]["content"]
