
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from livesrt.translate.remote_llm import RemoteLLM
from livesrt.translate.base import TranslationReceiver

@pytest.fixture
def mock_receiver():
    return MagicMock(spec=TranslationReceiver)

@pytest.mark.asyncio
async def test_remote_llm_process_manages_client(mock_receiver):
    """Test that RemoteLLM.process creates and cleans up the client."""
    translator = RemoteLLM(api_key="fake-key", lang_to="fr")
    
    # Mock super().process to return immediately (or throw CancelledError to stop loop)
    # But since we want to check if client is set DURING process, we need it to run at least a bit.
    # However, super().process runs forever until cancelled.
    
    # Let's mock LlmTranslator.process to just assert self._client is set
    with patch("livesrt.translate.remote_llm.LlmTranslator.process", new_callable=AsyncMock) as mock_super_process:
        with patch("livesrt.translate.remote_llm.httpx.AsyncClient") as mock_client_cls:
            mock_client_instance = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client_instance
            
            await translator.process(mock_receiver)
            
            # Verify client was created
            mock_client_cls.assert_called_once()
            
            # Verify super().process was called
            mock_super_process.assert_called_once_with(mock_receiver)
            
            # Verify client was cleaned up (implicit via context manager, but we can check if it was closed if we didn't use context manager)
            # Since we use context manager, __aexit__ is called.
            mock_client_cls.return_value.__aexit__.assert_called_once()
            
            # Verify _client is None after process
            assert translator._client is None

@pytest.mark.asyncio
async def test_remote_llm_completion_uses_client():
    """Test that RemoteLLM.completion passes the managed client to call_completion."""
    translator = RemoteLLM(api_key="fake-key", lang_to="fr")
    mock_client = AsyncMock()
    translator._client = mock_client
    
    with patch("livesrt.translate.remote_llm.call_completion", new_callable=AsyncMock) as mock_call_completion:
        mock_call_completion.return_value = {"choices": []}
        
        await translator.completion(messages=[], tools=[])
        
        mock_call_completion.assert_called_once()
        call_args = mock_call_completion.call_args
        assert call_args.kwargs["client"] is mock_client

@pytest.mark.asyncio
async def test_call_completion_uses_passed_client():
    """Test that call_completion uses the passed client."""
    from livesrt.translate.remote_llm import call_completion
    
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_client.post.return_value = mock_response
    
    await call_completion(
        model="openrouter/test/model",
        api_key="key",
        messages=[],
        tools=[],
        client=mock_client
    )
    
    mock_client.post.assert_called_once()
    mock_response.raise_for_status.assert_called_once()

@pytest.mark.asyncio
async def test_call_completion_creates_client_if_none():
    """Test that call_completion creates a new client if none is passed."""
    from livesrt.translate.remote_llm import call_completion
    
    with patch("livesrt.translate.remote_llm.httpx.AsyncClient") as mock_client_cls:
        mock_client_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_client_instance.post.return_value = mock_response
        
        await call_completion(
            model="openrouter/test/model",
            api_key="key",
            messages=[],
            tools=[],
            client=None
        )
        
        mock_client_cls.assert_called_once()
        mock_client_instance.post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
