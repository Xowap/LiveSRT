"""A translator that uses a local LLM."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import httpx
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
)

from ..errors import LiveSrtError
from .base import LlmTranslator, TranslatedTurn

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
    retry=(
        retry_if_exception(is_missing_tool_call)
        | retry_if_exception_type(httpx.TimeoutException)
    ),
    stop=stop_after_attempt(3),
)
async def call_completion(
    model: str,
    api_key: str,
    messages: list[dict],
    tools: list[dict],
    tool_choice: Literal["auto", "required", "none"] | dict = "auto",
) -> dict:
    """
    Calls the remote LLM API to get a completion.
    """
    provider, _, model_id = model.partition("/")

    try:
        base_url = {
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "mistral": "https://api.mistral.ai/v1/chat/completions",
            "google": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "deepinfra": "https://api.deepinfra.com/v1/openai/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "ollama": "http://localhost:11434/api/chat",
        }[provider]
    except KeyError as e:
        msg = f"Provider {provider!r} not found."
        raise LiveSrtError(msg) from e

    client: httpx.AsyncClient
    async with httpx.AsyncClient(
        headers={
            **({"Authorization": f"Bearer {api_key}"} if provider != "ollama" else {}),
            "HTTP-Referer": "https://github.com/Xowap/LiveSRT",
            "X-Title": "LiveSRT",
        },
        timeout=5,
    ) as client:
        # OpenRouter caching: Use structured system prompt with cache_control
        if provider == "openrouter" and "anthropic" in model_id:
            new_messages = []
            for item in messages:
                if item["role"] == "system" and isinstance(item["content"], str):
                    new_messages.append(
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": item["content"],
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    )
                else:
                    new_messages.append(item)
            messages = new_messages

        req_body = dict(
            model=model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        resp = await client.post(
            base_url,
            json=req_body,
        )

        if 400 <= resp.status_code < 500:
            logger.error(
                "API Request Error:\nRequest: %s\nResponse: %s",
                json.dumps(req_body, indent=2),
                resp.text,
            )

        resp.raise_for_status()

        return resp.json()


@dataclass(kw_only=True)
class RemoteLLM(LlmTranslator):
    """A translator that uses a remote LLM."""

    model: str = "openrouter/mistralai/ministral-8b-2512"
    api_key: str

    async def completion(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: Literal["auto", "required", "none"] | dict = "auto",
    ) -> dict:
        """
        Calling a remote LLM through their Completion endpoint
        """

        start = time.perf_counter()
        response = await call_completion(
            model=self.model,
            api_key=self.api_key,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        duration = time.perf_counter() - start
        logger.info("Remote LLM completion (%s) took %.2fs", self.model, duration)
        return response
