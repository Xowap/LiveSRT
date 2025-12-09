"""A translator that uses a local LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rich.console import Console

from livesrt.utils import ignore_stderr

from ..async_tools import sync_to_async
from .base import LlmTranslator

if TYPE_CHECKING:
    from llama_cpp import Llama


console = Console()


MODELS = {
    "qwen-3:14b:q4-k-m": ("unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf"),
    "ministral:8b:q4-k-m": (
        "bartowski/Ministral-8B-Instruct-2410-GGUF",
        "Ministral-8B-Instruct-2410-Q4_K_M.gguf",
    ),
    "ministral:3b:q4-k-m": (
        "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    ),
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


@dataclass(kw_only=True)
class LocalLLM(LlmTranslator):
    """A translator that uses a local LLM."""

    llm: Llama = field(init=False)

    async def init(self):
        """
        Initialize a local LLM.
        """
        model_path = await download_model("ministral:8b:q4-k-m")
        self.llm = await init_model(model_path)

    @sync_to_async
    def completion(  # type: ignore
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: Literal["auto", "required", "none"] | dict = "auto",
    ) -> dict:
        """
        Performs a completion call to the local LLM.
        """
        return self.llm.create_chat_completion(  # type: ignore
            messages=messages,  # type: ignore
            tools=tools,  # type: ignore
            tool_choice=tool_choice,  # type: ignore
        )
