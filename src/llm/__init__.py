"""LLM Integration Layer."""

from .base import BaseLLM, LLMResponse
from .ollama_client import OllamaClient

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "OllamaClient"
]
