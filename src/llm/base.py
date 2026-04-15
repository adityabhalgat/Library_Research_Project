"""Abstract base class for LLM integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Literal


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    usage: dict[str, int] | None = None
    raw_response: Any = None
    
    def to_dict(self) -> dict:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage
        }


@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    image_base64: str | None = None
    image_media_type: str = "image/jpeg"


class BaseLLM(ABC):
    """Abstract base class defining the LLM interface."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434", max_tokens: int = 4096, temperature: float = 0.7):
        """Initialize the LLM client.
        
        Args:
            model: Model name to use
            base_url: Base URL for the LLM service (Ollama endpoint)
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
        """
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this LLM provider."""
        pass
    
    @abstractmethod
    def generate_critique(
        self,
        prompt: str,
        image_base64: str | None = None,
        image_media_type: str = "image/jpeg"
    ) -> LLMResponse:
        """Generate a critique based on the prompt and optional image.
        
        Args:
            prompt: The complete prompt to send to the LLM
            image_base64: Base64 encoded image (optional)
            image_media_type: MIME type of the image
            
        Returns:
            LLMResponse containing the critique
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[ChatMessage]) -> LLMResponse:
        """Send a chat history to the LLM and get a response.
        
        Args:
            messages: List of ChatMessage objects representing the conversation history
            
        Returns:
            LLMResponse containing the assistant's reply
        """
        pass
    
    def validate_connection(self) -> bool:
        """Validate that the LLM service is reachable."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
