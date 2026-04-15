"""Ollama LLM integration via LangChain."""

import base64
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from .base import BaseLLM, LLMResponse, ChatMessage


class OllamaClient(BaseLLM):
    """Ollama client with vision capabilities via LangChain."""
    
    def __init__(
        self,
        model: str = "llava",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 4096,
        temperature: float = 0.7
    ):
        """Initialize Ollama client via LangChain.
        
        Args:
            model: Ollama model name (default: llava for vision support)
            base_url: Ollama server URL
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
        """
        super().__init__(model, base_url, max_tokens, temperature)
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )
    
    @property
    def name(self) -> str:
        """Get the name of this LLM provider."""
        return "ollama"
    
    def _build_content_blocks(
        self,
        text: str,
        image_base64: str | None = None,
        image_media_type: str = "image/jpeg"
    ) -> list:
        """Build LangChain content blocks with optional image.
        
        Args:
            text: The text content
            image_base64: Optional base64 encoded image
            image_media_type: MIME type of the image
            
        Returns:
            List of content blocks for LangChain message
        """
        content = []
        
        # Add image if provided
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": f"data:{image_media_type};base64,{image_base64}"
            })
        
        # Add text
        content.append({
            "type": "text",
            "text": text
        })
        
        return content
    
    def generate_critique(
        self,
        prompt: str,
        image_base64: str | None = None,
        image_media_type: str = "image/jpeg"
    ) -> LLMResponse:
        """Generate a critique using Ollama.
        
        Args:
            prompt: The complete prompt to send
            image_base64: Base64 encoded image (optional, requires vision model like llava)
            image_media_type: MIME type of the image
            
        Returns:
            LLMResponse containing the critique
        """
        # Build message with optional image
        content = self._build_content_blocks(prompt, image_base64, image_media_type)
        message = HumanMessage(content=content)
        
        # Invoke the model
        response = self.llm.invoke([message])
        
        return LLMResponse(
            content=response.content,
            model=self.model,
            usage=self._extract_usage(response),
            raw_response=response
        )
    
    def chat(self, messages: List[ChatMessage]) -> LLMResponse:
        """Send a chat history to Ollama using LangChain message history.
        
        Uses LangChain's InMemoryChatMessageHistory to maintain conversation
        context across multiple turns — essential for multi-step strategies
        like CoVe, RCoT, and Two-Model.
        
        Args:
            messages: List of ChatMessage objects representing the conversation
            
        Returns:
            LLMResponse containing the assistant's reply
        """
        # Build LangChain message history
        history = InMemoryChatMessageHistory()
        langchain_messages = []
        
        for msg in messages:
            if msg.role == "system":
                lc_msg = SystemMessage(content=msg.content)
            elif msg.role == "assistant":
                lc_msg = AIMessage(content=msg.content)
            else:  # user
                content = self._build_content_blocks(
                    msg.content,
                    msg.image_base64,
                    msg.image_media_type
                )
                lc_msg = HumanMessage(content=content)
            
            history.add_message(lc_msg)
            langchain_messages.append(lc_msg)
        
        # Invoke the model with full conversation history
        response = self.llm.invoke(langchain_messages)
        
        return LLMResponse(
            content=response.content,
            model=self.model,
            usage=self._extract_usage(response),
            raw_response=response
        )
    
    def _extract_usage(self, response) -> dict | None:
        """Extract token usage from LangChain response if available."""
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            meta = response.usage_metadata
            usage = {
                "prompt_tokens": meta.get("input_tokens", 0),
                "completion_tokens": meta.get("output_tokens", 0),
                "total_tokens": meta.get("total_tokens", 0)
            }
        elif hasattr(response, 'response_metadata') and response.response_metadata:
            meta = response.response_metadata
            if 'prompt_eval_count' in meta or 'eval_count' in meta:
                usage = {
                    "prompt_tokens": meta.get("prompt_eval_count", 0),
                    "completion_tokens": meta.get("eval_count", 0),
                    "total_tokens": meta.get("prompt_eval_count", 0) + meta.get("eval_count", 0)
                }
        return usage
