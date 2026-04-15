"""Monolithic prompting strategy."""

from dataclasses import asdict
import json
from typing import Any

from src.llm.base import BaseLLM, LLMResponse
from src.prompts.strategy import PromptStrategy
from src.prompts.base import BasePromptBuilder


class MonolithicStrategy(PromptStrategy):
    """Single comprehensive prompt strategy."""
    
    @property
    def name(self) -> str:
        return "monolithic"
    
    @property
    def description(self) -> str:
        return "Single comprehensive prompt covering all aspects"
    
    def _run_technique(
        self, 
        llm: BaseLLM, 
        prompt: str, 
        image_base64: str | None, 
        image_media_type: str | None, 
        instruction_override: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """Execute monolithic strategy."""
        
        final_prompt = prompt
        if instruction_override:
            final_prompt = prompt + "\n" + instruction_override
            
        return llm.generate_critique(
            prompt=final_prompt,
            image_base64=image_base64,
            image_media_type=image_media_type or "image/jpeg"
        )
