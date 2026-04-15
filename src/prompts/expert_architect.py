"""Expert Software Architect prompting strategy."""

from dataclasses import asdict
import json
from typing import Any

from src.llm.base import BaseLLM, LLMResponse
from src.prompts.strategy import PromptStrategy
from src.prompts.base import BasePromptBuilder


class ExpertArchitectStrategy(PromptStrategy):
    """Specific persona: Expert Software Architect (20 years exp)."""
    
    @property
    def name(self) -> str:
        return "expert"
    
    @property
    def description(self) -> str:
        return "Expert Software Architect (20y exp) with specific critique focus"
    
    def _run_technique(
        self, 
        llm: BaseLLM, 
        prompt: str, 
        image_base64: str | None, 
        image_media_type: str | None, 
        instruction_override: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """Execute expert architect strategy."""
        
        if instruction_override:
             persona = """
    You are an expert software architect with 20 years of experience. Your task is to analyze ONLY the System Architecture Description text and the Architecture Diagram image with extreme scrutiny.

"""
             final_prompt = persona + "\n" + prompt + "\n" + instruction_override
        else:
            persona = """
You are an expert software architect with 20 years of experience. Your task is to provide a
brutally honest, deep-dive technical critique. Evaluate the architecture diagram, 
project title, abstract, and system description with extreme scrutiny.

Specifically identify:
1. INCONSISTENCIES between diagrams and text (e.g. component mentioned but not drawn).
2. NAMING CONVENTIONS: Are component names vague or technically inaccurate?
3. BOTTLENECKS: Identify potential performance issues or single points of failure.
4. REDUNDANCY: Are there overlapping functionalities or unnecessary layers?
5. GAPS: What is missing for this to be a production-ready system?

Your feedback should be professional, technical, and detailed. Provide multi-sentence 
explanations for every observation.

IMPORTANT: Structure your findings according to the JSON schema provided below. DO NOT invent new keys. Place your detailed analysis in the 'detailed_critique' section and your summary in 'overall_assessment'.
"""
            final_prompt = persona + "\n" + prompt
        
        return llm.generate_critique(
            prompt=final_prompt,
            image_base64=image_base64,
            image_media_type=image_media_type or "image/jpeg"
        )
