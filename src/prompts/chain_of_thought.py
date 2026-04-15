"""Chain-of-Thought prompting strategy."""

from src.llm.base import BaseLLM, LLMResponse, ChatMessage
from src.prompts.strategy import PromptStrategy
from src.prompts.base import BasePromptBuilder
from typing import Any, List


class ChainOfThoughtStrategy(PromptStrategy):
    """Refined Chain-of-Thought strategy: one aspect per step."""
    
    @property
    def name(self) -> str:
        return "cot"
    
    @property
    def description(self) -> str:
        return "Strict step-by-step reasoning (one aspect per step)"
    
    def _run_technique(
        self, 
        llm: BaseLLM, 
        prompt: str, 
        image_base64: str | None, 
        image_media_type: str | None, 
        instruction_override: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """Execute chain-of-thought strategy."""
        
        if instruction_override:
            evidence_prompt = f"""
{prompt}

You are running Chain-of-Thought parameter detection.
Step 1: For each required parameter, collect concise evidence from the architecture description text and architecture diagram image only.
Step 2: If evidence is missing, mark it clearly as missing evidence.
Do not output final JSON yet.
"""
            messages = [
                ChatMessage(
                    role="user",
                    content=evidence_prompt,
                    image_base64=image_base64,
                    image_media_type=image_media_type or "image/jpeg",
                )
            ]
            evidence_response = llm.chat(messages)

            messages.append(ChatMessage(role="assistant", content=evidence_response.content))
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Based on your evidence analysis above, output ONLY the final strict JSON.\n"
                        f"{instruction_override}"
                    ),
                )
            )
            return llm.chat(messages)
        else:
            cot_instructions = """
Analyze the above using a strict step-by-step Chain-of-Thought approach.
Do not jump to conclusions. Evaluate each aspect individually before forming the final critique.

Follow this reasoning path exactly (tailor to specific phase criteria):
Step 1: Analyze the current scope (Content, Image, or Combined). What are the key elements?
Step 2: Evaluate the quality/completeness of each element.
Step 3: Synthesize findings into concrete strengths, weaknesses, and scores.
Step 4: Output the final synthesized critique matching the JSON schema.
"""
            final_prompt = prompt + "\n" + cot_instructions
        
        return llm.generate_critique(
            prompt=final_prompt,
            image_base64=image_base64,
            image_media_type=image_media_type or "image/jpeg"
        )
