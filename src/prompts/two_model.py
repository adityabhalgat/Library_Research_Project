"""Two-Model prompting strategy."""

from src.llm.base import BaseLLM, LLMResponse
from src.prompts.strategy import PromptStrategy


class TwoModelStrategy(PromptStrategy):
    """Two-Model strategy."""
    
    judge_llm = None
    
    @property
    def name(self) -> str:
        return "two_model"
    
    @property
    def description(self) -> str:
        return "Primary LLM Generates -> Judge LLM Critiques -> Primary LLM Refines"
    
    def _run_technique(
        self, 
        llm: BaseLLM, 
        prompt: str, 
        image_base64: str | None, 
        image_media_type: str | None, 
        instruction_override: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Two-Model strategy using LangChain chat history."""
        from src.llm.base import ChatMessage
        judge_llm = kwargs.get('judge_llm')
        
        if instruction_override:
            if not judge_llm:
                judge_llm = llm  # Fallback to same LLM if judge not provided

            generator_messages = [
                ChatMessage(
                    role="user",
                    content=(
                        f"{prompt}\n\n"
                        "Generator Step: Produce an initial parameter-detection JSON draft."
                    ),
                    image_base64=image_base64,
                    image_media_type=image_media_type or "image/jpeg",
                )
            ]
            draft_response = llm.chat(generator_messages)

            judge_messages = [
                ChatMessage(
                    role="user",
                    content=prompt,
                    image_base64=image_base64,
                    image_media_type=image_media_type or "image/jpeg",
                ),
                ChatMessage(role="assistant", content=draft_response.content),
                ChatMessage(
                    role="user",
                    content=(
                        "Judge Step: Evaluate the draft for false positives/negatives and key mismatches. "
                        "Return concrete correction notes only."
                    ),
                ),
            ]
            judge_feedback = judge_llm.chat(judge_messages)

            generator_messages.append(ChatMessage(role="assistant", content=draft_response.content))
            generator_messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Judge feedback to apply:\n"
                        f"{judge_feedback.content}\n\n"
                        "Refinement Step: Output ONLY the corrected final strict JSON.\n"
                        f"{instruction_override}"
                    ),
                )
            )
            return llm.chat(generator_messages)

        # Default critique logic
        
        # turn 1: Primary generates
        primary_response = llm.generate_critique(
            prompt=prompt,
            image_base64=image_base64,
            image_media_type=image_media_type or "image/jpeg"
        )
        
        # turn 2: Judge critiques (Judge gets the history too)
        judge_msg = "You are an expert evaluator assessing the quality of the AI response above. Provide detailed feedback explaining why the response is strong or weak compared to the original task. Identify missing elements or logical gaps, and suggest concrete improvements."
        
        messages = [
            ChatMessage(role="user", content=prompt, image_base64=image_base64, image_media_type=image_media_type or "image/jpeg"),
            ChatMessage(role="assistant", content=primary_response.content),
            ChatMessage(role="user", content=judge_msg)
        ]
        
        judge_response = judge_llm.chat(messages)
        
        # turn 3: Primary refines
        messages.append(ChatMessage(role="assistant", content=judge_response.content))
        messages.append(ChatMessage(role="user", content="Now, revise your initial critique based on that expert feedback. Ensure you output in the originally requested JSON format with high technical detail."))
        
        return llm.chat(messages)
