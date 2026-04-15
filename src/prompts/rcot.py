"""Reverse Chain-of-Thought (RCoT) prompting strategy."""

from src.llm.base import BaseLLM, LLMResponse
from src.prompts.strategy import PromptStrategy


class ReverseChainOfThoughtStrategy(PromptStrategy):
    """Reverse Chain-of-Thought strategy."""
    
    @property
    def name(self) -> str:
        return "rcot"
    
    @property
    def description(self) -> str:
        return "Generate -> Abstract -> Compare with constraints -> Refine"
    
    def _run_technique(
        self, 
        llm: BaseLLM, 
        prompt: str, 
        image_base64: str | None, 
        image_media_type: str | None, 
        instruction_override: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Reverse Chain-of-Thought strategy using LangChain chat history."""
        from src.llm.base import ChatMessage
        
        if instruction_override:
            messages = [
                ChatMessage(
                    role="user",
                    content=(
                        f"{prompt}\n\n"
                        "Reverse Chain-of-Thought Step 1: infer what evidence would be required to mark each "
                        "parameter as Detected. Do not output final JSON yet."
                    ),
                    image_base64=image_base64,
                    image_media_type=image_media_type or "image/jpeg",
                )
            ]
            criteria_response = llm.chat(messages)

            messages.append(ChatMessage(role="assistant", content=criteria_response.content))
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Reverse Chain-of-Thought Step 2: compare required evidence vs actual evidence in the "
                        "inputs, then output ONLY the final strict JSON.\n"
                        f"{instruction_override}"
                    ),
                )
            )
            return llm.chat(messages)

        # turn 1: Initial Response
        initial_response = llm.generate_critique(
            prompt=prompt,
            image_base64=image_base64,
            image_media_type=image_media_type or "image/jpeg"
        )
        
        # turn 2: Generate questions backwards from response
        q_msg = "Based entirely on your own response above, what original questions, evaluation criteria, or constraints must have been in my prompt to lead to that exact response? Generate a list of these implied criteria."
        messages = [
            ChatMessage(role="user", content=prompt, image_base64=image_base64, image_media_type=image_media_type or "image/jpeg"),
            ChatMessage(role="assistant", content=initial_response.content),
            ChatMessage(role="user", content=q_msg)
        ]
        q_response = llm.chat(messages)
        
        # turn 3: Compare and Refine
        messages.append(ChatMessage(role="assistant", content=q_response.content))
        messages.append(ChatMessage(role="user", content="Now, compare those implied criteria with my actual requested criteria in the very first message. Find any inconsistencies or omissions in your draft, correct them, and produce the final, refined critique in the requested JSON format with high technical detail."))
        
        return llm.chat(messages)
