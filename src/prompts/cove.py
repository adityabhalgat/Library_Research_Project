"""Chain-of-Verification (CoVe) prompting strategy."""

from src.llm.base import BaseLLM, LLMResponse
from src.prompts.strategy import PromptStrategy


class ChainOfVerificationStrategy(PromptStrategy):
    """Chain-of-Verification strategy."""
    
    @property
    def name(self) -> str:
        return "cove"
    
    @property
    def description(self) -> str:
        return "Generate -> Ask verification questions -> Verify -> Refine"
    
    def _run_technique(
        self, 
        llm: BaseLLM, 
        prompt: str, 
        image_base64: str | None, 
        image_media_type: str | None, 
        instruction_override: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """Execute Chain-of-Verification strategy using LangChain chat history."""
        from src.llm.base import ChatMessage
        
        # If parameter analysis (instruction_override present), simplify behavior
        if instruction_override:
            messages = [
                ChatMessage(
                    role="user",
                    content=(
                        f"{prompt}\n\n"
                        "Step 1 (Draft): Produce an initial parameter-detection JSON draft."
                    ),
                    image_base64=image_base64,
                    image_media_type=image_media_type or "image/jpeg",
                )
            ]
            draft_resp = llm.chat(messages)

            messages.append(ChatMessage(role="assistant", content=draft_resp.content))
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Step 2 (Verification Questions): Ask 5 focused verification questions that challenge "
                        "possible false positives or false negatives in the draft."
                    ),
                )
            )
            verify_q_resp = llm.chat(messages)

            messages.append(ChatMessage(role="assistant", content=verify_q_resp.content))
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Step 3 (Verification Answers): Answer your own verification questions using only "
                        "the original architecture text/image evidence."
                    ),
                )
            )
            verify_a_resp = llm.chat(messages)

            messages.append(ChatMessage(role="assistant", content=verify_a_resp.content))
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Step 4 (Finalize): Output ONLY the corrected final strict JSON.\n"
                        f"{instruction_override}"
                    ),
                )
            )
            return llm.chat(messages)

        # turn 1: Generate initial draft
        initial_response = llm.generate_critique(
            prompt=prompt,
            image_base64=image_base64,
            image_media_type=image_media_type or "image/jpeg"
        )
        
        # turn 2: Generate verification questions
        q_msg = "Based on your own draft critique above, generate 4-5 targeted verification questions to check factual correctness, logic, and comprehensive coverage. Be critical."
        messages = [
            ChatMessage(role="user", content=prompt, image_base64=image_base64, image_media_type=image_media_type or "image/jpeg"),
            ChatMessage(role="assistant", content=initial_response.content),
            ChatMessage(role="user", content=q_msg)
        ]
        q_response = llm.chat(messages)
        
        # turn 3: Answer verification questions
        # We append turn 2 to the history
        messages.append(ChatMessage(role="assistant", content=q_response.content))
        messages.append(ChatMessage(role="user", content="Now, answer these verification questions objectively based on the original project context provided in the first message."))
        
        a_response = llm.chat(messages)
        
        # turn 4: Finalize
        messages.append(ChatMessage(role="assistant", content=a_response.content))
        messages.append(ChatMessage(role="user", content="Finally, provide a refined, definitive critique taking into account the verification Q&A above. Ensure you output in the originally requested JSON format with high technical detail."))
        
        return llm.chat(messages)
