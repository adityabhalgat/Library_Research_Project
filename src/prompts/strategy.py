"""Base class for prompting strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json

from src.llm.base import BaseLLM, LLMResponse


from src.prompts.base import BasePromptBuilder

class PromptStrategy(ABC):
    """Abstract base class for prompting strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this strategy."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of this strategy."""
        pass
    
    def execute(
        self,
        llm: BaseLLM,
        inputs: Any,  # ProjectInputs
        **kwargs
    ) -> LLMResponse:
        """Execute the strategy in 3 phases: Content, Image, Combined."""

        run_logger = kwargs.pop("run_logger", None)
        log_context = kwargs.pop("log_context", {}) or {}
        project_id = log_context.get("project_id")
        project_title = log_context.get("project_title") or getattr(inputs, "project_title", None)
        
        system_instruction = BasePromptBuilder.JSON_OUTPUT_INSTRUCTION
        
        # Phase 1: Architecture-description-only critique
        architecture_description = (inputs.architecture_description or "").strip()
        if not architecture_description:
            architecture_description = (
                "No system architecture description was provided. Be explicit in the JSON that the"
                " text is missing and avoid inventing details. If an architecture diagram is"
                " available, note that only the image can be reviewed."
            )

        content_prompt = f"""
    You are an expert reviewer. Review ONLY the System Architecture Description text below and provide a technical critique STRICTLY in the required JSON schema.

    System Architecture Description:
    {architecture_description}

    Focus exclusively on the architecture description text. If the text indicates it is missing, clearly state that in the JSON and do not fabricate system details.
    
    {system_instruction}
    """
        content_response = self._run_technique(llm, content_prompt, None, None, **kwargs)
        if run_logger:
            run_logger.log_llm_exchange(
                project_id=project_id,
                project_title=project_title,
                strategy_name=self.name,
                phase_name="text_only_critique",
                prompt_text=content_prompt,
                response_text=content_response.content,
                has_image=False,
            )

        # Helper to extract and normalize JSON from response
        def extract_json(response: LLMResponse) -> Dict[str, Any]:
            import re
            content = response.content
            try:
                # Attempt 1: Find JSON block containing specific schema keys
                # This helps avoid capturing "Chain of Thought" JSONs if they exist earlier in output
                match = re.search(r'(\{.*"overall_assessment".*\})', content, re.DOTALL)
                
                # Attempt 2: Standard JSON block search
                if not match:
                    match = re.search(r'({.*})', content, re.DOTALL)
                
                if not match:
                    return {"raw_text": content}

                json_str = match.group(1)
                # Cleanup common LLM issues
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                # Fix escaped underscores often seen in Ollama/Llava output
                json_str = json_str.replace('\\_', '_')

                # Remove invalid control characters (except whitespace)
                import string
                json_str = ''.join(ch for ch in json_str if ch in string.printable or ch in '\n\r\t')

                # Attempt to parse
                try:
                    data = json.loads(json_str)
                except Exception:
                    # Try a more aggressive cleanup if simple parse fails
                    # Remove trailing commas before closing braces
                    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                    data = json.loads(json_str)

                # Normalize schema
                normalized = {}

                # 1. Overall Assessment (MUST BE STRING)
                oa = data.get("overall_assessment", data.get("overall", ""))
                if isinstance(oa, dict):
                    # Flatten dict to string if LLM got creative
                    oa = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in oa.items()])
                normalized["overall_assessment"] = str(oa)

                # 2. Scores
                scores = data.get("scores", {})
                if not isinstance(scores, dict): scores = {}
                normalized["scores"] = scores

                # 3. Lists
                for list_key in ["strengths", "weaknesses", "suggestions"]:
                    val = data.get(list_key, [])
                    if isinstance(val, str): val = [val]
                    elif not isinstance(val, list): val = []
                    normalized[list_key] = val

                # 4. Detailed Critique
                detailed = data.get("detailed_critique", {})
                if not isinstance(detailed, dict): detailed = {}
                # Ensure values are strings
                for k, v in detailed.items():
                    if isinstance(v, (dict, list)):
                        detailed[k] = str(v)
                normalized["detailed_critique"] = detailed

                # 5. Capture any other keys just in case
                for k, v in data.items():
                    if k not in normalized and k not in ["metadata", "usage"]:
                        normalized[k] = v

                return normalized

            except Exception as e:
                # Return wrapped raw text as ultimate fallback
                return {"raw_text": content, "parse_error": str(e)}

        # If no architecture image, return the parsed text critique as a valid JSON response
        # NOTE: Structured to match the expected multi-phase format even if only one phase ran
        if not inputs.has_architecture_image:
            parsed_content = extract_json(content_response)
            combined_result = {
                "text_only_critique": parsed_content,
                "image_only_critique": {"raw_text": "No architecture image provided."},
                "final_combined_critique": parsed_content
            }
            if run_logger:
                run_logger.log_note("No architecture image provided; image and combined phases were skipped.")
            return LLMResponse(
                content=json.dumps(combined_result, indent=2),
                model=content_response.model,
                usage=content_response.usage,
                raw_response=content_response.raw_response
            )

        # Phase 2: Image Only Critique
        image_prompt = f"""
    You are an expert reviewer. Review the attached BE Project architecture diagram and provide a technical critique STRICTLY in the required JSON schema.

    Inputs:
    [See attached image]

    Evaluate the architecture based on:
    1. LOGICAL FLOW: Is the data movement and process sequence sound?
    2. COMPLETENESS OF COMPONENTS: Are essential parts like databases, APIs, workers, or security layers missing?
    3. FEASIBILITY: Can this be built with current technologies given the description?
    4. INNOVATION FACTOR: How does this approach differ from standard solutions?

    BE AS DETAILED AS POSSIBLE. Point out specific missing connections, edge cases, or potential bottlenecks in the diagram.
    
    {system_instruction}
    """
        image_media_type = inputs.architecture_media_type
        image_response = self._run_technique(
            llm, 
            image_prompt, 
            inputs.architecture_image_base64, 
            image_media_type, 
            **kwargs
        )
        if run_logger:
            run_logger.log_llm_exchange(
                project_id=project_id,
                project_title=project_title,
                strategy_name=self.name,
                phase_name="image_only_critique",
                prompt_text=image_prompt,
                response_text=image_response.content,
                has_image=True,
            )

        # Phase 3: Combined Critique
        combined_prompt = f"""
    You are an expert reviewer. Review the complete BE Project details (text and architecture diagram) and synthesize a final, definitive, highly detailed technical critique STRICTLY in the required JSON schema.

    Previous Content Critique (for reference):
    {content_response.content}

    Previous Architecture Critique (for reference):
    {image_response.content}

    Current Inputs:
    System Architecture Description:
    {architecture_description}
    Architecture Diagram: See attached image

    Your task is to synthesize the previous findings into a single, comprehensive final evaluation. 
    Evaluate based on:
    1. OVERALL CONSISTENCY: Does the text description match the diagram? Mark any contradictions.
    2. TECHNICAL DEPTH: Is the abstract and description deep enough for an engineering project?
    3. FEASIBILITY AND SCALING: Can this system handle real-world load?
    4. INNOVATION AND IMPACT: What is the specific contribution of this project?

    CRITICAL: You MUST use the EXACT JSON structure provided below. DO NOT create new keys for the categories above; instead, incorporate those analyses into the "overall_assessment" and "detailed_critique" sections of the schema.
    
    {system_instruction}
    """
        final_response = self._run_technique(
            llm, 
            combined_prompt, 
            inputs.architecture_image_base64, 
            image_media_type, 
            **kwargs
        )
        if run_logger:
            run_logger.log_llm_exchange(
                project_id=project_id,
                project_title=project_title,
                strategy_name=self.name,
                phase_name="final_combined_critique",
                prompt_text=combined_prompt,
                response_text=final_response.content,
                has_image=True,
            )
        
        # Combine all 3 responses into a single JSON object
        combined_result = {
            "text_only_critique": extract_json(content_response),
            "image_only_critique": extract_json(image_response),
            "final_combined_critique": extract_json(final_response)
        }
        
        # Create a new LLMResponse to hold the combined data
        # We convert it back to a JSON string so the rest of the pipeline handles it normally
        combined_content = json.dumps(combined_result, indent=2)
        
        # Aggregate usage if available
        total_usage = None
        for resp in [content_response, image_response, final_response]:
            if resp.usage:
                if total_usage is None:
                    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                total_usage["prompt_tokens"] += resp.usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += resp.usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += resp.usage.get("total_tokens", 0)

        return LLMResponse(
            content=combined_content,
            model=final_response.model,
            usage=total_usage,
            raw_response=final_response.raw_response
        )

    @abstractmethod
    def _run_technique(
        self, 
        llm: BaseLLM, 
        prompt: str, 
        image_base64: str | None, 
        image_media_type: str | None, 
        **kwargs
    ) -> LLMResponse:
        """Run the specific prompting technique on the given prompt.
        
        Args:
            llm: The LLM client to use
            prompt: Base prompt for the current phase
            image_base64: Optional base64 encoded image
            image_media_type: MIME type of the image if present
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse containing the output for this phase
        """
        pass
