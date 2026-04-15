"""Base prompt builder with common utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProjectInputs:
    """Container for project inputs.
    
    Fields:
        project_title: Title of the BE project
        abstract: Project abstract/summary
        architecture_description: System architecture description text
        has_architecture_image: Whether an architecture diagram image is included
        architecture_image_base64: Base64-encoded architecture diagram image
        architecture_media_type: MIME type of the architecture image
    """
    project_title: str | None = None
    abstract: str | None = None
    architecture_description: str | None = None
    has_architecture_image: bool = False
    architecture_image_base64: str | None = None
    architecture_media_type: str = "image/jpeg"
    
    def to_text_only_block(self) -> str:
        """Convert only the text inputs to a formatted text block, ignoring images."""
        sections = []
        if self.project_title:
            sections.append(f"## Project Title\n{self.project_title}")
        if self.abstract:
            sections.append(f"## Abstract\n{self.abstract}")
        if self.architecture_description:
            sections.append(f"## System Architecture Description\n{self.architecture_description}")
        return "\n\n".join(sections)
        
    def to_text_block(self) -> str:
        """Convert inputs to a formatted text block, including reference to architecture image."""
        sections = []
        text_block = self.to_text_only_block()
        if text_block:
            sections.append(text_block)
        
        if self.has_architecture_image:
            sections.append("## Architecture Diagram\n[See attached image]")
        
        return "\n\n".join(sections)


class BasePromptBuilder(ABC):
    """Abstract base class for prompt builders."""
    
    # Strict JSON output instruction for LLM
    JSON_OUTPUT_INSTRUCTION = """
### CRITICAL JSON OUTPUT INSTRUCTIONS ###
1. Your response MUST be a single, valid JSON object. DO NOT use markdown, code blocks, or any filler text. Only output the JSON.
2. Start your response with '{' and end it with '}'.
3. DO NOT include any explanations, headers, or conversational text outside the JSON.
4. Every field in the JSON MUST be detailed and comprehensive. Provide multi-sentence explanations, specifically identifying technical details, inconsistencies, and improvements.
5. If you cannot produce valid JSON, retry and correct your output. Validate your response before returning.
6. The JSON MUST follow this EXACT schema:
{
    "overall_assessment": "4-6 sentence summary of project quality, potential, and major areas for improvement.",
    "scores": {
        "title_clarity": <integer 1-10>,
        "abstract_quality": <integer 1-10>,
        "architecture_design": <integer 1-10>,
        "architecture_description_completeness": <integer 1-10>,
        "feasibility": <integer 1-10>,
        "innovation": <integer 1-10>
    },
    "strengths": [
        "Major strength #1 with technical reasoning (2-3 sentences)",
        "Major strength #2..."
    ],
    "weaknesses": [
        "Major weakness/risk #1 with technical reasoning (2-3 sentences)",
        "Major weakness #2..."
    ],
    "suggestions": [
        "Suggestion #1: Explain EXACTLY how to fix the issue identified...",
        "Suggestion #2..."
    ],
    "detailed_critique": {
        "project_title": "3-5 sentence critique of the project title's clarity, scope, and technical accuracy.",
        "abstract": "4-6 sentence analysis of the abstract, checking if it covers problem, solution, methodology, and results.",
        "architecture": "6-10 sentence technical critique of the system architecture. Evaluate component choices, data flow, scaling bottlenecks, and potential points of failure.",
        "architecture_description": "4-6 sentence review of the architectural description text, checking for consistency with the diagram and clarity of technical concepts."
    }
}
7. If you reference previous critiques or diagrams, incorporate those findings into the appropriate fields, but DO NOT add new keys or change the schema.
8. Validate your output for proper JSON formatting before returning.
"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this prompting technique."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get a brief description of this prompting technique."""
        pass
    
    @abstractmethod
    def build_prompt(self, inputs: ProjectInputs) -> str:
        """Build the complete prompt for the LLM.
        
        Args:
            inputs: The project inputs to critique
            
        Returns:
            Complete prompt string
        """
        pass
