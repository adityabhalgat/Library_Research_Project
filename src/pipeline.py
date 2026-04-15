"""Main pipeline for generating critiques."""

import os
import time
import uuid
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

from src.input_handlers import ImageHandler, NetworkXHandler, TextHandler
from src.llm import OllamaClient
from src.llm.base import BaseLLM, LLMResponse
from src.prompts import get_prompt_strategy, ProjectInputs
from src.output import JSONStorage, CritiqueResult, RunTextLogger
from config import settings


class CritiquePipeline:
    """Orchestrator for the critique generation process."""
    
    def __init__(self, output_dir: str | None = None):
        """Initialize pipeline.
        
        Args:
            output_dir: Directory to save output files
        """
        self.storage = JSONStorage(output_dir or settings.output_dir)
        
    def _get_llm_client(self, model: str | None = None) -> BaseLLM:
        """Get Ollama LLM client.
        
        Args:
            model: Optional model override (defaults to settings.ollama_model)
        """
        return OllamaClient(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature
        )

    def _prepare_inputs(
        self,
        architecture_image: str | None = None,
        architecture_networkx: str | None = None,
        project_title: str | None = None,
        abstract: str | None = None,
        architecture_description: str | None = None
    ) -> Tuple[ProjectInputs, str | None, str]:
        """Prepare project inputs."""
        
        # Handle image/graph
        image_base64 = None
        media_type = "image/jpeg"
        
        if architecture_image:
            handler = ImageHandler(architecture_image)
            image_base64 = handler.to_base64()
            media_type = handler.media_type
        elif architecture_networkx:
            handler = NetworkXHandler(architecture_networkx)
            image_base64 = handler.to_base64()
            media_type = handler.media_type
            
        # Handle text inputs
        title_text = TextHandler.load(project_title) if project_title else None
        abstract_text = TextHandler.load(abstract) if abstract else None
        arch_desc_text = TextHandler.load(architecture_description) if architecture_description else None
        
        inputs = ProjectInputs(
            project_title=title_text,
            abstract=abstract_text,
            architecture_description=arch_desc_text,
            has_architecture_image=bool(image_base64),
            architecture_image_base64=image_base64,
            architecture_media_type=media_type
        )
        
        return inputs, image_base64, media_type

    def generate_critique(
        self,
        prompt_technique: str,
        project_id: str | None = None,
        architecture_image: str | None = None,
        architecture_networkx: str | None = None,
        project_title: str | None = None,
        abstract: str | None = None,
        architecture_description: str | None = None,
        save: bool = True,
        model: str | None = None,
        judge_model: str | None = None
    ) -> CritiqueResult:
        """Generate a single critique.
        
        Args:
            prompt_technique: The prompting strategy to use
            architecture_image: Path to architecture diagram image
            architecture_networkx: Path to NetworkX graph file
            project_title: Project title (text or file path)
            abstract: Project abstract (text or file path)
            architecture_description: System architecture description (text or file path)
            save: Whether to save the result to disk
            model: Optional Ollama model override
            judge_model: Optional judge model for two-model strategy
        """
        
        # Prepare inputs
        inputs, image_base64, media_type = self._prepare_inputs(
            architecture_image, architecture_networkx,
            project_title, abstract, architecture_description
        )

        run_logger = RunTextLogger(
            run_type=f"critique_{prompt_technique}",
            logs_dir="ogs",
            header_context={
                "Model": model or settings.ollama_model,
                "Technique": prompt_technique,
                "ProjectId": project_id or "N/A",
            },
        )
        run_logger.log_project_header(
            project_id=project_id,
            project_title=inputs.project_title,
        )
        
        # Get LLM client
        client = self._get_llm_client(model)
        
        # Get Strategy
        strategy = get_prompt_strategy(prompt_technique)
        
        # Special handling for TwoModelJudgeStrategy to inject judge
        if prompt_technique == "two_model" and hasattr(strategy, 'judge_llm'):
            if judge_model:
                strategy.judge_llm = self._get_llm_client(judge_model)
            elif settings.ollama_judge_model:
                strategy.judge_llm = self._get_llm_client(settings.ollama_judge_model)
            else:
                strategy.judge_llm = client  # Use same model as judge

        # Execute Strategy
        start_time = time.time()
        response = strategy.execute(
            client,
            inputs,
            run_logger=run_logger,
            log_context={"project_id": project_id, "project_title": inputs.project_title},
        )
        execution_time = time.time() - start_time

        run_logger.close(
            summary={
                "Execution Sec": round(execution_time, 2),
                "Response Model": response.model,
            }
        )
        
        # Save output
        output_path = None
        pdf_output_path = None
        if save:
            try:
                import json
                clean_content = response.content.replace('```json', '').replace('```', '').strip()
                critique_data = json.loads(clean_content)
            except Exception as e:
                critique_data = response.content
            
            output_path = self.storage.save(
                critique=critique_data,
                llm="ollama",
                model=response.model,
                prompt_technique=prompt_technique,
                execution_time=execution_time
            )
            
            # Generate PDF Report
            if isinstance(critique_data, dict) and output_path:
                try:
                    from src.output.pdf_generator import generate_pdf_report
                    pdf_path = output_path.replace('.json', '.pdf')
                    pdf_output_path = generate_pdf_report(
                        critique_data,
                        pdf_path,
                        metadata={"model": response.model, "prompt_technique": prompt_technique}
                    )
                except Exception as e:
                    print(f"Warning: Failed to generate PDF: {e}")
            
        return CritiqueResult(
            llm="ollama",
            model=response.model,
            prompt_technique=prompt_technique,
            response=response,
            execution_time=execution_time,
            output_path=output_path,
            pdf_output_path=pdf_output_path,
            log_output_path=str(run_logger.file_path)
        )

    def list_saved_critiques(self) -> List[Dict[str, Any]]:
        """List all saved critiques."""
        return self.storage.list_critiques()
        
    def load_critique(self, filename: str) -> Dict[str, Any]:
        """Load a specific critique."""
        return self.storage.load(filename)
