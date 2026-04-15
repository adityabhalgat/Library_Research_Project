"""JSON storage for critique outputs."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from src.llm.base import LLMResponse


@dataclass
class CritiqueResult:
    """Result of a critique generation."""
    llm: str
    model: str
    prompt_technique: str
    response: LLMResponse
    execution_time: float
    output_path: str | None = None
    pdf_output_path: str | None = None
    log_output_path: str | None = None


class JSONStorage:
    """Handler for saving critiques as JSON files."""
    
    def __init__(self, output_dir: str = "./outputs"):
        """Initialize JSON storage.
        
        Args:
            output_dir: Directory for saving output files
        """
        self.output_dir = Path(output_dir)
        self._ensure_dir_exists()
    
    def _ensure_dir_exists(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(
        self,
        llm: str,
        prompt_technique: str,
        timestamp: datetime | None = None
    ) -> str:
        """Generate a unique filename for the critique.
        
        Args:
            llm: Name of the LLM used
            prompt_technique: Prompting technique used
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Generated filename
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{ts_str}_{llm}_{prompt_technique}.json"
    
    def save(
        self,
        critique: dict | str,
        llm: str,
        model: str,
        prompt_technique: str,
        execution_time: float,
        inputs_summary: dict | None = None,
        usage: dict | None = None,
        custom_filename: str | None = None
    ) -> str:
        """Save a critique to a JSON file.
        
        Args:
            critique: The critique content (dict or JSON string)
            llm: Name of the LLM provider
            model: Specific model used
            prompt_technique: Prompting technique used
            execution_time: Time taken in seconds
            inputs_summary: Optional summary of inputs used
            usage: Optional token usage information
            custom_filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now()
        
        # Parse critique if it's a string
        if isinstance(critique, str):
            try:
                critique_data = json.loads(critique)
            except json.JSONDecodeError:
                # If not valid JSON, wrap in a dict
                critique_data = {"raw_response": critique}
        else:
            critique_data = critique
        
        # Build output structure
        output = {
            "metadata": {
                "llm": llm,
                "model": model,
                "prompt_technique": prompt_technique,
                "timestamp": timestamp.isoformat(),
                "execution_time_seconds": round(execution_time, 2),
                "usage": usage
            },
            "inputs_summary": inputs_summary,
            "critique": critique_data
        }
        
        # Generate filename
        if custom_filename:
            filename = custom_filename
        else:
            filename = self._generate_filename(llm, prompt_technique, timestamp)
        
        # Save file
        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def load(self, filename: str) -> dict:
        """Load a critique from a JSON file.
        
        Args:
            filename: Name of the file (or full path)
            
        Returns:
            The loaded critique data
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        # Check if it's a full path or just filename
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = self.output_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Critique file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_critiques(self) -> list[dict]:
        """List all saved critiques.
        
        Returns:
            List of dictionaries with file info
        """
        critiques = []
        
        for file_path in sorted(self.output_dir.glob("*.json"), reverse=True):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    metadata = data.get("metadata", {})
                    critiques.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "llm": metadata.get("llm"),
                        "model": metadata.get("model"),
                        "prompt_technique": metadata.get("prompt_technique"),
                        "timestamp": metadata.get("timestamp"),
                        "execution_time": metadata.get("execution_time_seconds")
                    })
            except (json.JSONDecodeError, KeyError):
                # Skip invalid files
                critiques.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "error": "Invalid or corrupted file"
                })
        
        return critiques
    
    def get_latest(self, llm: str | None = None) -> dict | None:
        """Get the most recent critique.
        
        Args:
            llm: Optional filter by LLM name
            
        Returns:
            The latest critique data or None
        """
        critiques = self.list_critiques()
        
        if llm:
            critiques = [c for c in critiques if c.get("llm") == llm]
        
        if not critiques:
            return None
        
        return self.load(critiques[0]["filename"])
