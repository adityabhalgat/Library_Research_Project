"""Configuration management for the BE Project Critique Pipeline."""

import os
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llava",
        description="Ollama model name (use a vision model like llava for image support)"
    )
    ollama_judge_model: str = Field(
        default="",
        description="Optional separate model for Two-Model judge strategy"
    )
    
    # Default Settings
    default_prompt_technique: Literal["monolithic", "cot", "expert", "cove", "rcot", "two_model"] = Field(
        default="cot",
        description="Default prompting technique"
    )
    output_dir: str = Field(default="./outputs", description="Output directory")

    # Parameter analysis prompt matrix
    parameter_prompt_table_path: str = Field(
        default="parameters.xlsx",
        description="Excel/CSV table containing pair-wise prompts for parameter analysis"
    )
    parameter_prompt_table_sheet: str = Field(
        default="",
        description="Optional sheet name for the prompt table (Excel only)"
    )
    
    # Generation Configuration
    max_tokens: int = Field(default=4096, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    
    def is_ollama_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.ollama_base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    def get_available_models(self) -> list[str]:
        """Get list of models available on the Ollama server."""
        try:
            import urllib.request
            import json
            req = urllib.request.Request(f"{self.ollama_base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# Global settings instance
settings = Settings()
