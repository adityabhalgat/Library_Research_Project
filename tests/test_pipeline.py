"""Tests for the BE Project Critique Pipeline."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts.base import ProjectInputs
from src.prompts import get_prompt_strategy
from src.input_handlers import TextHandler, ImageHandler
from src.output import JSONStorage


class TestProjectInputs:
    """Tests for ProjectInputs dataclass."""
    
    def test_to_text_block_full(self):
        """Test text block with all inputs."""
        inputs = ProjectInputs(
            project_title="Test Project Title",
            abstract="Test abstract",
            architecture_description="Test architecture description",
            has_architecture_image=True
        )
        
        text = inputs.to_text_block()
        assert "Test Project Title" in text
        assert "Test abstract" in text
        assert "Test architecture description" in text
        assert "Architecture Diagram" in text
    
    def test_to_text_block_partial(self):
        """Test text block with partial inputs."""
        inputs = ProjectInputs(
            project_title="Test Project Title"
        )
        
        text = inputs.to_text_block()
        assert "Test Project Title" in text
        assert "Abstract" not in text
        assert "Architecture Description" not in text
    
    def test_to_text_only_block(self):
        """Test text-only block excludes image reference."""
        inputs = ProjectInputs(
            project_title="Title",
            abstract="Abstract text",
            architecture_description="Arch desc",
            has_architecture_image=True
        )
        
        text = inputs.to_text_only_block()
        assert "Title" in text
        assert "Abstract text" in text
        assert "Arch desc" in text
        assert "Architecture Diagram" not in text  # Should NOT include image reference


class TestPromptBuilders:
    """Tests for prompt builders."""
    
    def test_get_monolithic_builder(self):
        """Test getting monolithic prompt builder."""
        builder = get_prompt_strategy("monolithic")
        assert builder.name == "monolithic"
    
    def test_get_cot_builder(self):
        """Test getting chain-of-thought prompt builder."""
        builder = get_prompt_strategy("cot")
        assert builder.name == "cot"
    
    def test_get_cove_builder(self):
        """Test getting cove prompt builder."""
        builder = get_prompt_strategy("cove")
        assert builder.name == "cove"
    
    def test_get_rcot_builder(self):
        """Test getting rcot prompt builder."""
        builder = get_prompt_strategy("rcot")
        assert builder.name == "rcot"
    
    def test_get_two_model_builder(self):
        """Test getting two_model prompt builder."""
        builder = get_prompt_strategy("two_model")
        assert builder.name == "two_model"
    
    def test_get_expert_builder(self):
        """Test getting expert prompt builder."""
        builder = get_prompt_strategy("expert")
        assert builder.name == "expert"
    
    def test_get_invalid_builder(self):
        """Test getting invalid prompt builder raises error."""
        with pytest.raises(ValueError):
            get_prompt_strategy("invalid")


class TestTextHandler:
    """Tests for TextHandler."""
    
    def test_load_inline_text(self):
        """Test loading inline text."""
        text = TextHandler.load("This is inline text")
        assert text == "This is inline text"
    
    def test_load_from_file(self):
        """Test loading from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("File content")
            f.flush()
            
            text = TextHandler.load(f.name)
            assert text == "File content"
            
            os.unlink(f.name)
    
    def test_is_file_path(self):
        """Test file path detection."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            assert TextHandler.is_file_path(f.name) == True
            os.unlink(f.name)
        
        assert TextHandler.is_file_path("not a file") == False
    
    def test_validate_not_empty(self):
        """Test empty text validation."""
        result = TextHandler.validate_not_empty("valid text", "field")
        assert result == "valid text"
        
        with pytest.raises(ValueError):
            TextHandler.validate_not_empty("", "field")
    
    def test_truncate(self):
        """Test text truncation."""
        short = TextHandler.truncate("short", max_chars=100)
        assert short == "short"
        
        long = "x" * 200
        truncated = TextHandler.truncate(long, max_chars=100)
        assert len(truncated) < 200
        assert "truncated" in truncated.lower()


class TestJSONStorage:
    """Tests for JSONStorage."""
    
    def test_save_and_load(self):
        """Test saving and loading critique."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = JSONStorage(tmpdir)
            
            # Save
            path = storage.save(
                critique={"test": "content"},
                llm="ollama",
                model="llava",
                prompt_technique="cot",
                execution_time=1.5
            )
            
            assert os.path.exists(path)
            
            # Load
            data = storage.load(os.path.basename(path))
            assert data["critique"]["test"] == "content"
            assert data["metadata"]["llm"] == "ollama"
    
    def test_save_string_critique(self):
        """Test saving string critique (parses as JSON)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = JSONStorage(tmpdir)
            
            path = storage.save(
                critique='{"test": "string content"}',
                llm="ollama",
                model="llava",
                prompt_technique="monolithic",
                execution_time=2.0
            )
            
            data = storage.load(os.path.basename(path))
            assert data["critique"]["test"] == "string content"
    
    def test_list_critiques(self):
        """Test listing critiques."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = JSONStorage(tmpdir)
            
            # Save two critiques
            storage.save(
                critique={"id": 1},
                llm="ollama",
                model="llava",
                prompt_technique="cot",
                execution_time=1.0
            )
            storage.save(
                critique={"id": 2},
                llm="ollama",
                model="llava-llama3",
                prompt_technique="monolithic",
                execution_time=2.0
            )
            
            critiques = storage.list_critiques()
            assert len(critiques) == 2


class TestImageHandler:
    """Tests for ImageHandler."""
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        with pytest.raises(FileNotFoundError):
            ImageHandler("/nonexistent/path.jpg")
    
    def test_validate_unsupported_format(self):
        """Test validation of unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            with pytest.raises(ValueError):
                ImageHandler(f.name)
            os.unlink(f.name)
    
    def test_media_type(self):
        """Test media type detection."""
        # Create a minimal valid JPEG file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # Write minimal JPEG header
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF')
            f.flush()
            
            handler = ImageHandler(f.name)
            assert handler.media_type == "image/jpeg"
            
            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
