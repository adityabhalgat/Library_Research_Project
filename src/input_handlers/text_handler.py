"""Text input handler for problem statement, objectives, and SRS."""

import os
from pathlib import Path


class TextHandler:
    """Handler for text inputs - files or inline text."""
    
    def __init__(self):
        """Initialize text handler."""
        pass
    
    @staticmethod
    def load(text_or_path: str) -> str:
        """Load text from file or return as-is if it's inline text.
        
        The method determines if the input is a file path or inline text:
        - If it's a valid file path that exists, load from file
        - Otherwise, treat as inline text
        
        Args:
            text_or_path: Either a file path or inline text
            
        Returns:
            The text content
        """
        # Check if it looks like a file path and exists
        if os.path.exists(text_or_path):
            return TextHandler.load_from_file(text_or_path)
        
        # Treat as inline text
        return text_or_path.strip()
    
    @staticmethod
    def load_from_file(file_path: str) -> str:
        """Load text from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            The text content
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    @staticmethod
    def is_file_path(text_or_path: str) -> bool:
        """Check if the input is a valid file path.
        
        Args:
            text_or_path: Input to check
            
        Returns:
            True if it's an existing file path
        """
        return os.path.exists(text_or_path) and os.path.isfile(text_or_path)
    
    @staticmethod
    def validate_not_empty(text: str, field_name: str) -> str:
        """Validate that text is not empty.
        
        Args:
            text: Text to validate
            field_name: Name of the field for error message
            
        Returns:
            The validated text
            
        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError(f"{field_name} cannot be empty")
        return text.strip()
    
    @staticmethod
    def truncate(text: str, max_chars: int = 10000) -> str:
        """Truncate text to a maximum number of characters.
        
        Args:
            text: Text to truncate
            max_chars: Maximum number of characters
            
        Returns:
            Truncated text with indicator if truncated
        """
        if len(text) <= max_chars:
            return text
        
        return text[:max_chars] + "\n\n[... Content truncated due to length ...]"
