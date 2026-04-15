"""Image input handler for architecture diagrams."""

import base64
import os
from pathlib import Path


class ImageHandler:
    """Handler for loading and processing image files."""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    MEDIA_TYPES = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    
    def __init__(self, image_path: str):
        """Initialize with image path.
        
        Args:
            image_path: Path to the image file
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        self.path = Path(image_path)
        self._validate()
    
    def _validate(self) -> None:
        """Validate the image file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")
        
        ext = self.path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {ext}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
    
    @property
    def media_type(self) -> str:
        """Get the MIME type of the image."""
        ext = self.path.suffix.lower()
        return self.MEDIA_TYPES.get(ext, 'image/jpeg')
    
    def to_base64(self) -> str:
        """Convert image to base64 string.
        
        Returns:
            Base64 encoded string of the image
        """
        with open(self.path, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def get_file_size(self) -> int:
        """Get the file size in bytes."""
        return os.path.getsize(self.path)
    
    def get_file_size_mb(self) -> float:
        """Get the file size in megabytes."""
        return self.get_file_size() / (1024 * 1024)
    
    @classmethod
    def from_bytes(cls, image_bytes: bytes, media_type: str = "image/jpeg") -> tuple[str, str]:
        """Create base64 string from raw bytes.
        
        Args:
            image_bytes: Raw image bytes
            media_type: MIME type of the image
            
        Returns:
            Tuple of (base64_string, media_type)
        """
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return base64_str, media_type
