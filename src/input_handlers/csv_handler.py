"""CSV handler for loading project data."""

import csv
from pathlib import Path
from typing import Dict, Any, Optional

from src.prompts.base import ProjectInputs
from src.input_handlers.image_handler import ImageHandler
from src.input_handlers.text_handler import TextHandler


class CSVHandler:
    """Handler for loading project data from CSV."""
    
    def __init__(self, csv_path: str = "be_project_dataset.csv"):
        """Initialize CSV handler.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
    def get_group_data(self, group_id: str) -> Dict[str, Any] | None:
        """Get project data for a specific group ID.
        
        Args:
            group_id: The group ID (e.g., 'group_1' or '1')
            
        Returns:
            Dictionary with project data or None if not found
        """
        # Normalize group ID (ensure 'group_' prefix if just number)
        target_id = group_id if group_id.startswith("group_") else f"group_{group_id}"
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['group_id'] == target_id:
                    return self._process_row(row)
        
        return None

    def _process_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Process a CSV row into input format.
        
        Maps CSV columns to the new input fields:
        - project_title: Extracted from pdf_file_name or first line of problem definition
        - abstract: The full problem_definition_and_objectives text
        - architecture_description: Combined SRS fields as architecture context
        """
        
        # Extract project title from group_id or PDF filename
        group_id = row.get('group_id', '')
        pdf_name = row.get('pdf_file_name', '')
        project_title = pdf_name.replace('.pdf', '').replace('_', ' ').title() if pdf_name else group_id
        
        # Use problem definition + objectives as abstract
        abstract = row.get('problem_definition_and_objectives', '')
        
        # Prefer explicit architecture description/system description if present; fall back to SRS fields
        architecture_description = (
            row.get('architecture_description')
            or row.get('system_description')
            or ""
        ).strip()

        if not architecture_description:
            srs_parts = [
                row.get('srs_assumptions_and_dependencies', ''),
                row.get('srs_functional_requirements', ''),
                row.get('srs_external_interface_requirements', ''),
                row.get('srs_nonfunctional_requirements', ''),
                row.get('srs_system_requirements', '')
            ]
            architecture_description = "\n\n".join([p for p in srs_parts if p])
        
        # Image path
        img_path = row.get('system_architecture_image_path', '')
        
        return {
            "group_id": group_id,
            "project_title": project_title,
            "abstract": abstract,
            "architecture_description": architecture_description,
            "architecture_image_path": img_path
        }

    def load_inputs(self, group_id: str) -> tuple[ProjectInputs, str, str | None, str]:
        """Load inputs for pipeline usage.
        
        Returns:
            Tuple of (ProjectInputs, group_id, image_base64, media_type)
        """
        data = self.get_group_data(group_id)
        if not data:
            raise ValueError(f"Group {group_id} not found in CSV.")
            
        # Handle Image
        image_base64 = None
        media_type = "image/jpeg"
        image_path = data['architecture_image_path']
        
        if image_path:
            if Path(image_path).exists():
                handler = ImageHandler(image_path)
                image_base64 = handler.to_base64()
                media_type = handler.media_type
            else:
                print(f"Warning: Image not found at {image_path}")
        
        # Create ProjectInputs
        inputs = ProjectInputs(
            project_title=data['project_title'],
            abstract=data['abstract'],
            architecture_description=data['architecture_description'],
            has_architecture_image=bool(image_base64),
            architecture_image_base64=image_base64,
            architecture_media_type=media_type
        )
        
        return inputs, data['group_id'], image_base64, media_type
