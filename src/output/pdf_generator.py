"""PDF Generator for project critiques."""

import os
import re
import json
from typing import Dict, Any, List
from fpdf import FPDF


def _safe_text(text) -> str:
    """Safely encode text for PDF output (handles emojis, special chars)."""
    return str(text).encode('latin-1', 'replace').decode('latin-1')


def _extract_json_from_raw_text(raw_text: str) -> Dict[str, Any] | None:
    """Try to extract a JSON object from a raw_text string."""
    try:
        match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
        if match:
            json_str = match.group(1).replace('```json', '').replace('```', '').strip()
            return json.loads(json_str)
    except Exception:
        pass
    return None


def _clean_raw_text_for_display(text: str) -> str:
    """Removes JSON blocks and code fences from text to make it readable in PDF."""
    if not text:
        return ""
    
    # Remove markdown code fences for JSON
    text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Find and remove any remaining JSON blocks { ... } at the end or middle
    # We look for blocks that start with { and end with } and contain key-value looking patterns
    # This is a bit aggressive but helps remove hallucinations
    text = re.sub(r'\{[^{]*"[^"]*"\s*:\s*[^{}]*\}', '', text, flags=re.DOTALL)
    
    # Try to find the largest { } block if it looks like JSON and remove it
    try:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            # Only remove if it contains typical JSON syntax indicators
            block = match.group(1)
            if '"' in block and ':' in block:
                text = text.replace(block, "")
    except Exception:
        pass

    # Clean up whitespace and introductory/concluding phrase artifacts
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


class PDFReport(FPDF):
    """Custom PDF class for critique reports."""

    def __init__(self, project_title: str = "BE Project Critique Report"):
        super().__init__()
        self.project_title_text = project_title

    def header(self):
        self.set_font("helvetica", "B", 16)
        self.set_text_color(0, 51, 102)
        self.cell(0, 12, _safe_text(self.project_title_text), border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 51, 102)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str, color=(0, 51, 102)):
        self.set_font("helvetica", "B", 13)
        self.set_text_color(*color)
        self.set_fill_color(235, 240, 250)
        self.cell(0, 9, _safe_text(title), border="B", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def subsection_title(self, title: str):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(30, 30, 100)
        self.cell(0, 7, _safe_text(title), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def body_text(self, text: str):
        self.set_font("helvetica", "", 10)
        self.multi_cell(0, 6, _safe_text(text))
        self.ln(3)

    def add_scores(self, scores: Dict[str, Any]):
        self.section_title("Evaluation Scores")
        self.set_font("helvetica", "B", 10)
        # Table header
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        self.cell(120, 8, "Criterion", border=1, fill=True)
        self.cell(30, 8, "Score / 10", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

        for i, (k, v) in enumerate(scores.items()):
            formatted_key = str(k).replace('_', ' ').title()
            fill = i % 2 == 0
            self.set_fill_color(240, 245, 255) if fill else self.set_fill_color(255, 255, 255)
            self.set_font("helvetica", "", 10)
            self.cell(120, 7, _safe_text(formatted_key), border=1, fill=fill)
            
            # Color-code the score
            try:
                score_num = float(str(v).replace('/10', '').strip())
                if score_num >= 8:
                    self.set_text_color(0, 128, 0)   # green
                elif score_num >= 6:
                    self.set_text_color(180, 100, 0)  # orange
                else:
                    self.set_text_color(180, 0, 0)    # red
            except Exception:
                self.set_text_color(0, 0, 0)
            
            self.cell(30, 7, _safe_text(f"{v}"), border=1, fill=fill, align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
        self.ln(5)

    def add_list_section(self, title: str, items: List, color=(0, 51, 102)):
        if not items:
            return
        self.section_title(title, color)
        self.set_font("helvetica", "", 10)
        for item in items:
            self.multi_cell(0, 6, _safe_text(f"  \u2022  {item}"))
            self.ln(1)
        self.ln(4)

    def add_detailed_section(self, title: str, content):
        """Add a detailed section with sub-fields if content is a dict, or plain text if string."""
        self.section_title(title)
        if isinstance(content, dict):
            for k, v in content.items():
                label = str(k).replace('_', ' ').title()
                self.subsection_title(label)
                self.body_text(str(v))
        elif isinstance(content, str):
            self.body_text(content)
        self.ln(3)

    def add_raw_text_section(self, title: str, raw_text: str):
        """Render a raw_text fallback section."""
        self.section_title(title, color=(80, 80, 80))
        self.body_text(raw_text)
        self.ln(3)


def _render_single_critique_block(pdf: PDFReport, data: Dict[str, Any], phase_label: str = ""):
    """Render a 'critique block' (dict with optional assessment/scores/strengths etc)."""
    
    # --- If only raw_text is available --- 
    if "raw_text" in data and len(data) == 1:
        # Try to extract JSON from inside the raw_text first
        extracted = _extract_json_from_raw_text(data["raw_text"])
        if extracted:
            _render_single_critique_block(pdf, extracted, phase_label)
            return
        
        # Fallback: render cleaned text
        label = f"{phase_label} - Full Critique Text" if phase_label else "Full Critique Text"
        cleaned_text = _clean_raw_text_for_display(data["raw_text"])
        if cleaned_text:
            pdf.add_raw_text_section(label, cleaned_text)
        return
    
    if "overall_assessment" in data:
        lbl = f"Overall Assessment" + (f" ({phase_label})" if phase_label else "")
        pdf.section_title(lbl)
        pdf.body_text(data["overall_assessment"])

    if "scores" in data and isinstance(data["scores"], dict):
        pdf.add_scores(data["scores"])

    if "strengths" in data and isinstance(data["strengths"], list):
        pdf.add_list_section("Strengths", data["strengths"], (0, 100, 0))

    if "weaknesses" in data and isinstance(data["weaknesses"], list):
        pdf.add_list_section("Weaknesses", data["weaknesses"], (180, 0, 0))

    if "suggestions" in data and isinstance(data["suggestions"], list):
        pdf.add_list_section("Suggestions for Improvement", data["suggestions"], (180, 100, 0))

    if "detailed_critique" in data and isinstance(data["detailed_critique"], dict):
        pdf.add_detailed_section("Detailed Critique", data["detailed_critique"])

    if "raw_text" in data:
        # There are other keys too; render cleaned raw_text as supplementary
        cleaned_text = _clean_raw_text_for_display(data["raw_text"])
        if cleaned_text:
            pdf.add_raw_text_section("Additional Observations", cleaned_text)

    # Handle any other structured sub-dicts (from model's creative JSON keys)
    skip_keys = {"overall_assessment", "scores", "strengths", "weaknesses",
                 "suggestions", "detailed_critique", "raw_text"}
    for k, v in data.items():
        if k not in skip_keys:
            label = str(k).replace('_', ' ').title()
            if isinstance(v, dict):
                pdf.add_detailed_section(label, v)
            elif isinstance(v, list):
                pdf.add_list_section(label, [str(i) for i in v])
            elif isinstance(v, str) and v.strip():
                pdf.section_title(label)
                pdf.body_text(v)


def generate_pdf_report(
    critique_data: Dict[str, Any],
    output_path: str,
    metadata: Dict[str, Any] = None
) -> str:
    """Generate a PDF report from critique data.

    Args:
        critique_data: The critique dict (can be raw critique output or the full JSON with metadata wrapper)
        output_path: File path to save the PDF to
        metadata: Optional metadata dict with model, prompt_technique, etc.

    Returns:
        Absolute path to the generated PDF
    """
    
    # --- Unwrap if full file structure was passed ---
    # The pipeline saves: {"metadata":..., "inputs_summary":..., "critique": {...}}
    # but may also pass the raw critique dict directly. Handle both.
    if "critique" in critique_data:
        critique_content = critique_data["critique"]
        if metadata is None and "metadata" in critique_data:
            metadata = critique_data["metadata"]
    else:
        critique_content = critique_data

    # Determine project title for header
    project_title = "BE Project Critique Report"
    if metadata:
        technique = metadata.get('prompt_technique', '')
        model = metadata.get('model', '')
        if technique or model:
            project_title = f"BE Project Critique — {technique.upper()} [{model}]"

    pdf = PDFReport(project_title)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Metadata banner ---
    if metadata:
        pdf.set_font("helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        ts = metadata.get('timestamp', '')
        exec_time = metadata.get('execution_time_seconds', metadata.get('execution_time', ''))
        meta_line = f"Technique: {metadata.get('prompt_technique', '')}  |  Model: {metadata.get('model', '')}  |  Time: {exec_time}s  |  {ts}"
        pdf.cell(0, 5, _safe_text(meta_line), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # --- Three-phase format (text_only + image_only + final_combined) ---
    if isinstance(critique_content, dict) and (
        "text_only_critique" in critique_content or "final_combined_critique" in critique_content
    ):
        # Phase 1: Text-only
        if "text_only_critique" in critique_content:
            data = critique_content["text_only_critique"]
            if "raw_text" in data and len(data) == 1:
                extracted = _extract_json_from_raw_text(data["raw_text"])
                data = extracted or data
            pdf.section_title("Phase 1: Text-Only Critique", (0, 51, 102))
            pdf.ln(2)
            _render_single_critique_block(pdf, data)

        # Phase 2: Image-only
        if "image_only_critique" in critique_content:
            pdf.add_page()
            data = critique_content["image_only_critique"]
            if "raw_text" in data and len(data) == 1:
                extracted = _extract_json_from_raw_text(data["raw_text"])
                data = extracted or data
            pdf.section_title("Phase 2: Architecture Diagram Critique", (0, 80, 50))
            pdf.ln(2)
            _render_single_critique_block(pdf, data)

        # Phase 3: Final Combined — always on new page
        if "final_combined_critique" in critique_content:
            pdf.add_page()
            data = critique_content["final_combined_critique"]
            if "raw_text" in data and len(data) == 1:
                extracted = _extract_json_from_raw_text(data["raw_text"])
                data = extracted or data
            pdf.section_title("Phase 3: Final Combined Critique", (120, 0, 80))
            pdf.ln(2)
            _render_single_critique_block(pdf, data)

    else:
        # Simple / single-phase format — render directly
        _render_single_critique_block(pdf, critique_content)

    pdf.output(output_path)
    return os.path.abspath(output_path)
