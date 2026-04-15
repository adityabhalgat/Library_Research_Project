"""Tests for pair-wise prompt matrix parsing."""

from pathlib import Path
import sys

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.input_handlers.prompt_table_handler import PromptTableHandler


def test_wide_format_excel_with_multi_step_cell(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")

    file_path = tmp_path / "prompt_matrix.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Prompts"

    sheet.append(["criteria", "monolithic", "cot"])
    sheet.append(
        [
            "C1",
            "Monolithic prompt for C1",
            "Step 1: Gather evidence || Step 2: Decide output",
        ]
    )
    sheet.append(
        [
            "C2",
            "Monolithic prompt for C2",
            "Prompt 1: Parse text\nPrompt 2: Validate diagram\nPrompt 3: Final JSON",
        ]
    )

    workbook.save(file_path)

    matrix = PromptTableHandler(str(file_path), sheet_name="Prompts").load()

    assert matrix.criteria == ["C1", "C2"]
    assert matrix.get_prompts("C1", "monolithic") == ["Monolithic prompt for C1"]
    assert matrix.get_prompts("C1", "cot") == ["Step 1: Gather evidence", "Step 2: Decide output"]
    assert matrix.get_prompts("C2", "cot") == [
        "Parse text",
        "Validate diagram",
        "Final JSON",
    ]


def test_long_format_excel_with_json_prompt_list(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")

    file_path = tmp_path / "prompt_long.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Table"

    sheet.append(["criterion", "prompt_technique", "prompt"])
    sheet.append(["C1", "cot", '["collect evidence", "final json"]'])
    sheet.append(["C1", "expert", "single expert prompt"])

    workbook.save(file_path)

    matrix = PromptTableHandler(str(file_path), sheet_name="Table").load()

    assert matrix.criteria == ["C1"]
    assert matrix.get_prompts("C1", "cot") == ["collect evidence", "final json"]
    assert matrix.get_prompts("C1", "expert") == ["single expert prompt"]
