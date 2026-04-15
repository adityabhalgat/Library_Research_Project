"""Extract system architecture text from PDFs and augment the dataset CSV.

- Reads the existing dataset CSV (default: be_project_dataset.csv).
- For each row, loads the corresponding PDF from renamed_pdfs/ and extracts the
  text under the "System Architecture" section.
- Writes the architecture_description column alongside existing fields without
  altering other values.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import List

from pypdf import PdfReader

SECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?:^|\n)\s*\d+(?:\.\d+)*\s*System Architecture\b[:\s]*", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*System Architecture\b[:\s]*", re.IGNORECASE),
]

# Heuristic to find the next section heading (numbered or ALL CAPS heading on its own line).
NEXT_HEADING_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*[A-Z][A-Za-z0-9 /&-]{3,80}|[A-Z][A-Z /&-]{3,80})\s*(?:\n|$)",
    re.MULTILINE,
)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF, returning an empty string on failure."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[warn] Could not open PDF {pdf_path}: {exc}")
        return ""

    chunks: List[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            # "layout" mode keeps relative positioning better for headings; fallback to default if it fails.
            text = page.extract_text(extraction_mode="layout")  # type: ignore[arg-type]
            if not text:
                text = page.extract_text() or ""
            chunks.append(text)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[warn] Failed to read page {idx + 1} of {pdf_path}: {exc}")
    return "\n".join(chunks)


def normalize_text(text: str) -> str:
    """Normalize PDF text for more reliable regex extraction."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("-\n", "")  # undo hyphenation line breaks
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def extract_system_architecture_section(text: str) -> str:
    """Extract the system architecture section based on heading heuristics."""
    if not text:
        return ""

    normalized = normalize_text(text)

    starts: List[int] = []
    for pattern in SECTION_PATTERNS:
        starts.extend(match.end() for match in pattern.finditer(normalized))

    if not starts:
        return ""

    best_section = ""

    def is_toc_like(candidate: str) -> bool:
        dot_runs = candidate.count(". . .")
        dot_ratio = candidate.count(".") / max(len(candidate), 1)
        return (dot_runs >= 3 and dot_runs * 8 > 0.05 * len(candidate)) or dot_ratio > 0.05

    for start in sorted(starts):
        next_heading = NEXT_HEADING_PATTERN.search(normalized, pos=start)
        end = next_heading.start() if next_heading else len(normalized)
        candidate = normalized[start:end].strip()
        if not candidate:
            continue
        if is_toc_like(candidate):
            continue
        if len(candidate) > len(best_section):
            best_section = candidate

    return best_section


def load_rows(csv_path: Path) -> tuple[List[dict], List[str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames


def write_rows(output_path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_backup(input_path: Path, output_path: Path, enable_backup: bool) -> None:
    if enable_backup and input_path == output_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy(input_path, backup_path)
            print(f"[info] Backup created at {backup_path}")


def process_dataset(
    input_csv: Path,
    pdf_dir: Path,
    output_csv: Path,
    enable_backup: bool = True,
) -> None:
    rows, fieldnames = load_rows(input_csv)
    output_fields = list(fieldnames)
    if "architecture_description" not in output_fields:
        output_fields.append("architecture_description")

    processed = 0
    extracted = 0

    ensure_backup(input_csv, output_csv, enable_backup)

    for row in rows:
        processed += 1
        pdf_name = (row.get("pdf_file_name") or "").strip()
        pdf_path = pdf_dir / pdf_name if pdf_name else None
        arch_desc = ""

        if pdf_path and pdf_path.exists():
            raw_text = extract_pdf_text(pdf_path)
            arch_desc = extract_system_architecture_section(raw_text)
            if arch_desc:
                extracted += 1
        else:
            if pdf_name:
                print(f"[warn] PDF not found for {pdf_name}")

        row["architecture_description"] = arch_desc

    write_rows(output_csv, rows, output_fields)

    print(
        f"[done] Wrote {len(rows)} rows to {output_csv} | "
        f"extracted architecture description for {extracted}/{processed} PDFs"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract system architecture text from PDFs into the dataset CSV."
    )
    parser.add_argument(
        "--input",
        default="be_project_dataset.csv",
        type=Path,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=Path("renamed_pdfs"),
        type=Path,
        help="Directory containing PDFs referenced by pdf_file_name.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output CSV path (defaults to overwriting --input).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic .bak backup when overwriting the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input
    output_csv = args.output or input_csv
    pdf_dir = args.pdf_dir

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    process_dataset(
        input_csv=input_csv,
        pdf_dir=pdf_dir,
        output_csv=output_csv,
        enable_backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
