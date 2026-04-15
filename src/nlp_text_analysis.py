#!/usr/bin/env python3
"""Corpus NLP analysis for BE project dataset.

This script computes:
1. Summary statistics for word and sentence counts across text types.
2. POS tags per token for each project text.
3. Noun-only lemmatized corpora and word clouds.
4. Verb-only lemmatized corpora and word clouds.

The script is intentionally data-driven: it reads all rows from a CSV each run,
so adding/removing projects only requires re-running this command.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

TITLE_FIELDS = ["project_title", "title", "paper_title", "project_name"]
ABSTRACT_FIELDS = ["abstract", "problem_definition_and_objectives", "summary"]
SYSTEM_REQ_FIELDS = [
    "system_requirements_text",
    "system_requirements",
    "srs_system_requirements",
]
ARCH_DESC_FIELDS = ["architecture_description", "system_description", "architecture_text"]
ARCH_FALLBACK_FIELDS = [
    "srs_assumptions_and_dependencies",
    "srs_functional_requirements",
    "srs_external_interface_requirements",
    "srs_nonfunctional_requirements",
    "srs_system_requirements",
]
TOKEN_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z'-]*$")


@dataclass
class TextRecord:
    """A single text document tied to a project and text type."""

    project_id: str
    text_type: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NLP statistics, POS tags, and word clouds for project corpus."
    )
    parser.add_argument(
        "--csv-path",
        default="be_project_dataset.csv",
        help="Input CSV path (default: be_project_dataset.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/nlp_analysis",
        help="Directory for JSON/text/image outputs",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum lemma frequency included in word cloud (default: 2)",
    )
    return parser.parse_args()


def ensure_nltk_resources() -> None:
    resources: list[tuple[str, str]] = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)

    # Newer NLTK versions may require these package aliases.
    for optional_package, optional_path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(optional_path)
        except LookupError:
            try:
                nltk.download(optional_package, quiet=True)
            except Exception:
                pass


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def get_project_id(row: dict[str, str], idx: int) -> str:
    for key in ("group_id", "project_id", "id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return f"project_{idx}"


def first_non_empty(row: dict[str, str], fields: Iterable[str]) -> str:
    for field in fields:
        value = (row.get(field) or "").strip()
        if value:
            return value
    return ""


def derive_title(row: dict[str, str]) -> str:
    direct_title = first_non_empty(row, TITLE_FIELDS)
    if direct_title:
        return direct_title

    pdf_name = (row.get("pdf_file_name") or "").strip()
    if not pdf_name:
        return ""

    stem = Path(pdf_name).stem
    normalized = stem.replace("_", " ").replace("-", " ").strip()
    return normalized.title()


def build_text_records(rows: list[dict[str, str]]) -> list[TextRecord]:
    records: list[TextRecord] = []

    for idx, row in enumerate(rows, start=1):
        project_id = get_project_id(row, idx)

        title = derive_title(row)
        abstract = first_non_empty(row, ABSTRACT_FIELDS)
        title_abstract = "\n\n".join([part for part in [title, abstract] if part]).strip()
        if title_abstract:
            records.append(TextRecord(project_id=project_id, text_type="title_abstract", text=title_abstract))

        system_requirements = first_non_empty(row, SYSTEM_REQ_FIELDS)
        if system_requirements:
            records.append(
                TextRecord(
                    project_id=project_id,
                    text_type="system_requirements",
                    text=system_requirements,
                )
            )

        architecture_description = first_non_empty(row, ARCH_DESC_FIELDS)
        if not architecture_description:
            fallback_parts = [
                (row.get(field) or "").strip()
                for field in ARCH_FALLBACK_FIELDS
                if (row.get(field) or "").strip()
            ]
            architecture_description = "\n\n".join(fallback_parts).strip()

        if architecture_description:
            records.append(
                TextRecord(
                    project_id=project_id,
                    text_type="architecture_description",
                    text=architecture_description,
                )
            )

    return records


def word_tokens(text: str) -> list[str]:
    tokens = word_tokenize(text)
    return [token for token in tokens if any(char.isalnum() for char in token)]


def sentence_count(text: str) -> int:
    return len([s for s in sent_tokenize(text) if s.strip()])


def numeric_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "average": 0.0,
            "std_dev": 0.0,
            "first_quartile": 0.0,
            "median": 0.0,
            "third_quartile": 0.0,
        }

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        q1 = median = q3 = float(sorted_values[0])
    else:
        quartiles = statistics.quantiles(sorted_values, n=4, method="inclusive")
        q1 = float(quartiles[0])
        median = float(statistics.median(sorted_values))
        q3 = float(quartiles[2])

    std_dev = statistics.stdev(sorted_values) if len(sorted_values) > 1 else 0.0

    return {
        "count": len(sorted_values),
        "min": int(min(sorted_values)),
        "max": int(max(sorted_values)),
        "average": round(float(statistics.mean(sorted_values)), 4),
        "std_dev": round(float(std_dev), 4),
        "first_quartile": round(q1, 4),
        "median": round(median, 4),
        "third_quartile": round(q3, 4),
    }


def clean_token(token: str) -> str:
    lowered = token.lower().strip()
    return lowered if TOKEN_PATTERN.match(lowered) else ""


def lemmatize_token(lemmatizer: WordNetLemmatizer, token: str, coarse_pos: str) -> str:
    cleaned = clean_token(token)
    if not cleaned:
        return ""
    return lemmatizer.lemmatize(cleaned, pos=coarse_pos)


def make_wordcloud(counter: Counter[str], output_file: Path) -> None:
    if not counter:
        return

    wc = WordCloud(width=1600, height=900, background_color="white", collocations=False)
    wc.generate_from_frequencies(counter)
    wc.to_file(str(output_file))


def run_analysis(csv_path: Path, output_dir: Path, min_frequency: int) -> None:
    ensure_nltk_resources()

    rows = read_rows(csv_path)
    records = build_text_records(rows)

    by_type: dict[str, list[TextRecord]] = {
        "title_abstract": [record for record in records if record.text_type == "title_abstract"],
        "system_requirements": [record for record in records if record.text_type == "system_requirements"],
        "architecture_description": [record for record in records if record.text_type == "architecture_description"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pos_dir = output_dir / "pos_tags"
    filtered_dir = output_dir / "filtered_tokens"
    clouds_dir = output_dir / "wordclouds"
    for folder in [pos_dir, filtered_dir, clouds_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv_path": str(csv_path),
        "projects_in_csv": len(rows),
        "text_types": {},
    }

    lemmatizer = WordNetLemmatizer()

    for text_type, type_records in by_type.items():
        word_counts: list[int] = []
        sentence_counts: list[int] = []

        noun_counter: Counter[str] = Counter()
        verb_counter: Counter[str] = Counter()

        pos_file = pos_dir / f"{text_type}_pos_tags.jsonl"
        nouns_file = filtered_dir / f"{text_type}_nouns.jsonl"
        verbs_file = filtered_dir / f"{text_type}_verbs.jsonl"

        with pos_file.open("w", encoding="utf-8") as pos_handle, nouns_file.open(
            "w", encoding="utf-8"
        ) as noun_handle, verbs_file.open("w", encoding="utf-8") as verb_handle:
            for record in type_records:
                tokens = word_tokens(record.text)
                tags = pos_tag(tokens)

                word_counts.append(len(tokens))
                sentence_counts.append(sentence_count(record.text))

                noun_lemmas: list[str] = []
                verb_lemmas: list[str] = []

                for token, tag in tags:
                    if tag.startswith("NN"):
                        lemma = lemmatize_token(lemmatizer, token, "n")
                        if lemma:
                            noun_lemmas.append(lemma)
                            noun_counter[lemma] += 1
                    elif tag.startswith("VB"):
                        lemma = lemmatize_token(lemmatizer, token, "v")
                        if lemma:
                            verb_lemmas.append(lemma)
                            verb_counter[lemma] += 1

                pos_row = {
                    "project_id": record.project_id,
                    "text_type": text_type,
                    "token_count": len(tokens),
                    "tags": [{"token": token, "pos": tag} for token, tag in tags],
                }
                pos_handle.write(json.dumps(pos_row, ensure_ascii=False) + "\n")

                noun_row = {
                    "project_id": record.project_id,
                    "text_type": text_type,
                    "lemmas": noun_lemmas,
                    "text": " ".join(noun_lemmas),
                }
                noun_handle.write(json.dumps(noun_row, ensure_ascii=False) + "\n")

                verb_row = {
                    "project_id": record.project_id,
                    "text_type": text_type,
                    "lemmas": verb_lemmas,
                    "text": " ".join(verb_lemmas),
                }
                verb_handle.write(json.dumps(verb_row, ensure_ascii=False) + "\n")

        noun_cloud_counter = Counter({k: v for k, v in noun_counter.items() if v >= min_frequency})
        verb_cloud_counter = Counter({k: v for k, v in verb_counter.items() if v >= min_frequency})

        make_wordcloud(noun_cloud_counter, clouds_dir / f"{text_type}_nouns.png")
        make_wordcloud(verb_cloud_counter, clouds_dir / f"{text_type}_verbs.png")

        summary["text_types"][text_type] = {
            "projects_with_text": len(type_records),
            "word_count_summary": numeric_summary(word_counts),
            "sentence_count_summary": numeric_summary(sentence_counts),
            "noun_lemmas_unique": len(noun_counter),
            "verb_lemmas_unique": len(verb_counter),
            "noun_lemmas_total": int(sum(noun_counter.values())),
            "verb_lemmas_total": int(sum(verb_counter.values())),
        }

    summary_path = output_dir / "summary_statistics.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved summary statistics to: {summary_path}")
    print(f"Saved POS tag outputs to: {pos_dir}")
    print(f"Saved noun/verb filtered outputs to: {filtered_dir}")
    print(f"Saved word clouds to: {clouds_dir}")


def main() -> None:
    args = parse_args()
    run_analysis(csv_path=Path(args.csv_path), output_dir=Path(args.output_dir), min_frequency=args.min_frequency)


if __name__ == "__main__":
    main()
