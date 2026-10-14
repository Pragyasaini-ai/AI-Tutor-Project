from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import pdfplumber

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "class8_science.jsonl"

CHAPTER_TITLES = {
    "ch01": "Crop Production and Management",
    "ch02": "Microorganisms: Friend and Foe",
    "ch03": "Synthetic Fibres and Plastics",
    "ch04": "Materials: Metals and Non-Metals",
    "ch05": "Coal and Petroleum",
    "ch06": "Combustion and Flame",
    "ch07": "Conservation of Plants and Animals",
    "ch08": "Cell — Structure and Functions",
    "ch09": "Reproduction in Animals",
    "ch10": "Reaching the Age of Adolescence",
    "ch11": "Force and Pressure",
    "ch12": "Friction",
    "ch13": "Sound",
}


def clean_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"\d+\s*[|/\\-]+\s*\d+", line):
            continue
        line = re.sub(r"\bPage\s+\d+\b", "", line, flags=re.IGNORECASE)
        line = re.sub(r"Rationalised\s+\d{4}[-–]\d{2}", "", line, flags=re.IGNORECASE)
        line = line.replace("©", "")
        lines.append(line)
    collapsed = " ".join(lines)
    collapsed = re.sub(r"\s+", " ", collapsed)
    return collapsed.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return clean_text("\n".join(text_parts))


def chunk_text(text: str, max_words: int = 250, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_demo_corpus() -> List[Dict[str, object]]:
    documents: List[Dict[str, object]] = []
    for chapter_id, chapter_title in CHAPTER_TITLES.items():
        sample_content = (
            f"{chapter_title} introduces important ideas in Class 8 Science. "
            "Students learn to observe natural phenomena carefully, ask scientific questions, "
            "and connect observations to explanations in simple language. "
            "The chapter also highlights safe practices and the importance of curiosity in science."
        )
        for idx, chunk in enumerate(chunk_text(sample_content)):
            documents.append(
                {
                    "id": f"{chapter_id}_{idx}",
                    "content": chunk,
                    "meta": {"chapter": chapter_id, "chapter_title": chapter_title},
                }
            )
    return documents


def ingest_documents(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH, use_demo_if_empty: bool = True) -> List[Dict[str, object]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(raw_dir.glob("*.pdf"))

    documents: List[Dict[str, object]] = []
    for pdf_path in pdf_paths:
        chapter_id = pdf_path.stem
        chapter_title = CHAPTER_TITLES.get(chapter_id, pdf_path.stem)
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception:
            text = ""
        if not text:
            continue
        for idx, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f"{chapter_id}_{idx}",
                    "content": chunk,
                    "meta": {"chapter": chapter_id, "chapter_title": chapter_title},
                }
            )

    if not documents and use_demo_if_empty:
        documents = build_demo_corpus()

    with output_path.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NCERT PDFs into a chunked JSONL corpus")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    ingest_documents(raw_dir=args.raw_dir, output_path=args.output)


if __name__ == "__main__":
    main()
