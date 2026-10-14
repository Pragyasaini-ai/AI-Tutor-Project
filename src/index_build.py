from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
JSONL_PATH = BASE_DIR / "data" / "processed" / "class8_science.jsonl"
INDEX_PATH = BASE_DIR / "data" / "processed" / "faiss.index"
DOC_LIST_PATH = BASE_DIR / "data" / "processed" / "documents.pkl"


def load_documents(jsonl_path: Path = JSONL_PATH) -> List[Dict[str, object]]:
    documents: List[Dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def build_index(jsonl_path: Path = JSONL_PATH, index_path: Path = INDEX_PATH, doc_list_path: Path = DOC_LIST_PATH, model_name: str = "all-MiniLM-L6-v2") -> List[Dict[str, object]]:
    documents = load_documents(jsonl_path)
    if not documents:
        raise ValueError(f"No documents found in {jsonl_path}")

    model = SentenceTransformer(model_name)
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with doc_list_path.open("wb") as handle:
        pickle.dump(documents, handle)

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a JSONL corpus and build a FAISS index")
    parser.add_argument("--jsonl", type=Path, default=JSONL_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--docs", type=Path, default=DOC_LIST_PATH)
    args = parser.parse_args()
    build_index(jsonl_path=args.jsonl, index_path=args.index, doc_list_path=args.docs)


if __name__ == "__main__":
    main()
