from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import faiss
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

BASE_DIR = Path(__file__).resolve().parents[1]
JSONL_PATH = BASE_DIR / "data" / "processed" / "class8_science.jsonl"
INDEX_PATH = BASE_DIR / "data" / "processed" / "faiss.index"
DOC_LIST_PATH = BASE_DIR / "data" / "processed" / "documents.pkl"

load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "all-MiniLM-L6-v2"
SYSTEM_PROMPT = (
    "You answer only from the provided context. Use simple Class 8 science language. "
    "Always name the source chapter. If the context does not contain the answer, return: "
    "I'm focused on Class 8 Science; try re-phrasing your question."
)
FALLBACK_TEXT = "I'm focused on Class 8 Science; try re-phrasing your question."

_MODEL = None
_INDEX = None
_DOCS: List[Dict[str, Any]] = []


def _ensure_assets() -> None:
    global _MODEL, _INDEX, _DOCS
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    if _INDEX is None:
        if not JSONL_PATH.exists():
            from src.ingest import ingest_documents
            ingest_documents(output_path=JSONL_PATH)
        if not INDEX_PATH.exists() or not DOC_LIST_PATH.exists():
            from src.index_build import build_index
            build_index(jsonl_path=JSONL_PATH, index_path=INDEX_PATH, doc_list_path=DOC_LIST_PATH)
        _INDEX = faiss.read_index(str(INDEX_PATH))
    if not _DOCS:
        with DOC_LIST_PATH.open("rb") as handle:
            _DOCS = pickle.load(handle)


def retrieve(query: str, k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
    _ensure_assets()
    if not _DOCS:
        return []
    query_embedding = _MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = _INDEX.search(query_embedding, min(k, len(_DOCS)))
    results: List[Tuple[Dict[str, Any], float]] = []
    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue
        results.append((_DOCS[int(index)], float(score)))
    return results


def _source_titles(documents: List[Dict[str, Any]]) -> List[str]:
    seen = []
    for doc in documents:
        meta = doc.get("meta", {})
        title = meta.get("chapter_title") or meta.get("chapter") or "Unknown chapter"
        if title not in seen:
            seen.append(title)
    return seen


def _call_gemini(context: str, query: str, max_retries: int = 3) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")
    if not api_key or not model_name or genai is None or types is None:
        raise RuntimeError("GEMINI_API_KEY or GEMINI_MODEL not set in .env")

    client = genai.Client(api_key=api_key)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                contents=f"Context:\n{context}\n\nQuestion: {query}",
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            raise RuntimeError("Gemini returned an empty response")
        except Exception as exc:
            msg = str(exc).lower()
            if ("429" in msg or "rate limit" in msg or "too many requests" in msg or "quota" in msg) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def _call_ollama(context: str, query: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    prompt = (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    resp = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def answer(
    query: str,
    k: int = 4,
    threshold: float = 0.3,
    backend: str = "gemini",
) -> Tuple[str, List[str]]:
    retrieved = retrieve(query, k=k)
    if not retrieved:
        return FALLBACK_TEXT, []

    top_doc, top_score = retrieved[0]
    if top_score < threshold:
        return FALLBACK_TEXT, []

    context_chunks = []
    for doc, _ in retrieved:
        chapter = doc.get("meta", {}).get("chapter_title") or doc.get("meta", {}).get("chapter") or "Unknown chapter"
        context_chunks.append(f"[{chapter}] {doc['content']}")
    context = "\n\n".join(context_chunks)
    sources = _source_titles([doc for doc, _ in retrieved])

    try:
        if backend == "ollama":
            text = _call_ollama(context, query)
        else:
            text = _call_gemini(context, query)
        return text, sources
    except Exception as exc:
        print(f"[rag] {backend} error: {exc}", file=sys.stderr)
        return f"⚠️ {backend.capitalize()} error: {exc}", sources
