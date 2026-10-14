# AI Tutor for NCERT Class 8 Science
### A Retrieval-Augmented Generation (RAG) System

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Component Deep-Dive](#5-component-deep-dive)
   - 5.1 [Text Ingestion (ingest.py)](#51-text-ingestion-ingestpy)
   - 5.2 [Vector Indexing (index_build.py)](#52-vector-indexing-index_buildpy)
   - 5.3 [RAG Engine (rag.py)](#53-rag-engine-ragpy)
   - 5.4 [Terminal Interface (cli.py)](#54-terminal-interface-clipy)
6. [Data Pipeline](#6-data-pipeline)
7. [Retrieval Strategy](#7-retrieval-strategy)
8. [Answer Generation — Dual Backend](#8-answer-generation--dual-backend)
9. [Out-of-Syllabus Fallback](#9-out-of-syllabus-fallback)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [Project Structure](#11-project-structure)
12. [Setup and Usage](#12-setup-and-usage)
13. [Design Decisions and Trade-offs](#13-design-decisions-and-trade-offs)
14. [Limitations and Future Work](#14-limitations-and-future-work)

---

## 1. Project Overview

This project builds a **curriculum-locked AI tutor** for NCERT Class 8 Science. Given a question from a student, the system retrieves the most relevant passages from the official Class 8 Science textbook and uses a large language model to compose a simple, accurate, grade-appropriate answer — always citing the source chapter.

The system supports **two interchangeable generation backends**:
- **Gemini Flash** (Google, cloud, free tier) — ~1,500 requests/day at no cost
- **Ollama** (local, fully offline) — no API key or internet connection required

Retrieval always runs **locally** using sentence-transformers and FAISS, so no textbook content is ever sent to an embedding API. The interface is a **terminal REPL** (`cli.py`) — no browser or web server required.

**Corpus stats (real PDFs):** 13 NCERT chapters → 370 chunks → 555 KB FAISS index

---

## 2. Problem Statement

Generic large language models (LLMs) hallucinate, cover topics far beyond the school syllabus, and cannot cite which page of a textbook an answer came from. For a Class 8 student:

- Answers must be **factually grounded** in the NCERT textbook
- Language must be **simple and age-appropriate**
- The tutor must **refuse** out-of-syllabus questions gracefully rather than fabricating answers
- The source chapter must always be **cited** so a student can verify

Retrieval-Augmented Generation (RAG) addresses all four requirements: the LLM is shown only the retrieved textbook passages as context and is instructed not to go beyond them.

---

## 3. System Architecture

```
 Student Question (terminal input)
       │
       ▼
 ┌─────────────┐
 │  Embed query │  ← all-MiniLM-L6-v2 (local, 384-dim, no API call)
 └──────┬──────┘
        │ query vector
        ▼
 ┌─────────────────┐
 │  FAISS Search   │  ← IndexFlatIP (cosine similarity, exact)
 │  top-k chunks   │
 └──────┬──────────┘
        │ top score < 0.3 ?
        ├──── YES ──► Fallback: "I'm focused on Class 8 Science…" (no LLM call)
        │
        │ top score ≥ 0.3
        ▼
 ┌──────────────────────────┐
 │  Build context block     │
 │  [Chapter] chunk text × k│
 └──────┬───────────────────┘
        │
        ├── backend = "gemini" ──► Gemini Flash API (cloud, free tier)
        │
        └── backend = "ollama" ──► Ollama REST API (localhost, fully offline)
                │
                ▼ (both paths use the same system prompt)
        System: answer from context only, cite chapter,
                Class-8 language, fallback if context is insufficient
                │
                ▼
 Answer + Deduplicated Source Chapter List → terminal output
```

The pipeline is **two-phase at build time** (ingest → index) and **single-pass at query time** (embed → retrieve → generate). The backend can be switched mid-session with `/switch`.

---

## 4. Technology Stack

| Component | Library / Tool | Why |
|---|---|---|
| PDF extraction | `pdfplumber` | Reliable column-aware text extraction from NCERT PDFs |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Free, runs locally, 384-dim, strong semantic similarity for English |
| Vector store | `faiss-cpu` `IndexFlatIP` | Exact cosine search, no server, deterministic |
| LLM — cloud | `google-genai` Gemini Flash | Free tier (~1,500 req/day), fast, strong instruction-following |
| LLM — local | Ollama REST API (`requests`) | Fully offline, no API key, works with any pulled model |
| Interface | `cli.py` (stdlib only) | Lightweight terminal REPL, no browser or server dependency |
| Evaluation | `rouge-score`, `nltk` | Standard NLP metrics for QA systems |
| Config | `python-dotenv` | Keeps API keys out of source code |
| Setup | `start.bat` | One-click Windows setup and launch script |

**Key design principle:** embeddings and retrieval always run **locally and offline**. The LLM is the only network-dependent component, and even that can be replaced with a local Ollama model.

---

## 5. Component Deep-Dive

### 5.1 Text Ingestion (`ingest.py`)

**Input:** PDF files in `data/raw/` named `ch01.pdf` … `ch13.pdf`

**Steps:**

1. **Extract** — `pdfplumber` opens each PDF and extracts raw text page by page.

2. **Clean** — A regular-expression pass removes:
   - Bare page numbers (lines that are just a digit string)
   - NCERT rationalisation notices (`Rationalised 2023-24`)
   - Page-number separators (`42 | 43`)
   - Copyright symbols
   - Excess whitespace and blank lines
   - All remaining lines are joined into a single space-separated string

3. **Chunk** — The cleaned text is windowed into overlapping passages:
   - Window size: **250 words**
   - Overlap: **50 words** (the last 50 words of chunk N become the first 50 of chunk N+1)
   - Overlap prevents facts that straddle a boundary from being lost

4. **Write** — Each chunk is written as one JSON line to `data/processed/class8_science.jsonl`:
   ```json
   {"id": "ch03_7", "content": "Synthetic fibres are made from chemical substances...",
    "meta": {"chapter": "ch03", "chapter_title": "Synthetic Fibres and Plastics"}}
   ```

**Demo fallback:** If `data/raw/` contains no PDFs, the function generates one synthetic chunk per chapter so all downstream components still run end-to-end.

**Real corpus result:** 13 PDFs → **370 chunks** (~250 words each)

---

### 5.2 Vector Indexing (`index_build.py`)

**Input:** `data/processed/class8_science.jsonl`

**Steps:**

1. Load all chunks from the JSONL file into memory.
2. Encode every chunk's `content` field using `SentenceTransformer("all-MiniLM-L6-v2")` with `normalize_embeddings=True`. Normalisation ensures every vector has unit L2 length, so inner-product distance equals cosine similarity directly.
3. Build a `faiss.IndexFlatIP` (Inner Product) index and add all 370 normalised vectors.
4. Save two artifacts:
   - `data/processed/faiss.index` — the binary FAISS index (555 KB)
   - `data/processed/documents.pkl` — the ordered list of doc dicts, so index row *N* maps back to `documents[N]`

**Why IndexFlatIP?** Exact (not approximate) search. For 370 chunks, brute-force search completes in under a millisecond on CPU and never misses the closest vector.

---

### 5.3 RAG Engine (`rag.py`)

The core module. Assets are loaded lazily on first call and auto-bootstrapped if missing.

#### `retrieve(query, k=4)`

1. Encodes the query with the same normalised embedding model.
2. Runs `index.search(query_vector, k)` → top-k (doc, cosine_score) pairs.
3. Returns `[(doc_dict, score), …]` in descending score order.

#### `answer(query, k=4, threshold=0.3, backend="gemini")`

1. Calls `retrieve()`.
2. **Fallback gate:** If the top cosine score < `0.3`, the question is too far from any textbook content — returns the fallback string immediately with no LLM call.
3. Builds a labelled context block:
   ```
   [Friction] Friction is the force that opposes relative motion between...

   [Sound] Sound is produced by vibrating objects. The vibrations create...
   ```
4. Dispatches to the selected backend:
   - `backend="gemini"` → `_call_gemini()` via `google-genai` SDK
   - `backend="ollama"` → `_call_ollama()` via HTTP POST to `localhost:11434`
5. Both backends use the same system prompt and context format.
6. Gemini path handles HTTP 429 with exponential backoff (2 s, 4 s, 6 s).
7. Any exception is printed to stderr and surfaced in the answer as `⚠️ <backend> error: …` so it is never silently swallowed.
8. Returns `(answer_text, deduplicated_source_chapter_titles)`.

**System prompt:**
> You answer only from the provided context. Use simple Class 8 science language. Always name the source chapter. If the context does not contain the answer, return: I'm focused on Class 8 Science; try re-phrasing your question.

---

### 5.4 Terminal Interface (`cli.py`)

A minimal interactive REPL — no web framework, no browser dependency.

**Session flow:**

```
$ python cli.py

 ============================================
  NCERT Class 8 Science | AI Tutor (CLI)
 ============================================

Select LLM backend:
  [1] Gemini (Cloud)
  [2] Ollama (Local)
Choice [1/2, default=1]: _

You: What causes friction?

Tutor: Friction is caused by the interlocking of irregularities on the
       surfaces in contact... (Chapter: Friction)

  Sources: Friction
──────────────────────────────────────────────────
You: _
```

**In-session commands:**

| Command | Action |
|---|---|
| Any question | Get a grounded answer |
| `/switch` | Toggle between Gemini and Ollama without restarting |
| `/quit` or `/exit` | Exit the tutor |
| `Ctrl+C` | Emergency exit |

---

## 6. Data Pipeline

```
data/raw/ch01.pdf  ┐
data/raw/ch02.pdf  │
       …           ├── src/ingest.py ──► data/processed/class8_science.jsonl (370 chunks)
data/raw/ch13.pdf  ┘                                     │
                                                          ▼
                                             src/index_build.py
                                                          │
                                        ┌─────────────────┴──────────────────┐
                                        ▼                                     ▼
                             data/processed/faiss.index          data/processed/documents.pkl
                             (555 KB, 370 × 384-dim vectors)     (ordered doc list, pickle)
```

Both `ingest.py` and `index_build.py` are importable modules and runnable as scripts:

```bash
python -m src.ingest
python -m src.index_build
```

`rag.py` auto-triggers both if their outputs are missing, so a fresh run of `cli.py` will build the index on first query without manual intervention.

---

## 7. Retrieval Strategy

**Model:** `all-MiniLM-L6-v2`
- Trained specifically for semantic sentence similarity
- 384-dimensional embeddings — compact for CPU inference, rich enough for the task
- Downloads once (~90 MB) then runs fully offline

**Similarity measure:** Cosine similarity via normalised inner product. Score `1.0` = identical direction; `0.0` = orthogonal (no semantic overlap).

**Threshold (`0.3`):** Questions from Class 8 Science reliably score above 0.35 against relevant chunks. General-knowledge questions (capitals, planet counts, historical facts) score below 0.25 against science text — safely below the gate.

**k=4 chunks:** 4 × 250 words ≈ 1,000 words of context. Enough detail for multi-step answers without overloading the prompt length.

**Deduplication:** Source chapter titles in the output are deduplicated — if three retrieved chunks come from the same chapter, only one citation is shown.

---

## 8. Answer Generation — Dual Backend

The generation layer is fully swappable at runtime via the `backend` parameter.

### Gemini (Cloud)

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL"),          # e.g. gemini-2.5-flash
    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    contents=f"Context:\n{context}\n\nQuestion: {query}",
)
```

- Requires `GEMINI_API_KEY` (free personal key from `aistudio.google.com`, starts with `AIza`)
- Model ID stored in `GEMINI_MODEL` in `.env` — verify in AI Studio as IDs change
- Rate limit: ~10–15 req/min on free tier; 429 errors are retried with backoff

### Ollama (Local)

```python
import requests

resp = requests.post(
    f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate",
    json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "prompt": prompt, "stream": False},
    timeout=120,
)
text = resp.json()["response"].strip()
```

- Requires [Ollama](https://ollama.com) installed and running locally
- Pull any model: `ollama pull llama3.2`
- No API key, no internet dependency during queries
- `OLLAMA_MODEL` in `.env` controls which local model is used

**Switching backends** mid-session: type `/switch` in the CLI — no restart needed.

---

## 9. Out-of-Syllabus Fallback

The system uses a two-layer fallback:

**Layer 1 — Cosine threshold gate (`rag.py`):**
If `top_score < 0.3`, no LLM call is made. Returns immediately:
> "I'm focused on Class 8 Science; try re-phrasing your question."

**Layer 2 — LLM instruction fallback:**
The system prompt instructs both Gemini and Ollama to return the same fallback string if the retrieved context does not actually answer the question. This catches cases where a question is semantically close to science vocabulary but not answerable from the retrieved text.

**Test questions that trigger the fallback** (from `eval/test_questions.json`):
- "What is the capital of India?"
- "How many planets are in the solar system?"
- "Who wrote the national anthem of India?"

---

## 10. Evaluation Methodology

**Test set:** 20 questions in `eval/test_questions.json`
- 16 in-syllabus questions covering all 13 chapters
- 4 out-of-syllabus questions to verify the fallback

**Metrics computed per question:**

| Metric | Formula | Interpretation |
|---|---|---|
| BLEU | Modified n-gram precision (with smoothing) | Measures exact word-overlap. Penalises paraphrase even when meaning is correct. |
| ROUGE-L | F-measure of longest common subsequence | Captures word-order similarity. More robust to paraphrase than BLEU. |

**Implementation:**
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

bleu   = sentence_bleu([ref.split()], gen.split(), smoothing_function=SmoothingFunction().method1)
rougeL = scorer.score(ref, gen)["rougeL"].fmeasure
```

**Output:** `eval/evaluation.csv` — columns: `question, generated, reference, sources, bleu, rougeL, human_review`

The `human_review` column is intentionally blank. BLEU and ROUGE-L underestimate quality for free-form answers, so manual review is essential for a full assessment.

**Running the evaluation:**
```bash
venv\Scripts\python eval\run_evaluation.py
# Results written to eval/evaluation.csv
```

---

## 11. Project Structure

```
ai-tutor-class8/
├── data/
│   ├── raw/                        # Place ch01.pdf … ch13.pdf here
│   └── processed/
│       ├── class8_science.jsonl    # Chunked corpus — 370 chunks (generated)
│       ├── faiss.index             # FAISS binary index, 555 KB (generated)
│       └── documents.pkl           # Ordered doc list (generated)
├── notebooks/
│   └── ai_tutor_class8.ipynb      # End-to-end walkthrough notebook
├── src/
│   ├── __init__.py
│   ├── ingest.py                   # PDF → clean → chunk → JSONL
│   ├── index_build.py              # Embed corpus → FAISS index
│   └── rag.py                      # retrieve() + answer() (Gemini & Ollama)
├── eval/
│   ├── test_questions.json         # 20 Q&A pairs (16 in + 4 out-of-syllabus)
│   ├── run_evaluation.py           # Batch BLEU/ROUGE-L evaluation
│   └── evaluation.csv             # Generated results (gitignored)
├── cli.py                          # Terminal REPL — main entry point
├── start.bat                       # One-click Windows setup + launch
├── report.md                       # This document
├── requirements.txt                # Python dependencies (no Streamlit)
├── .env                            # API keys (gitignored)
└── .env.example                    # Config template
```

---

## 12. Setup and Usage

### Option A — Windows (recommended): one-click setup

1. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY` (or leave blank to use Ollama only).
2. Drop `ch01.pdf` … `ch13.pdf` into `data/raw/`.
3. Double-click **`start.bat`**.

`start.bat` automatically:
- Creates a Python virtual environment (`venv/`) if it does not exist
- Installs all dependencies from `requirements.txt`
- Warns if `GEMINI_API_KEY` is missing
- Launches `cli.py`

On every subsequent run, double-clicking `start.bat` goes straight to the tutor (setup is skipped).

---

### Option B — Manual setup (any OS)

**Step 1 — Install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

**Step 2 — Configure `.env`**
```
# Gemini (cloud) — free key from aistudio.google.com
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash

# Ollama (local) — install from ollama.com, then: ollama pull llama3.2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**Step 3 — Add PDFs**
Place `ch01.pdf` … `ch13.pdf` in `data/raw/`. If omitted, a built-in demo corpus is used.

**Step 4 — Build the index** *(one-time; auto-runs on first query if skipped)*
```bash
python -m src.ingest
python -m src.index_build
```

**Step 5 — Run the tutor**
```bash
python cli.py
```

**Running the evaluation:**
```bash
python eval/run_evaluation.py
```

**Running the notebook:**
```bash
jupyter notebook notebooks/ai_tutor_class8.ipynb
```

---

## 13. Design Decisions and Trade-offs

| Decision | Alternative considered | Reason for choice |
|---|---|---|
| Terminal REPL (cli.py) | Streamlit web UI | No browser or server dependency; lighter install; easier to zip and share |
| Dual backend (Gemini + Ollama) | Gemini only | Ollama enables fully offline use; single `/switch` command to toggle |
| Local embeddings (all-MiniLM-L6-v2) | Google text-embedding-004 (free) | Runs offline, no quota consumed, no textbook content sent to any API |
| IndexFlatIP (exact search) | HNSW (approximate) | Corpus is 370 chunks — exact search is sub-millisecond and never misses a result |
| Cosine threshold = 0.3 | Learned classifier | Simple, interpretable, no training data needed; validated against 20 test questions |
| k = 4 retrieved chunks | k = 2 or k = 8 | 4 × 250 words ≈ 1,000 words of context — sufficient detail without overloading the prompt |
| 250-word chunks / 50-word overlap | Sentence-level or character chunking | Word-level windows are language-agnostic and map naturally to one topic per chunk |
| `start.bat` for setup | Makefile / shell script | Windows-native; no additional tools (make, bash) required |

---

## 14. Limitations and Future Work

**Current limitations:**

1. **Demo corpus quality** — Without real PDFs the demo corpus has one generic sentence per chapter, producing vague answers. Always use real NCERT PDFs for meaningful results.

2. **BLEU/ROUGE limitations** — Both metrics penalise paraphrase even when meaning is preserved. The blank `human_review` column in `evaluation.csv` exists for this reason.

3. **No conversation memory** — Each question is answered independently. Follow-up questions that rely on a previous answer are not supported.

4. **Image and diagram content** — NCERT textbooks contain important diagrams. `pdfplumber` extracts only text; diagram labels and figure descriptions may be incomplete.

5. **Ollama response speed** — Local models are slower than the Gemini API, especially on CPU. Response time depends on the hardware and model size.

**Potential improvements:**

- **Multi-turn context:** Append the last N turns to the prompt so the tutor can handle follow-up questions
- **BM25 hybrid retrieval:** Combine dense (embedding) and sparse (keyword) retrieval with reciprocal rank fusion for better recall on exact-term questions
- **Diagram descriptions:** Use a vision model to generate text descriptions of figures and include them in the corpus
- **Cross-encoder reranking:** Add a reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) after FAISS retrieval to improve precision
- **Response caching:** Cache answers to repeated questions to reduce API calls and latency

---

*Built using NCERT Class 8 Science textbook content. All answers are grounded in official curriculum material.*
