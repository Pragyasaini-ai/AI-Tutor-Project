# AI Tutor for NCERT Class 8 Science

An AI-powered, curriculum-focused tutor that answers NCERT Class 8 Science questions using Retrieval-Augmented Generation (RAG).

## Project Overview

This project helps students learn Class 8 Science by retrieving relevant information from NCERT textbook chapters and using a language model to generate simple, context-based answers.

The tutor is designed to stay focused on the Class 8 Science syllabus and identify the textbook chapters used to answer questions.

## Key Features

- **RAG-based question answering:** Retrieves relevant textbook passages before generating an answer.
- **Semantic search:** Uses sentence embeddings to find relevant content.
- **FAISS vector index:** Enables efficient similarity search.
- **Source chapter references:** Displays relevant chapter names.
- **Dual LLM support:** Supports Google Gemini and local Ollama models.
- **Out-of-syllabus fallback:** Returns a fallback response when retrieved content is not sufficiently relevant.
- **Command-line interface:** Lets students ask questions directly from the terminal.

## Technology Stack

- Python
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Google Gemini API
- Ollama
- PDFPlumber
- Pandas
- Python-dotenv

## Project Structure

```text
ai-tutor-class8/
├── data/
│   ├── raw/              # NCERT chapter PDFs
│   └── processed/        # Processed textbook text
├── eval/                 # Evaluation scripts and test questions
├── notebooks/            # Jupyter notebook
├── src/
│   ├── ingest.py         # Extracts and chunks PDF text
│   ├── index_build.py    # Builds the vector index
│   └── rag.py            # Retrieval and answer generation
├── cli.py                # Terminal-based tutor
├── requirements.txt      # Python dependencies
├── report.md             # Detailed project report
└── .env.example          # Example configuration
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Pragyasaini-ai/AI-Tutor-Project.git
cd AI-Tutor-Project
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the language model

Copy `.env.example` to `.env` and add your Gemini API key, or configure a locally installed Ollama model.

Never commit your `.env` file or expose API keys publicly.

### 5. Run the tutor

```bash
python cli.py
```

The first run may download the sentence-transformer model and build the required vector index.

## Evaluation

The project includes evaluation scripts and test questions under the `eval/` directory. The evaluation can be run and reviewed to assess the tutor's answer quality.

## Limitations

- Answer quality depends on the textbook content retrieved and the selected language model.
- A generated answer may still contain errors and should be checked against the textbook.
- Running local language models requires sufficient system resources.
- API-based generation requires a valid API key and may be subject to usage limits.

## Future Improvements

- Add a Streamlit or Gradio web interface.
- Improve answer evaluation with reference-based metrics and human review.
- Add more curriculum-specific subjects and grades.
- Improve retrieval quality and citation detail.

## Author

**Pragya Saini**

GitHub: https://github.com/Pragyasaini-ai
