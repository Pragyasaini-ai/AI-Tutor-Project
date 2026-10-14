from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag import answer

BANNER = """
╔══════════════════════════════════════════════╗
║    NCERT Class 8 Science — AI Tutor (CLI)   ║
╚══════════════════════════════════════════════╝
Type your question and press Enter.
Commands:  /switch   toggle Gemini ↔ Ollama
           /quit     exit
"""


def pick_backend() -> str:
    print("Select LLM backend:")
    print("  [1] Gemini (Cloud)  — requires GEMINI_API_KEY in .env")
    print("  [2] Ollama (Local)  — requires Ollama running locally")
    while True:
        choice = input("Choice [1/2, default=1]: ").strip()
        if choice in ("", "1"):
            return "gemini"
        if choice == "2":
            return "ollama"
        print("Please enter 1 or 2.")


def main() -> None:
    print(BANNER)
    backend = pick_backend()
    print(f"\nUsing backend: {backend.upper()}")
    print("─" * 50)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if query.lower() == "/switch":
            backend = "ollama" if backend == "gemini" else "gemini"
            print(f"Switched to: {backend.upper()}")
            continue

        print("\nTutor: ", end="", flush=True)
        ans, sources = answer(query, backend=backend)
        print(ans)
        if sources:
            print(f"\n  Sources: {', '.join(sources)}")
        print("─" * 50)


if __name__ == "__main__":
    main()
