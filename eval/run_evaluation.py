from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.rag import answer

BASE_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = BASE_DIR / "eval" / "test_questions.json"
OUTPUT_PATH = BASE_DIR / "eval" / "evaluation.csv"


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not hyp:
        return 0.0
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))
    log_avg = 0.0
    for n in range(1, 5):
        ref_counts = Counter(_ngrams(ref, n))
        hyp_counts = Counter(_ngrams(hyp, n))
        clipped = sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
        total = sum(hyp_counts.values())
        log_avg += math.log((clipped + 1) / (total + 1))
    return round(math.exp(log_avg / 4) * bp, 4)


def rouge_l(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ref[i - 1] == hyp[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def run_evaluation() -> None:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)

    rows = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        reference = item["reference"]
        print(f"[{i:02d}/{len(questions)}] {question[:65]}...")
        generated, sources = answer(question, backend="ollama")
        rows.append({
            "question":     question,
            "generated":    generated,
            "reference":    reference,
            "sources":      ", ".join(sources),
            "bleu":         bleu(reference, generated),
            "rougeL":       rouge_l(reference, generated),
            "human_review": "",
        })
        time.sleep(0.8)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows → {OUTPUT_PATH}")

    in_syllabus = df[~df["generated"].str.startswith("I'm focused")]
    print(f"In-syllabus answered : {len(in_syllabus)}/{len(df)}")
    if len(in_syllabus):
        print(f"Mean BLEU            : {in_syllabus['bleu'].mean():.4f}")
        print(f"Mean ROUGE-L         : {in_syllabus['rougeL'].mean():.4f}")


if __name__ == "__main__":
    run_evaluation()
