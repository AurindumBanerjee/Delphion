#!/usr/bin/env python
"""Draft a question set from the corpus so you only have to edit, not invent.

Samples chunks, asks the LLM for a question each chunk uniquely answers plus the
verbatim span that answers it, verifies the span really appears in the chunk,
and writes eval/questions.draft.json.

    python eval/bootstrap_questions.py --n 30

Review every question before renaming it to questions.json. A benchmark is only
worth the ground truth behind it, and the model will happily write a question
its own retriever finds trivially easy.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import normalize  # noqa: E402
from src.config import settings  # noqa: E402
from src.extract_from_pdf import load_and_chunk  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

PROMPT = (
    "You write evaluation questions for a document retrieval benchmark.\n"
    "Given one passage, return JSON with exactly two keys:\n"
    '  "question" — a specific question this passage answers, phrased as a user '
    "would ask it, without referring to 'the passage' or 'the document'\n"
    '  "answer_span" — a VERBATIM substring copied exactly from the passage '
    "(8-25 words) that contains the answer\n"
    "The span must appear character-for-character in the passage. "
    "Return only the JSON object."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="how many to draft")
    parser.add_argument("--method", default="recursive")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=400,
                        help="skip chunks too short to hold a real answer")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not settings.llm_configured:
        sys.exit("No LLM endpoint configured — set LLM_API_KEY (see .env.example).")

    from src.models import chat

    pdfs = sorted(str(p) for p in (EVAL_DIR / "corpus").glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {EVAL_DIR / 'corpus'}.")

    chunks = [c for c in load_and_chunk(pdfs, args.method, args.size, 200)
              if len(c.page_content) >= args.min_chars]
    if not chunks:
        sys.exit("No chunks long enough. Lower --min-chars.")

    random.seed(args.seed)
    sample = random.sample(chunks, min(args.n, len(chunks)))
    print(f"Drafting from {len(sample)} of {len(chunks)} chunks…\n")

    drafted, rejected = [], 0
    for i, chunk in enumerate(sample, 1):
        try:
            raw = chat(PROMPT, f"Passage:\n{chunk.page_content}")
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            span = data["answer_span"]

            # The model often paraphrases despite instructions. Verify.
            if normalize(span) not in normalize(chunk.page_content):
                rejected += 1
                print(f"  {i:>3}. rejected — span not verbatim")
                continue

            drafted.append({
                "question": data["question"],
                "answer_span": span,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            })
            print(f"  {i:>3}. {data['question'][:72]}")
        except Exception as exc:
            rejected += 1
            print(f"  {i:>3}. rejected — {type(exc).__name__}: {exc}")

    out = EVAL_DIR / "questions.draft.json"
    out.write_text(json.dumps(drafted, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(drafted)} drafted, {rejected} rejected → {out}")
    print("Edit it, delete the weak ones, then rename to eval/questions.json.")


if __name__ == "__main__":
    main()
