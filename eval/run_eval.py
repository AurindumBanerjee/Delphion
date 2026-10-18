#!/usr/bin/env python
"""Retrieval evaluation sweep.

Answers the question the three chunking strategies were built to raise and
never measured: which one actually retrieves the right passage?

    python eval/run_eval.py                      # default sweep
    python eval/run_eval.py --methods recursive --sizes 500 1000 --k 1 3 5
    python eval/run_eval.py --judge              # also score answer quality (uses the LLM)
    python eval/run_eval.py --modes dense hybrid_rerank   # before/after comparison

Put PDFs in eval/corpus/ and ground truth in eval/questions.json first.
Embedding is local and free, so the sweep costs nothing but time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import aggregate, is_relevant, score  # noqa: E402
from src.extract_from_pdf import load_and_chunk  # noqa: E402
from src.retrieval import HybridIndex  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
RESULTS = EVAL_DIR / "results"


def load_questions(path: Path):
    if not path.exists():
        sys.exit(
            f"No question set at {path}.\n"
            f"Copy eval/questions.example.json to eval/questions.json and edit it, "
            f"or run: python eval/bootstrap_questions.py"
        )
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not questions:
        sys.exit("Question set is empty.")
    return questions


def corpus_paths():
    pdfs = sorted(str(p) for p in CORPUS.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {CORPUS}. Add the documents your questions refer to.")
    return pdfs


def evaluate(store, questions, k_values, mode="dense"):
    """Retrieve once at the largest k, then score every smaller cutoff from it."""
    max_k = max(k_values)
    per_k = {k: [] for k in k_values}
    misses = []
    latencies = []

    for question in questions:
        t0 = time.perf_counter()
        docs = store.search(question["question"], k=max_k, mode=mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        for k in k_values:
            per_k[k].append(score(docs, question, k))

        if not any(is_relevant(d, question) for d in docs):
            misses.append(
                {
                    "question": question["question"],
                    "expected": question.get("answer_span", "")[:90],
                    "got": [
                        {
                            "source": d.metadata.get("source"),
                            "page": d.metadata.get("page"),
                            "preview": " ".join(d.page_content.split())[:90],
                        }
                        for d in docs[:3]
                    ],
                }
            )

    latencies.sort()
    lat = {"p50_ms": latencies[len(latencies) // 2],
           "p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]}
    return {k: aggregate(rows) for k, rows in per_k.items()}, misses, lat


JUDGE_PROMPT = (
    "You grade a RAG answer. Reply with exactly two words separated by a space.\n"
    "Word 1 (correctness vs the reference): CORRECT, PARTIAL or WRONG.\n"
    "Word 2 (faithfulness: is every claim supported by the passages?): FAITHFUL or UNFAITHFUL."
)


def judge(store, questions, k, mode):
    """Optional answer-quality pass. Costs one LLM call per question."""
    from src.config import settings
    from src.rag import answer

    if not settings.llm_configured:
        print("  (skipping --judge: no LLM endpoint configured)")
        return None

    from src.models import chat
    from src.rag import format_context, retrieve

    correct, faithful = [], []
    for question in questions:
        docs = retrieve(store, question["question"], k, mode)
        reply, _ = answer(store, question["question"], k, mode=mode)
        verdict = chat(
            JUDGE_PROMPT,
            f"Question: {question['question']}\n"
            f"Reference: {question.get('answer_span', '')}\n"
            f"Passages:\n{format_context(docs)}\n\n"
            f"Answer: {reply}",
        )
        words = (verdict or "").strip().upper().split()
        correct.append({"CORRECT": 1.0, "PARTIAL": 0.5}.get(words[0] if words else "", 0.0))
        faithful.append(1.0 if len(words) > 1 and words[1].startswith("FAITHFUL") else 0.0)
    n = len(questions) or 1
    return {"correct": sum(correct) / n, "faithful": sum(faithful) / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["recursive", "semantic", "markdown"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 1000, 2000])
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json")
    parser.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid", "hybrid_rerank"],
                        help="retrieval modes to compare: dense bm25 hybrid hybrid_rerank")
    parser.add_argument("--judge", action="store_true", help="also grade generated answers")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    pdfs = corpus_paths()
    k_values = sorted(set(args.k))

    print(f"{len(questions)} questions · {len(pdfs)} PDF(s) · "
          f"{len(args.methods) * len(args.sizes)} chunk configs × {len(args.modes)} modes\n")

    reranker = None
    if "hybrid_rerank" in args.modes:
        from src.models import get_reranker

        reranker = get_reranker()
        if reranker is None:
            sys.exit("hybrid_rerank requested but the reranker model could not be loaded.")

    runs = []
    for method in args.methods:
        for size in args.sizes:
            label = f"{method}/{size}"
            started = time.perf_counter()
            try:
                chunks = load_and_chunk(pdfs, method, size, args.overlap)
                if not chunks:
                    print(f"  {label:<20} no chunks produced — skipped")
                    continue
                store = HybridIndex.build(chunks, reranker=reranker)
            except Exception as exc:
                print(f"  {label:<20} FAILED: {exc}")
                continue

            build = time.perf_counter() - started
            for mode in args.modes:
                by_k, misses, lat = evaluate(store, questions, k_values, mode)
                run = {
                    "method": method,
                    "chunk_size": size,
                    "mode": mode,
                    "overlap": args.overlap,
                    "chunks": len(store.docs),
                    "build_seconds": round(build, 1),
                    "latency": lat,
                    "metrics": {str(k): by_k[k] for k in k_values},
                    "misses": misses,
                }
                if args.judge:
                    run["answer_score"] = judge(store, questions, 4, mode)
                runs.append(run)

                best = by_k[max(k_values)]
                print(f"  {label:<16} {mode:<14} {len(store.docs):>5} chunks  "
                      f"hit@1={by_k[min(k_values)]['hit']:.2f}  "
                      f"hit@{max(k_values)}={best['hit']:.2f}  mrr={best['mrr']:.2f}  "
                      f"p50={lat['p50_ms']:.0f}ms")

    if not runs:
        sys.exit("\nNo configuration completed.")

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    report = render(runs, k_values, len(questions))
    (RESULTS / f"{stamp}.md").write_text(report, encoding="utf-8")
    (RESULTS / "latest.md").write_text(report, encoding="utf-8")

    readme = ROOT / "README.md"
    text = readme.read_bytes().decode("utf-8").replace("\r\n", "\n")
    start, end = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"
    if start in text and end in text:
        body = report.replace("## Retrieval benchmark\n\n", "", 1)
        head, rest = text.split(start, 1)
        readme.write_text(head + start + "\n" + body + end + rest.split(end, 1)[1],
                          encoding="utf-8", newline="\n")
        print("README.md results section updated.")

    print("\n" + report)
    print(f"Written to eval/results/{stamp}.md")


def render(runs, k_values, n_questions):
    """Markdown report — paste the tables straight into the README."""
    headline = 5 if 5 in k_values else max(k_values)
    tag = lambda r: f"`{r['method']}` @ {r['chunk_size']}"

    lines = [
        "## Retrieval benchmark",
        "",
        f"{n_questions} questions · embeddings `BAAI/bge-small-en-v1.5` (local) · "
        f"ground truth is a verbatim span, so it survives re-chunking.",
        "",
        "| Chunking | Retrieval | Chunks | " + " | ".join(f"hit@{k}" for k in k_values)
        + f" | MRR@{headline} | nDCG@{headline} | p50 query |",
        "|---|---|---:|" + "---:|" * len(k_values) + "---:|---:|---:|",
    ]
    ranked = sorted(runs, key=lambda r: (-r["metrics"][str(headline)]["mrr"],
                                         -r["metrics"][str(headline)]["hit"]))
    for run in ranked:
        cells = [f"{run['metrics'][str(k)]['hit']:.2f}" for k in k_values]
        head = run["metrics"][str(headline)]
        lines.append(
            f"| {tag(run)} | {run['mode']} | {run['chunks']} | " + " | ".join(cells)
            + f" | {head['mrr']:.2f} | {head['ndcg']:.2f} | {run['latency']['p50_ms']:.0f} ms |"
        )

    # Before / after: same chunking, baseline mode vs best mode.
    modes = list(dict.fromkeys(r["mode"] for r in runs))
    if len(modes) > 1:
        base, new = modes[0], modes[-1]
        lines += ["", f"### Before ({base}) → after ({new})", "",
                  f"| Chunking | hit@1 | hit@{headline} | MRR@{headline} |", "|---|---|---|---|"]
        by = {(r["method"], r["chunk_size"], r["mode"]): r for r in runs}
        for (m, c, mode), r in by.items():
            if mode != base or (m, c, new) not in by:
                continue
            a, b = r["metrics"], by[(m, c, new)]["metrics"]
            lo, hi = str(min(k_values)), str(headline)
            fmt = lambda x, y: f"{x:.2f} → **{y:.2f}** ({y - x:+.2f})"
            lines.append(f"| {tag(r)} | {fmt(a[lo]['hit'], b[lo]['hit'])} | "
                         f"{fmt(a[hi]['hit'], b[hi]['hit'])} | {fmt(a[hi]['mrr'], b[hi]['mrr'])} |")

    judged = [r for r in ranked if r.get("answer_score")]
    if judged:
        lines += ["", "| Chunking | Retrieval | Answer correct | Faithful |", "|---|---|---:|---:|"]
        for run in judged:
            sc = run["answer_score"]
            lines.append(f"| {tag(run)} | {run['mode']} | {sc['correct']:.2f} | {sc['faithful']:.2f} |")

    winner = ranked[0]
    lines += ["", f"**Best:** {tag(winner)} with {winner['mode']} — "
              f"hit@{headline} {winner['metrics'][str(headline)]['hit']:.2f}, "
              f"MRR@{headline} {winner['metrics'][str(headline)]['mrr']:.2f}."]
    if winner["misses"]:
        lines += ["", f"**Still missed in top {max(k_values)} ({len(winner['misses'])}):**", ""]
        lines += [f"- {m['question']}" for m in winner["misses"][:8]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
