# Retrieval benchmark

Three chunking strategies were built and never compared. This measures them.

```bash
pip install -r requirements.txt
cp your-documents/*.pdf eval/corpus/
python eval/bootstrap_questions.py --n 30     # drafts questions (needs an LLM key)
#   → review eval/questions.draft.json, delete the weak ones,
#     rename it to eval/questions.json
python eval/run_eval.py                        # runs the sweep — free, local, offline
```

Output goes to `eval/results/<timestamp>.{md,json}` and `eval/results/latest.md`.
The markdown table is meant to be pasted straight into the root README.

## Retrieval modes

Each chunking configuration is indexed once and queried four ways — `dense`,
`bm25`, `hybrid` (RRF) and `hybrid_rerank` — so the before/after comparison
holds chunking constant and isolates the retrieval change. Pick a subset with
`--modes dense hybrid_rerank`. The run also writes the results table into the
root README between the `BENCHMARK` markers.

`--judge` adds answer-level scoring (needs an LLM key): correctness against the
reference span and faithfulness (every claim supported by the passages).

## Ground truth is a span, not a chunk id

Each question carries a short **verbatim substring** of the source document:

```json
{
  "question": "What throughput did the cluster reach across eight nodes?",
  "answer_span": "1.4 million operations per second across eight nodes",
  "source": "benchmark-report.pdf",
  "page": 7
}
```

This is the design decision the whole harness rests on. The obvious approach —
recording which chunk is the right one — breaks immediately, because chunk ids
change every time you alter the chunking strategy or size. That is precisely the
variable being tested, so the question set would invalidate itself on the first
comparison. A span survives re-chunking, so one question set measures every
configuration.

A chunk counts as relevant if the span appears in it (whitespace- and
case-normalised, since PDF extraction inserts line breaks mid-sentence), or if
`source` and `page` both match when no span is given.

## Metrics

| Metric | Question it answers |
|---|---|
| **hit@k** | Did the right passage make the cut at all? |
| **MRR** | How near the top was it? Rewards ranking it first, not just including it. |
| **precision@k** | How much of what we retrieved was noise? |
| **nDCG@k** | Rank-discounted gain, normalised to one relevant passage at rank 1. |

hit@k is the headline number; MRR is the one that moves when you add reranking.

## Interpreting it

- **hit@1 low but hit@5 high** — retrieval finds the passage but ranks it badly.
  That is a reranking problem, not an embedding problem.
- **hit@5 low across every config** — the embedding model or the question set is
  the issue, not chunking. Check the `Still missed` list first; badly worded
  questions are the usual cause.
- **Configs within ~0.05 of each other** — with 25-40 questions that is noise.
  Prefer the cheaper configuration and say so.
- **semantic barely beating recursive** — likely the correct result. Semantic
  chunking costs an embedding pass per sentence; if it does not buy a real gain,
  ship recursive and be able to explain why.

## Costs

Embedding runs locally on CPU, so the sweep is free and works offline. Only
`bootstrap_questions.py` and `run_eval.py --judge` call an LLM — one request per
question each. At a typical free-tier limit of ~30 requests/minute, a 30-question
draft takes about a minute.

## Caveats

- 25-40 questions is enough to rank configurations, not to publish a number.
  Report it as a comparison, not an absolute.
- The bootstrapper writes questions from a passage it can see, which biases
  toward questions retrieval finds easy. Editing the draft is not optional —
  rewrite questions to use the vocabulary a real user would, not the document's.
- Every question here is answerable from a single passage. Multi-hop questions
  need a different ground-truth format.
