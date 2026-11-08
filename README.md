# Offline Wikipedia RAG — domain benchmark

A local, evidence-grounded evaluation of a small RAG stack on an offline English
Wikipedia ZIM. The purpose is to show **where the pipeline works and where it
fails by domain**, not to claim a Wikipedia-wide ranking.

## Stack

- Device: Steam Deck (Arch/Omarchy, 8 threads, 11.5 GiB RAM, 4 GiB VRAM).
- Corpus: `wikipedia_en_all_nopic_2026-06.zim`
  (`52,690,706,555` bytes; images removed).
- Serving: `kiwix-serve` on `127.0.0.1:8080`.
- Generator: LFM3-1.7B-Q4_K_M via llama.cpp/Vulkan (`127.0.0.1:11437`).
- Embedding: `nomic-embed-text` via Ollama (`127.0.0.1:11434`).
- Runtime: ~2–7 s per generated answer, depending on context/model state.

## Dataset

- 8 domains: mathematics, physics/astronomy, chemistry/materials,
  life/medicine, geography/environment, history, social science, arts/humanities.
- Per domain: 40 test-index articles + 10 dev-index articles + 6 extra
  sources. 448 articles were fetched; 320 test-index, 80 dev-index.
- Test set: **193 questions** — 160 answerable + 33 unanswerable.
- Dev set: **79 questions** — 64 answerable + 15 unanswerable.
- Article selection: six broad search queries per domain, five result pages
  each; stable hash sampling, with cross-domain de-duplication. This is a
  **search-coverage sample**, not a uniform Wikipedia sample.
- Questions: short extractive/fill-in-the-blank questions generated
  deterministically from source sentences; each has a source span, accepted
  answer(s), and gold evidence chunk(s). No model was used to generate labels.
- The test index is separate from the dev index. Unanswerable source articles
  are excluded from the split index and the target string is checked against
  every chunk in that split.

## Arms

- `base`: closed-book LFM3 (no retrieved context).
- `retrieval`: embedding-only retrieval metrics; no generation.
- `embed`: LFM3 + top-5 retrieved chunks, JSON answer + chunk citations.

## Metrics

- Answerable `answer_ok`: normalized exact accepted answer **and** at least one
  valid citation covering the gold evidence.
- Unanswerable `answer_ok`: `abstain=true`, empty answer, no citations.
- Also: exact match, token F1, citation coverage, abstention accuracy,
  article hit, evidence hit, MRR, latency.
- Results are reported per domain as well as overall.

## Reproduce

```bash
# on the Deck
cd ~/deck-rag/pilot3
python3 scripts/select_wiki.py
python3 scripts/fetch_wiki.py
python3 scripts/build_cases.py
EMBED_MODEL=nomic-embed-text python3 scripts/embed_index.py
python3 scripts/run_wiki.py --index index/test_embedded.json \
  --cases cases/test.jsonl --arms base,retrieval,embed --out runs/locked-v1
```

## Results

See [`reports/RESULTS.md`](reports/RESULTS.md). One run on 193 test cases:
answerable `answer_ok` 0.231, retrieval evidence hit 0.513, and **0/33
unanswerable cases correctly abstained**. The per-domain and base/retrieval
breakdowns are in the report.

- Speed measurements: [`reports/SPEED.md`](reports/SPEED.md)
- Research and next experiments: [`reports/RESEARCH-PLAN.md`](reports/RESEARCH-PLAN.md)
- Independent review and known measurement bugs: [`reviews/OPUS-REVIEW.md`](reviews/OPUS-REVIEW.md)

## Known measurement issues

Before using these numbers for a new comparison, read the Opus review. Two
confirmed bugs are:

- the RAG runner's 7,000-character context cap supplied only ~3.17 chunks on
  average, not the intended five;
- the nomic index was built without `search_document:` / `search_query:`
  prefixes.

The benchmark remains a useful negative/exploratory result, but the next run
must repair these before claiming retrieval or prompt improvements.

## Limitations

- This is an **extractive, automatically constructed** benchmark, not a
  human-written open-ended QA set. The fill-in-the-blank format is deliberate:
  it makes the gold answer and source span mechanically verifiable.
- The search-derived article pool is biased toward pages that rank for the six
  broad queries; some articles are peripheral to the nominal domain.
- Only one generation run is stored unless stated otherwise; LFM3 output has
  run-to-run variance even with a fixed seed.
- `nomic-embed-text` was used for the large index because the qwen3 embedding
  build on this Deck was impractically slow for 4.5k chunks.
- The test set is public in this repository after the run. It is not a hidden
  leaderboard; do not tune on it and report it as a general ranking.

## Wikipedia attribution

Article text and quotations originate from English Wikipedia, licensed under
CC BY-SA 4.0. See <https://www.wikipedia.org/> and the source article links in
`cases/`. Code in this repository is MIT-licensed.
