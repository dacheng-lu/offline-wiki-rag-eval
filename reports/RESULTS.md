# Results — offline Wikipedia domain benchmark

Run: `runs/locked-v1` (one run; LFM3-1.7B-Q4_K_M, nomic-embed-text, top-5).
Test set: 193 questions (160 answerable, 33 unanswerable), 8 domains.

## Overall

| arm | n | answer_ok | answerable answer_ok | exact (answerable) | token F1 | citation_ok | unanswerable answer_ok | mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 193 | 0.0000 | 0.0000 | 0.0063 | 0.0563 | 0.0000 | 0.0000 | 1.47 s |
| embed | 193 | 0.1917 | 0.2313 | 0.2938 | 0.4271 | 0.4750 | 0.0000 | 4.71 s |
| retrieval | 193 | — | — | — | — | — | — | — |

Retrieval-only: article hit **0.6943**, evidence hit **0.5130**, MRR **0.4312**.

## By domain — answerable `answer_ok`

| domain | answerable n | embed answer_ok | exact | citation_ok | retrieval evidence_hit |
|---|---:|---:|---:|---:|---:|
| mathematics | 20 | 0.1500 | 0.1500 | 0.6000 | 0.5417 |
| physics_astronomy | 20 | 0.1500 | 0.2000 | 0.3000 | 0.5200 |
| chemistry_materials | 20 | 0.3000 | 0.4500 | 0.6000 | 0.6087 |
| life_medicine | 20 | 0.1000 | 0.1500 | 0.3500 | 0.4000 |
| geography_environment | 20 | 0.3500 | 0.5000 | 0.5000 | 0.6667 |
| history | 20 | 0.2000 | 0.2000 | 0.5000 | 0.5000 |
| social_science | 20 | 0.1500 | 0.2500 | 0.3500 | 0.3913 |
| arts_humanities | 20 | 0.4500 | 0.4500 | 0.6000 | 0.4800 |

## What stands out

- **Unanswerable cases: 0/33 correct across every domain.** LFM3 almost never
  returned the required empty-answer/empty-citation JSON; it answered anyway.
  This is the largest safety failure in the run.
- **Arts/humanities** is the strongest domain for this extractive pipeline
  (0.45 answerable answer_ok); **life/medicine** is the weakest (0.10).
- Retrieval is moderate, not catastrophic: article hit 0.69 and evidence hit
  0.51. The pipeline still loses substantially between retrieved evidence and
  a correct cited answer (0.23 overall answerable answer_ok).
- Base closed-book is essentially zero on this metric; the one exact-match
  answer still fails because base has no citation.
- The result is a **single run** and a search-derived, automatically constructed
  test set. It is not a Wikipedia-wide estimate and not a human-verified
  open-ended QA benchmark.

## Artifacts

- `metrics.json` — overall summary.
- `domain-metrics.json` — per-domain summary.
- `rows.json` — all 579 per-case rows (base/retrieval/embed).
- `requests.jsonl`, `responses.jsonl` — raw request/response records.
- `cases/test.jsonl` — frozen questions, labels, source spans, gold chunks.
- `data/selection.json` — article-selection audit trail.
- `data/build-report.json` — article/case/chunk counts.
