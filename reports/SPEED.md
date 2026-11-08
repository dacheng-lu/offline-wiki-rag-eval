# Deck speed measurements

Measured on the Steam Deck OLED, current LFM3-1.7B-Q4_K_M + nomic-embed-text
setup, while the Wikipedia ZIM and Kiwix server were running. Raw data:
[`speed-measure.json`](speed-measure.json). Script:
`scripts/measure_speed.py`.

## LFM3 generation

| run | completion tokens | predicted tok/s | wall |
|---|---:|---:|---:|
| 1 | 79 | 36.28 | 2.34 s |
| 2 | 79 | 35.60 | 2.36 s |
| 3 | 79 | 36.10 | 2.33 s |
| mean | 79 | **35.99** | 2.34 s |

The server's `timings.predicted_per_second` is the decode rate. Short outputs
look slower because fixed request/load overhead dominates; a 4-token probe was
about 30 tok/s.

## Embedding — nomic-embed-text

| workload | prompt tokens | duration | tokens/s |
|---|---:|---:|---:|
| short query | ~10–12 | 0.032–0.035 s | — |
| one 571-token chunk | 571 | 0.27–1.00 s | 572–2,121 |
| batch of 16 chunks | 9,195 | 2.99–3.07 s | **2,998–3,070** |

Single-chunk latency varied with whether the model had just been loaded and
with memory pressure. Batch throughput is the more stable number.

## Retrieval

- 3,693 test chunks × 768 dimensions, exact cosine top-5.
- Mean **3.22 ms**, p95 **3.33 ms** (20 repeated queries).

## End-to-end RAG (5 chunks)

Latest run:

- question case: `test-mathematics-01`
- query embedding: **0.032 s**
- retrieval: ~0.003 s
- prompt: **1,732 tokens**
- completion: 77 tokens
- LFM3 wall: **3.11 s**
- server prompt processing: **500 tok/s**
- server decode: **36.8 tok/s**
- total: **3.14 s**

An earlier run with 42 completion tokens took 4.45 s; the difference is mostly
prompt/model state and output length. The stable conclusion is:

**Embedding and retrieval are negligible. LFM3 prefill plus decode dominates.**

## Caveats

- These are Deck-specific measurements, not a hardware-independent benchmark.
- The ZIM, Kiwix server, and model servers were active; memory pressure can
  change both prefill and decode.
- `prompt_eval_count` comes from the nomic/Ollama response and is not a
  cross-model tokenizer comparison.
- The current RAG runner truncated context to a 7,000-character budget, so the
  end-to-end number above is not a clean five-chunk benchmark. See
  `reviews/OPUS-REVIEW.md`; the planned measurement repair is in
  `reports/RESEARCH-PLAN.md`.
