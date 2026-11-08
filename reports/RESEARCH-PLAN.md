# Research and test blueprint — speed + answer quality

Status: research note, not executed. Baseline: `runs/locked-v1` (193 cases,
8 domains, one run). Current speed: `reports/SPEED.md`. Measurement caveats:
`reviews/OPUS-REVIEW.md`.

## Baseline in one paragraph

LFM3 decodes at ~36 tok/s and processes the RAG prompt at ~500 tok/s.
Embedding is ~3k tokens/s in batches and retrieval is ~3 ms; neither is the
bottleneck. The measured RAG case was 1,732 prompt tokens, 77 output tokens,
and 3.14 s end-to-end. Quality is limited by evidence coverage (0.513), citation
coverage (0.475), and refusal (0/33), not by embedding throughput.

## Stream A — speed

### What the sources say

- llama.cpp exposes separate logical batch (`--batch-size`) and physical batch
  (`--ubatch-size`), KV-cache types (`--cache-type-k/v`), Flash Attention
  (`--flash-attn`), and speculative decoding (`--spec-type`, draft options).
  These are the first knobs to measure, not guess. [1]
- Vulkan has configuration-specific pathologies; one reported case collapses
  prompt processing with a particular `ubatch` + f16 KV + MTP combination, so
  every flag change must be A/B measured on this Deck. [2]
- KV quantization reduces cache memory but can cost decode throughput depending
  on workload; test both prompt-heavy and decode-heavy cases. [3]
- LFM3-1.7B is explicitly positioned for local agents and retrieval/search for
  knowledge-heavy work; it is not a large reasoning model. [4]

### Experiments

| id | change | measure | accept if | stop if |
|---|---|---|---|---|
| S1 | `--flash-attn on` vs auto/off | pp tok/s, tg tok/s, output equality | +10% pp or +5% tg, no quality change | crash, garbage, or >5% slowdown |
| S2 | `-b 2048 -ub 512/1024/2048` matrix | pp/tg, peak RSS/VRAM | best p95 with stable output | OOM or regression >10% |
| S3 | `-ctk q8_0 -ctv q8_0` vs f16 | memory, pp/tg, answer_ok on dev | memory saving with <5% quality loss | >5% answer_ok drop |
| S4 | context budget: 5 vs 4 vs 3 chunks / 1,200 vs 700 chars per chunk | prompt tokens, answer_ok, citation | same answer_ok at lower tokens | quality drop >3 points |
| S5 | speculative ngram (`--spec-type ngram-simple`) or a small draft model | tg tok/s, acceptance rate | >15% decode gain, identical outputs | no gain or instability |
| S6 | prompt caching / slot reuse (`--cache-reuse`, warm requests) | p50/p95 end-to-end | lower p95 with same answers | cache contamination |

Speculative decoding is lower priority than prefill reduction: at 1,732 prompt
tokens and 77 output tokens, prefill is the larger share. Do S4/S6 before S5.

## Stream B — answer quality

### What the sources say

- Context compression can reduce generator input, but citations that look
  correct against a compressed summary may not be attributable to the original
  source span; report citation validity against source spans, not summaries. [5]
- Long-context readers underuse evidence in the middle; retrieval should
  deliver a compact evidence set and place critical spans deliberately. [6]
- A validation layer between retrieval and delivery (citation validity,
  grounding checks, refusal/regeneration) can sharply improve grounded-answer
  and refusal behavior; retrieval accuracy alone is insufficient. [7]
- Grounded answer + abstention should be treated as one decision boundary;
  answerability gating and citation-aware generation are separate controls. [8]
- LFM3 supports structured output/tool calling and retrieval/search; the model
  card recommends retrieval for knowledge-heavy tasks. [4]

### Experiments

| id | change | measure | accept if | stop if |
|---|---|---|---|---|
| Q1 | repair measurement: 5 chunks actually supplied, nomic prefixes, aliases, BM25 sanity | valid context count, evidence hit, answer_ok | BM25 evidence hit ≥0.90; JSON valid ≥0.95 | BM25 <0.80 → fix extraction/chunking |
| Q2 | oracle arm: gold chunk + 4 distractors | answer_ok, exact, citation | separates retrieval loss from generation loss | cannot classify failures |
| Q3 | answerability gate: first yes/no on context sufficiency, then answer | true refusal, false refusal, answer_ok | true refusal ↑ with false refusal ≤10% | gate refuses >20% answerable |
| Q4 | hybrid retrieval: dense + BM25, then rerank to 5 | article hit, evidence hit, answer_ok | evidence hit +0.10 absolute | latency >+1.5 s with no quality gain |
| Q5 | extractive sentence compression: keep query-relevant sentences, preserve spans | prompt tokens, answer_ok, citation validity | −30% prompt tokens, answer_ok no worse | citation validity drops >3 points |
| Q6 | citation-first JSON + validation/one retry | valid JSON, citation coverage, answer_ok | valid JSON ≥0.95; no answer_ok drop | retries exceed 10% |
| Q7 | nomic prefix + qwen3 subset comparison | retrieval metrics, answer_ok | one model wins on dev with paired CI | no separation → keep nomic |
| Q8 | LoRA/domain adaptation | answer_ok, citation, cost | only after Q1–Q7 plateau | no data or no gain |

## Stream C — test protocol

- Dev: 79 cases for all tuning. Locked: 193 cases; do not tune on them.
- Run every accepted change on dev first; run locked once per version.
- Repeat 3 times for generation arms; report mean and range; cluster bootstrap
  by article/family, not by question.
- Report per-domain n and Wilson/bootstrap intervals. With 20 answerable cases
  per domain, do not claim a domain ranking unless intervals separate.
- Freeze test questions, labels, prompts, model digests, embedding prefixes,
  and run order before the locked run.
- Keep one failure taxonomy: retrieval miss / wrong extraction / citation
  failure / formatting / refusal / scoring error.

## Priority

1. **P0 measurement repair** (Q1) — current context truncation and missing
   nomic prefixes invalidate clean comparisons.
2. **P0 oracle arm** (Q2) — attributes the 0.231 between retrieval and
   generation.
3. **P1 answerability gate** (Q3) — directly targets 0/33.
4. **P1 retrieval + compression** (Q4/Q5) — raise evidence coverage and cut
   prefill tokens.
5. **P2 inference flags** (S1–S3), then speculative decoding (S5).
6. **P2 embedding/model comparison** (Q7), LoRA only if needed (Q8).

## If only one change is made

Run the **oracle-context arm plus re-score**. It needs no new index and answers
whether LFM3 can answer when the right evidence is present. Use the result to
decide whether the next investment is retrieval or generation.

## Sources

1. llama.cpp server options — <https://github.com/ggml-org/llama.cpp/blob/HEAD/tools/server/README.md>
2. Vulkan ubatch/KV pathology report — <https://github.com/ggml-org/llama.cpp/issues/27420>
3. KV cache quantization throughput note — <https://sotaaz.com/post/llamacpp-server-kv-cache-quantization-en>
4. LFM3-1.7B docs — <https://docs.liquid.ai/lfm/models/lfm3-1.7b>
5. Attribution–compression frontier — <https://arxiv.org/html/2609.14245>
6. Lost in the middle (ACL) — <https://aclanthology.org/2024.findings-acl.890.pdf>
7. CiteGuard-RAG — <https://arxiv.org/html/2609.15830>
8. GRACE grounded response and abstention — <https://arxiv.org/pdf/2601.04525>
9. EviScope evidence-action diagnostics — <https://arxiv.org/html/2609.17081>
