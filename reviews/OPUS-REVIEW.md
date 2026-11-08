# Opus review — offline-wiki-rag-eval

Review date: 2026-11-08. The full benchmark is in
`/home/dav/Projects/deck-rag/pilot3` and public at
<https://github.com/dacheng-lu/offline-wiki-rag-eval>.

## Reviewer verdict

The benchmark supports only narrow claims: under this specific configuration,
end-to-end answer success is low, the model almost never abstains, and latency
is around 4.7 s. It does **not** support domain rankings or a general claim
about LFM3's knowledge.

The largest methodological weakness is the automatically generated
fill-in-the-blank format. It has one accepted answer, no aliases, formula
residue, and exact-match scoring is sensitive to formatting. The second is
sample size: 20 answerable questions per domain cannot separate domains.

## What I verified independently

- `embed` rows: 193; mean context chunks supplied = **3.17** (min 3, max 5;
  163/193 cases got only 3 chunks). The context builder's 7,000-character cap
  truncated the intended top-5. This invalidates the "retrieval hit" numbers as
  pure top-5 retrieval metrics.
- `unanswerable` rows: 33. `abstain=true`: **0**; empty answer: **0**; empty
  citations: 8; all three conditions: **0**. The 0/33 is therefore a real
  failure of the current output/decision behavior, not only a scoring artifact.
- `embed` valid JSON: 164/193; 29 responses were malformed or did not follow the
  JSON contract.
- `base`: 193 rows, all non-empty, exactly 1 exact answer; `answer_ok` is 0
  because base has no citation. The zero is format-driven, not a clean measure
  of LFM3 knowledge.
- Per-domain Wilson 95% intervals for answerable `answer_ok` overlap heavily:
  arts 0.45 [0.26, 0.66], life 0.10 [0.03, 0.30], math/physics/social 0.15
  [0.05, 0.36]. Do not report a domain ranking as established.

## Minimal diagnosis experiment

1. Re-score the existing outputs separately for `abstain`, empty answer, and
   empty citations; do not rerun the model.
2. On the 33 unanswerable cases, run three contexts: empty, unrelated-domain
   chunks, and the current same-domain neighbours. If empty context abstains
   but neighbour context does not, the failure is prompt/retrieval interference,
   not lack of model knowledge.
3. Add an answerability gate: first ask yes/no whether the context contains the
   answer, then answer only if yes. Test on the same 33 unanswerable + 33
   answerable cases.

## Recommended next steps

### P0 — fix measurement before more modeling

- Fix context truncation: either raise the budget enough for five chunks or
  shrink chunks; exclude truncated chunks from retrieval-hit statistics.
- Rebuild the nomic index with the documented `search_document:` /
  `search_query:` prefixes; the current index uses raw text.
- Add BM25 over the source sentences as a sanity baseline. If BM25 evidence hit
  is below 0.9, stop and fix extraction/chunking before touching the model.
- Improve scoring: numeric separators, articles (`a/an/the`), entity aliases;
  publish exact matching rules.
- Human-audit 40 random questions. Accept only if the bad-question rate is
  below 10%; if above 20%, rebuild the question generator.

### P1 — add an oracle-context arm

Give LFM3 the gold chunk plus four distractors, and separately run the
abstention 2×3 experiment. This separates retrieval loss from generation loss.
Acceptance: every failure can be classified as retrieval failure / wrong
extraction / format or citation failure / scoring error.

### P2 — expand and compare

- At least 60 answerable questions per domain, one question per article, with
  article-level bootstrap intervals.
- Compare nomic and qwen3 on a subset; run three seeds; tune only on dev.
- Only then decide whether to swap embedding or change the generator.

## If only one thing can be done

Add the **oracle-context arm** and re-score the existing outputs. It needs no
index rebuild and can be run in roughly 15–20 minutes at the observed speed. It
answers whether LFM3 can extract the answer when the right evidence is present,
and how much of the current 0.231 is a retrieval problem.

## Public-reporting risks

- Readers may interpret 0/33 and domain differences as model-capability
  conclusions. Add confidence intervals and state that the domains are not
  statistically separable.
- The test set and outputs are public; later models may have seen them. Do not
  treat this as a hidden leaderboard.
- The MIT license covers code only; Wikipedia-derived text/quotes require
  CC BY-SA 4.0 attribution in `cases/` and `runs/`.
- README should disclose: context truncation, nomic prefix omission, malformed
  JSON count, automatic-question construction, and the fact that base's zero is
  citation-format-driven.

## Reference

This review was produced from the benchmark description and result tables; the
independent checks above were run against `runs/locked-v1/rows.json`.
