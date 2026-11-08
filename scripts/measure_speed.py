#!/usr/bin/env python3
"""Measure LFM3 generation, nomic embedding, retrieval, and end-to-end RAG speed."""
import json, time, urllib.request, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LFM = "http://127.0.0.1:11437"
EMB = "http://127.0.0.1:11434"


def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = json.loads(r.read().decode())
    return raw, time.monotonic() - t0


def embed(texts):
    raw, wall = post(EMB + "/api/embed", {"model": "nomic-embed-text", "input": texts, "keep_alive": "2m"})
    return raw, wall


def lfm(messages, max_tokens=128):
    raw, wall = post(LFM + "/v1/chat/completions", {
        "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        "seed": 20261108, "stream": False})
    return raw, wall


def main():
    idx = json.loads((ROOT / "index/test_embedded.json").read_text())
    cases = [json.loads(line) for line in (ROOT / "cases/test.jsonl").open()]
    case = next(c for c in cases if c["answerable"])
    out = {}

    # 1) LFM3 unconstrained generation: warm, 3 samples.
    gens = []
    for _ in range(3):
        raw, wall = lfm([{"role": "user", "content":
            "Write a concise 60-word paragraph explaining why local RAG needs a generator as well as an embedding model."}], 128)
        usage = raw.get("usage", {})
        timings = raw.get("timings", {})
        gens.append({"wall_s": round(wall, 3),
                     "completion_tokens": usage.get("completion_tokens"),
                     "predicted_per_second": round(timings.get("predicted_per_second", 0), 3),
                     "predicted_ms": timings.get("predicted_ms"),
                     "prompt_tokens": usage.get("prompt_tokens")})
    out["lfm_generation"] = gens
    out["lfm_generation_mean_tok_s"] = round(statistics.mean(x["predicted_per_second"] for x in gens), 3)

    # 2) Embedding warm single + batch 16 (chunks from index).
    texts = [ch["text"] for ch in idx["chunks"][:16]]
    emb = {}
    for label, batch in [("single", texts[:1]), ("batch16", texts)]:
        raw, wall = embed(batch)
        n = raw.get("prompt_eval_count")
        dur = raw.get("total_duration", 0) / 1e9
        emb[label] = {"texts": len(batch), "prompt_eval_count": n,
                      "total_duration_s": round(dur, 4), "wall_s": round(wall, 4),
                      "tokens_per_s": round(n / dur, 1) if dur and n else None}
    out["embedding"] = emb

    # 3) Retrieval over 3693 x 768 vectors, 20 repeats.
    import numpy as np
    mat = np.asarray(idx["vectors"], dtype=np.float32)
    q = np.asarray(embed([case["question"]])[0]["embeddings"][0], dtype=np.float32)
    times = []
    for _ in range(20):
        t0 = time.monotonic()
        sims = mat @ q
        order = np.argpartition(-sims, 5)[:5]
        _ = order[np.argsort(-sims[order])]
        times.append(time.monotonic() - t0)
    out["retrieval"] = {"chunks": len(idx["chunks"]), "dim": int(mat.shape[1]),
                        "mean_ms": round(statistics.mean(times) * 1000, 3),
                        "p95_ms": round(sorted(times)[int(.95 * len(times))] * 1000, 3)}

    # 4) End-to-end RAG for one case: query embedding + retrieval + generation.
    qraw, qwall = embed([case["question"]])
    qv = np.asarray(qraw["embeddings"][0], dtype=np.float32)
    sims = mat @ qv
    order = np.argpartition(-sims, 5)[:5]
    order = order[np.argsort(-sims[order])]
    hits = [(float(sims[i]), idx["chunks"][i]) for i in order]
    context = "\n\n".join(f"[{ch['chunk_id']}] {ch['text'][:1400]}" for _, ch in hits)
    messages = [
        {"role": "system", "content": "Answer ONLY from the context. Return JSON only: {\"answer\":\"...\",\"abstain\":false,\"citations\":[\"chunk_id\"]}."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {case['question']}"},
    ]
    lraw, lwall = lfm(messages, 128)
    out["rag_end_to_end"] = {
        "case": case["id"],
        "query_embed_wall_s": round(qwall, 4),
        "context_chunks": len(hits),
        "lfm_wall_s": round(lwall, 3),
        "prompt_tokens": lraw.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": lraw.get("usage", {}).get("completion_tokens"),
        "lfm_predicted_tok_s": round(lraw.get("timings", {}).get("predicted_per_second", 0), 3),
        "lfm_prompt_tok_s": round(lraw.get("timings", {}).get("prompt_per_second", 0), 3),
        "total_wall_s": round(qwall + lwall, 3),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
