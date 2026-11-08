#!/usr/bin/env python3
"""Run the offline-Wikipedia domain benchmark.

Arms:
  base       closed-book LFM3
  retrieval  embedding-only retrieval metrics (no generation)
  embed      LFM3 + retrieved chunks
"""
import argparse
import json
import math
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LFM_BASE = "http://127.0.0.1:11437"
EMBED_BASE = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def embed(text):
    return post(EMBED_BASE + "/api/embed",
                {"model": EMBED_MODEL, "input": [text], "keep_alive": "2m"})["embeddings"][0]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(index, qvec, k):
    scored = sorted(((cosine(qvec, v), ch) for ch, v in zip(index["chunks"], index["vectors"])),
                    key=lambda x: x[0], reverse=True)
    return scored[:k]


def normalize(s):
    s = s.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(answer, golds):
    a = normalize(answer).split()
    best = 0.0
    for gold in golds:
        g = normalize(gold).split()
        if not a or not g:
            continue
        common = {}
        for t in a:
            common[t] = common.get(t, 0) + 1
        overlap = 0
        for t in g:
            if common.get(t, 0):
                overlap += 1
                common[t] -= 1
        p, r = overlap / len(a), overlap / len(g)
        best = max(best, 2 * p * r / (p + r) if p + r else 0.0)
    return round(best, 4)


def parse_json_answer(raw):
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Best-effort extraction, but mark malformed.
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"answer": text, "abstain": False, "citations": [], "_malformed": True}


def prompt(context, question, with_context):
    if with_context:
        system = (
            "Answer ONLY from the supplied context. Return JSON only, no markdown:\n"
            '{"answer":"<short answer>","abstain":false,"citations":["<chunk_id>"]}\n'
            "If the context does not contain the answer, return exactly:\n"
            '{"answer":"","abstain":true,"citations":[]}\n'
            "Every citation must be a chunk_id present in the context."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        system = (
            "Answer from your own knowledge. Return JSON only, no markdown:\n"
            '{"answer":"<short answer>","abstain":false,"citations":[]}\n'
            "If you are not sure, return exactly:\n"
            '{"answer":"","abstain":true,"citations":[]}'
        )
        user = question
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_lfm(messages, max_tokens=256):
    started = time.monotonic()
    raw = post(LFM_BASE + "/v1/chat/completions", {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 20261108,
        "stream": False,
    })
    latency = time.monotonic() - started
    content = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    return content, latency, raw


def score_case(case, answer_obj, context_ids):
    answer = str(answer_obj.get("answer", ""))
    citations = []
    for c in (answer_obj.get("citations") or []):
        c = re.sub(r"^[\[\]\"'`\s]+|[\[\]\"'`\s]+$", "", str(c))
        if c:
            citations.append(c)
    abstain = bool(answer_obj.get("abstain"))
    accepted = case.get("accepted_answers") or []
    exact = normalize(answer) in {normalize(a) for a in accepted} if accepted else False
    valid = [c for c in citations if c in context_ids]
    gold = set(case.get("gold_chunk_ids") or [])
    covered = bool(gold.intersection(valid))
    if case.get("answerable"):
        answer_ok = bool(exact and covered)
        abstention_ok = abstain
    else:
        answer_ok = bool(abstain and not answer.strip() and not citations)
        abstention_ok = answer_ok
    return {
        "answer": answer,
        "abstain": abstain,
        "citations": citations,
        "valid_citations": valid,
        "gold_chunks": sorted(gold),
        "answer_exact": exact,
        "citation_ok": covered,
        "answer_ok": answer_ok,
        "token_f1": token_f1(answer, accepted),
        "answerable": bool(case.get("answerable")),
        "abstention_ok": abstention_ok,
    }


def summarize(rows, arms):
    out = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        n = len(rs)
        ans = [r for r in rs if r["scores"]["answerable"]]
        unans = [r for r in rs if not r["scores"]["answerable"]]
        out[arm] = {
            "n": n,
            "answer_ok": round(sum(r["scores"]["answer_ok"] for r in rs) / n, 4),
            "answerable_answer_ok": round(sum(r["scores"]["answer_ok"] for r in ans) / len(ans), 4) if ans else None,
            "unanswerable_answer_ok": round(sum(r["scores"]["answer_ok"] for r in unans) / len(unans), 4) if unans else None,
            "answer_exact": round(sum(r["scores"]["answer_exact"] for r in ans) / len(ans), 4) if ans else None,
            "token_f1": round(sum(r["scores"]["token_f1"] for r in ans) / len(ans), 4) if ans else None,
            "citation_ok": round(sum(r["scores"]["citation_ok"] for r in ans) / len(ans), 4) if ans else None,
            "abstention_ok": round(sum(r["scores"]["abstention_ok"] for r in unans) / len(unans), 4) if unans else None,
            "mean_latency_s": round(sum(r["latency_s"] for r in rs) / n, 4),
            "p50_latency_s": round(sorted(r["latency_s"] for r in rs)[n // 2], 4),
        }
    return out


def build_context(hits, max_chars=7000):
    parts = []
    used = 0
    ids = set()
    for _, ch in hits:
        text = ch["text"]
        remaining = max_chars - used
        if remaining <= 200:
            break
        if len(text) > remaining:
            text = text[:remaining]
        parts.append(f"[{ch['chunk_id']}] {text}")
        ids.add(ch["chunk_id"])
        used += len(text) + len(ch["chunk_id"]) + 4
    return "\n\n".join(parts), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index/test_embedded.json")
    ap.add_argument("--cases", default="cases/test.jsonl")
    ap.add_argument("--out", default="runs/locked-v1")
    ap.add_argument("--arms", default="base,retrieval,embed")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-context-chars", type=int, default=7000)
    args = ap.parse_args()
    index = json.loads((ROOT / args.index).read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (ROOT / args.cases).open(encoding="utf-8")]
    if args.limit:
        cases = cases[:args.limit]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    rows, requests, responses = [], [], []

    for case in cases:
        qvec = embed(case["question"])
        hits = retrieve(index, qvec, args.top_k)
        context, context_ids = build_context(hits, args.max_context_chars)
        gold = set(case.get("gold_chunk_ids") or [])

        if "retrieval" in arms:
            hit_articles = {ch["article_id"] for _, ch in hits}
            row = {
                "case": case["id"], "arm": "retrieval", "question": case["question"],
                "domain": case["domain"], "answerable": bool(case.get("answerable")),
                "retrieved": [{"chunk_id": ch["chunk_id"], "article_id": ch["article_id"],
                               "score": round(sc, 6)} for sc, ch in hits],
                "latency_s": 0.0,
                "scores": {
                    "answerable": bool(case.get("answerable")),
                    "article_hit": case["source"]["article_id"] in hit_articles,
                    "evidence_hit": bool(gold.intersection(context_ids)),
                    "mrr": None,
                    "answer_ok": None,
                },
            }
            # MRR of the first chunk overlapping the gold evidence.
            for rank, (_, ch) in enumerate(hits, 1):
                if ch["chunk_id"] in gold:
                    row["scores"]["mrr"] = round(1.0 / rank, 4)
                    break
            if row["scores"]["mrr"] is None:
                row["scores"]["mrr"] = 0.0
            rows.append(row)

        for arm in ("base", "embed"):
            if arm not in arms:
                continue
            messages = prompt(context if arm == "embed" else "", case["question"], arm == "embed")
            answer, latency, raw = call_lfm(messages, args.max_tokens)
            obj = parse_json_answer(answer)
            scores = score_case(case, obj, context_ids if arm == "embed" else set())
            scores["valid_json"] = not obj.get("_malformed", False)
            rows.append({
                "case": case["id"], "arm": arm, "question": case["question"],
                "domain": case["domain"], "answer": answer, "parsed": obj,
                "context_chunk_ids": sorted(context_ids) if arm == "embed" else [],
                "latency_s": round(latency, 4), "scores": scores,
            })
            requests.append({"case": case["id"], "arm": arm, "payload": messages})
            responses.append({"case": case["id"], "arm": arm, "raw": raw})
            print(f"{arm} {case['id']} {latency:.2f}s ok={scores['answer_ok']} exact={scores['answer_exact']}", flush=True)

    metrics = summarize(rows, [a for a in arms if a != "retrieval"])
    if "retrieval" in arms:
        rs = [r for r in rows if r["arm"] == "retrieval"]
        metrics["retrieval"] = {
            "n": len(rs),
            "article_hit": round(sum(r["scores"]["article_hit"] for r in rs) / len(rs), 4) if rs else None,
            "evidence_hit": round(sum(r["scores"]["evidence_hit"] for r in rs) / len(rs), 4) if rs else None,
            "mrr": round(sum(r["scores"]["mrr"] for r in rs) / len(rs), 4) if rs else None,
        }
    (outdir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (outdir / "requests.jsonl").open("w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (outdir / "responses.jsonl").open("w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (outdir / "environment.json").write_text(json.dumps({
        "index": args.index, "cases": args.cases, "arms": arms,
        "top_k": args.top_k, "embedding_model": EMBED_MODEL,
        "lfm_endpoint": LFM_BASE, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
