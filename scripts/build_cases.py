#!/usr/bin/env python3
"""Stage 3: build stratified test/dev cases and chunk indexes from fetched articles."""
import argparse
import hashlib
import json
import re
from collections import defaultdict, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_ORDER = [
    "mathematics", "physics_astronomy", "chemistry_materials", "life_medicine",
    "geography_environment", "history", "social_science", "arts_humanities",
]
DOMAIN_TERMS = {
    "mathematics": ["mathemat", "algebra", "geometry", "calculus", "number", "statistic", "theorem", "equation", "function"],
    "physics_astronomy": ["physic", "quantum", "astronom", "planet", "star", "thermodynam", "mechanic", "cosmolog", "relativ"],
    "chemistry_materials": ["chem", "molecul", "polymer", "metallurg", "material", "element", "crystal"],
    "life_medicine": ["biolog", "medic", "disease", "genetic", "anatom", "pharmac", "cell", "health"],
    "geography_environment": ["geograph", "river", "mountain", "climate", "ecosystem", "country", "lake", "island", "region"],
    "history": ["histor", "rome", "middle ages", "war", "revolution", "archaeolog", "empire", "ancient", "medieval"],
    "social_science": ["sociolog", "econom", "psycholog", "political", "anthropolog", "law", "social"],
    "arts_humanities": ["art", "music", "literature", "philosoph", "paint", "poet", "theater", "film", "culture"],
}
TITLE_BLOCK = {
    "music", "album", "song", "band", "film", "movie", "television", "season",
    "team", "rugby", "basketball", "baseball", "football", "game", "comic",
    "franchise", "novel", "award", "prize", "hotel", "magazine", "newspaper",
    "journal", "review", "school", "university", "college", "department",
    "faculty", "institute", "society", "association", "company",
}
BLOCK_WORDS = {
    "music", "album", "song", "band", "film", "movie", "television", "tv",
    "season", "team", "rugby", "basketball", "baseball", "football", "game",
    "comic", "comics", "franchise", "novel", "award", "awards", "prize",
    "hotel", "magazine", "newspaper", "museum", "society", "institute",
    "university", "college", "journal", "school", "department", "faculty",
    "ship", "line", "company", "band", "poem", "poetry", "painting",
}


def stable(text):
    return hashlib.sha256(text.encode()).hexdigest()


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def choose_target(sentence):
    # Prefer a numeric/date fact; otherwise a multiword capitalized entity.
    nums = [m for m in re.finditer(r"\b(?:1[0-9]{3}|20[0-9]{2}|\d{1,3}(?:[.,]\d+)?)\b", sentence)
            if 0 < m.start() < len(sentence) - 1]
    for m in sorted(nums, key=lambda m: stable(sentence + m.group(0))):
        target = m.group(0)
        if target not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}:
            return m.start(), m.end(), target
    ents = list(re.finditer(r"\b[A-Z][A-Za-z0-9'’-]+(?:\s+[A-Z][A-Za-z0-9'’-]+){1,3}\b", sentence))
    for m in sorted(ents, key=lambda m: stable(sentence + m.group(0))):
        target = m.group(0)
        if len(target) >= 6 and not any(w.lower() in BLOCK_WORDS for w in target.split()):
            return m.start(), m.end(), target
    return None


def make_question(article, sentence, target_start, target_end, target, category):
    blanked = sentence[:target_start] + "_____" + sentence[target_end:]
    q = "Fill in the blank using the article: " + blanked
    return {
        "question": q,
        "accepted_answers": [target],
        "keywords": [target],
        "category": category,
    }


def source_span(article, sentence):
    start = article["text"].find(sentence)
    if start < 0:
        return None
    return {"article_id": article["article_id"], "start": start,
            "end": start + len(sentence), "quote": sentence}


BAD_SENTENCE = (
    "isbn", "doi", "issn", "retrieved", "archived", "creative commons",
    "the text is available", "this article is part", "founded",
    "references", "external links", "see also", "categories", "edit",
    "bourbaki", "springer-verlag", "awards", "author", "citation",
)


def candidate_sentences(article):
    out = []
    # Lead/body first; this avoids most reference-list and license boilerplate.
    body = article["text"][:6000]
    for para in body.split("\n\n"):
        if len(para) < 80 or len(para) > 1200:
            continue
        for sent in split_sentences(para):
            if len(sent) < 60 or len(sent) > 500:
                continue
            low = sent.lower()
            if any(b in low for b in BAD_SENTENCE):
                continue
            if sent.strip().startswith("^"):
                continue
            target = choose_target(sent)
            if target:
                out.append((sent, target))
    return out


def build_chunks(article, max_words=384, overlap=64):
    words = list(re.finditer(r"\S+", article["text"]))
    if not words:
        return []
    chunks = []
    step = max_words - overlap
    i = 0
    while i < len(words):
        j = min(len(words), i + max_words)
        start = words[i].start()
        end = words[j - 1].end()
        text = article["text"][start:end]
        chunks.append({
            "chunk_id": f"{article['article_id']}:c{len(chunks):03d}",
            "article_id": article["article_id"],
            "domain": article["domain"],
            "start": start,
            "end": end,
            "text": text,
        })
        if j == len(words):
            break
        i += step
    return chunks


def relevant_title(domain, title):
    low = title.lower()
    if any(b in low for b in TITLE_BLOCK):
        return False
    return any(t in low for t in DOMAIN_TERMS[domain])


def select_articles(by_domain):
    # 40 test index + 10 dev index + 6 unanswerable sources per domain.
    out = {}
    for domain in DOMAIN_ORDER:
        out[domain] = by_domain.get(domain, [])[:56]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="text/articles.jsonl")
    ap.add_argument("--out-dir", default="cases")
    ap.add_argument("--index-dir", default="index")
    args = ap.parse_args()
    by_domain = defaultdict(list)
    with (ROOT / args.articles).open(encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            by_domain[a["domain"]].append(a)
    selected = select_articles(by_domain)
    cases = {"test": [], "dev": []}
    index_articles = {"test": [], "dev": []}
    chunks = {"test": [], "dev": []}
    diagnostics = {}
    for domain in DOMAIN_ORDER:
        arts = selected[domain]
        if len(arts) < 56:
            diagnostics[domain] = {"error": f"only {len(arts)} articles", "count": len(arts)}
        test_arts = arts[:40]
        dev_arts = arts[40:50]
        extra_arts = arts[50:56]
        if len(test_arts) < 40 or len(dev_arts) < 10:
            raise SystemExit(f"{domain}: insufficient articles {len(arts)}")
        # Question articles: draw from the frozen split until the quota is met.
        test_question_arts = test_arts
        dev_question_arts = dev_arts
        for split, qarts, index_arts, n_extra, target_q in [
            ("test", test_question_arts, test_arts, 5, 20),
            ("dev", dev_question_arts, dev_arts, 2, 8),
        ]:
            index_articles[split].extend(index_arts)
            for a in index_arts:
                chunks[split].extend(build_chunks(a))
            qnum = 0
            for a in qarts:
                if qnum >= target_q:
                    break
                cands = candidate_sentences(a)
                # Two distinct sentences per article when available.
                chosen = []
                for sent, target in cands:
                    if any(sent == x[0] for x in chosen):
                        continue
                    chosen.append((sent, target))
                    if len(chosen) >= 2:
                        break
                for sent, (start, end, target) in chosen:
                    if qnum >= target_q:
                        break
                    span = source_span(a, sent)
                    if not span:
                        continue
                    qnum += 1
                    row = {
                        "id": f"{split}-{domain}-{qnum:02d}",
                        "split": split,
                        "domain": domain,
                        "family_id": a["article_id"],
                        "answerable": True,
                        "source": span,
                    }
                    row.update(make_question(a, sent, start, end, target, "extractive"))
                    cases[split].append(row)
            # Unanswerable: source article is not in this split's index.
            unum = 0
            for a in extra_arts[:n_extra]:
                cands = candidate_sentences(a)
                for sent, (start, end, target) in cands:
                    if target.lower() in "\n".join(c["text"].lower() for c in chunks[split]):
                        continue
                    span = source_span(a, sent)
                    if not span:
                        continue
                    unum += 1
                    row = {
                        "id": f"{split}-{domain}-u{unum:02d}",
                        "split": split,
                        "domain": domain,
                        "family_id": a["article_id"],
                        "answerable": False,
                        "source": span,
                        "absence_audit": {
                            "target": target,
                            "target_in_index": False,
                            "note": "article excluded from split index; target checked against all split chunks",
                        },
                    }
                    row.update(make_question(a, sent, start, end, target, "extractive"))
                    cases[split].append(row)
                    break
        diagnostics[domain] = {
            "articles": len(arts),
            "test_index": len(test_arts),
            "dev_index": len(dev_arts),
            "test_cases": sum(1 for c in cases["test"] if c["domain"] == domain),
            "dev_cases": sum(1 for c in cases["dev"] if c["domain"] == domain),
        }
    # Drop duplicate question strings and enforce planned counts.
    for split in cases:
        seen = set()
        unique = []
        for c in cases[split]:
            key = c["question"].lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
        cases[split] = unique
    # Attach gold chunk IDs to answerable cases.
    chunk_by_article = defaultdict(list)
    for split in chunks:
        for ch in chunks[split]:
            chunk_by_article[(split, ch["article_id"])].append(ch)
    for split in cases:
        for c in cases[split]:
            if not c["answerable"]:
                c["gold_chunk_ids"] = []
                continue
            st = c["source"]["start"]
            en = c["source"]["end"]
            c["gold_chunk_ids"] = [
                ch["chunk_id"] for ch in chunk_by_article[(split, c["source"]["article_id"])]
                if ch["start"] < en and ch["end"] > st
            ]
            c["gold_doc_ids"] = [c["source"]["article_id"]]
    outdir = ROOT / args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "dev"):
        with (outdir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for c in cases[split]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    idxdir = ROOT / args.index_dir
    idxdir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "dev"):
        with (idxdir / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump({"split": split, "chunks": chunks[split]}, f, ensure_ascii=False)
    diag = {"articles_by_domain": {d: len(by_domain[d]) for d in DOMAIN_ORDER},
            "selected": diagnostics,
            "cases": {s: len(cases[s]) for s in cases},
            "chunks": {s: len(chunks[s]) for s in chunks},
            "answerable": {s: sum(1 for c in cases[s] if c["answerable"]) for s in cases},
            "unanswerable": {s: sum(1 for c in cases[s] if not c["answerable"]) for s in cases}}
    (ROOT / "data/build-report.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diag, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
