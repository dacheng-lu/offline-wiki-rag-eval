#!/usr/bin/env python3
"""Stage 1: stratified candidate search against the local Kiwix server."""
import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "http://127.0.0.1:8080"
BOOK = "wikipedia_en_all_nopic_2026-06"
ROOT = Path(__file__).resolve().parents[1]

DOMAINS = {
    "mathematics": ["mathematics", "algebra", "geometry", "calculus", "number theory", "statistics"],
    "physics_astronomy": ["physics", "quantum mechanics", "astronomy", "planet", "star", "thermodynamics"],
    "chemistry_materials": ["chemistry", "chemical element", "materials science", "molecule", "polymer", "metallurgy"],
    "life_medicine": ["biology", "medicine", "disease", "genetics", "anatomy", "pharmacology"],
    "geography_environment": ["geography", "country", "river", "mountain", "climate", "ecosystem"],
    "history": ["history", "ancient Rome", "Middle Ages", "world war", "revolution", "archaeology"],
    "social_science": ["sociology", "economics", "psychology", "political science", "anthropology", "law"],
    "arts_humanities": ["art", "music", "literature", "philosophy", "painting", "poetry"],
}
SKIP_PREFIXES = (
    "List_of_", "Lists_of_", "Outline_of_", "Index_of_", "Timeline_of_",
    "Category:", "File:", "Image:", "Portal:", "Wikipedia:", "Template:",
    "Help:", "Special:", "Talk:", "Draft:", "Module:", "Main_Page",
)
SKIP_SUBSTRINGS = ("disambiguation", "_(disambiguation)")


class SearchParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href", "")
        if href.startswith(f"/content/{BOOK}/"):
            self._href = href
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            text = re.sub(r"\s+", " ", "".join(self._text)).strip()
            self.links.append((self._href, text))
            self._href = None
            self._text = []


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "deck-wiki-rag-eval/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def search(query, start, length=20):
    url = (BASE + "/search?" + urllib.parse.urlencode({
        "content": BOOK, "pattern": query, "pageStart": start, "pageLength": length,
    }))
    parser = SearchParser()
    parser.feed(get(url))
    out = []
    for href, text in parser.links:
        title = urllib.parse.unquote(href.rsplit("/", 1)[-1])
        if not title or title.startswith(SKIP_PREFIXES):
            continue
        if any(s in title.lower() for s in SKIP_SUBSTRINGS):
            continue
        if any(x in title for x in ("/", "#")):
            continue
        out.append({"href": href, "title": title.replace("_", " "), "title_slug": title})
    return out


def stable_key(domain, slug):
    return hashlib.sha256(f"{domain}\0{slug}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/selection.json")
    ap.add_argument("--per-query-pages", type=int, default=5)
    ap.add_argument("--index-per-domain", type=int, default=50)
    ap.add_argument("--extra-per-domain", type=int, default=6)
    args = ap.parse_args()
    domain_pools = {}
    used_global = set()
    for domain, queries in DOMAINS.items():
        found = {}
        query_log = []
        for q in queries:
            for page in range(args.per_query_pages):
                start = page * 20
                try:
                    links = search(q, start)
                except Exception as e:
                    query_log.append({"query": q, "start": start, "error": f"{type(e).__name__}: {e}"})
                    continue
                query_log.append({"query": q, "start": start, "count": len(links)})
                for x in links:
                    if x["href"] not in found:
                        found[x["href"]] = x
                time.sleep(0.02)
        # Stable hash sampling, then greedy cross-domain dedupe.
        ordered = sorted(found.values(), key=lambda x: stable_key(domain, x["title_slug"]))
        chosen = []
        for x in ordered:
            if x["href"] in used_global:
                continue
            x = dict(x)
            x["domain"] = domain
            x["selection_key"] = stable_key(domain, x["title_slug"])
            chosen.append(x)
            used_global.add(x["href"])
            if len(chosen) >= args.index_per_domain + args.extra_per_domain:
                break
        domain_pools[domain] = {
            "queries": queries,
            "candidates_total": len(found),
            "chosen": chosen,
            "query_log": query_log,
        }
        print(f"{domain}: candidates={len(found)} chosen={len(chosen)}", flush=True)
    payload = {
        "book": BOOK,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "domains": domain_pools,
        "limits": {"index_per_domain": args.index_per_domain,
                   "extra_per_domain": args.extra_per_domain},
        "selection_note": "Search-result coverage sample, not a uniform Wikipedia sample.",
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
