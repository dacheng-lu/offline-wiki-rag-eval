#!/usr/bin/env python3
"""Stage 2: fetch selected articles from the local Kiwix server and extract text."""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "http://127.0.0.1:8080"
ROOT = Path(__file__).resolve().parents[1]
SKIP_TAGS = {"script", "style", "sup", "noscript", "template"}
BLOCK_TAGS = {"p", "li", "h2", "h3", "h4", "h5", "h6", "blockquote", "dt", "dd"}


class ContentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_content = False
        self.content_div_depth = 0
        self.skip_depth = 0
        self.current_block = []
        self.blocks = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag == "div" and attrs.get("id") == "mw-content-text":
            self.in_content = True
            self.content_div_depth = 1
            return
        if self.in_content:
            if tag == "div":
                self.content_div_depth += 1
            if tag in SKIP_TAGS:
                self.skip_depth += 1
            elif self.skip_depth == 0 and tag in BLOCK_TAGS:
                self._flush_block()

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if not self.in_content:
            return
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if tag in BLOCK_TAGS and self.skip_depth == 0:
            self._flush_block()
        if tag == "div":
            self.content_div_depth -= 1
            if self.content_div_depth <= 0:
                self.in_content = False
                self._flush_block()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self.in_content and self.skip_depth == 0:
            self.current_block.append(data)

    def _flush_block(self):
        text = re.sub(r"\s+", " ", "".join(self.current_block)).strip()
        text = re.sub(r"\s*\[(?:edit|citation needed)\]\s*", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 2:
            self.blocks.append(text)
        self.current_block = []


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "deck-wiki-rag-eval/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="data/selection.json")
    ap.add_argument("--out", default="text/articles.jsonl")
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--limit-per-domain", type=int, default=56)
    args = ap.parse_args()
    selection = json.loads((ROOT / args.selection).read_text(encoding="utf-8"))
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = []
    with out_path.open("w", encoding="utf-8") as out:
        for domain, payload in selection["domains"].items():
            for item in payload["chosen"][:args.limit_per_domain]:
                url = BASE + item["href"]
                try:
                    raw = fetch(url)
                except Exception as e:
                    skipped.append({"domain": domain, "href": item["href"], "error": f"{type(e).__name__}: {e}"})
                    continue
                html = raw.decode("utf-8", errors="replace")
                parser = ContentParser()
                parser.feed(html)
                parser.close()
                text = "\n\n".join(parser.blocks).strip()
                if len(text) < 400:
                    skipped.append({"domain": domain, "href": item["href"], "error": f"short text {len(text)}"})
                    continue
                slug = item["title_slug"]
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)[:180]
                (raw_dir / f"{safe}.html").write_bytes(raw)
                record = {
                    "article_id": f"wiki:{slug}",
                    "domain": domain,
                    "title": parser.title.strip() or item["title"],
                    "title_slug": slug,
                    "url": url,
                    "html_sha256": hashlib.sha256(raw).hexdigest(),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "text": text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                print(f"{count} {domain} {slug} {len(text)} chars", flush=True)
                time.sleep(0.01)
    meta = {"articles": count, "skipped": skipped, "selection": args.selection}
    (ROOT / "data/fetch-report.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"articles": count, "skipped": len(skipped)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
