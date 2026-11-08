#!/usr/bin/env python3
"""Embed test/dev chunk indexes with qwen3-embedding:0.6b."""
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:11434"
MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")


def post(payload):
    req = urllib.request.Request(
        BASE + "/api/embed",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def main():
    for split in ("test", "dev"):
        path = ROOT / "index" / f"{split}.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        chunks = obj["chunks"]
        vectors = []
        started = time.monotonic()
        for i in range(0, len(chunks), 16):
            batch = [c["text"] for c in chunks[i:i + 16]]
            out = post({"model": MODEL, "input": batch, "keep_alive": "2m"})
            vectors.extend(out["embeddings"])
            print(f"{split} {min(i + 16, len(chunks))}/{len(chunks)}", flush=True)
        if len(vectors) != len(chunks):
            raise SystemExit("embedding count mismatch")
        obj["embedding_model"] = MODEL
        obj["vectors"] = vectors
        obj["build_seconds"] = round(time.monotonic() - started, 3)
        outpath = ROOT / "index" / f"{split}_embedded.json"
        outpath.write_text(json.dumps(obj), encoding="utf-8")
        print(f"wrote {outpath} {len(chunks)} chunks in {obj['build_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
