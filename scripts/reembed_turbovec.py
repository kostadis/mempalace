#!/usr/bin/env python3
"""reembed_turbovec.py — re-embed every turbovec collection in one or more
MemPalace palaces with a new embedding model, changing vector dimension.

WHY THIS EXISTS
  The built-in `mempalace repair` rebuild path is ChromaDB-specific (it reads
  chroma.sqlite3 via the chromadb client). The turbovec backend stores the
  verbatim text in each collection's `docs.document` column, so we can
  re-embed in place — but turbovecdb has no drop-collection API and a
  collection's dimension is fixed once created. So per collection we:
    1. read all (ids, documents, metadatas) from the existing collection,
    2. re-embed the documents with the new model (new dim),
    3. delete the collection directory,
    4. recreate it fresh and re-add with the new vectors (rebuilds the
       4-bit ANN index correctly).

  RAW TEXT, openai-compat /v1/embeddings — matches exactly how MemPalace's
  production query path embeds (config: embedding_provider=openai-compat),
  so stored vectors and query vectors share one space. No instruction
  prefixes (the EF embeds raw); add them here only if you also add them to
  the EF's query path.

SAFETY
  This DELETES and recreates collection directories. Back up the palaces
  first (cp -a). A full backup of the 6 live palaces is ~2.6 GB.

USAGE  (run with the env that has turbovecdb: ~/.venvs/main/bin/python)
  ~/.venvs/main/bin/python scripts/reembed_turbovec.py --dry-run \
      ~/.mempalace/palaces/{chat,phandalin,abyss,campaign-dev,toee,memories}
  ~/.venvs/main/bin/python scripts/reembed_turbovec.py \
      --model qwen3-embedding:0.6b --endpoint http://192.168.1.147:11434 \
      ~/.mempalace/palaces/{chat,phandalin,abyss,campaign-dev,toee,memories}

AFTER a real run, flip the live model so queries match the new vectors:
  set "embedding_model" to the new model in ~/.mempalace/config.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request

import turbovecdb

DEFAULT_ENDPOINT = "http://192.168.1.147:11434"
DEFAULT_MODEL = "qwen3-embedding:0.6b"
BIT_WIDTH = 4          # matches mempalace's _DEFAULT_BIT_WIDTH
METRIC = "cosine"


def embed(endpoint: str, model: str, texts: list[str], batch: int = 64) -> list[list[float]]:
    """Embed via the OpenAI-compatible /v1/embeddings endpoint (raw text)."""
    url = endpoint.rstrip("/") + "/v1/embeddings"
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        body = json.dumps({"model": model, "input": chunk}).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        # OpenAI shape: data["data"] is index-ordered.
        out.extend(item["embedding"] for item in data["data"])
    if len(out) != len(texts):
        raise RuntimeError(f"embed returned {len(out)} vectors for {len(texts)} inputs")
    return out


def reembed_palace(palace_path: str, endpoint: str, model: str, dry_run: bool) -> dict:
    tv = os.path.join(palace_path, "turbovec")
    if not os.path.isdir(tv):
        print(f"  ! no turbovec dir at {tv} — skipping")
        return {}
    db = turbovecdb.connect(tv)
    names = db.list_collections()
    print(f"  collections: {names}")

    # Phase 1: read everything out first (so a mid-run failure leaves the
    # still-768d collections untouched until we have their data in hand).
    snapshots = []
    for name in names:
        col = db.collection(name, create=False)
        res = col.get(include=("documents", "metadatas"))
        snapshots.append((name, list(res.ids), list(res.documents), list(res.metadatas)))
        print(f"    {name}: {len(res.ids)} docs")
    db.close()

    if dry_run:
        return {name: len(ids) for name, ids, _, _ in snapshots}

    # Phase 2: per collection — embed, drop dir, recreate, re-add.
    counts = {}
    for name, ids, docs, metas in snapshots:
        if not ids:
            print(f"    {name}: empty — skipping")
            counts[name] = 0
            continue
        t0 = time.time()
        vectors = embed(endpoint, model, docs)
        dim = len(vectors[0])
        coll_dir = os.path.join(tv, name)
        shutil.rmtree(coll_dir)
        db = turbovecdb.connect(tv)
        newcol = db.collection(name, dim=dim, bit_width=BIT_WIDTH,
                               metric=METRIC, create=True)
        newcol.add(ids=ids, documents=docs, metadatas=metas, vectors=vectors)
        n = newcol.count()
        db.close()
        counts[name] = n
        print(f"    {name}: re-embedded {n} docs -> dim {dim}  ({time.time()-t0:.1f}s)")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("palaces", nargs="+", help="palace directory paths")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"{'DRY-RUN' if args.dry_run else 'RE-EMBED'} model={args.model} endpoint={args.endpoint}\n")
    grand = {}
    for p in args.palaces:
        p = os.path.expanduser(p)
        print(f"palace: {p}")
        grand[p] = reembed_palace(p, args.endpoint, args.model, args.dry_run)
        print()
    total = sum(sum(c.values()) for c in grand.values())
    print(f"{'would re-embed' if args.dry_run else 're-embedded'} {total} docs across "
          f"{len(args.palaces)} palaces")
    if not args.dry_run:
        print("\nNEXT: set embedding_model to the new model in ~/.mempalace/config.json "
              "so queries match the new vectors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
