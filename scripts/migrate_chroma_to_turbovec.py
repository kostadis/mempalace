#!/usr/bin/env python3
"""chroma -> turbovec migration for a single palace.

Copies every drawer/closet/index record (id, verbatim document, metadata, and
the embedding vector) out of a palace's ChromaDB collections and into turbovec
collections of the same name, under ``<palace>/turbovec/``.

Per collection, the migrator picks a vector strategy automatically:

* **copy** — when chroma can return the stored vectors, they are handed to
  turbovec as precomputed ``embeddings`` (lossless, no embedding endpoint
  needed).
* **reembed** — when chroma cannot return the vectors (e.g. a corrupt vector
  segment), the intact verbatim documents are re-embedded with MemPalace's
  configured embedding function and those vectors are written instead. The
  verbatim text is preserved exactly; only the vectors are rebuilt. This needs
  the embedding endpoint to be reachable.

The source ChromaDB store is never modified; this only writes the new
``<palace>/turbovec/`` directory. Re-runnable: a collection whose turbovec count
already matches chroma is skipped.

Usage:
    python scripts/migrate_chroma_to_turbovec.py --palace /path/to/palace [--dry-run]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from mempalace.backends import PalaceRef, get_backend

BATCH = 1000
EMBED_BATCH = 128

_ef = None


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via MemPalace's configured embedding function,
    in sub-batches, returning one vector per text in order."""
    global _ef
    if _ef is None:
        from mempalace.embedding import get_embedding_function

        _ef = get_embedding_function()
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i : i + EMBED_BATCH]
        out.extend(list(v) for v in _ef(chunk))
    return out


def _embeddings_readable(src) -> bool:
    """True if chroma can return stored vectors for this collection."""
    try:
        r = src.get(limit=1, offset=0, include=["embeddings"])
    except Exception:
        return False
    return r.embeddings is not None and len(r.embeddings) > 0


def _chroma_collection_names(palace: str) -> list[str]:
    """Read collection names straight from the palace's chroma sqlite."""
    db = Path(palace) / "chroma.sqlite3"
    if not db.exists():
        raise SystemExit(f"no chroma.sqlite3 in {palace} — nothing to migrate")
    con = sqlite3.connect(str(db))
    try:
        return [r[0] for r in con.execute("SELECT name FROM collections ORDER BY name")]
    finally:
        con.close()


def _migrate_collection(name: str, ref: PalaceRef, chroma, turbo, dry_run: bool) -> tuple[int, int]:
    src = chroma.get_collection(palace=ref, collection_name=name, create=False)
    total = src.count()

    dst = turbo.get_collection(palace=ref, collection_name=name, create=True)
    already = dst.count()
    if already == total and total > 0:
        print(f"  [{name}] already migrated ({already}/{total}) — skipping")
        return total, already
    if already and already != total:
        print(
            f"  [{name}] WARNING: turbovec already has {already} (chroma has {total}); "
            f"continuing will upsert/extend — inspect before trusting"
        )

    mode = "copy" if _embeddings_readable(src) else "reembed"
    print(
        f"  [{name}] migrating {total} records (turbovec has {already}) "
        f"via {mode}" + (" [dry-run]" if dry_run else "")
    )
    include = ["documents", "metadatas"]
    if mode == "copy":
        include.append("embeddings")

    moved = 0
    offset = 0
    while offset < total:
        page = src.get(limit=BATCH, offset=offset, include=include)
        ids = page.ids
        if not ids:
            break

        if mode == "copy":
            embs = page.embeddings
            if embs is None or len(embs) != len(ids):
                raise SystemExit(
                    f"  [{name}] chroma returned no/short embeddings at offset {offset}"
                )
            vectors = [list(v) for v in embs]
        else:
            docs = page.documents
            if any(d is None for d in docs):
                raise SystemExit(
                    f"  [{name}] missing document at offset {offset}; cannot re-embed losslessly"
                )
            # Don't burn thousands of embedding calls just to dry-run.
            vectors = None if dry_run else _embed(docs)

        if not dry_run:
            dst.add(
                ids=ids,
                documents=page.documents,
                metadatas=page.metadatas,
                embeddings=vectors,
            )
        moved += len(ids)
        offset += len(ids)
        print(f"    {moved}/{total}", end="\r", flush=True)
    print(f"    {moved}/{total} done                ")

    final = dst.count() if not dry_run else already + moved
    return total, final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--palace", required=True, help="path to the palace dir")
    ap.add_argument("--dry-run", action="store_true", help="read + report, write nothing")
    args = ap.parse_args()

    palace = str(Path(args.palace).expanduser().resolve())
    ref = PalaceRef(id=palace, local_path=palace)

    names = _chroma_collection_names(palace)
    print(f"palace: {palace}")
    print(f"chroma collections: {names}")

    chroma = get_backend("chroma")
    turbo = get_backend("turbovec")

    ok = True
    for name in names:
        total, final = _migrate_collection(name, ref, chroma, turbo, args.dry_run)
        if not args.dry_run and final != total:
            ok = False
            print(f"  [{name}] COUNT MISMATCH: chroma={total} turbovec={final}")

    turbo.close()
    chroma.close()

    if args.dry_run:
        print("dry-run complete — no data written")
        return 0
    print("migration complete" if ok else "migration FAILED (count mismatch above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
