"""turbovec storage backend for MemPalace (experimental).

turbovec (https://github.com/RyanCodrai/turbovec) is a CPU vector index using
TurboQuant 4-bit quantization (SIMD kernels — NEON on ARM, AVX on x86). It runs
on whatever machine MemPalace runs on; MemPalace is local-first, so this backend
is a local, CPU-resident index — no GPU, no remote box. It maps ``uint64`` ids
to quantized vectors and does fast approximate nearest-neighbour search — but it
stores *only* vectors. It has no notion of documents, metadata, or string ids.
So a turbovec backend is "half a backend": this module pairs it with a SQLite
sidecar that is the durable, verbatim source of truth for text + metadata +
the string-id↔uint64 map + the (L2-normalized) float32 vectors.

Design (the turbovec quality eval that motivated this lives in
~/src/dgx/turbovec-observations.md, merged PR #5 — that eval happened to run on
a DGX Spark, but nothing here is Spark-specific):

* **SQLite is the source of truth.** Every drawer's text, metadata, and exact
  float32 vector live there. The ``.tvim`` turbovec index is a *derived,
  rebuildable cache* — if it is missing or stale, it is rebuilt from SQLite on
  open. This is what makes "100% recall / verbatim always" a store property,
  not an index property: lossy 4-bit quantization can only reorder neighbours,
  never change what text comes back.
* **Exact re-rank.** turbovec returns unnormalized scores, but searcher
  (``searcher.py`` ``_hybrid_rank``) consumes an *absolute* cosine distance via
  ``1 - distance``. So we use turbovec only to fetch a candidate pool, then
  recompute true cosine from the stored float32 vectors and return
  ``distance = 1 - cosine`` in ``[0, 2]``. Storing the float32 vectors forfeits
  turbovec's ~8x *index* compression, but the backend is still smaller on disk
  than ChromaDB end-to-end — the 4-bit ``.tvim`` is far lighter than an HNSW
  graph. Measured local parity at 15.8k docs / MiniLM-384: turbovec backend vs
  chroma backend — build 12x faster, query p50/p95 ~3x faster, ~2.3x smaller on
  disk, retrieval quality a wash. See ~/src/dgx/turbovec-backend-parity.py.
* **BM25/hybrid is above this layer** (searcher.py). This backend only does
  vector ANN + returns documents + correct cosine distances.

This is an experimental backend (RFC 001 §3): registered via the
``mempalace.backends`` entry point, selectable through ``MEMPALACE_BACKEND`` /
``get_backend("turbovec")``, but not the in-tree default.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Optional

import numpy as np
import turbovec

from .base import (
    BaseBackend,
    BaseCollection,
    GetResult,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
)

# turbovec's bit menu is {2, 3, 4}; 4 is the recall ceiling (PR #5).
_DEFAULT_BIT_WIDTH = 4
# How many turbovec candidates to pull before exact-cosine re-rank. Headroom
# above n_results recovers the few true neighbours turbovec mis-ranks near ties.
_RERANK_FANOUT = 5
_RERANK_FLOOR = 50

_VALID_INCLUDE = frozenset({"documents", "metadatas", "distances", "embeddings"})


def _l2_normalize(matrix) -> np.ndarray:
    """Row-wise L2 normalization → float32, with a zero-vector guard."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(m / norms, dtype=np.float32)


def _resolve_include(include: Optional[list]) -> dict:
    if include is None:
        return {"documents": True, "metadatas": True, "distances": True, "embeddings": False}
    keys = {k for k in include if k in _VALID_INCLUDE}
    return {
        "documents": "documents" in keys,
        "metadatas": "metadatas" in keys,
        "distances": "distances" in keys,
        "embeddings": "embeddings" in keys,
    }


# ---------------------------------------------------------------------------
# where / where_document → SQL
# ---------------------------------------------------------------------------


def _clause_to_sql(field: str, value) -> tuple:
    """Translate one ``{field: value}`` clause into (sql, params).

    Supported (MVP): scalar equality and ``{"$in": [...]}``. Anything else
    raises ``UnsupportedFilterError`` — silent dropping is forbidden by spec.
    """
    path = f"$.{field}"
    if isinstance(value, dict):
        if set(value.keys()) != {"$in"}:
            raise UnsupportedFilterError(
                f"turbovec backend supports only equality and $in on field {field!r}, "
                f"got {sorted(value.keys())}"
            )
        members = value["$in"]
        if not isinstance(members, (list, tuple)) or not members:
            raise UnsupportedFilterError(f"$in on {field!r} requires a non-empty list")
        placeholders = ",".join("?" for _ in members)
        return (f"json_extract(metadata, ?) IN ({placeholders})", [path, *members])
    # scalar equality
    return ("json_extract(metadata, ?) = ?", [path, value])


def _where_to_sql(where: Optional[dict]) -> tuple:
    """Translate a ``where`` dict into (sql, params). Empty/None → ("", [])."""
    if not where:
        return ("", [])
    if "$and" in where:
        if set(where.keys()) != {"$and"}:
            raise UnsupportedFilterError("turbovec backend cannot mix $and with sibling keys")
        frags, params = [], []
        for sub in where["$and"]:
            f, p = _where_to_sql(sub)
            if f:
                frags.append(f)
                params.extend(p)
        return (" AND ".join(frags), params)
    # Reject other top-level operators ($or, $ne, $gt, ...) explicitly.
    for key in where:
        if key.startswith("$"):
            raise UnsupportedFilterError(f"turbovec backend does not support operator {key!r}")
    frags, params = [], []
    for field, value in where.items():
        f, p = _clause_to_sql(field, value)
        frags.append(f)
        params.extend(p)
    return (" AND ".join(frags), params)


def _where_document_to_sql(where_document: Optional[dict]) -> tuple:
    if not where_document:
        return ("", [])
    if set(where_document.keys()) != {"$contains"}:
        raise UnsupportedFilterError(
            f"turbovec backend supports only $contains on where_document, "
            f"got {sorted(where_document.keys())}"
        )
    needle = where_document["$contains"]
    return ("document LIKE ? ESCAPE '\\'", [f"%{_like_escape(needle)}%"])


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _combine(*fragments) -> tuple:
    frags, params = [], []
    for f, p in fragments:
        if f:
            frags.append(f"({f})")
            params.extend(p)
    return (" AND ".join(frags), params)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class TurboVecCollection(BaseCollection):
    """A turbovec index + SQLite sidecar for one collection.

    SQLite layout (the durable store)::

        docs(uid INTEGER PRIMARY KEY,     -- turbovec external uint64 id
             str_id TEXT UNIQUE NOT NULL, -- mempalace string id
             document TEXT, metadata TEXT,-- metadata is JSON
             vector BLOB)                 -- float32 bytes, L2-normalized
        meta(key TEXT PRIMARY KEY, value TEXT)

    ``meta`` carries ``dim``, ``bit_width``, ``next_uid``, and two generation
    counters: ``store_gen`` (bumped on every committed write) and ``tvim_gen``
    (set to ``store_gen`` whenever the ``.tvim`` cache is flushed). On open, a
    ``tvim_gen != store_gen`` (or a missing ``.tvim``) triggers a rebuild from
    the stored vectors — the crash-safety property.
    """

    def __init__(self, coll_dir: str, *, bit_width: int):
        self._dir = coll_dir
        os.makedirs(coll_dir, exist_ok=True)
        self._db_path = os.path.join(coll_dir, "store.sqlite3")
        self._tvim_path = os.path.join(coll_dir, "index.tvim")
        self._lock = threading.RLock()
        self._dirty = False
        self._ef = None  # lazily built embedding function

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            "uid INTEGER PRIMARY KEY, str_id TEXT UNIQUE NOT NULL, "
            "document TEXT, metadata TEXT, vector BLOB)"
        )
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

        self._bit_width = int(self._meta_get("bit_width", bit_width))
        self._meta_set("bit_width", self._bit_width)
        self._dim = self._meta_get("dim", None)
        self._dim = int(self._dim) if self._dim is not None else None
        self._next_uid = int(self._meta_get("next_uid", 0))
        self._conn.commit()

        self._index = None
        self._open_or_rebuild_index()

    # -- meta helpers ---------------------------------------------------------

    def _meta_get(self, key: str, default=None):
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row is not None else default

    def _meta_set(self, key: str, value) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    # -- index lifecycle ------------------------------------------------------

    def _open_or_rebuild_index(self) -> None:
        """Load the .tvim cache if it is consistent with the store; else rebuild."""
        if self._dim is None:
            self._index = None  # lazy: dim commits on first add
            return
        store_gen = int(self._meta_get("store_gen", 0))
        tvim_gen = int(self._meta_get("tvim_gen", -1))
        if os.path.exists(self._tvim_path) and tvim_gen == store_gen:
            try:
                self._index = turbovec.IdMapIndex.load(self._tvim_path)
                return
            except Exception:
                pass  # fall through to rebuild
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        idx = turbovec.IdMapIndex(dim=self._dim, bit_width=self._bit_width)
        rows = self._conn.execute("SELECT uid, vector FROM docs").fetchall()
        if rows:
            uids = np.array([r[0] for r in rows], dtype=np.uint64)
            vecs = np.ascontiguousarray(
                np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]),
                dtype=np.float32,
            )
            idx.add_with_ids(vecs, uids)
        self._index = idx
        self._dirty = True  # cache is now out of sync with on-disk .tvim

    def _flush_index(self) -> None:
        if self._index is None or not self._dirty:
            return
        tmp = self._tvim_path + ".tmp"
        self._index.write(tmp)
        os.replace(tmp, self._tvim_path)
        self._meta_set("tvim_gen", self._meta_get("store_gen", 0))
        self._conn.commit()
        self._dirty = False

    # -- embedding ------------------------------------------------------------

    def _embed(self, texts: list) -> np.ndarray:
        if self._ef is None:
            from ..embedding import get_embedding_function

            self._ef = get_embedding_function()
        return np.asarray(self._ef(list(texts)), dtype=np.float32)

    # -- writes ---------------------------------------------------------------

    def add(self, *, documents, ids, metadatas=None, embeddings=None) -> None:
        self._write(documents=documents, ids=ids, metadatas=metadatas,
                    embeddings=embeddings, replace=False)

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None) -> None:
        self._write(documents=documents, ids=ids, metadatas=metadatas,
                    embeddings=embeddings, replace=True)

    def _write(self, *, documents, ids, metadatas, embeddings, replace) -> None:
        n = len(ids)
        if len(documents) != n:
            raise ValueError(f"documents length {len(documents)} != ids length {n}")
        if metadatas is not None and len(metadatas) != n:
            raise ValueError(f"metadatas length {len(metadatas)} != ids length {n}")
        if embeddings is not None and len(embeddings) != n:
            raise ValueError(f"embeddings length {len(embeddings)} != ids length {n}")
        if n == 0:
            return

        vecs = embeddings if embeddings is not None else self._embed(documents)
        vecs = _l2_normalize(vecs)

        with self._lock:
            if self._dim is None:
                self._dim = int(vecs.shape[1])
                self._meta_set("dim", self._dim)
                self._index = turbovec.IdMapIndex(dim=self._dim, bit_width=self._bit_width)
            elif vecs.shape[1] != self._dim:
                from .base import DimensionMismatchError

                raise DimensionMismatchError(
                    f"embedding dim {vecs.shape[1]} != collection dim {self._dim}"
                )

            new_uids, new_vecs = [], []
            for i, str_id in enumerate(ids):
                meta_json = json.dumps(metadatas[i] if metadatas else {})
                vec_bytes = vecs[i].tobytes()
                row = self._conn.execute(
                    "SELECT uid FROM docs WHERE str_id=?", (str_id,)
                ).fetchone()
                if row is not None:
                    if not replace:
                        raise ValueError(f"id {str_id!r} already exists (use upsert)")
                    uid = int(row[0])
                    self._index.remove(np.uint64(uid))
                    self._conn.execute(
                        "UPDATE docs SET document=?, metadata=?, vector=? WHERE uid=?",
                        (documents[i], meta_json, vec_bytes, uid),
                    )
                else:
                    uid = self._next_uid
                    self._next_uid += 1
                    self._conn.execute(
                        "INSERT INTO docs(uid, str_id, document, metadata, vector) "
                        "VALUES(?, ?, ?, ?, ?)",
                        (uid, str_id, documents[i], meta_json, vec_bytes),
                    )
                new_uids.append(uid)
                new_vecs.append(vecs[i])

            self._index.add_with_ids(
                np.ascontiguousarray(np.stack(new_vecs), dtype=np.float32),
                np.array(new_uids, dtype=np.uint64),
            )
            self._meta_set("next_uid", self._next_uid)
            self._bump_store_gen()
            self._conn.commit()
            self._dirty = True

    def delete(self, *, ids=None, where=None) -> None:
        with self._lock:
            uids = self._select_uids(ids=ids, where=where)
            if not uids:
                return
            for uid in uids:
                try:
                    self._index.remove(np.uint64(uid))
                except Exception:
                    pass
            qmarks = ",".join("?" for _ in uids)
            self._conn.execute(f"DELETE FROM docs WHERE uid IN ({qmarks})", uids)
            self._bump_store_gen()
            self._conn.commit()
            self._dirty = True

    def _bump_store_gen(self) -> None:
        self._meta_set("store_gen", int(self._meta_get("store_gen", 0)) + 1)

    def _select_uids(self, *, ids=None, where=None) -> list:
        frags, params = [], []
        if ids is not None:
            qmarks = ",".join("?" for _ in ids)
            frags.append(f"str_id IN ({qmarks})")
            params.extend(ids)
        wsql, wparams = _where_to_sql(where)
        if wsql:
            frags.append(wsql)
            params.extend(wparams)
        clause = (" WHERE " + " AND ".join(frags)) if frags else ""
        rows = self._conn.execute(f"SELECT uid FROM docs{clause}", params).fetchall()
        return [int(r[0]) for r in rows]

    # -- reads ----------------------------------------------------------------

    def query(self, *, query_texts=None, query_embeddings=None, n_results=10,
              where=None, where_document=None, include=None) -> QueryResult:
        if (query_texts is None) == (query_embeddings is None):
            raise ValueError("exactly one of query_texts / query_embeddings is required")
        spec = _resolve_include(include)
        emb_req = spec["embeddings"]

        if query_texts is not None:
            if len(query_texts) == 0:
                raise ValueError("query_texts must be non-empty")
            q = self._embed(query_texts)
        else:
            if len(query_embeddings) == 0:
                raise ValueError("query_embeddings must be non-empty")
            q = np.asarray(query_embeddings, dtype=np.float32)
        q = _l2_normalize(q)
        nq = q.shape[0]

        with self._lock:
            if self._index is None or self._dim is None:
                return QueryResult.empty(num_queries=nq, embeddings_requested=emb_req)

            # Filter → allowlist of uids. Empty filter → search whole index.
            allow = None
            if where or where_document:
                fsql, fparams = _combine(_where_to_sql(where), _where_document_to_sql(where_document))
                clause = (" WHERE " + fsql) if fsql else ""
                rows = self._conn.execute(f"SELECT uid FROM docs{clause}", fparams).fetchall()
                allow_ids = [int(r[0]) for r in rows]
                if not allow_ids:
                    return QueryResult.empty(num_queries=nq, embeddings_requested=emb_req)
                allow = np.array(allow_ids, dtype=np.uint64)

            pool = max(n_results, min(_RERANK_FLOOR, n_results * _RERANK_FANOUT))
            ids_out, docs_out, metas_out, dists_out, embs_out = [], [], [], [], []
            for qi in range(nq):
                qrow = q[qi].reshape(1, -1)
                _, cand = self._index.search(qrow, k=pool, allowlist=allow)
                cand_uids = [int(u) for u in cand[0].tolist()]
                hits = self._rerank(qrow[0], cand_uids, n_results, spec)
                ids_out.append([h["str_id"] for h in hits])
                docs_out.append([h["document"] for h in hits] if spec["documents"] else [])
                metas_out.append([h["metadata"] for h in hits] if spec["metadatas"] else [])
                dists_out.append([h["distance"] for h in hits] if spec["distances"] else [])
                if emb_req:
                    embs_out.append([h["vector"] for h in hits])

            return QueryResult(
                ids=ids_out,
                documents=docs_out,
                metadatas=metas_out,
                distances=dists_out,
                embeddings=embs_out if emb_req else None,
            )

    def _rerank(self, qvec: np.ndarray, cand_uids: list, n_results: int, spec: dict) -> list:
        """Exact-cosine re-rank of a turbovec candidate pool. Returns top n_results."""
        if not cand_uids:
            return []
        qmarks = ",".join("?" for _ in cand_uids)
        rows = self._conn.execute(
            f"SELECT uid, str_id, document, metadata, vector FROM docs WHERE uid IN ({qmarks})",
            cand_uids,
        ).fetchall()
        by_uid = {int(r[0]): r for r in rows}
        scored = []
        for uid in cand_uids:  # preserve turbovec's order as a tiebreak
            r = by_uid.get(uid)
            if r is None:
                continue
            v = np.frombuffer(r[4], dtype=np.float32)
            cosine = float(np.dot(qvec, v))
            distance = 1.0 - cosine  # qvec and v are L2-normalized → cosine ∈ [-1, 1]
            scored.append({
                "str_id": r[1],
                "document": r[2] if spec["documents"] else None,
                "metadata": json.loads(r[3]) if spec["metadatas"] and r[3] else {},
                "distance": distance,
                "vector": v.tolist() if spec["embeddings"] else None,
            })
        scored.sort(key=lambda h: h["distance"])
        return scored[:n_results]

    def get(self, *, ids=None, where=None, where_document=None, limit=None,
            offset=None, include=None) -> GetResult:
        spec = _resolve_include(include)
        frags, params = [], []
        if ids is not None:
            qmarks = ",".join("?" for _ in ids)
            frags.append(f"str_id IN ({qmarks})")
            params.extend(ids)
        wsql, wparams = _where_to_sql(where)
        if wsql:
            frags.append(wsql)
            params.extend(wparams)
        wdsql, wdparams = _where_document_to_sql(where_document)
        if wdsql:
            frags.append(wdsql)
            params.extend(wdparams)
        clause = (" WHERE " + " AND ".join(frags)) if frags else ""
        sql = f"SELECT str_id, document, metadata, vector FROM docs{clause} ORDER BY uid"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
            if offset is not None:
                sql += " OFFSET ?"
                params.append(int(offset))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out_ids, out_docs, out_metas, out_embs = [], [], [], []
        for r in rows:
            out_ids.append(r[0])
            out_docs.append(r[1] if spec["documents"] else None)
            out_metas.append(json.loads(r[2]) if spec["metadatas"] and r[2] else {})
            if spec["embeddings"]:
                out_embs.append(np.frombuffer(r[3], dtype=np.float32).tolist())
        return GetResult(
            ids=out_ids,
            documents=out_docs,
            metadatas=out_metas,
            embeddings=out_embs if spec["embeddings"] else None,
        )

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            try:
                self._flush_index()
            finally:
                self._conn.commit()
                self._conn.close()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TurboVecBackend(BaseBackend):
    """Factory serving turbovec-backed collections, one SQLite+.tvim per palace."""

    name = "turbovec"
    spec_version = "1.0"
    capabilities = frozenset({
        "supports_embeddings_in",
        "supports_embeddings_out",
        "supports_metadata_filters",
        "local_mode",
    })

    def __init__(self):
        self._collections: dict = {}
        self._lock = threading.Lock()

    @staticmethod
    def _palace_path(palace) -> str:
        if isinstance(palace, PalaceRef):
            return palace.local_path or palace.id
        return str(palace)

    def get_collection(self, palace=None, *, collection_name, create=False,
                       options=None) -> TurboVecCollection:
        path = self._palace_path(palace)
        if not create and not os.path.isdir(path):
            raise PalaceNotFoundError(path)
        os.makedirs(path, exist_ok=True)

        bit_width = _DEFAULT_BIT_WIDTH
        if options and options.get("bit_width"):
            bit_width = int(options["bit_width"])

        key = (os.path.abspath(path), collection_name)
        with self._lock:
            coll = self._collections.get(key)
            if coll is None:
                coll_dir = os.path.join(path, "turbovec", collection_name)
                coll = TurboVecCollection(coll_dir, bit_width=bit_width)
                self._collections[key] = coll
            return coll

    def close_palace(self, palace) -> None:
        target = os.path.abspath(self._palace_path(palace))
        with self._lock:
            for key in [k for k in self._collections if k[0] == target]:
                self._collections.pop(key).close()

    def close(self) -> None:
        with self._lock:
            for coll in self._collections.values():
                try:
                    coll.close()
                except Exception:
                    pass
            self._collections.clear()

    @classmethod
    def detect(cls, path: str) -> bool:
        return os.path.isdir(os.path.join(path, "turbovec"))
