"""MemPalace storage backend backed by turbovecdb.

This is a thin adapter: the storage engine (turbovec 4-bit ANN + a durable
SQLite sidecar, exact-cosine re-rank, metadata filters, persistence, and
multi-process safety) lives in the standalone ``turbovecdb`` library
(https://github.com/kostadis/turbovecdb). MemPalace is turbovecdb's first
customer; this module maps the RFC 001 ``BaseBackend`` / ``BaseCollection``
contract onto turbovecdb's API.

Per palace, turbovecdb collections live under ``<palace>/turbovec/<name>/``.
Vectors come from MemPalace's configured embedding function: the miner passes
precomputed embeddings; query-by-text is embedded lazily via the same function
(handed to turbovecdb as its ``embedder``). Distances are true cosine in
``[0, 2]``, matching what ``searcher._hybrid_rank`` expects.

Selectable via ``MEMPALACE_BACKEND=turbovec`` / ``get_backend("turbovec")``; not
the in-tree default.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import turbovecdb

from .base import (
    BaseBackend,
    BaseCollection,
    DimensionMismatchError,
    GetResult,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
)

_DEFAULT_BIT_WIDTH = 4


def _lazy_embedder():
    """An embedder callable that resolves MemPalace's embedding function on first
    use — so adding precomputed vectors never pays the model-load cost."""
    cache = {}

    def embed(texts):
        ef = cache.get("ef")
        if ef is None:
            from ..embedding import get_embedding_function

            ef = cache["ef"] = get_embedding_function()
        return ef(list(texts))

    return embed


def _map_include(include: Optional[list]) -> Optional[list]:
    """MemPalace's ``embeddings`` include key is ``vectors`` in turbovecdb."""
    if include is None:
        return None
    return ["vectors" if k == "embeddings" else k for k in include]


class TurboVecCollection(BaseCollection):
    """Maps a turbovecdb ``Collection`` onto the MemPalace ``BaseCollection``."""

    def __init__(self, col):
        self._col = col

    # -- writes ---------------------------------------------------------------

    def add(self, *, documents, ids, metadatas=None, embeddings=None) -> None:
        try:
            self._col.add(ids=ids, documents=documents, metadatas=metadatas,
                          vectors=embeddings)
        except turbovecdb.DimensionMismatchError as e:
            raise DimensionMismatchError(str(e)) from e

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None) -> None:
        try:
            self._col.upsert(ids=ids, documents=documents, metadatas=metadatas,
                             vectors=embeddings)
        except turbovecdb.DimensionMismatchError as e:
            raise DimensionMismatchError(str(e)) from e

    def delete(self, *, ids=None, where=None) -> None:
        try:
            self._col.delete(ids=ids, where=where)
        except turbovecdb.UnsupportedFilterError as e:
            raise UnsupportedFilterError(str(e)) from e

    # -- reads ----------------------------------------------------------------

    def query(self, *, query_texts=None, query_embeddings=None, n_results=10,
              where=None, where_document=None, include=None) -> QueryResult:
        if (query_texts is None) == (query_embeddings is None):
            raise ValueError("exactly one of query_texts / query_embeddings is required")
        items = query_texts if query_texts is not None else query_embeddings
        if len(items) == 0:
            raise ValueError("query input list must be non-empty")
        emb_requested = include is not None and "embeddings" in include
        tv_include = _map_include(include) if include is not None else \
            ["documents", "metadatas", "distances"]

        ids, docs, metas, dists, embs = [], [], [], [], []
        for i, item in enumerate(items):
            kwargs = {"text": item} if query_texts is not None else {"vector": item}
            try:
                r = self._col.query(k=n_results, where=where, where_document=where_document,
                                    include=tv_include, **kwargs)
            except turbovecdb.UnsupportedFilterError as e:
                raise UnsupportedFilterError(str(e)) from e
            ids.append(r.ids)
            docs.append(r.documents)
            metas.append(r.metadatas)
            dists.append(r.distances)
            if emb_requested:
                embs.append(r.vectors if r.vectors is not None else [])
        return QueryResult(
            ids=ids, documents=docs, metadatas=metas, distances=dists,
            embeddings=embs if emb_requested else None,
        )

    def get(self, *, ids=None, where=None, where_document=None, limit=None,
            offset=None, include=None) -> GetResult:
        emb_requested = include is not None and "embeddings" in include
        tv_include = _map_include(include) if include is not None else ["documents", "metadatas"]
        try:
            r = self._col.get(ids=ids, where=where, where_document=where_document,
                              limit=limit, offset=offset, include=tv_include)
        except turbovecdb.UnsupportedFilterError as e:
            raise UnsupportedFilterError(str(e)) from e
        return GetResult(
            ids=r.ids, documents=r.documents, metadatas=r.metadatas,
            embeddings=r.vectors if emb_requested else None,
        )

    def count(self) -> int:
        return self._col.count()

    def close(self) -> None:
        self._col.close()


class TurboVecBackend(BaseBackend):
    """Factory serving turbovecdb-backed collections, one DB dir per palace."""

    name = "turbovec"
    spec_version = "1.0"
    capabilities = frozenset({
        "supports_embeddings_in",
        "supports_embeddings_out",
        "supports_metadata_filters",
        "local_mode",
    })

    def __init__(self):
        self._dbs: dict = {}  # palace_path -> turbovecdb.Database
        self._collections: dict = {}  # (palace_path, name) -> TurboVecCollection
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
        bit_width = int(options["bit_width"]) if options and options.get("bit_width") \
            else _DEFAULT_BIT_WIDTH

        abspath = os.path.abspath(path)
        key = (abspath, collection_name)
        with self._lock:
            existing = self._collections.get(key)
            if existing is not None:
                return existing
            db = self._dbs.get(abspath)
            if db is None:
                db = turbovecdb.connect(os.path.join(path, "turbovec"))
                self._dbs[abspath] = db
            try:
                col = db.collection(collection_name, bit_width=bit_width, metric="cosine",
                                    embedder=_lazy_embedder(), create=create)
            except turbovecdb.CollectionNotFoundError as e:
                raise PalaceNotFoundError(path) from e
            wrapped = TurboVecCollection(col)
            self._collections[key] = wrapped
            return wrapped

    def close_palace(self, palace) -> None:
        target = os.path.abspath(self._palace_path(palace))
        with self._lock:
            for key in [k for k in self._collections if k[0] == target]:
                self._collections.pop(key)
            db = self._dbs.pop(target, None)
            if db is not None:
                db.close()

    def close(self) -> None:
        with self._lock:
            for db in self._dbs.values():
                try:
                    db.close()
                except Exception:
                    pass
            self._dbs.clear()
            self._collections.clear()

    @classmethod
    def detect(cls, path: str) -> bool:
        return os.path.isdir(os.path.join(path, "turbovec"))
