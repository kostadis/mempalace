#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Two-tier retrieval. ``primary`` is verbatim drawer cosine + BM25 hybrid;
its ranking depends only on the drawers themselves. ``themes`` is the
LLM-judgment layer — closet rows (prose + taxonomy) that semantically
match the query — surfaced as a separate list so bad closets cannot
corrupt the primary answer. The two lists answer different questions:
``primary`` returns the user's exact words; ``themes`` suggests
conceptual neighbors to drill into.
"""

import functools
import logging
import math
import os
import re
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Optional

from .backends import BackendError, BackendMismatchError
from .config import MempalaceConfig, sqlite_read_uri
from .date_window import filed_at_in_window, parse_window
from .i18n import _canonical_lang, get_stopwords
from .palace import (
    _open_collection_or_explain,
    get_closets_collection,
    get_collection,
    resolve_backend_name,
)

# Closet pointer line format: "topic|entities|→drawer_id_a,drawer_id_b"
# Multiple lines may join with newlines inside one closet document.
_CLOSET_DRAWER_REF_RE = re.compile(r"→([\w,]+)")

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def _first_or_empty(results, key: str) -> list:
    """Return the first inner list of a query result field, or [].

    Accepts both the typed :class:`QueryResult` (attribute access) and the
    pre-typed chroma dict shape; this polymorphism is retained so test mocks
    still work and callers mid-migration do not crash. Preserves the empty-
    collection semantics from issue #195: when no queries returned hits, the
    outer list may be empty and indexing ``[0]`` would raise.
    """
    outer = getattr(results, key, None) if not isinstance(results, dict) else results.get(key)
    if not outer:
        return []
    return outer[0] or []


def _aligned_query_ids(results, document_count: int) -> list:
    """Return query IDs padded to match the document result column.

    Production backends return an ID for every document. Some legacy test
    mocks omit IDs, so pad with ``None`` instead of letting ``zip`` discard
    otherwise valid mocked results.
    """
    ids = list(_first_or_empty(results, "ids"))
    if len(ids) < document_count:
        ids.extend([None] * (document_count - len(ids)))
    return ids[:document_count]


def _result_drawer_id(meta, stored_drawer_id):
    """Return the ID that round-trips through ``mempalace_get_drawer``.

    Chunk metadata carries the logical-group id under ``parent_drawer_id``
    (``tool_add_drawer``) or ``parent_entry_id`` (``tool_diary_write``);
    resolving both means a hit on a chunked diary entry reports the id that
    fetches the WHOLE entry rather than the one chunk that matched (#2185).
    Kept in sync with ``mcp_server._PARENT_ID_KEYS``.
    """
    meta = meta or {}
    return meta.get("parent_drawer_id") or meta.get("parent_entry_id") or stored_drawer_id


def _tokenize(text: str, stop_words: frozenset = frozenset()) -> list:
    """Lowercase + strip to alphanumeric tokens of length ≥ 2.

    Tolerates ``None`` documents — Chroma can return ``None`` in the
    ``documents`` field for drawers without text content, which would
    otherwise raise ``AttributeError`` mid-rerank.

    When ``stop_words`` is non-empty, filters tokens that match any entry.
    The set is expected to already be lowercased so callers can share one
    instance across query + document tokenization.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    if stop_words:
        return [t for t in tokens if t not in stop_words]
    return tokens


@functools.lru_cache(maxsize=16)
def _stopwords_for_canonical(canonical_lang: str) -> frozenset:
    """Cached stop-word set keyed by a canonical locale code.

    Splitting canonicalization out of the cache key avoids thrashing when
    callers pass equivalent variants (``"EN"``, ``"en"``, ``"en-US"``) —
    they all hit the same cache slot.
    """
    return frozenset(get_stopwords(canonical_lang))


def _stopwords_for_lang(lang: str) -> frozenset:
    """Resolve raw ``lang`` to its canonical form before cache lookup.

    Kept as the public-shaped helper (callers and tests reach for this
    name) while the lru_cache lives on ``_stopwords_for_canonical`` to
    keep the cache key normalized.
    """
    canonical = _canonical_lang(lang) or lang.lower()
    return _stopwords_for_canonical(canonical)


def _resolve_stop_words(lang: Optional[str]) -> frozenset:
    """Return the BM25 stop-word set for ``lang`` as an opt-in feature.

    When ``lang`` is an explicit string, loads that locale's stop words.
    When ``lang`` is ``None``, resolution order is:

    1. ``MEMPALACE_LANG`` / ``MEMPAL_LANG`` environment variable.
    2. ``MempalaceConfig().lang_explicit`` (which itself reads the env vars
       first, then ``config.json["lang"]``).

    The env-var fast path avoids constructing ``MempalaceConfig`` (which
    reads ``config.json`` from disk) on the hot search path when the user
    has set the env var — the common case for explicit-locale palaces.
    Palaces that never configured a language get an empty set, preserving
    pre-PR scoring byte-for-byte.
    """
    if lang is None:
        env_val = os.environ.get("MEMPALACE_LANG") or os.environ.get("MEMPAL_LANG")
        if env_val and env_val.strip():
            lang = env_val.strip()
        else:
            try:
                lang = MempalaceConfig().lang_explicit
            except Exception:
                logger.debug("lang resolution failed, skipping stop-word filter", exc_info=True)
                return frozenset()
        if lang is None:
            return frozenset()
    return _stopwords_for_lang(lang)


def _bm25_scores(
    query: str,
    documents: list,
    k1: float = 1.5,
    b: float = 0.75,
    stop_words: frozenset = frozenset(),
) -> list:
    """Compute Okapi-BM25 scores for ``query`` against each document.

    IDF is computed over the *provided corpus* using the Lucene/BM25+
    smoothed formula ``log((N - df + 0.5) / (df + 0.5) + 1)``, which is
    always non-negative. This is well-defined for re-ranking a small
    candidate set returned by vector retrieval — IDF then reflects how
    discriminative each query term is *within the candidates*, exactly
    what's needed to reorder them.

    Parameters mirror Okapi-BM25 conventions:
        k1 — term-frequency saturation (1.2-2.0 typical, 1.5 default)
        b  — length normalization (0.0 = none, 1.0 = full, 0.75 default)

    Returns a list of scores in the same order as ``documents``.
    """
    n_docs = len(documents)
    query_terms = set(_tokenize(query, stop_words))
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d, stop_words) for d in documents]
    doc_lens = [len(toks) for toks in tokenized]
    if not any(doc_lens):
        return [0.0] * n_docs
    avgdl = sum(doc_lens) / n_docs or 1.0

    # Document frequency: how many docs contain each query term?
    df = {term: 0 for term in query_terms}
    for toks in tokenized:
        seen = set(toks) & query_terms
        for term in seen:
            df[term] += 1

    idf = {term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in query_terms}

    scores = []
    for toks, dl in zip(tokenized, doc_lens):
        if dl == 0:
            scores.append(0.0)
            continue
        tf: dict = {}
        for t in toks:
            if t in query_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * num / den
        scores.append(score)
    return scores


def _distance_to_similarity(distance, metric: str = "cosine") -> float:
    """Map a backend-reported ``distance`` to a [0, 1]-ish similarity.

    The backend contract for the ``distances`` field is *lower = closer*
    regardless of metric (RFC 001, backend metric declaration), so every
    mapping here is monotonic decreasing in ``distance``. The output stays
    bounded so it is
    commensurable with the min-max-normalized BM25 term in
    :func:`_hybrid_rank`.

    * ``cosine`` — distance ∈ [0, 2], 0 = identical: ``max(0, 1 - d)``.
    * ``l2`` — Euclidean ∈ [0, ∞): ``1 / (1 + d)`` (1 at d=0, →0 as d→∞).
    * ``ip`` — inner-product distance (e.g. pgvector ``<#>`` = -dot, lower =
      closer), unbounded and signed: logistic squash ``1 / (1 + e^d)``.
      Provisional until a real ip backend exercises it; no in-tree backend
      uses ip today.

    ``distance is None`` (vector-unknown, e.g. a BM25-only candidate) maps to
    0.0 so the candidate scores on its BM25 contribution alone.
    """
    if distance is None:
        return 0.0
    m = (metric or "cosine").lower()
    if m == "l2":
        return 1.0 / (1.0 + max(0.0, distance))
    if m == "ip":
        # Clamp the exponent so a large positive distance can't overflow.
        return 1.0 / (1.0 + math.exp(min(60.0, distance)))
    # cosine (default)
    return max(0.0, 1.0 - distance)


def _metric_for_collection(col) -> str:
    """Resolve a collection's declared distance metric, defaulting to cosine.

    Reads the ``distance_metric`` exposed by the backend collection (the
    RFC 001 backend metric declaration). ``EmbeddingCollection`` delegates the
    attribute to its inner collection; legacy Chroma palaces report their
    actual ``hnsw:space``.
    Any failure falls back to ``"cosine"`` — the value all in-tree backends
    use and the only metric MemPalace created palaces with historically.
    """
    try:
        metric = getattr(col, "distance_metric", "cosine")
    except Exception:
        return "cosine"
    metric = str(metric or "cosine").lower()
    return metric if metric in ("cosine", "l2", "ip") else "cosine"


def _hybrid_rank(
    results: list,
    query: str,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    metric: str = "cosine",
    stop_words: frozenset = frozenset(),
) -> list:
    """Re-rank ``results`` by a convex combination of vector similarity and BM25.

    * Vector similarity is derived from each candidate's backend-reported
      ``distance`` via :func:`_distance_to_similarity`, interpreted in the
      collection's declared ``metric`` (per RFC 001) rather than assuming
      cosine. Absolute (not relative-to-max) means adding/removing a
      candidate can't reshuffle the others.
    * BM25 is real Okapi-BM25 with corpus-relative IDF over the candidates
      themselves. Since the absolute scale is unbounded, BM25 is min-max
      normalized within the candidate set so weights are commensurable.

    Candidates with ``distance=None`` are treated as vector-unknown
    (no vector signal available) and scored on BM25 contribution alone.
    Used by candidate-union mode to merge BM25-only candidates that the
    vector index didn't surface.

    Mutates each result dict to add ``bm25_score`` and reorders the list
    in place. Returns the same list for convenience.
    """
    if not results:
        return results

    docs = [r.get("text", "") for r in results]
    bm25_raw = _bm25_scores(query, docs, stop_words=stop_words)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 for s in bm25_raw] if max_bm25 > 0 else [0.0] * len(bm25_raw)

    scored = []
    for r, raw, norm in zip(results, bm25_raw, bm25_norm):
        vec_sim = _distance_to_similarity(r.get("distance"), metric)
        r["bm25_score"] = round(raw, 3)
        scored.append((vector_weight * vec_sim + bm25_weight * norm, r))

    # Break exact score ties toward the more recently authored drawer so equal-score
    # candidates rank chronologically instead of in arbitrary backend order. ISO-8601
    # ``authored_at`` strings sort chronologically; missing dates sort oldest.
    # authored_at lives at the top level on the search_memories path and nested under
    # "metadata" on the candidate-union path; check both so the tie-break works for each.
    scored.sort(
        key=lambda pair: (
            pair[0],
            pair[1].get("authored_at") or pair[1].get("metadata", {}).get("authored_at") or "",
        ),
        reverse=True,
    )
    results[:] = [r for _, r in scored]
    return results


def build_where_filter(wing: str = None, room: str = None, source_file: str = None) -> dict:
    """Build a ChromaDB where filter from optional wing/room/source_file.

    ChromaDB needs a ``$and`` only when ≥2 clauses are present; a single
    clause is returned bare and zero clauses yield an empty filter (#1815).
    """
    clauses = []
    if wing:
        clauses.append({"wing": wing})
    if room:
        clauses.append({"room": room})
    if source_file:
        clauses.append({"source_file": source_file})
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _build_where_filter_multi(wing_filters=None, room_filters=None, source_file=None) -> dict:
    """Build ChromaDB where filter supporting multi-valued wing/room scopes.

    Used by ``search_within`` to express ``wing IN {a, b, c}`` and/or
    ``room IN {x, y}`` in a single query — the common shape for the leaf
    step of hierarchical descent, where wing/room pruning has already
    identified several candidate scopes.

    Single-value lists degrade to a plain ``{"field": value}`` so older
    Chroma versions (which don't accept ``$in`` on single values in all
    positions) behave identically to ``build_where_filter``.
    """

    def _clause(field: str, values) -> dict:
        if not values:
            return {}
        vals = [v for v in values if v]
        if not vals:
            return {}
        if len(vals) == 1:
            return {field: vals[0]}
        return {field: {"$in": list(vals)}}

    wing_clause = _clause("wing", wing_filters)
    room_clause = _clause("room", room_filters)
    # ``source_file`` is an exact single-value equality on the full stored
    # value (#1815) — matching ``build_where_filter``'s semantics.
    source_clause = {"source_file": source_file} if source_file else {}
    clauses = [c for c in (wing_clause, room_clause, source_clause) if c]
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _extract_drawer_ids_from_closet(closet_doc: str) -> list:
    """Parse all `→drawer_id_a,drawer_id_b` pointers out of a closet document.

    Preserves order and dedupes.
    """
    seen: dict = {}
    for match in _CLOSET_DRAWER_REF_RE.findall(closet_doc):
        for did in match.split(","):
            did = did.strip()
            if did and did not in seen:
                seen[did] = None
    return list(seen.keys())


def _scoped_source_filter(source_file: str, parent_drawer_id=None) -> dict:
    """Build a Chroma ``where`` clause that scopes a query to ``source_file``,
    additionally constrained by ``parent_drawer_id`` when one is supplied.

    Two unrelated oversized ``tool_add_drawer`` writes (chunked path from
    #1539) can pass the same ``source_file`` (e.g. two pastes tagged
    ``"chat.log"``); each call stores its own ``parent_drawer_id`` group
    of chunks but the bare ``source_file`` filter pulls chunks from both
    groups as if they were siblings (#1580). When the matched chunk
    carries a ``parent_drawer_id`` the filter narrows to that logical
    group. Otherwise (pre-#1539 drawers, single-chunk writes, and
    ``diary_ingest`` drawers grouped by real file path) the original
    file-global shape is preserved. Mirrors the conditional-``$and``
    precedent in ``build_where_filter``.
    """
    if parent_drawer_id:
        return {
            "$and": [
                {"source_file": source_file},
                {"parent_drawer_id": parent_drawer_id},
            ]
        }
    return {"source_file": source_file}


def _expand_with_neighbors(drawers_col, matched_doc: str, matched_meta: dict, radius: int = 1):
    """Expand a matched drawer with its ±radius sibling chunks in the same source file.

    Motivation — "drawer-grep context" feature: a closet hit returns one
    drawer, but the chunk boundary may clip mid-thought (e.g., the matched
    chunk says "here's a breakdown:" and the actual breakdown lives in the
    next chunk). Fetching the small neighborhood around the match gives
    callers enough context without forcing a follow-up ``get_drawer`` call.

    Returns a dict with:
        ``text``            combined chunks in chunk_index order
        ``drawer_index``    the matched chunk's index in the source file
        ``total_drawers``   total drawer count for the source file (or None)

    On any ChromaDB failure or missing metadata, falls back to returning the
    matched drawer alone so search never breaks because neighbor expansion
    failed.
    """
    src = matched_meta.get("source_file")
    chunk_idx = matched_meta.get("chunk_index")
    if not src or not isinstance(chunk_idx, int):
        return {"text": matched_doc, "drawer_index": chunk_idx, "total_drawers": None}

    # Narrow by ``parent_drawer_id`` when present so chunks from unrelated
    # logical drawers sharing ``source_file`` do not stitch (#1580). See
    # ``_scoped_source_filter`` for the contract.
    parent_id = matched_meta.get("parent_drawer_id")
    target_indexes = [chunk_idx + offset for offset in range(-radius, radius + 1)]
    neighbor_clauses = [
        {"source_file": src},
        {"chunk_index": {"$in": target_indexes}},
    ]
    if parent_id:
        neighbor_clauses.append({"parent_drawer_id": parent_id})
    try:
        neighbors = drawers_col.get(
            where={"$and": neighbor_clauses},
            include=["documents", "metadatas"],
        )
    except Exception:
        return {"text": matched_doc, "drawer_index": chunk_idx, "total_drawers": None}

    indexed_docs = []
    for doc, meta in zip(neighbors.documents, neighbors.metadatas):
        ci = meta.get("chunk_index")
        if isinstance(ci, int):
            indexed_docs.append((ci, doc))
    indexed_docs.sort(key=lambda pair: pair[0])

    if not indexed_docs:
        combined_text = matched_doc
    else:
        combined_text = "\n\n".join(doc for _, doc in indexed_docs)

    # Cheap total_drawers lookup. When ``parent_drawer_id`` is present the
    # count is scoped to that group so the returned number matches the
    # text the caller gets back. Without a parent id, the legacy
    # file-global count is preserved.
    total_drawers = None
    try:
        all_meta = drawers_col.get(
            where=_scoped_source_filter(src, parent_id),
            include=["metadatas"],
        )
        total_drawers = len(all_meta.ids) if all_meta.ids else None
    except Exception:
        logger.debug("total_drawers lookup failed for %s", src, exc_info=True)

    return {
        "text": combined_text,
        "drawer_index": chunk_idx,
        "total_drawers": total_drawers,
    }


def _warn_if_legacy_metric(col) -> None:
    """Print a one-line notice if the palace was created without
    ``hnsw:space=cosine``.

    ChromaDB's default is L2 (Euclidean), under which cosine-based
    similarity interpretation falls apart — distances routinely exceed
    1.0 and the display ``max(0, 1 - dist)`` floors every result to 0.
    Legacy palaces (mined before this metadata was consistently set)
    need ``mempalace repair`` to rebuild with the correct metric.

    The warning fires only for palaces that clearly have the wrong
    metric; palaces with no metadata table at all (empty dict) also
    fall under this check since that is the signal of a pre-metadata
    palace.
    """
    try:
        meta = getattr(col, "metadata", None)
    except Exception:
        return
    if not isinstance(meta, dict):
        return
    space = meta.get("hnsw:space")
    if space == "cosine":
        return
    # Either missing or set to something else — both are suspect.
    import sys as _sys

    detail = f"hnsw:space={space!r}" if space else "no hnsw:space metadata"
    print(
        f"\n  NOTICE: this palace was created without cosine distance ({detail}).\n"
        "          Semantic similarity scores will not be meaningful.\n"
        "          Run `mempalace repair` to rebuild the index with the correct metric.",
        file=_sys.stderr,
    )


def _hnsw_capacity_diverged(palace_path: str) -> bool:
    """Return True if HNSW divergence is severe enough to crash ChromaDB.

    Thin, exception-safe wrapper around
    :func:`mempalace.backends.chroma.hnsw_capacity_status`. Used by the
    CLI search path to short-circuit to the BM25-only fallback before
    opening a Chroma client. Client construction and collection identity
    checks can themselves touch the damaged index, so guarding only
    ``col.query()`` is too late (#1222 covers the MCP path via the module-level
    ``_vector_disabled`` flag; this covers the CLI path).

    A probe that raises falls through to ``False`` so the caller proceeds
    to the normal vector path — the underlying query then either succeeds
    (probe was a false negative) or raises its own diagnostic error. The
    probe itself must never be the thing that crashes search.
    """
    try:
        from .backends.chroma import hnsw_capacity_status
        from .config import get_configured_collection_name

        info = hnsw_capacity_status(palace_path, get_configured_collection_name())
        return bool(info.get("diverged"))
    except Exception:
        logger.debug("HNSW capacity probe raised; proceeding to vector path", exc_info=True)
        return False


def _print_search_results_bm25_only(
    query: str,
    palace_path: str,
    wing: str,
    room: str,
    n_results: int,
    stop_words: frozenset = frozenset(),
    since_dt=None,
    before_dt=None,
) -> None:
    """CLI fallback printer for when HNSW divergence fences off vector search.

    Mirrors the vector-path output shape so users get lexical matches in
    the format they expect, plus a clear notice pointing at
    ``mempalace repair``. Replaces the silent SIGBUS users otherwise hit
    when the CLI called ``col.query()`` against a diverged segment.

    ``stop_words`` reaches the BM25 scorer here for the same reason
    :func:`_vector_disabled_search` forwards it on the MCP side: this path
    still ranks by BM25, so dropping the filter would rank a diverged
    palace by different rules than a healthy one.

    An active ``[since_dt, before_dt)`` window is forwarded to the BM25
    reader, which post-filters on it. A diverged index degrades the
    ranking; it must never widen the result set past the window the
    caller asked for.
    """
    result = _bm25_only_via_sqlite(
        query=query,
        palace_path=palace_path,
        wing=wing,
        room=room,
        n_results=n_results,
        stop_words=stop_words,
        since_dt=since_dt,
        before_dt=before_dt,
    )
    hits = result.get("results", [])

    print(
        "\n  NOTICE: vector search disabled — HNSW index has diverged from SQLite.\n"
        "          Showing BM25-only results. Run `mempalace repair` to restore "
        "vector search.\n"
    )
    print(f"{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    if not hits:
        print(f'  No results found for: "{query}"')
        return

    for i, hit in enumerate(hits, 1):
        bm25 = hit.get("bm25_score", 0.0)
        wing_name = hit.get("wing", "?")
        room_name = hit.get("room", "?")
        source = Path(hit.get("source_file", "?")).name

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  bm25={bm25}  (vector disabled)")
        print()
        for line in (hit.get("text", "") or "").strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'-' * 56}")

    print()


def _candidate_pool_size(n_results: int, date_window_active: bool) -> int:
    """Rerank-pool size for the drawer vector query.

    Without a date window this is the historical ``n_results * 3``
    over-fetch. With one, the window filters the pool AFTER retrieval
    (ChromaDB rejects string operands for ``$gte``/``$lt``, so ``filed_at``
    can't be range-filtered server-side), and a narrow window over a large
    palace would starve a 3x pool even though matching drawers exist —
    recall is the design requirement. Widen to ``n_results * 15``, capped
    at 500 (the ceiling the filter-fallback path already uses) — except
    the pool never drops below ``n_results`` itself, or an oversized
    request could return fewer rows than an unfiltered query would.
    """
    if not date_window_active:
        return n_results * 3
    return max(min(n_results * 15, 500), n_results)


def search(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    since: str = None,
    before: str = None,
):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect), and/or narrow to
    drawers whose ``filed_at`` falls in the ``[since, before)`` window —
    same semantics as ``search_memories``/``list_drawers`` (#1128/#463).
    """
    # Resolved before the fence below: both exits from this function rank by
    # BM25, so the filter has to be in hand on either branch.
    stop_words = _resolve_stop_words(None)

    # Parse the window before probing the palace: an inverted or malformed
    # bound is a caller error and must raise identically whether or not the
    # index turns out to be diverged.
    try:
        since_dt, before_dt = parse_window(since, before)
    except ValueError as e:
        print(f"\n  {e}")
        raise SearchError(str(e)) from e
    date_window_active = since_dt is not None or before_dt is not None

    # Probe a Chroma palace before get_collection(). Opening the client can
    # load native index state, and embedder-identity enforcement may call
    # collection.count(); both happen before the old query-only guard and can
    # hit the same native crash. Non-Chroma backends never use Chroma's HNSW
    # files or sqlite-specific fallback and proceed normally.
    try:
        backend_name = resolve_backend_name(palace_path)
    except (BackendMismatchError, KeyError):
        # Preserve _open_collection_or_explain's state-specific diagnostics
        # for mixed artifacts and unknown backend selections. This probe is
        # only an early Chroma safety fence; it must not become a second,
        # less-helpful backend validation path.
        backend_name = None

    if backend_name == "chroma" and _hnsw_capacity_diverged(palace_path):
        return _print_search_results_bm25_only(
            query,
            palace_path,
            wing,
            room,
            n_results,
            stop_words=stop_words,
            since_dt=since_dt,
            before_dt=before_dt,
        )

    col = _open_collection_or_explain(palace_path, opener=get_collection)
    if col is None:
        if not os.path.isdir(palace_path):
            raise SearchError(f"No palace found at {palace_path}")
        raise SearchError(f"No palace database at {palace_path}")

    # Alert the user if this palace predates hnsw:space=cosine being set on
    # creation — their similarity scores will be junk until they run repair.
    _warn_if_legacy_metric(col)

    where = build_where_filter(wing, room)

    try:
        kwargs = {
            "query_texts": [query],
            # The window is a post-filter (ChromaDB can't range-compare
            # string metadata), so widen the fetch the same way the
            # programmatic path does and trim back after filtering.
            "n_results": _candidate_pool_size(n_results, date_window_active)
            if date_window_active
            else n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

    except Exception as e:
        print(f"\n  Search error: {e}")
        raise SearchError(f"Search error: {e}") from e

    docs = _first_or_empty(results, "documents")
    metas = _first_or_empty(results, "metadatas")
    dists = _first_or_empty(results, "distances")

    if date_window_active:
        kept = [
            (doc, meta, dist)
            for doc, meta, dist in zip(docs, metas, dists)
            if filed_at_in_window((meta or {}).get("filed_at"), since_dt, before_dt)
        ]
        # Keep the whole in-window pool here; the hybrid re-rank below must
        # see every survivor before the display cut to n_results, or a
        # BM25-strong drawer deep in the pool could never surface.
        docs = [k[0] for k in kept]
        metas = [k[1] for k in kept]
        dists = [k[2] for k in kept]

    if not docs:
        print(f'\n  No results found for: "{query}"')
        return

    # Pure-cosine retrieval on the CLI path was missing lexical matches:
    # a drawer whose text contains every query term can still score distance
    # >= 1.0 against the natural-language query when the drawer is a
    # mechanical artifact (directory listing, diff, log fragment) that
    # embeds as file-tree noise rather than as prose about its subject.
    # The MCP tool path already hybridizes BM25 with vector sim via
    # `_hybrid_rank`; do the same here so CLI results match what agents
    # see via `mempalace_search`.
    metric = _metric_for_collection(col)
    hits = [
        {"text": doc or "", "distance": float(dist), "metadata": meta or {}}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    hits = _hybrid_rank(hits, query, metric=metric, stop_words=stop_words)
    if date_window_active:
        # The widened fetch exists only to survive the window filter; the
        # display contract stays "top n_results", now cut AFTER the re-rank.
        hits = hits[:n_results]

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    if since:
        print(f"  Since: {since}")
    if before:
        print(f"  Before: {before}")
    print(f"{'=' * 60}\n")

    for i, hit in enumerate(hits, 1):
        vec_sim = round(_distance_to_similarity(hit["distance"], metric), 3)
        bm25 = hit.get("bm25_score", 0.0)
        meta = hit["metadata"]
        source = Path(meta.get("source_file", "?")).name
        wing_name = meta.get("wing", "?")
        room_name = meta.get("room", "?")

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  {metric}_sim={vec_sim}  bm25={bm25}")
        print()
        # Print the verbatim text, indented
        for line in hit["text"].strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'-' * 56}")

    print()

    # ── Themes: LLM-tagged conceptual neighbors ──────────────────────
    # The themes path queries the closets collection separately and is
    # presented as a distinct block. These hits reflect LLM judgment and
    # should be reviewed before drilling in — they're suggestions, not
    # verbatim matches.
    theme_hits = _themes_for_cli(query, palace_path, where, n_results)
    if theme_hits:
        print(f"{'=' * 60}")
        print("  Related (LLM-tagged themes — review before drilling in)")
        print(f"{'=' * 60}\n")
        for i, t in enumerate(theme_hits, 1):
            sim = t["similarity"]
            gen = t["generated_by"] or "unknown"
            print(f"  [{i}] {t['wing']} / {t['room']}")
            print(f"      Source: {t['source_file']}")
            print(f"      Theme:  similarity={sim}  generated_by={gen}")
            print()
            for line in t["closet_text"].strip().split("\n"):
                print(f"      {line}")
            print()
            print(f"  {'─' * 56}")
        print()


def _themes_for_cli(query: str, palace_path: str, where: dict, n_results: int) -> list:
    """Fetch the themes block for the CLI ``search`` printer.

    Mirrors the closet path inside :func:`search_within` so the CLI shows
    the same conceptual-neighbors layer. Failures degrade silently to an
    empty list — themes are advisory, never load-bearing.
    """
    try:
        closets_col = get_closets_collection(palace_path, create=False)
    except Exception:
        return []
    try:
        ckwargs = {
            "query_texts": [query],
            "n_results": max(n_results * 4, 10),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            ckwargs["where"] = where
        closet_results = closets_col.query(**ckwargs)
    except Exception:
        return []

    taxonomy_pen = _taxonomy_penalty()
    seen: set = set()
    candidates: list = []
    for cdoc, cmeta, cdist in zip(
        _first_or_empty(closet_results, "documents"),
        _first_or_empty(closet_results, "metadatas"),
        _first_or_empty(closet_results, "distances"),
    ):
        cmeta = cmeta or {}
        source = cmeta.get("source_file", "") or ""
        if not source or source in seen:
            continue
        seen.add(source)
        generated_by = cmeta.get("generated_by", "") or ""
        sort_dist = cdist + taxonomy_pen if generated_by.startswith("taxonomy:") else cdist
        candidates.append(
            (
                sort_dist,
                {
                    "source_file": Path(source).name if source else "?",
                    "wing": cmeta.get("wing", "unknown"),
                    "room": cmeta.get("room", "unknown"),
                    "closet_text": (cdoc or "")[:500],
                    "similarity": round(max(0.0, 1 - float(cdist)), 3),
                    "generated_by": generated_by,
                },
            )
        )
    candidates.sort(key=lambda p: p[0])
    return [c[1] for c in candidates[:n_results]]


def _window_sql_prefilters(since_dt, before_dt) -> list:
    """(operator, bound-string) pairs for the SQL date-window narrowing.

    A SQL-side *narrowing* on the ISO ``filed_at`` string, kept at
    whole-DAY granularity so it is provably wider than the window for
    every ISO-8601 spelling that shares the YYYY-MM-DD prefix (bare date,
    space separator, minute precision, Z/offset suffixes) — a
    full-isoformat bound would sort after some of those on the boundary
    day and drop an in-window row at the SQL layer, where the
    authoritative Python re-filter (offset drop, unparseable exclusion —
    mirroring the wing/room double-check) can't recover it. Day
    granularity costs at most one extra day of candidates per bound;
    Python decides the exact window.
    """
    prefilters = []
    if since_dt is not None:
        prefilters.append((">=", since_dt.date().isoformat()))
    if before_dt is not None:
        try:
            upper = (before_dt + timedelta(days=1)).date().isoformat()
        except OverflowError:
            # before at the calendar ceiling ("9999-12-31" as an open-ended
            # sentinel): there is no next day to bound by, so skip the SQL
            # narrowing entirely — the Python re-filter stays authoritative
            # and such a window is effectively unbounded above anyway.
            upper = None
        if upper is not None:
            prefilters.append(("<", upper))
    return prefilters


def _search_error_result(error: str, **extra) -> dict:
    """Error envelope for programmatic search callers.

    Always includes ``results: []`` so callers can safely index
    ``result["results"]`` without a KeyError when the palace failed to
    open or the query raised mid-flight (Windows CI flake surface).
    """
    out = {"error": error, "results": []}
    out.update(extra)
    return out


def _bm25_only_via_sqlite(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    source_file: str = None,
    n_results: int = 5,
    max_candidates: int = 500,
    _include_internal: bool = False,
    collection_name: str = None,
    stop_words: frozenset = frozenset(),
    since_dt=None,
    before_dt=None,
) -> dict:
    """BM25-only search reading drawers directly from chroma.sqlite3.

    Used when HNSW is diverged or unloadable (#1222). Bypasses chromadb's
    Python client entirely so a corrupt vector segment can't segfault the
    MCP server. Routes through chromadb's own FTS5 trigram index
    (``embedding_fulltext_search``) for candidate selection, then re-ranks
    with the same Okapi-BM25 used in :func:`_hybrid_rank` so the result
    shape matches the vector path.

    The query is split into ≥3-char trigram-tokens and OR-joined for the
    FTS5 MATCH — chromadb writes the index with ``tokenize='trigram'``,
    so single-character tokens never match. When no usable token survives
    (e.g. "is a"), candidate selection falls back to the most-recent
    ``max_candidates`` rows so we still return *something* rather than
    nothing.
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return _search_error_result(
            "No palace found",
            hint="Run: mempalace init <dir> && mempalace mine <dir>",
        )
    if collection_name is None:
        from .config import get_configured_collection_name

        collection_name = get_configured_collection_name()

    def _metadata_filter_sql(row_id_expr: str) -> tuple[str, list[str]]:
        clauses = []
        params = []
        for key, value in (("wing", wing), ("room", room), ("source_file", source_file)):
            if not value:
                continue
            clauses.append(
                f"""
                AND EXISTS (
                    SELECT 1
                    FROM embedding_metadata mf
                    WHERE mf.id = {row_id_expr}
                      AND mf.key = ?
                      AND COALESCE(
                        mf.string_value,
                        CAST(mf.int_value AS TEXT),
                        CAST(mf.float_value AS TEXT),
                        CAST(mf.bool_value AS TEXT)
                      ) = ?
                )
                """
            )
            params.extend([key, value])
        for op, sql_bound in _window_sql_prefilters(since_dt, before_dt):
            clauses.append(
                f"""
                AND EXISTS (
                    SELECT 1
                    FROM embedding_metadata mf
                    WHERE mf.id = {row_id_expr}
                      AND mf.key = 'filed_at'
                      AND mf.string_value {op} ?
                )
                """
            )
            params.append(sql_bound)
        return "".join(clauses), params

    try:
        conn = sqlite3.connect(sqlite_read_uri(db_path), uri=True)
    except sqlite3.Error as e:
        return _search_error_result(f"sqlite open failed: {e}")

    window_active = since_dt is not None or before_dt is not None
    try:
        # FTS5 MATCH expects whitespace-separated tokens. Drop tokens
        # shorter than 3 chars (trigram tokenizer can't match them).
        tokens = [t for t in _tokenize(query) if len(t) >= 3]
        candidate_ids: list[int] = []
        use_recency_fallback = not tokens
        if tokens:
            fts_query = " OR ".join(tokens)
            filter_sql, filter_params = _metadata_filter_sql("embedding_fulltext_search.rowid")
            try:
                rows = conn.execute(
                    f"""
                    SELECT embedding_fulltext_search.rowid
                    FROM embedding_fulltext_search
                    JOIN embeddings e ON e.id = embedding_fulltext_search.rowid
                    JOIN segments s ON e.segment_id = s.id
                    JOIN collections c ON s.collection = c.id
                    WHERE embedding_fulltext_search MATCH ?
                      AND c.name = ?
                    {filter_sql}
                    LIMIT ?
                    """,
                    (fts_query, collection_name, *filter_params, max_candidates),
                ).fetchall()
                candidate_ids = [r[0] for r in rows]
            except sqlite3.Error:
                # FTS5 tokenizer mismatch or syntax error — fall through
                # to the recency-window selector below.
                logger.debug("FTS5 MATCH failed; using recency fallback", exc_info=True)
                use_recency_fallback = True

        if not candidate_ids and use_recency_fallback:
            # No usable FTS tokens, or FTS itself failed — pull the most
            # recent rows for the drawers segment so we can BM25-rank
            # something rather than return empty-handed. A clean FTS miss
            # must stay empty, especially after wing/room filtering, because
            # recency fallback would return unrelated scoped drawers.
            # Wrapped in try/except because the schema may differ on legacy
            # palaces (older chromadb without ``created_at``, missing
            # ``segments`` rows after partial restore, etc.); on schema
            # mismatch we fall back to ordering by primary-key id and finally
            # to an empty result rather than letting search raise.
            try:
                filter_sql, filter_params = _metadata_filter_sql("e.id")
                rows = conn.execute(
                    f"""
                    SELECT e.id
                    FROM embeddings e
                    JOIN segments s ON e.segment_id = s.id
                    JOIN collections c ON s.collection = c.id
                    WHERE c.name = ?
                    {filter_sql}
                    ORDER BY e.created_at DESC
                    LIMIT ?
                    """,
                    (collection_name, *filter_params, max_candidates),
                ).fetchall()
                candidate_ids = [r[0] for r in rows]
            except sqlite3.Error:
                logger.debug(
                    "recency-window query failed; trying id-ordered fallback",
                    exc_info=True,
                )
                try:
                    filter_sql, filter_params = _metadata_filter_sql("e.id")
                    rows = conn.execute(
                        f"""
                        SELECT e.id
                        FROM embeddings e
                        JOIN segments s ON e.segment_id = s.id
                        JOIN collections c ON s.collection = c.id
                        WHERE c.name = ?
                        {filter_sql}
                        ORDER BY e.id DESC
                        LIMIT ?
                        """,
                        (collection_name, *filter_params, max_candidates),
                    ).fetchall()
                    candidate_ids = [r[0] for r in rows]
                except sqlite3.Error:
                    logger.debug("id-ordered fallback also failed", exc_info=True)
                    candidate_ids = []

        # A full candidate page means rows beyond it never got a chance to
        # match the window — mirror the vector path's truncation honesty
        # (``date_filter_pool_truncated``) instead of a silently thin result.
        window_pool_truncated = window_active and len(candidate_ids) >= max_candidates

        if not candidate_ids:
            return {
                "query": query,
                "filters": {"wing": wing, "room": room, "source_file": source_file},
                "total_before_filter": 0,
                "primary": [],
                "results": [],
                "themes": [],
                "fallback": "bm25_only_via_sqlite",
            }

        placeholders = ",".join(["?"] * len(candidate_ids))
        meta_rows = conn.execute(
            f"""
            SELECT m.id, e.embedding_id, m.key, m.string_value, m.int_value
            FROM embedding_metadata AS m
            JOIN embeddings AS e ON e.id = m.id
            WHERE m.id IN ({placeholders})
            """,
            candidate_ids,
        ).fetchall()
    finally:
        conn.close()

    # Group metadata rows into per-drawer dicts.
    drawers: dict[int, dict] = {}
    for emb_id, stored_drawer_id, key, sval, ival in meta_rows:
        d = drawers.setdefault(
            emb_id,
            {
                "_id": emb_id,
                "_stored_drawer_id": stored_drawer_id,
                "metadata": {},
                "text": "",
            },
        )
        if key == "chroma:document":
            d["text"] = sval or ""
        else:
            d["metadata"][key] = sval if sval is not None else ival

    # Apply wing/room filters in Python (FTS5 candidates may include
    # entries from other wings).
    candidates = []
    for d in drawers.values():
        meta = d["metadata"]
        if wing and meta.get("wing") != wing:
            continue
        if room and meta.get("room") != room:
            continue
        if source_file and meta.get("source_file") != source_file:
            continue
        if window_active and not filed_at_in_window(meta.get("filed_at"), since_dt, before_dt):
            continue
        full_source = meta.get("source_file", "") or ""
        candidates.append(
            {
                "drawer_id": _result_drawer_id(meta, d["_stored_drawer_id"]),
                "text": d["text"],
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(full_source).name if full_source else "?",
                "source_path": full_source,
                "created_at": meta.get("filed_at", "unknown"),
                "authored_at": meta.get("authored_at", meta.get("filed_at", "unknown")),
                # No vector distance available in BM25-only mode.
                "similarity": None,
                "distance": None,
                "matched_via": "bm25_sqlite",
                # Internal: full path + chunk_index let callers (notably
                # candidate_strategy="union") dedupe at chunk granularity
                # rather than basename — two files in different directories
                # may share a basename, and one source_file is split across
                # multiple chunks. Stripped before this helper returns.
                "_source_file_full": full_source,
                "_chunk_index": meta.get("chunk_index"),
            }
        )

    # Local BM25 over the candidate set.
    docs = [c["text"] for c in candidates]
    bm25_raw = _bm25_scores(query, docs, stop_words=stop_words)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    for c, raw in zip(candidates, bm25_raw):
        c["bm25_score"] = round(raw, 3)
        c["_score"] = (raw / max_bm25) if max_bm25 > 0 else 0.0
    candidates.sort(key=lambda c: c["_score"], reverse=True)
    hits = candidates[:n_results]
    for h in hits:
        h.pop("_score", None)
        # Strip internal fields by default so the public BM25-only fallback
        # response stays clean. Callers that need chunk-precise dedup
        # (notably the union-merge path) opt in via _include_internal.
        if not _include_internal:
            h.pop("_source_file_full", None)
            h.pop("_chunk_index", None)

    result = {
        "query": query,
        "filters": {"wing": wing, "room": room, "source_file": source_file},
        "total_before_filter": len(candidates),
        "primary": hits,
        # ``results`` is an alias for ``primary`` kept so v3.3.5 callers that
        # consumed the BM25 fallback under the old key keep working without
        # losing the palace-isolation-era ``primary`` consumers.
        "results": hits,
        "themes": [],
        "fallback": "bm25_only_via_sqlite",
        "fallback_reason": "vector_search_disabled",
    }
    if window_pool_truncated:
        result["date_filter_pool_truncated"] = True
    return result


def _taxonomy_penalty() -> float:
    """Distance penalty applied to taxonomy-generated closet rows in ``themes``
    ranking. Prose rows are file-specific and usually more informative than
    the generic taxonomy labels, so we discount taxonomy hits by a small
    constant before sorting. Env-overridable via
    ``MEMPALACE_THEME_TAXONOMY_PENALTY``; default 0.05."""
    raw = os.environ.get("MEMPALACE_THEME_TAXONOMY_PENALTY")
    if raw is None:
        return 0.05
    try:
        return float(raw)
    except ValueError:
        return 0.05


def _query_drawers_with_filter_fallback(
    drawers_col, dkwargs, query, n_results, wing_list, room_list, source_file=None
):
    """Run the filtered drawer query, falling back to an unfiltered query plus a
    Python-side post-filter when ChromaDB raises on the filtered query.

    A ChromaDB HNSW/SQLite index mismatch makes filtered queries fail with
    "Error finding id" even when unfiltered search works fine — it happens when
    drawers are ingested via two different paths (e.g. bulk import vs MCP tool
    calls), leaving the vector index inconsistent with the metadata store. We
    retry unfiltered (over-fetching) and re-apply the wing/room/source_file filter in Python.
    See #1245 / #1035.

    Adapted to the local two-list ``search_within`` shape: ``wing_list`` /
    ``room_list`` are the normalized multi-value filters, so the post-filter is
    a set-membership test rather than upstream's single-value equality.
    """
    where = dkwargs.get("where")
    try:
        return drawers_col.query(**dkwargs)
    except Exception as filter_err:
        if not where:
            raise
        logger.warning(
            "Filtered search failed (%s); falling back to unfiltered + post-filter",
            filter_err,
        )
        raw = drawers_col.query(
            query_texts=[query],
            n_results=min(n_results * 15, 500),
            include=["documents", "metadatas", "distances"],
        )
        raw_docs = _first_or_empty(raw, "documents")
        raw_ids = _aligned_query_ids(raw, len(raw_docs))
        wing_set = set(wing_list or [])
        room_set = set(room_list or [])
        fids, fdocs, fmetas, fdists = [], [], [], []
        for stored_drawer_id, doc, meta, dist in zip(
            raw_ids,
            raw_docs,
            _first_or_empty(raw, "metadatas"),
            _first_or_empty(raw, "distances"),
        ):
            meta = meta or {}
            if wing_set and meta.get("wing") not in wing_set:
                continue
            if room_set and meta.get("room") not in room_set:
                continue
            if source_file and meta.get("source_file") != source_file:
                continue
            fids.append(stored_drawer_id)
            fdocs.append(doc)
            fmetas.append(meta)
            fdists.append(dist)
        return {
            "ids": [fids],
            "documents": [fdocs],
            "metadatas": [fmetas],
            "distances": [fdists],
        }


def _backend_capabilities(col) -> frozenset:
    backend = getattr(col, "_backend", None)
    if backend is None:
        inner = getattr(col, "_inner", None)
        backend = getattr(inner, "_backend", None) if inner is not None else None
    caps = getattr(backend, "capabilities", None) if backend is not None else None
    return caps if isinstance(caps, (set, frozenset)) else frozenset()


def _fetch_closet_candidates(closets_col, *, query: str, n_results: int, where: dict):
    """Return raw ``(documents, metadatas, distances)`` candidate lists for the
    themes pass, routed by backend capability.

    sqlite_exact (and any lexical-only backend) has no real cosine index over
    the closets collection — closets are pointer lines, so BM25 via
    ``lexical_search`` is the correct signal, not a vector ``.query()`` the
    backend would otherwise have to fake. A lexical hit's relevance ``score``
    isn't a distance, so it's converted to a rank-order pseudo-distance
    (``1 - 1/(rank+2)``: 0.5, 0.667, 0.75, ...) that sorts the same way
    downstream as the cosine-distance branch below, without claiming a
    magnitude the backend never reported.
    """
    if "supports_lexical_search" in _backend_capabilities(closets_col):
        result = closets_col.lexical_search(query=query, n_results=n_results, where=where or None)
        hits = getattr(result, "hits", None) or []
        docs = [h.document for h in hits]
        metas = [h.metadata for h in hits]
        dists = [1.0 - (1.0 / (rank + 2)) for rank in range(len(hits))]
        return docs, metas, dists

    ckwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        ckwargs["where"] = where
    closet_results = closets_col.query(**ckwargs)
    return (
        _first_or_empty(closet_results, "documents"),
        _first_or_empty(closet_results, "metadatas"),
        _first_or_empty(closet_results, "distances"),
    )


def search_within(
    query: str,
    palace_path: str,
    *,
    wing_filters=None,
    room_filters=None,
    source_file: str = None,
    ids=None,
    n_results: int = 5,
    n_themes: int = None,
    max_distance: float = 0.0,
    collection_name: str = None,
    lang: Optional[str] = None,
) -> dict:
    """Generic scoped search primitive — the leaf of hierarchical descent.

    Returns two distinct lists. ``primary`` is verbatim drawer cosine +
    BM25 hybrid; its ranking depends only on the drawers themselves.
    ``themes`` is the LLM-judgment layer — closet rows (prose +
    taxonomy) that semantically match the query, deduped to one per
    source_file. Closet content does not influence ``primary`` ranking.

    Args:
        query: Natural-language search query.
        palace_path: Path to the ChromaDB palace directory.
        wing_filters: Optional iterable of wing names to restrict the search to.
        room_filters: Optional iterable of room names to restrict the search to.
        source_file: Optional exact source_file filter. Matches the full
            stored ``source_file`` metadata value verbatim, scoping both
            the drawer (primary) and closet (themes) queries (#1815).
        ids: Optional iterable of drawer IDs; results are post-filtered to
            this set. Use when a prior pruning step has chosen specific
            drawers and you want to rerank them against a fresh query.
        n_results: Maximum primary hits to return.
        n_themes: Maximum theme hits to return. Defaults to ``n_results``.
        max_distance: Cosine-distance cutoff for primary hits (0 disables).

    Returns a dict with ``query``, ``filters``, ``total_before_filter``,
    ``primary`` (drawer hits), and ``themes`` (closet hits).
    """
    try:
        if collection_name is not None:
            drawers_col = get_collection(palace_path, collection_name=collection_name, create=False)
        else:
            drawers_col = get_collection(palace_path, create=False)
    except BackendError as e:
        # Distinguish a backend that failed to open (service down, mismatch)
        # from a palace that simply doesn't exist yet — collapsing both to
        # "No palace found" hid real backend failures behind an init hint.
        return {
            "error": "Backend error",
            "details": str(e),
            "hint": "The configured storage backend failed to open. "
            "Check the backend service and configuration.",
        }
    except Exception as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }

    # Normalize filter args to deterministic lists.
    wing_list = [w for w in (wing_filters or []) if w]
    room_list = [r for r in (room_filters or []) if r]
    id_set = set(ids) if ids else None
    theme_limit = n_themes if n_themes is not None else n_results

    where = _build_where_filter_multi(wing_list, room_list, source_file)

    # Primary path: drawer cosine + BM25 hybrid. Closet content plays no
    # role here — that's what makes ``primary`` immune to bad closets.
    try:
        dkwargs = {
            "query_texts": [query],
            "n_results": n_results * 3,  # over-fetch for re-ranking
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            dkwargs["where"] = where
        drawer_results = _query_drawers_with_filter_fallback(
            drawers_col, dkwargs, query, n_results, wing_list, room_list, source_file
        )
    except Exception as e:
        # Callers index result["results"] unconditionally, error or not
        # (Windows KeyError flake) — never omit it, even on a mid-query
        # exception.
        return {"error": f"Search error: {e}", "results": [], "primary": [], "themes": []}

    scored: list = []
    for drawer_id, doc, meta, dist in zip(
        _first_or_empty(drawer_results, "ids"),
        _first_or_empty(drawer_results, "documents"),
        _first_or_empty(drawer_results, "metadatas"),
        _first_or_empty(drawer_results, "distances"),
    ):
        if max_distance > 0.0 and dist > max_distance:
            continue
        if id_set is not None and drawer_id not in id_set:
            continue
        meta = meta or {}
        source = meta.get("source_file", "") or ""
        scored.append(
            {
                # Resolve chunk ids to their logical parent (#2080/#2185) so a
                # hit on one chunk reports the id that round-trips the whole
                # drawer through mempalace_get_drawer, not just that chunk.
                "drawer_id": _result_drawer_id(meta, drawer_id),
                "text": doc,
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(source).name if source else "?",
                # Full stored source_file value; lets a caller round-trip a
                # result back into an exact ``source_file`` filter (#1815).
                "source_path": source,
                "created_at": meta.get("filed_at", "unknown"),
                "authored_at": meta.get("authored_at", meta.get("filed_at", "unknown")),
                "similarity": round(max(0.0, 1 - dist), 3),
                "distance": round(dist, 4),
                "_sort_key": dist,
            }
        )

    scored.sort(key=lambda h: h["_sort_key"])
    primary = scored[:n_results]
    primary = _hybrid_rank(primary, query, stop_words=_resolve_stop_words(lang))
    for h in primary:
        h.pop("_sort_key", None)

    # Themes path: closet rows (prose + taxonomy) that semantically match
    # the query, deduped to top row per source_file. These are the
    # LLM-judgment layer — useful as conceptual neighbors, but kept
    # cleanly separated from ``primary`` so weak closets cannot mislead
    # the verbatim answer.
    themes: list = []
    try:
        closets_col = get_closets_collection(palace_path, create=False)
        closet_docs, closet_metas, closet_dists = _fetch_closet_candidates(
            closets_col, query=query, n_results=max(theme_limit * 4, 10), where=where
        )

        taxonomy_pen = _taxonomy_penalty()
        seen_sources: set = set()
        candidate_themes: list = []
        for cdoc, cmeta, cdist in zip(closet_docs, closet_metas, closet_dists):
            cmeta = cmeta or {}
            source = cmeta.get("source_file", "") or ""
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            generated_by = cmeta.get("generated_by", "") or ""
            drawer_ids = _extract_drawer_ids_from_closet(cdoc or "")
            sort_dist = cdist
            if generated_by.startswith("taxonomy:"):
                sort_dist = cdist + taxonomy_pen
            candidate_themes.append(
                {
                    "source_file": Path(source).name if source else "?",
                    "wing": cmeta.get("wing", "unknown"),
                    "room": cmeta.get("room", "unknown"),
                    "closet_text": (cdoc or "")[:500],
                    "closet_distance": round(float(cdist), 4),
                    "similarity": round(max(0.0, 1 - float(cdist)), 3),
                    "generated_by": generated_by,
                    "drawer_ids": drawer_ids,
                    "_sort_key": sort_dist,
                }
            )
        candidate_themes.sort(key=lambda t: t["_sort_key"])
        themes = candidate_themes[:theme_limit]
        for t in themes:
            t.pop("_sort_key", None)
    except Exception:
        # No closets collection yet, or it errored — themes degrades to [].
        logger.debug("Closet collection unavailable; using drawer-only search", exc_info=True)
        themes = []

    return {
        "query": query,
        "filters": {
            "wing_filters": wing_list or None,
            "room_filters": room_list or None,
            "source_file": source_file,
            "ids": list(id_set) if id_set is not None else None,
        },
        "total_before_filter": len(_first_or_empty(drawer_results, "documents")),
        "primary": primary,
        # ``results`` alias kept for v3.3.5-era consumers that read the
        # hit list under the old key.
        "results": primary,
        "themes": themes,
    }


def search_memories(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    source_file: str = None,
    since: str = None,
    before: str = None,
    n_results: int = 5,
    max_distance: float = 0.0,
    vector_disabled: bool = False,
    collection_name: str = None,
    lang: Optional[str] = None,
) -> dict:
    """Programmatic search — single-wing, single-room convenience wrapper.

    Thin shim over :func:`search_within` that preserves the historical
    single-value filter shape in the ``filters`` return block, so MCP
    tools and scripts that peek at it don't break.

    When ``vector_disabled`` is True the call routes to
    :func:`_bm25_only_via_sqlite` instead — the MCP server sets the flag
    when the HNSW capacity probe detects a divergence that would segfault
    chromadb on segment load (#1222).

    ``lang`` selects the BM25 stop-word locale (opt-in; see
    :func:`_resolve_stop_words`) and is threaded through to
    :func:`search_within`.

    ``since`` / ``before`` are accepted for call-signature compatibility
    with upstream but are **not yet implemented** on this branch — passing
    them is currently a no-op. Upstream's date-window filtering
    (``_window_and_fallback_gate``, ``_candidate_pool_size``-driven pool
    widening, ``date_filter_pool_truncated`` reporting) has not been ported
    into the local ``search_within`` two-list shape. Tracked as a follow-up.

    Used by the MCP server and other callers that need data rather than
    printed output.

    """
    if vector_disabled:
        return _bm25_only_via_sqlite(
            query,
            palace_path,
            wing=wing,
            room=room,
            source_file=source_file,
            n_results=n_results,
            collection_name=collection_name,
            stop_words=_resolve_stop_words(lang),
        )
    result = search_within(
        query,
        palace_path,
        wing_filters=[wing] if wing else None,
        room_filters=[room] if room else None,
        source_file=source_file,
        n_results=n_results,
        max_distance=max_distance,
        collection_name=collection_name,
        lang=lang,
    )
    # Preserve the pre-search_within return shape for existing consumers.
    if "filters" in result:
        result["filters"] = {"wing": wing, "room": room, "source_file": source_file}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Virtual line numbering — read-time grid for drawers (3.3.6).
#
# Drawers are stored verbatim on disk. The reader applies a line-number grid
# at read time so any drawer — numbered or not — can be sectioned by a closet
# pointer like ``→2026-01-18:L55-L72`` without rewriting the corpus. Pure
# functions, no I/O. Source drawer text is never mutated.
# See docs/virtual-line-numbering.md for the full design rationale.
# ─────────────────────────────────────────────────────────────────────────────


# A line is "already numbered" iff it starts with [<digits>].
_ALREADY_NUMBERED_RE = re.compile(r"^\[\d+\]")


def render_with_line_numbers(text: "str | None", start_line: int = 1) -> str:
    """Prefix each line of ``text`` with ``[N] `` for read-time grid display.

    Lines that already begin with ``[<digits>]`` pass through unchanged,
    but the counter still advances on them so callers can rely on positional
    alignment with the original line indices.

    ``None`` is treated as empty string. Pure function.
    """
    if not text:
        return ""
    out = []
    for i, line in enumerate(text.split("\n"), start=start_line):
        if _ALREADY_NUMBERED_RE.match(line):
            out.append(line)
        else:
            out.append(f"[{i}] {line}")
    return "\n".join(out)


def extract_line_range(text: str, line_start: int, line_end: int) -> str:
    """Return the 1-indexed inclusive slice ``[line_start, line_end]`` rendered with line numbers.

    This is the closet-pointer read path. A pointer like ``→2026-01-18:L55-L72``
    resolves by opening the day-drawer and calling ``extract_line_range(drawer_text, 55, 72)``.
    Out-of-bounds ranges are clamped. Invalid ranges return ``""``.
    """
    if not text:
        return ""
    if line_end < line_start:
        return ""

    lines = text.split("\n")
    effective_start = max(1, line_start)
    effective_end = min(len(lines), line_end)

    if effective_start > effective_end:
        return ""

    section = "\n".join(lines[effective_start - 1 : effective_end])
    return render_with_line_numbers(section, start_line=effective_start)
