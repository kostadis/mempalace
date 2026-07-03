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

import logging
import math
import os
import re
import sqlite3
from pathlib import Path

from .backends import CollectionNotInitializedError, PalaceNotFoundError
from .palace import get_closets_collection, get_collection

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


def _tokenize(text: str) -> list:
    """Lowercase + strip to alphanumeric tokens of length ≥ 2.

    Tolerates ``None`` documents — Chroma can return ``None`` in the
    ``documents`` field for drawers without text content, which would
    otherwise raise ``AttributeError`` mid-rerank.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(
    query: str,
    documents: list,
    k1: float = 1.5,
    b: float = 0.75,
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
    query_terms = set(_tokenize(query))
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d) for d in documents]
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


def _hybrid_rank(
    results: list,
    query: str,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list:
    """Re-rank ``results`` by a convex combination of vector similarity and BM25.

    * Vector similarity uses absolute cosine sim ``max(0, 1 - distance)`` —
      ChromaDB's hnsw cosine distance lives in ``[0, 2]`` (0 = identical).
      Absolute (not relative-to-max) means adding/removing a candidate
      can't reshuffle the others.
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
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 for s in bm25_raw] if max_bm25 > 0 else [0.0] * len(bm25_raw)

    scored = []
    for r, raw, norm in zip(results, bm25_raw, bm25_norm):
        distance = r.get("distance")
        if distance is None:
            vec_sim = 0.0
        else:
            vec_sim = max(0.0, 1.0 - distance)
        r["bm25_score"] = round(raw, 3)
        scored.append((vector_weight * vec_sim + bm25_weight * norm, r))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results[:] = [r for _, r in scored]
    return results


def build_where_filter(wing: str = None, room: str = None) -> dict:
    """Build ChromaDB where filter for wing/room filtering."""
    if wing and room:
        return {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        return {"wing": wing}
    elif room:
        return {"room": room}
    return {}


def _build_where_filter_multi(wing_filters=None, room_filters=None) -> dict:
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
    if wing_clause and room_clause:
        return {"$and": [wing_clause, room_clause]}
    return wing_clause or room_clause or {}


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

    target_indexes = [chunk_idx + offset for offset in range(-radius, radius + 1)]
    try:
        neighbors = drawers_col.get(
            where={
                "$and": [
                    {"source_file": src},
                    {"chunk_index": {"$in": target_indexes}},
                ]
            },
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

    # Cheap total_drawers lookup: metadata-only scan of the source file.
    total_drawers = None
    try:
        all_meta = drawers_col.get(where={"source_file": src}, include=["metadatas"])
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


def search(query: str, palace_path: str, wing: str = None, room: str = None, n_results: int = 5):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect).
    """
    # Filesystem-first checks distinguish State A / State B before reaching
    # chromadb. PersistentClient lazily creates chroma.sqlite3 on first open
    # of an empty palace dir, so without these checks State B collapses into
    # the "initialized but empty" State C message and mutates the dir as a
    # side effect of a read-only search call (#1498).
    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        raise SearchError(f"No palace found at {palace_path}")
    if not os.path.isfile(os.path.join(palace_path, "chroma.sqlite3")):
        print(f"\n  Palace dir at {palace_path} exists but has no chroma.sqlite3 yet.")
        print("  Run: mempalace mine <dir>")
        raise SearchError(f"No palace database at {palace_path}")
    try:
        col = get_collection(palace_path, create=False)
    except CollectionNotInitializedError as e:
        # State C from #1498: palace initialized but never mined.
        print(f"\n  Palace at {palace_path} is initialized but empty (no drawers yet).")
        print("  Run: mempalace mine <dir>")
        raise SearchError(f"Palace at {palace_path} is initialized but empty") from e
    except PalaceNotFoundError as e:
        # Backend filesystem-race fallback: dir was deleted between our
        # check above and the backend call. Same message as State A.
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        raise SearchError(f"No palace found at {palace_path}") from e

    # Alert the user if this palace predates hnsw:space=cosine being set on
    # creation — their similarity scores will be junk until they run repair.
    _warn_if_legacy_metric(col)

    where = build_where_filter(wing, room)

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
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
    hits = [
        {"text": doc or "", "distance": float(dist), "metadata": meta or {}}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    hits = _hybrid_rank(hits, query)

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    for i, hit in enumerate(hits, 1):
        vec_sim = round(max(0.0, 1 - hit["distance"]), 3)
        bm25 = hit.get("bm25_score", 0.0)
        meta = hit["metadata"]
        source = Path(meta.get("source_file", "?")).name
        wing_name = meta.get("wing", "?")
        room_name = meta.get("room", "?")

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  cosine={vec_sim}  bm25={bm25}")
        print()
        # Print the verbatim text, indented
        for line in hit["text"].strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

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


def _bm25_only_via_sqlite(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    max_candidates: int = 500,
    _include_internal: bool = False,
    collection_name: str = None,
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
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }
    if collection_name is None:
        from .config import get_configured_collection_name

        collection_name = get_configured_collection_name()

    def _metadata_filter_sql(row_id_expr: str) -> tuple[str, list[str]]:
        clauses = []
        params = []
        for key, value in (("wing", wing), ("room", room)):
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
        return "".join(clauses), params

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return {"error": f"sqlite open failed: {e}"}

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

        if not candidate_ids:
            return {
                "query": query,
                "filters": {"wing": wing, "room": room},
                "total_before_filter": 0,
                "primary": [],
                "results": [],
                "themes": [],
                "fallback": "bm25_only_via_sqlite",
            }

        placeholders = ",".join(["?"] * len(candidate_ids))
        meta_rows = conn.execute(
            f"""
            SELECT id, key, string_value, int_value
            FROM embedding_metadata
            WHERE id IN ({placeholders})
            """,
            candidate_ids,
        ).fetchall()
    finally:
        conn.close()

    # Group metadata rows into per-drawer dicts.
    drawers: dict[int, dict] = {}
    for emb_id, key, sval, ival in meta_rows:
        d = drawers.setdefault(emb_id, {"_id": emb_id, "metadata": {}, "text": ""})
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
        full_source = meta.get("source_file", "") or ""
        candidates.append(
            {
                "text": d["text"],
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(full_source).name if full_source else "?",
                "created_at": meta.get("filed_at", "unknown"),
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
    bm25_raw = _bm25_scores(query, docs)
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

    return {
        "query": query,
        "filters": {"wing": wing, "room": room},
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


def search_within(
    query: str,
    palace_path: str,
    *,
    wing_filters=None,
    room_filters=None,
    ids=None,
    n_results: int = 5,
    n_themes: int = None,
    max_distance: float = 0.0,
    collection_name: str = None,
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

    where = _build_where_filter_multi(wing_list, room_list)

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
        drawer_results = drawers_col.query(**dkwargs)
    except Exception as e:
        return {"error": f"Search error: {e}"}

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
                "drawer_id": drawer_id,
                "text": doc,
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(source).name if source else "?",
                "created_at": meta.get("filed_at", "unknown"),
                "similarity": round(max(0.0, 1 - dist), 3),
                "distance": round(dist, 4),
                "_sort_key": dist,
            }
        )

    scored.sort(key=lambda h: h["_sort_key"])
    primary = scored[:n_results]
    primary = _hybrid_rank(primary, query)
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
        ckwargs = {
            "query_texts": [query],
            "n_results": max(theme_limit * 4, 10),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            ckwargs["where"] = where
        closet_results = closets_col.query(**ckwargs)

        taxonomy_pen = _taxonomy_penalty()
        seen_sources: set = set()
        candidate_themes: list = []
        for cdoc, cmeta, cdist in zip(
            _first_or_empty(closet_results, "documents"),
            _first_or_empty(closet_results, "metadatas"),
            _first_or_empty(closet_results, "distances"),
        ):
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
    n_results: int = 5,
    max_distance: float = 0.0,
    vector_disabled: bool = False,
    collection_name: str = None,
) -> dict:
    """Programmatic search — single-wing, single-room convenience wrapper.

    Thin shim over :func:`search_within` that preserves the historical
    single-value filter shape in the ``filters`` return block, so MCP
    tools and scripts that peek at it don't break.

    When ``vector_disabled`` is True the call routes to
    :func:`_bm25_only_via_sqlite` instead — the MCP server sets the flag
    when the HNSW capacity probe detects a divergence that would segfault
    chromadb on segment load (#1222).

    Used by the MCP server and other callers that need data rather than
    printed output.
    """
    if vector_disabled:
        return _bm25_only_via_sqlite(
            query,
            palace_path,
            wing=wing,
            room=room,
            n_results=n_results,
        )
    result = search_within(
        query,
        palace_path,
        wing_filters=[wing] if wing else None,
        room_filters=[room] if room else None,
        n_results=n_results,
        max_distance=max_distance,
        collection_name=collection_name,
    )
    # Preserve the pre-search_within return shape for existing consumers.
    if "filters" in result:
        result["filters"] = {"wing": wing, "room": room}
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
