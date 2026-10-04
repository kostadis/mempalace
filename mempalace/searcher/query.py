# Loaded into mempalace.searcher via exec (see __init__.py). Not a standalone module.
if __name__ != "mempalace.searcher":
    raise ImportError(f"{__name__} is an implementation fragment; import mempalace.searcher")


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
            drawers_col = get_collection(
                palace_path, collection_name=collection_name, create=False, read_only=True
            )
        else:
            drawers_col = get_collection(palace_path, create=False, read_only=True)
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
    vector_weight, bm25_weight = _resolve_hybrid_rank_weights()
    primary = _hybrid_rank(
        primary,
        query,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        stop_words=_resolve_stop_words(lang),
    )
    for h in primary:
        h.pop("_sort_key", None)

    # Themes path: closet rows (prose + taxonomy) that semantically match
    # the query, deduped to top row per source_file. These are the
    # LLM-judgment layer — useful as conceptual neighbors, but kept
    # cleanly separated from ``primary`` so weak closets cannot mislead
    # the verbatim answer.
    themes: list = []
    try:
        closets_col = get_closets_collection(palace_path, create=False, read_only=True)
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
