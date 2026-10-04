# Loaded into mempalace.searcher via exec (see __init__.py). Not a standalone module.
if __name__ != "mempalace.searcher":
    raise ImportError(f"{__name__} is an implementation fragment; import mempalace.searcher")


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


def _search_error_result(error: str, **extra) -> dict:
    """Error envelope for programmatic search callers.

    Always includes ``results: []`` so callers can safely index
    ``result["results"]`` without a KeyError when the palace failed to
    open or the query raised mid-flight (Windows CI flake surface).
    """
    out = {"error": error, "results": []}
    out.update(extra)
    return out
