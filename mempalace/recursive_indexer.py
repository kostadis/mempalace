"""
recursive_indexer.py — deterministic room + wing index aggregator.

Leaf closets (``mempalace_closets``) are mined constantly; the retrieval
hierarchy needs a compact roll-up at the ``(wing, room)`` and ``wing``
levels so ``tool_search_hierarchical`` can prune the search space before
scoring individual drawers. This module produces those roll-ups by
*arithmetic projection over existing AAAK lines* — no LLM, no prose
rewrite, no drift.

Design rules (all enforced):

* Pure function of leaves. Given the same closet corpus the indexer
  always writes the same room/wing documents, byte-for-byte.
* Idempotent. Re-running on a clean palace is a no-op. Re-running after
  a crash picks up where it left off via the dirty-flag plumbing.
* Rebuildable. ``rebuild_all`` walks the drawer collection and rebuilds
  every index from scratch; intermediate levels contain no information
  that can't be regenerated from leaves.
* Serialized under ``mine_lock``. Concurrent miners + indexer runs never
  corrupt an index document.
* Drift-free. Intermediate levels are rank-bucketed projections of leaf
  AAAK — same grammar, same entity codes, same arrow pointers. A human
  reading a returned wing → room → drawer path can interpret it without
  opening drawers.

This module is the ``recursive_indexer.py`` called out in the MemPalace
sub-plan of the RLM integration plan (see
``CampaignGenerator/docs/rlm_integration_plan.md``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Iterable, Iterator

from .dialect import aggregate_entity_sets, project_closet_lines
from .palace import (
    clear_room_dirty,
    clear_wing_dirty,
    get_closets_collection,
    get_collection,
    get_room_indices_collection,
    get_wing_indices_collection,
    iter_dirty_rooms,
    iter_dirty_wings,
    mine_lock,
)

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────

# Index schema version. Bump when the projection algorithm changes in a way
# that existing index documents should be rebuilt to pick up. ``rebuild_all``
# treats any index doc with a lower ``aaak_version`` as stale.
AAAK_VERSION = 1

# How many top-ranked closet lines survive into each room/wing index.
# Room indices stay small — they are the middle pruning layer and are
# queried through vector similarity, so every extra line is noise. Wing
# indices are slightly larger because they have to cover more territory.
ROOM_INDEX_TOP_N = 100
WING_INDEX_TOP_N = 200

# Hard cap on the character length of the rendered index document body
# before upsert into the *_indices collections. The embedding model
# enforces its own token limit (nomic-embed-text-v1.5 = 2048 tokens).
#
# Why 3000 chars not 6000: closet lines have the shape
#   "<prose>|<entities>|→<drawer_id>,<drawer_id>,..."
# The drawer-ID tails tokenize *much* denser than prose (long
# alphanumeric SHA prefixes split into many tokens each). Empirically a
# 6000-char doc tokenizes to 2049+ tokens, blowing past nomic's 2K
# limit. 3000 chars gives roughly 1.4× safety margin under the cap.
#
# Override via env for embedders with larger context (e.g., bge-m3 at
# 8K supports ~24000 chars).
INDEX_DOC_MAX_CHARS = int(os.environ.get("MEMPALACE_INDEX_DOC_MAX_CHARS", "3000"))

# Paginate through the Chroma collection this many rows at a time. Keeps
# the O(n) scan honest on large palaces; without explicit pagination
# Chroma silently caps ``get`` at a few thousand rows.
_PAGE_SIZE = 1000

# Max entities surfaced in the index metadata ``top_entities`` field —
# a human-readable rollup used for interpretability at a glance.
_TOP_ENTITIES_PER_ROOM = 15
_TOP_ENTITIES_PER_WING = 25


# ── ID helpers ────────────────────────────────────────────────────────────


def _room_index_id(wing: str, room: str) -> str:
    h = hashlib.sha256(f"{wing}|{room}".encode("utf-8")).hexdigest()[:16]
    return f"room_idx_{h}"


def _wing_index_id(wing: str) -> str:
    h = hashlib.sha256(wing.encode("utf-8")).hexdigest()[:16]
    return f"wing_idx_{h}"


# ── Pagination helpers ────────────────────────────────────────────────────


def _iter_docs(col, where: dict | None, page_size: int = _PAGE_SIZE) -> Iterator[tuple]:
    """Yield ``(doc, metadata)`` tuples for every row matching ``where``.

    Pages with ``limit``/``offset`` because Chroma's unbounded ``get`` silently
    truncates large collections. Skips rows whose metadata is ``None``.
    """
    offset = 0
    while True:
        try:
            batch = col.get(
                where=where or None,
                limit=page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("indexer: Chroma get() failed at offset=%s: %s", offset, exc)
            return

        # Chroma returns a mix of raw dicts and typed QueryResult objects
        # depending on the adapter layer; tolerate both.
        docs = _field(batch, "documents") or []
        metas = _field(batch, "metadatas") or []
        ids = _field(batch, "ids") or []

        if not docs:
            return
        for doc, meta in zip(docs, metas):
            if doc is None:
                continue
            yield (doc, meta or {})

        # Exhausted — last page smaller than page_size.
        if len(ids) < page_size:
            return
        offset += page_size


def _field(batch, key: str):
    """Return ``batch[key]``/``batch.key`` polymorphically (dict vs typed result)."""
    if batch is None:
        return None
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


# ── Line extraction ───────────────────────────────────────────────────────


def _iter_lines(documents: Iterable[str]) -> Iterator[str]:
    """Yield individual AAAK pointer lines from a stream of multi-line docs."""
    for doc in documents:
        if not isinstance(doc, str):
            continue
        for line in doc.splitlines():
            s = line.strip()
            if s:
                yield s


# ── Room-level aggregation ────────────────────────────────────────────────


def build_room_index(
    palace_path: str,
    wing: str,
    room: str,
    *,
    top_n: int = ROOM_INDEX_TOP_N,
) -> dict:
    """Rebuild the index document for ``(wing, room)``.

    Reads every closet whose metadata matches the room, projects a
    rank-bucketed top-``n`` over the AAAK lines, and upserts a single
    document into ``mempalace_room_indices``. Returns a status dict:

        {"wing", "room", "status", "closet_count", "line_count",
         "kept_lines", "doc_id"}

    ``status`` is one of ``"written"``, ``"empty"`` (no closets found →
    index document deleted), or ``"error"``.
    """
    try:
        closets_col = get_closets_collection(palace_path, create=False)
    except Exception as exc:
        return {
            "wing": wing,
            "room": room,
            "status": "error",
            "error": f"closets collection missing: {exc}",
        }

    rows = list(_iter_docs(closets_col, where={"$and": [{"wing": wing}, {"room": room}]}))
    docs = [d for d, _m in rows]
    metas = [m for _d, m in rows]

    if not rows:
        # No closets left in this room — drop any stale room-index doc so
        # the retriever doesn't chase a dangling pointer.
        try:
            indices_col = get_room_indices_collection(palace_path, create=False)
            indices_col.delete(ids=[_room_index_id(wing, room)])
        except Exception:
            pass
        return {
            "wing": wing,
            "room": room,
            "status": "empty",
            "closet_count": 0,
            "line_count": 0,
            "kept_lines": 0,
        }

    all_lines = list(_iter_lines(docs))
    projected = project_closet_lines(all_lines, n=top_n)
    entity_fields = [m.get("top_entities", "") if isinstance(m, dict) else "" for m in metas]
    top_entities = aggregate_entity_sets(entity_fields, n=_TOP_ENTITIES_PER_ROOM)

    # Cap the rendered doc to INDEX_DOC_MAX_CHARS — the embedding model's
    # token limit (default nomic 2K) is enforced server-side, so we drop
    # the lowest-ranked lines if they'd push us over. project_closet_lines
    # is rank-ordered so trailing lines are the ones to lose.
    kept_lines = []
    running_chars = 0
    for line, _count in projected:
        line_len = len(line) + 1  # +1 for the joining newline
        if kept_lines and running_chars + line_len > INDEX_DOC_MAX_CHARS:
            break
        kept_lines.append(line)
        running_chars += line_len
    doc_text = "\n".join(kept_lines) if kept_lines else ""

    drawer_ids: set = set()
    for line in kept_lines:
        arrow = line.find("→")
        if arrow < 0:
            arrow_ascii = line.find("->")
            if arrow_ascii < 0:
                continue
            tail = line[arrow_ascii + 2 :]
        else:
            tail = line[arrow + 1 :]
        for did in tail.split(","):
            did = did.strip()
            if did:
                drawer_ids.add(did)

    doc_id = _room_index_id(wing, room)
    # Chroma's metadata values must be scalar strings/numbers/bools — encode
    # lists as semicolon-joined strings to stay safely inside that envelope.
    top_entity_str = ";".join(e for e, _c in top_entities)

    metadata = {
        "wing": wing,
        "room": room,
        "level": "room",
        "aaak_version": AAAK_VERSION,
        "closet_count": len(docs),
        "line_count": len(all_lines),
        "kept_lines": len(kept_lines),
        "drawer_count": len(drawer_ids),
        "top_entities": top_entity_str,
        "built_at": int(time.time()),
    }

    try:
        indices_col = get_room_indices_collection(palace_path, create=True)
        if doc_text:
            indices_col.upsert(ids=[doc_id], documents=[doc_text], metadatas=[metadata])
        else:
            # Projection produced nothing (all lines malformed); don't leave
            # an empty doc behind.
            try:
                indices_col.delete(ids=[doc_id])
            except Exception:
                pass
    except Exception as exc:
        return {
            "wing": wing,
            "room": room,
            "status": "error",
            "error": f"upsert failed: {exc}",
            "closet_count": len(docs),
            "line_count": len(all_lines),
            "kept_lines": len(kept_lines),
        }

    return {
        "wing": wing,
        "room": room,
        "status": "written" if doc_text else "empty",
        "closet_count": len(docs),
        "line_count": len(all_lines),
        "kept_lines": len(kept_lines),
        "drawer_count": len(drawer_ids),
        "doc_id": doc_id,
    }


# ── Wing-level aggregation ────────────────────────────────────────────────


def build_wing_index(
    palace_path: str,
    wing: str,
    *,
    top_n: int = WING_INDEX_TOP_N,
) -> dict:
    """Rebuild the index document for ``wing``.

    Reads every ``(wing, room)`` room index in the wing, unions their
    AAAK lines, and projects a top-``n``. The result is a *single*
    compact rollup of the wing — the outermost pruning layer.

    ``max_depth=0`` retrieval stops here; deeper retrieval steps descend
    into room indices using :func:`search_within` with ``wing_filters=[wing]``.
    """
    # create=True so a fresh palace with no room indices yet is treated as
    # "no rows to aggregate" (returns status=empty) rather than an error.
    try:
        indices_col = get_room_indices_collection(palace_path, create=True)
    except Exception as exc:
        return {
            "wing": wing,
            "status": "error",
            "error": f"room indices collection missing: {exc}",
        }

    rows = list(_iter_docs(indices_col, where={"wing": wing}))
    docs = [d for d, _m in rows]
    metas = [m for _d, m in rows]

    if not rows:
        try:
            wing_col = get_wing_indices_collection(palace_path, create=False)
            wing_col.delete(ids=[_wing_index_id(wing)])
        except Exception:
            pass
        return {"wing": wing, "status": "empty", "room_count": 0, "line_count": 0}

    all_lines = list(_iter_lines(docs))
    projected = project_closet_lines(all_lines, n=top_n)
    entity_fields = [m.get("top_entities", "") if isinstance(m, dict) else "" for m in metas]
    top_entities = aggregate_entity_sets(entity_fields, n=_TOP_ENTITIES_PER_WING)

    # See INDEX_DOC_MAX_CHARS comment above; same cap applies to wing
    # indices, which can be even larger than room indices because they
    # aggregate every room in the wing.
    kept_lines = []
    running_chars = 0
    for line, _count in projected:
        line_len = len(line) + 1
        if kept_lines and running_chars + line_len > INDEX_DOC_MAX_CHARS:
            break
        kept_lines.append(line)
        running_chars += line_len
    doc_text = "\n".join(kept_lines) if kept_lines else ""
    top_entity_str = ";".join(e for e, _c in top_entities)

    room_names = sorted({m.get("room", "") for m in metas if isinstance(m, dict) and m.get("room")})
    total_drawers = sum(
        int(m.get("drawer_count", 0) or 0) for m in metas if isinstance(m, dict)
    )

    doc_id = _wing_index_id(wing)
    metadata = {
        "wing": wing,
        "level": "wing",
        "aaak_version": AAAK_VERSION,
        "room_count": len(room_names),
        "rooms": ";".join(room_names),
        "drawer_count": total_drawers,
        "line_count": len(all_lines),
        "kept_lines": len(kept_lines),
        "top_entities": top_entity_str,
        "built_at": int(time.time()),
    }

    try:
        wing_col = get_wing_indices_collection(palace_path, create=True)
        if doc_text:
            wing_col.upsert(ids=[doc_id], documents=[doc_text], metadatas=[metadata])
        else:
            try:
                wing_col.delete(ids=[doc_id])
            except Exception:
                pass
    except Exception as exc:
        return {
            "wing": wing,
            "status": "error",
            "error": f"upsert failed: {exc}",
            "room_count": len(room_names),
            "line_count": len(all_lines),
        }

    return {
        "wing": wing,
        "status": "written" if doc_text else "empty",
        "room_count": len(room_names),
        "rooms": room_names,
        "line_count": len(all_lines),
        "kept_lines": len(kept_lines),
        "drawer_count": total_drawers,
        "doc_id": doc_id,
    }


# ── Orchestration ─────────────────────────────────────────────────────────


def _indexer_lock_key(palace_path: str) -> str:
    """Lock file key scoping mine_lock to a single-indexer run per palace."""
    return f"{palace_path}|__recursive_indexer__"


def rebuild_dirty(palace_path: str) -> dict:
    """Drain the dirty-flag queue: rebuild every dirty room/wing, clear flags.

    Guarded by ``mine_lock`` so two processes running the indexer against
    the same palace serialize cleanly. Room rebuilds run before wing
    rebuilds — any wing whose rooms changed is already flagged because
    ``mark_room_dirty`` cascades to ``mark_wing_dirty``.

    Returns a stats dict:

        {"rooms": [room_result, ...],
         "wings": [wing_result, ...],
         "room_count", "wing_count",
         "elapsed_seconds"}
    """
    start = time.perf_counter()
    with mine_lock(_indexer_lock_key(palace_path)):
        room_results = []
        for wing, room, _key in list(iter_dirty_rooms(palace_path)):
            result = build_room_index(palace_path, wing, room)
            room_results.append(result)
            if result.get("status") in ("written", "empty"):
                clear_room_dirty(palace_path, wing, room)

        wing_results = []
        for wing, _key in list(iter_dirty_wings(palace_path)):
            result = build_wing_index(palace_path, wing)
            wing_results.append(result)
            if result.get("status") in ("written", "empty"):
                clear_wing_dirty(palace_path, wing)

    return {
        "rooms": room_results,
        "wings": wing_results,
        "room_count": len(room_results),
        "wing_count": len(wing_results),
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }


def _discover_wings_and_rooms(palace_path: str) -> dict:
    """Walk the drawer collection and return ``{wing: {rooms...}}`` for every
    (wing, room) pair with at least one drawer. Used by :func:`rebuild_all`.
    """
    try:
        drawers_col = get_collection(palace_path, create=False)
    except Exception as exc:
        logger.warning("indexer: cannot open drawers collection: %s", exc)
        return {}

    scope: dict = {}
    for _doc, meta in _iter_docs(drawers_col, where=None):
        wing = meta.get("wing") if isinstance(meta, dict) else None
        room = meta.get("room") if isinstance(meta, dict) else None
        if isinstance(wing, str) and wing and isinstance(room, str) and room:
            scope.setdefault(wing, set()).add(room)
    return {w: sorted(rs) for w, rs in scope.items()}


def rebuild_all(palace_path: str) -> dict:
    """Rebuild every room + wing index from scratch by walking drawers.

    Equivalent to marking every (wing, room) dirty and running
    :func:`rebuild_dirty`, but does not depend on dirty flags — safe to
    call after upgrades, migrations, or manual palace surgery.

    Guarded by ``mine_lock`` so it doesn't race with ``rebuild_dirty``
    running in another process.
    """
    start = time.perf_counter()
    scope = _discover_wings_and_rooms(palace_path)
    room_results = []
    wing_results = []
    with mine_lock(_indexer_lock_key(palace_path)):
        for wing, rooms in scope.items():
            for room in rooms:
                room_results.append(build_room_index(palace_path, wing, room))
                clear_room_dirty(palace_path, wing, room)
        for wing in scope.keys():
            wing_results.append(build_wing_index(palace_path, wing))
            clear_wing_dirty(palace_path, wing)

    return {
        "rooms": room_results,
        "wings": wing_results,
        "room_count": len(room_results),
        "wing_count": len(wing_results),
        "scope": {w: list(rs) for w, rs in scope.items()},
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }
