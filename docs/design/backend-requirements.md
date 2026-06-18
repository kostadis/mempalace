# Backend Requirements & Evaluation Framework

A rubric for evaluating candidate storage backends for MemPalace, and a map of
what MemPalace actually requires of a backend to function correctly.

This document **describes the requirements; it does not evaluate any specific
backend.** Use the checklists and tables below to score a candidate (turbovec,
LanceDB, Qdrant, sqlite-vec, pgvector, etc.). Fill in the right-hand columns
per candidate; leave the left-hand "what MemPalace needs" columns as the fixed
reference.

Companion reading: [`ARCHITECTURE.md`](../ARCHITECTURE.md) (Layer 1 — Storage
backends), [`backends/base.py`](../../mempalace/backends/base.py) (the formal
contract, RFC 001).

---

## How to read this document

MemPalace's storage dependency comes in **two very different tiers**, and a
candidate backend must be scored against both:

1. **Tier A — the formal contract (RFC 001).** The backend-agnostic interface in
   `backends/base.py`. Any backend that implements this surface is *pluggable*.
   This is the clean seam.

2. **Tier B — the implementation-specific assumptions the default (ChromaDB)
   backend satisfies, and that callers outside the backend rely on.** These have
   **leaked past the contract** into `searcher.py`, `repair.py`, `migrate.py`,
   and `embedding.py`. A new backend either (a) provides an equivalent, or
   (b) forces those callers to be made backend-neutral first. **This tier is the
   real cost of adding a backend** and is where evaluation effort should focus.

A backend that satisfies Tier A but not Tier B will *load and run* but may
silently break recall guarantees, repair tooling, or cross-palace
compatibility. Score both.

---

## Tier A — The formal contract (RFC 001)

Source of truth: [`backends/base.py`](../../mempalace/backends/base.py).
A candidate MUST implement all of this. These are table-stakes, not
differentiators.

### A.1 `BaseCollection` (per-collection, kwargs-only)

| Method | Required | Notes |
|---|:---:|---|
| `add` | ✅ abstract | accepts precomputed `embeddings` (mining writes them) |
| `upsert` | ✅ abstract | idempotent re-mine depends on this |
| `query` | ✅ abstract | exactly one of `query_texts` / `query_embeddings` |
| `get` | ✅ abstract | by `ids`, `where`, `where_document`, `limit`, `offset` |
| `delete` | ✅ abstract | by `ids` or `where` |
| `count` | ✅ abstract | exact row count |
| `estimated_count` | default → `count()` | override if exact count is expensive |
| `close` | default no-op | release file/connection handles |
| `health` | default healthy | |
| `update` | default get+merge+upsert | override with atomic single-round-trip if `supports_update` |

### A.2 `BaseBackend` (per-palace factory)

| Method | Required | Notes |
|---|:---:|---|
| `get_collection(palace, collection_name, create, options)` | ✅ abstract | **no I/O in `__init__`** — defer all connection work to here |
| `close_palace(palace)` | default no-op | must release file locks if the backend holds any |
| `close()` | default no-op | |
| `health(palace)` | default healthy | |
| `detect(path)` | default `False` | on-disk auto-detection hint for `resolve_backend_for_palace()` |

Class attributes: `name` (registry key), `spec_version`, `capabilities`
(frozenset of capability strings).

### A.3 Value objects & errors

- Returns: `QueryResult` (outer = #queries, inner = hits) and `GetResult`,
  with attribute access (`result.ids`). Empty fields must preserve outer shape.
- Errors a backend must raise where applicable: `PalaceNotFoundError`,
  `BackendClosedError`, `UnsupportedFilterError` (**silent dropping of unknown
  where-operators is forbidden**), `DimensionMismatchError`,
  `EmbedderIdentityMismatchError`.

### A.4 Where-clause operators

| Operator class | Operators | Requirement |
|---|---|---|
| Required | `$eq $ne $in $nin $and $or $contains` | MUST support or raise `UnsupportedFilterError` |
| Optional | `$gt $gte $lt $lte` | MAY support; MUST raise if not |

### A.5 Registration

Register under the `mempalace.backends` entry-point group in `pyproject.toml`;
selection priority is explicit arg → per-palace config → `MEMPALACE_BACKEND` env
→ `detect()` → default `chroma`.

### A.6 The four collections per palace

A backend must serve all four by name; only the first is performance-critical.

| Collection | Holds | Read by |
|---|---|---|
| `mempalace_drawers` | verbatim chunks + metadata, vector-indexed | `searcher`, `layers` |
| `mempalace_closets` | AAAK pointer lines | `searcher` themes path |
| `mempalace_room_indices` | room-level roll-ups | hierarchical search depth 1 |
| `mempalace_wing_indices` | wing-level roll-ups | hierarchical search depth 0 |

---

## Tier B — Implementation assumptions that leak past the contract

**This is the evaluation that matters.** Each row is a ChromaDB-specific
behavior that some part of MemPalace *outside* `backends/base.py` currently
depends on. For each candidate backend, decide: **(P)rovide an equivalent**,
**(N)ot needed by this backend**, or **(R)efactor the caller to be neutral
first**.

### B.1 Distance metric & similarity math

| What MemPalace assumes | Where | Candidate must… |
|---|---|---|
| Vector space is **cosine**; collections created with `hnsw:space=cosine` | `chroma.py`, `searcher.py` | expose cosine, or a distance whose `[0,2]` range maps cleanly |
| Hybrid ranker hard-codes `vec_sim = max(0, 1 − distance)` (cosine distance ∈ `[0,2]`) | `searcher.py:_hybrid_rank` | return a distance for which this formula is correct, **or** the formula must be parameterized per backend |
| Legacy palaces created without cosine are detected via `collection.metadata["hnsw:space"]` | `searcher.py`, `ChromaCollection.metadata` | provide a way to report the active metric, or this guard must be generalized |

**Evaluation question:** does the candidate return a *normalized, documented*
distance, and is `1 − distance` valid? If not, `_hybrid_rank` must be made
metric-aware.

### B.2 The recall-survival fallback (the 100%-recall promise)

This is the highest-stakes coupling. When the vector index is corrupt/unloadable,
`searcher._bm25_only_via_sqlite()` bypasses the backend entirely and reads
`chroma.sqlite3` directly.

| What MemPalace assumes | Where | Candidate must… |
|---|---|---|
| A keyword/lexical fallback exists that does **not** require the vector index | `searcher.py` | provide an independent lexical retrieval path that survives vector-index damage |
| Full-text index uses FTS5 **`tokenize='trigram'`** → query tokens `<3` chars are dropped, OR-joined | `searcher.py:_bm25_only_via_sqlite` | document its tokenizer; the token-length / join logic is currently trigram-shaped |
| Document body is reconstructable from storage under the key `chroma:document` | `searcher.py` | expose document text without the vector segment |
| Recency fallback when no usable token: `ORDER BY created_at DESC`, then `id DESC` | `searcher.py` | provide a recency ordering reachable without the vector index |

**Evaluation question:** if the candidate's vector index is destroyed, can
MemPalace still return every drawer by keyword? If the answer routes through
backend-private SQL, that SQL is ChromaDB-specific and must be replaced by a
contract method (e.g. a `lexical_query()` capability) before the backend is
safe.

### B.3 Direct on-disk store access (the `chroma.sqlite3` dependency)

MemPalace opens the backend's underlying store **directly** (read-only SQLite)
in multiple modules, bypassing the backend object. A non-SQLite backend breaks
all of these unless they are first refactored.

| Internal relied upon | Where | Purpose |
|---|---|---|
| `segments(id, collection, scope='VECTOR')` | `chroma.py`, `repair.py` | find the vector segment without opening it |
| `collections(id, name)` | many | name → id |
| `collection_metadata(collection_id, key, int_value)` | `chroma.py:_read_sync_threshold` | read index tuning back |
| `embeddings(id, segment_id, seq_id, created_at)` | `chroma.py`, `repair.py`, `searcher.py` | counts, recency, write-cursor |
| `embedding_metadata(...string/int/float/bool_value)` | `searcher.py` | reconstruct drawers in fallback |
| `embedding_fulltext_search` (FTS5 shadow) | `searcher.py` | lexical candidates |
| `max_seq_id`, `embeddings_queue` | `repair.py`, `chroma.py` | write-cursor poisoning detection/repair |
| Special metadata key `chroma:document` | `searcher.py` | document body row |

**Evaluation question:** every one of these is a direct read of ChromaDB's
private schema. A new backend either ships an equivalent introspection surface
**or** these callers must move behind contract methods. This is the largest
single block of refactoring a non-ChromaDB backend implies.

### B.4 Index health, divergence, and corruption detection

The repair/maintenance ladder is built around ChromaDB's specific failure modes.

| What MemPalace assumes | Where | Candidate must… |
|---|---|---|
| Vector index can diverge from the row store and **segfault on open** | `chroma.py`, `repair.py` | declare whether its index can desync, and whether a bad index crashes the process |
| Divergence is measurable: SQLite row count vs HNSW pickle `id_to_label` count | `chroma.py:hnsw_capacity_status` | expose "rows the index actually holds" vs "rows that exist" |
| Divergence tolerance scales with `2 × hnsw:sync_threshold` (async flush lag) | `chroma.py` | if it flushes async, expose the flush threshold; if synchronous, this whole probe is N/A |
| Physical-file heuristics: `link_lists.bin` / `data_level0.bin` size ratio > 10× = corrupt | `chroma.py` | N/A unless it has analogous bloat-prone segment files |
| Index metadata is a byte-sniffable pickle (`0x80`…`0x2e`), unpickled via allowlist | `chroma.py` | N/A unless it persists pickled metadata |
| Repair ladder: `status → scan → prune → rebuild_index → rebuild_from_sqlite` | `repair.py` | provide a rebuild-from-row-store path that does not touch the broken vector index |

**Evaluation question:** what are *this* backend's corruption modes, and does it
provide a "rebuild the index from the durable row store" path? The entire
`repair.py` module is ChromaDB-shaped and would need a backend-specific sibling.

### B.5 Write concurrency model

| What MemPalace assumes | Where | Candidate must… |
|---|---|---|
| The vector index is **not thread-safe on insert** → single writer only | `parallel.py` (1 consumer thread), `palace.py:mine_palace_lock` | declare its concurrency model; if it supports concurrent writes, the single-consumer pipeline is an unnecessary bottleneck |
| All writes serialize through `mine_palace_lock` (a file lock) | `ChromaCollection` write methods | tolerate serialized writes (always safe), but may not *need* the lock |
| Insert threads pinned to 1 (`hnsw:num_threads=1`), re-applied every open | `chroma.py:_pin_hnsw_threads` | N/A unless it has a tunable insert-thread count with the same race |

**Evaluation question:** if the candidate supports safe concurrent writes,
MemPalace leaves throughput on the table with its one-consumer pipeline — an
*opportunity*, not a blocker.

### B.6 Index-build tuning to avoid pathological growth

| What MemPalace assumes | Where | Candidate must… |
|---|---|---|
| Large batch/sync thresholds (`50_000`) prevent `link_lists.bin` bloat on big mines | `chroma.py:_HNSW_BLOAT_GUARD` | N/A unless its index has a resize-feedback bloat mode; if so, expose equivalent tuning at creation |

### B.7 Embedding-function identity & persistence

| What MemPalace assumes | Where | Candidate must… |
|---|---|---|
| Backend persists/validates EF identity and **rejects reads from a differently-named EF** → MemPalace spoofs `name()="default"` | `embedding.py:_build_ef_class` | document whether it validates embedder identity; if it does, MemPalace needs the same compatibility shim |
| EF is **not persisted** with the collection → must be passed on every `get`/`create` | `chroma.py:_resolve_embedding_function` | document whether EF is persisted; mismatched EF on read = silent wrong results |
| Lazy embed at query time (`_lazy_embedder`) so precomputed-write mining skips model load | both backends today | support precomputed-embedding writes **and** text-query embedding |
| Embedding dimension must match collection; raise `DimensionMismatchError` | contract | enforce dimension match on write |

**Evaluation question:** how does the candidate decide two embedders are "the
same"? A silent EF mismatch is the worst failure class — plausible but wrong
results, no error.

### B.8 Process / lifecycle side-effects

| What MemPalace assumes | Where | Candidate must… |
|---|---|---|
| Client holds an exclusive file lock until `close()`; dict-eviction is insufficient | `chroma.py:close_palace` | release all handles/locks on `close_palace`, or the palace path stays unremovable in-process |
| Palace rebuilds on disk are detected via `chroma.sqlite3` inode+mtime freshness check | `chroma.py:_client` | provide a cheap "did the store change underneath me?" signal, or accept stale cached handles after external rebuild |
| Store file is created lazily (re-stat after constructor) | `chroma.py` | N/A — informational |
| Opening certain client states leaves WAL state that crashes the next open → ordered pre-open repair | `chroma.py:_prepare_palace_for_open` | declare ordering constraints on open; ideally none |

### B.9 Version migration

| What MemPalace assumes | Where | Candidate must… |
|---|---|---|
| Cross-version on-disk format breaks (e.g. BLOB→INTEGER `seq_id`, poisoned `max_seq_id`) need pre-open repair | `chroma.py:_fix_blob_seq_ids`, `repair.py`, `migrate.py` | declare its format-stability story across versions; ChromaDB's specific migrations are N/A but the *need for a migration hook* is the transferable requirement |

---

## Capability matrix (fill in per candidate)

Copy this block per backend under evaluation. `name` / `capabilities` come from
the backend's `BaseBackend` class attributes.

| Capability flag (Tier A) | ChromaDB (reference) | Candidate |
|---|:---:|:---:|
| `supports_embeddings_in` | ✅ | |
| `supports_embeddings_passthrough` | ✅ | |
| `supports_embeddings_out` | ✅ | |
| `supports_metadata_filters` | ✅ | |
| `supports_contains_fast` | ✅ | |
| `local_mode` | ✅ | |
| `supports_update` (atomic) | (default get+merge+upsert) | |

| Tier B dimension | ChromaDB (reference) | Candidate: P / N / R | Notes |
|---|---|:---:|---|
| B.1 Cosine + `1−distance` valid | yes (explicit) | | |
| B.2 Vector-independent lexical recall | yes (FTS5 trigram via direct SQL) | | |
| B.3 No direct private-schema reads needed | **violated** (reads `chroma.sqlite3`) | | |
| B.4 Rebuild-from-row-store path | yes (`rebuild_from_sqlite`) | | |
| B.4 Index can segfault on open | yes | | |
| B.5 Concurrent writes safe | no (single writer) | | |
| B.6 Index bloat tuning needed | yes (`sync_threshold`) | | |
| B.7 EF identity validated / persisted | validated; **not** persisted | | |
| B.8 Holds exclusive file lock | yes | | |
| B.9 Needs version-migration hook | yes | | |

Legend: **P** = candidate provides an equivalent · **N** = not applicable to
this backend's design · **R** = requires refactoring the MemPalace caller to be
backend-neutral first.

---

## Which MemPalace layers must change to admit a new backend

Ordered by how much each currently assumes ChromaDB specifics. A backend that
only needs the first row is cheap; one that trips the lower rows is expensive.

| Layer / module | Coupling to ChromaDB today | Work to admit a new backend |
|---|---|---|
| `backends/base.py` | none (this *is* the neutral contract) | implement it — table stakes |
| `backends/<new>.py` | n/a | new file; register in `mempalace.backends` |
| `backends/registry.py` | none | add `detect()` + selection entry |
| `embedding.py` | EF `name()` spoof, lazy embed, dim check | generalize EF-identity handling if the backend validates embedders differently |
| `searcher.py` | **`_hybrid_rank` cosine math; `_bm25_only_via_sqlite` reads `chroma.sqlite3` + FTS5 trigram + `chroma:document`** | biggest leak: metric-aware ranking + a contract-level lexical-fallback method |
| `repair.py` | entirely ChromaDB-shaped (HNSW files, `max_seq_id`, `embeddings_queue`, `rebuild_from_sqlite`) | needs a backend-specific repair sibling or a generalized repair interface |
| `migrate.py` | ChromaDB version migrations | backend-specific or no-op for backends with stable formats |
| `parallel.py` / `palace.py` | single-consumer + `mine_palace_lock` because HNSW isn't thread-safe | safe to keep; only an optimization opportunity if the backend allows concurrent writes |

**Summary of the refactor needed to make MemPalace truly backend-neutral
(independent of which backend wins):**

1. Add a **lexical-fallback capability** to the contract so `searcher.py` no
   longer reaches into `chroma.sqlite3` / FTS5 directly. *(closes B.2 + B.3)*
2. Make **`_hybrid_rank` metric-aware** instead of hard-coding cosine
   `1 − distance`. *(closes B.1)*
3. Add a **repair/introspection capability** (rows-in-index, rebuild-from-store)
   so `repair.py` stops reading private tables and pickles. *(closes B.4 + B.3)*
4. Move **EF-identity reconciliation** behind the contract so each backend
   declares its own embedder-matching rule. *(closes B.7)*

Until 1–4 land, every non-ChromaDB backend is only as functional as Tier A —
it will store and retrieve, but the recall-survival fallback, repair tooling,
and ranking correctness remain ChromaDB-specific.

---

## Glossary of the ChromaDB internals referenced

- **HNSW** — Hierarchical Navigable Small World, the ANN index. Files:
  `data_level0.bin` (vectors), `link_lists.bin` (graph edges),
  `index_metadata.pickle` (counts/dims).
- **`chroma.sqlite3`** — the durable row store: `collections`, `segments`,
  `embeddings`, `embedding_metadata`, `embedding_fulltext_search` (FTS5),
  `max_seq_id`, `embeddings_queue`.
- **`sync_threshold` / `batch_size`** — how often HNSW flushes to disk; drives
  async flush-lag and (mis-set) index bloat.
- **`max_seq_id` / `embeddings_queue`** — ChromaDB's write-cursor and pending
  write log; a poisoned cursor silently drops writes.
- **EF identity** — ChromaDB 1.5 stamps the embedding function's `name()` on the
  collection and refuses reads from a differently-named EF.
