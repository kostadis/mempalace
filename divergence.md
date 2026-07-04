# Divergence: `kostadis-dev` vs upstream `v3.5.0`

Context: `kostadis-dev` is a personal fork of MemPalace carrying **67 local commits** on top of upstream. It was brought current with upstream `v3.5.0` through a chain of five per-release merges on `merge/v3.5.0-into-kostadis-dev` — v3.3.6 (#25), v3.4.0 (#26), v3.4.1 (#27), v3.5.0 (#28) — tracked under issue #23. `version.py` = 3.5.0. This document explains, for offline review and potential discussion with the upstream maintainers, where the two branches diverge and why. (An earlier revision described the fork against v3.3.5; the seven structural divergences below originated there and **held unchanged through v3.5.0** — the local design absorbed each release's bugfixes without moving off its own shape. Sections 8–12 are new tensions that surfaced during the v3.4.0–v3.5.0 merges.)

The divergence is not the merge — the bulk of upstream's 525-commit v3.3.5→v3.5.0 delta applied cleanly. The divergence is **structural**: a set of upstream features were dropped, restructured, or replaced because the local branch had already moved in a different direction on the same surface area. This document enumerates those tensions.

> **A note on silent auto-merge defects.** Across all five merges, `git merge` repeatedly produced a *clean* (conflict-free) blend of a local-diverged file that was nonetheless *broken* — a dropped constant in `backends/chroma.py` (NameError, 15 failing tests), a poisoned `mcp_server.tool_reconnect`, dropped `_metadata_cache` scalar writes, a `tool_delete_by_source` writing a nonexistent global. Every one was caught only by running the full test suite plus a grep for references to removed symbols — never by the absence of conflict markers. Anyone repeating this kind of long-lived-fork merge should treat "no conflicts" as unproven, not safe.

---

## Local features that have no upstream counterpart

These were built on `kostadis-dev` and had no upstream equivalent when they landed. By v3.5.0, three grew a parallel upstream implementation and are now reconciled rather than purely local — the **pluggable backend** (see §8, converged onto upstream), **Cursor integration** (§9, coexisting duplication), and **remote/multilingual embedding** (§10, unified); the table entries below are kept for provenance. The rest remain local-only through v3.5.0.

| Area | Local feature | Why it was added |
|---|---|---|
| Storage | **Palace isolation** — strict separation between a "chat" palace (hook-fired writes) and "campaign" or curated palaces (explicit reads/writes). Invariant: hook writes never touch a curated palace via `mempalace.yaml` walk-up or `default_palace`. | Curated palaces were being silently contaminated by Stop-hook auto-mines that ran in a campaign workspace. |
| Storage | **`.mempalaceignore`** (replaces `.gitignore` as primary mining filter; gitignore is a fallback). | Mining wanted ignore rules independent of git's view — e.g. mine a config dir that's gitignored. |
| Storage | **DrvFs / 9P / CIFS / NFS warning** at palace path resolution. | ChromaDB + SQLite mmap/flock/fsync semantics break on WSL2 DrvFs mounts; silent corruption. |
| Storage | **Walk-up `mempalace.yaml` discovery** + **`default_palace` config key** + **alias resolver** + **loud-fail when no palace declared** (`PalaceNotDeclared`). | Removes the implicit-fallback footgun; users must declare which palace they want. |
| Storage | **KG co-located with palace directory** (`<palace>/knowledge_graph.sqlite3` instead of `~/.mempalace/knowledge_graph.sqlite3`). | Co-location is required for palace isolation — otherwise a palace move/copy leaves the KG dangling. |
| MCP | **Per-palace backend cache + optional `palace=` arg on all read tools**. | Cross-palace reads from a single MCP server (e.g. `mempalace_search(palace='chat')` from a campaign workspace). |
| Mining | **Parallel mining pipeline** — producer/consumer harness with `--workers` CLI flag. Mine splits into `_prepare_file` (CPU + IO), `_embed_prepared` (HTTP/ONNX), `_write_prepared` (single-writer under lock). | Single-threaded mine wastes the time spent waiting on remote embedding endpoints (Ollama, Spark). |
| Mining | **Parallel convo mining** + **parallel closet regeneration** + **parallel LLM refinement** via the same `ParallelPipeline` harness. | Same motivation — overlap remote-API waits. |
| Mining | **urllib3 PoolManager keep-alive** for the embedding and LLM HTTP clients. | Reuse connections across the parallel producers; new TCP handshake per request dominated wall time. |
| Embedding | **Remote embedding providers** (Ollama, OpenAI-compatible) via pluggable config. | Local-first principle preserved — Ollama on localhost is "your machine" — but the user can also point at a Spark or other host they own. |
| LLM | **`MEMPALACE_LLM_*` env / config-file mirroring** throughout closet_llm and friends. | Replaces ad-hoc CLI args; lets the same LLM endpoint serve every component without re-passing it. |
| Search | **`search_within` scoped-search primitive** with explicit `wing_filters` / `room_filters` / `ids` arguments, returning a **two-list `primary` / `themes` shape**. `search_memories` becomes a thin shim over it. | Hierarchical descent needs a primitive that can be called with a pre-pruned ID set. The two-list split is the load-bearing architectural change (see §"Closet boosting", below). |
| Search | **Hierarchical AAAK retrieval** + `mempalace_search_hierarchical` MCP tool. | Coarse-to-fine: scan compressed AAAK index → pick rooms → drill into drawers. |
| Search | **Closet taxonomy bridging** for paraphrased queries (fixed-vocabulary classification surfaces closets a literal-token search would miss). | Closet matches were brittle on paraphrased queries. |
| Search | **Recursive indexer** capped at the nomic 2048-token limit. | Local default embedder changed; chunks above the limit silently truncated. |
| Closets | **Closet LLM iter-06 prose prompt** + list-shape parser tolerance. | The hyphen-glued tag-chain output was hard for both humans and LLMs to read; prose forms with looser parsing won. |
| Closets | **Pagination of drawer loader** to bypass SQLite's `SQLITE_MAX_VARIABLE_NUMBER` (32766) on large palaces. | One-shot `get(limit=total)` blew up at 30k+ drawers. (v3.3.5 fixed the same bug independently with `batch_size=5000`; merged toward local's `PAGE=10000`.) |
| Hooks | **Cursor harness** for auto-save + dedicated stop/preCompact/sessionStart entry points + a `cursor.hooks.json` drop-in template. | Cursor's stop payload uses `conversation_id` and lacks `stop_hook_active`; needs a different parse path. |
| Hooks | **Chat-palace pinning** at the hook layer — all auto-mines and ingest paths pass `--palace <chat-palace>` to the spawned subprocess. | Enforces palace-isolation invariant #1 at the only layer that can. |
| Distribution | **Opencode MemPalace plugin** (`.codex-plugin/`). | First-class Codex/Opencode integration alongside the existing Claude Code plugin. |

---

## Where the two branches structurally diverge

These are the load-bearing differences — places where the same surface area has incompatible designs. Each merge (v3.3.6 → v3.5.0) resolved these by keeping the local design and adapting that release's bugfixes into it. Re-aligning with upstream would require an architectural conversation, not a code change. (Per-site "Upstream v3.3.5" descriptions below name the version where the divergence originated; each held through v3.5.0.)

### 1. Closet boosting on primary hits

- **Upstream v3.3.5**: closets boost primary hits in a single rerank pool. `effective_distance = max(0, min(2, dist − boost))` with rank-based boosts (0.40, 0.25, 0.15, 0.08, 0.04). `candidate_strategy="union"` widens that pool by also pulling top-K BM25 candidates from sqlite.
- **Local**: closets surface in a separate `themes` list. Primary is never boosted by closet rank ("Closet content does not influence `primary` ranking"). The rationale is that closets are an LLM-judgment layer, and a weak closet match shouldn't reshape the verbatim answer.

The local design makes `candidate_strategy="union"` moot (there's no merged pool to apply it to) and removes the `effective_distance` / `closet_boost` / `matched_via` fields from primary hits.

**Open question for upstream**: would there be appetite for a `themes` list as an alternative output shape — preserving the v3.3.5 boost path as the default but offering the two-list split for callers (notably hierarchical search) that want clean separation?

### 2. Mining pipeline: serial `process_file` vs parallel `_prepare_file` / `_embed_prepared` / `_write_prepared`

- **Upstream v3.3.5**: `mine()` calls `process_file(filepath, …)` in a loop. Each call reads, chunks, embeds, writes under the per-file `mine_lock`. Single-threaded by construction.
- **Local**: the loop is a `ParallelPipeline` of N producer threads (CPU + remote-embedding HTTP) and one consumer thread (chromadb writer under `mine_palace_lock`). `process_file` is split into three pure-function-ish stages so producers can do everything except the upsert.

This forced one real impl change (§4 below). It also breaks any v3.3.5 test that patches `process_file` to inject behavior mid-mine.

**Open question for upstream**: parallelism is going to keep coming up as remote-embedding setups proliferate. Is there an appetite for adopting a pipeline harness upstream, or for at least extracting `process_file` into prep/embed/write stages so both serial and parallel callers can reuse them?

### 3. Palace addressability: implicit global vs explicit argument

- **Upstream v3.3.5**: `_config.palace_path` is the single palace. Globals like `_client_cache`, `_collection_cache`, `_metadata_cache`, `_palace_db_inode`, `_palace_db_mtime` are scalars, not dicts.
- **Local**: every read/write tool can be addressed against any palace via a `palace=` arg. The MCP cache is a `{palace_path: entry}` map. `_get_collection(palace_path=None, …)` defaults to the active palace; non-default paths get their own entry. `_get_kg(palace_path=None)` similarly. `_resolve_palace_arg(palace)` and `_resolve_cli_palace(args)` normalize input.

The merge had to adapt v3.3.5's `_force_chroma_cache_reset()` and `tool_sync`'s `_metadata_cache = None` to write through `_invalidate_metadata_cache()` instead — the globals they targeted no longer exist.

**Open question for upstream**: this is the heaviest single divergence and the foundation of palace isolation. Upstream may have a different read on how to scope ChromaDB connection state across palaces. Worth a design conversation before any of this comes back upstream.

### 4. `mine_palace_lock` reentrance: per-thread vs per-process

- **Upstream v3.3.5**: `_palace_lock_holders = threading.local()`. Re-entrance is per-thread.
- **Local (post-merge)**: process-wide holder set under `threading.RLock`, refreshed on PID change. Re-entrance is per-process.

Necessary fix: v3.3.5 added `ChromaCollection._write_lock()` which acquires `mine_palace_lock` to protect MCP/direct callers against concurrent mines. In the local parallel pipeline, the consumer thread (which calls `collection.upsert` from inside `_write_prepared`) is NOT the orchestrator thread that took the outer `mine_palace_lock` from `mine()`. A per-thread reentrant set sends it down the `fcntl.flock` path, which fails because the same process already holds the lock on a different fd. Result: self-deadlock raised as `MineAlreadyRunning` mid-mine.

The process-wide holder set is consistent with `fcntl.flock`'s actual scope (per-process, not per-thread).

**Open question for upstream**: should `ChromaCollection._write_lock()` be a no-op when the current process already holds the outer mine lock, regardless of which thread acquired it? The "guard MCP against mine" contract still holds either way; the per-thread restriction was incidental.

### 5. Hook architecture: bash script vs Python module

- **Upstream v3.3.5**: `hooks/mempal_save_hook.sh` is the actual save logic — reads transcript path, classifies, fires `mempalace mine`.
- **Local**: `mempalace.hooks_cli` is the save logic; the bash hook is a thin wrapper that pipes stdin JSON to `mempalace hook --hook stop`. Multi-harness (Claude Code, Codex, Cursor) routes through `_parse_harness_input`.

The v3.3.5 test `test_mempal_dir_default_not_empty` asserts the bash script has a transcript fallback when `MEMPAL_DIR` is empty; locally that fallback lives in the Python module, so the test no longer reflects the active code path.

**Open question for upstream**: are the upstream hooks planning to grow multi-harness support? If so, the Python module is the natural place; if not, the bash script needs the fallback.

### 6. Ignore source: `.gitignore` vs `.mempalaceignore`

- **Upstream v3.3.5**: `is_gitignored(...)`, `load_gitignore_matcher(...)`, parameter `respect_gitignore=True`. Hardcodes git as the ignore source.
- **Local**: `is_ignored(...)`, `load_ignore_matcher(...)`, parameter `respect_ignore=True`. The matcher reads `.mempalaceignore` if present, falls back to `.gitignore`.

Aliased in the merge (`from .miner import is_ignored as is_gitignored, …`) so v3.3.5's new `sync.py` keeps working. The naming is the structural divergence — local says "what should mining ignore?" is its own question, not "what does git ignore?".

**Open question for upstream**: would `.mempalaceignore` (with `.gitignore` fallback) be acceptable as the documented mining filter? It's a one-line config change for users who don't care, and a foothold for users who do.

### 7. KG access pattern: single global vs lazy per-path

- **Upstream v3.3.5**: `_kg_by_path: dict[str, KnowledgeGraph]` keyed by absolute path, with `_call_kg(op)` that retries once on `sqlite3.ProgrammingError` (race with `tool_reconnect`).
- **Local**: a default-palace singleton `_kg` (which tests can monkeypatch) plus a `_kg_cache` dict for non-default palaces. `_get_kg(palace_path=None)` returns the singleton on default-path, otherwise builds-and-caches.

The merge kept the local structure and ported v3.3.5's retry helper on top (`_call_kg(op, palace_path=None)`). The shapes are isomorphic; the difference is whether the default palace gets a special-cased singleton (local: yes, to preserve test monkeypatching) or not (upstream: no).

**Open question for upstream**: convergence here is mostly tractable — both designs solve the same problem. Mainly a question of which monkeypatch contract to preserve for existing tests.

### 8. Pluggable backend layer — CONVERGED on upstream (v3.4.0)

Both branches independently built a pluggable backend layer on the shared v3.3.5 `backends/{base,chroma}.py`. Upstream shipped a full framework in v3.4.0 (`registry` + `pgvector`/`qdrant`/`sqlite_exact`/`embedding_wrapper`, RFC 001); the fork had added `turbovec` + `MEMPALACE_BACKEND` routing. **Resolution (user decision, the upstream RFC-001 work was done with the user's input): adopt upstream's framework.** This is the one place the fork deliberately gave up its own design. It was easy: `turbovec.py` already subclassed upstream's `BaseBackend`/`BaseCollection` and registered via a pyproject **entry point**, so almost no re-homing was needed. The only local retentions: chroma is registered **best-effort / imported lazily** (PEP 562 `__getattr__`) so a turbovec-only deploy never imports `chromadb` (fork #20); the new backends register eagerly (their heavy client deps import lazily inside methods). Not a standing divergence — the fork now tracks upstream here.

### 9. Cursor / IDE integration: Python harness vs bash hooks (semantic duplication, not a conflict)

The fork added Cursor support as a `cursor` harness inside `mempalace.hooks_cli` (Python) plus `hooks/mempal_cursor_*.sh`. Upstream added its own Cursor support in v3.4.1 under `hooks/cursor/*` (bash + `lib/common.sh`) and a `.cursor-plugin/`, plus Antigravity in the same release. These live on **different paths**, so they coexist without a git conflict — but the surface is duplicated and the design philosophy clashes (§5 all over again: Python module vs bash). Both were kept. **Open question for upstream**: fold the fork's Python multi-harness cursor path and upstream's bash `hooks/cursor/` into one.

### 10. Embedding selection: provider-first vs model-first — UNIFIED (v3.3.6)

The fork added **remote embedding providers** (`embedding_provider` = onnx/ollama/openai-compat — the DGX-Spark / LAN embedding path); upstream v3.3.6 added a **multilingual model selector** (`embedding_model` = minilm/embeddinggemma + `EmbeddinggemmaONNX`). Both reused the `embedding_model` config key with incompatible meaning. **Resolution (user decision): unify.** `embedding_provider` is the top-level switch; under `onnx`, `embedding_model` (case-insensitive) selects minilm vs embeddinggemma; under a remote provider it is the remote model name. Both capabilities preserved. The v3.4.1 embeddinggemma OOM sub-batching fix folded into the kept `EmbeddinggemmaONNX`. **Open question for upstream**: would a `embedding_provider` switch (onnx default) be acceptable, with the model selector nested under onnx?

### 11. HNSW segment quarantine — CONVERGED on upstream (v3.4.0)

The fork had diverged on chroma HNSW-segment health checks (fork #0991677 "don't quarantine unflushed metadata", #1532 tolerance). Upstream v3.4.0 refined the same logic to distinguish *never-persisted* (sub-threshold → healthy) from *partially-flushed-crashed* (→ quarantine) via `link_lists.bin` — which **subsumes** the fork's tolerance intent. Adopted upstream; local's blanket "missing pickle → healthy" early-return became redundant dead code and was dropped. Not a standing divergence.

### 12. `mine --limit` semantics — OPEN divergence (deferred)

Upstream v3.4.1 (#1535) defines `--limit N` as "stop after N files that produced **new** drawers." The fork's parallel miner applies `--limit N` as a **pre-slice** of the scanned file list. Concretely, `--limit 5` over a directory where 8 of 10 files are already mined yields 0 new drawers locally where upstream yields 2. The `#1535` limit tests are skipped (they patch the monolithic `process_file`, which the parallel consumer bypasses — §2). **This behavior difference is real and unresolved** — porting #1535 into the parallel consumer (as the chunk-cap #1455 port was done) is the fix. Tracked as a follow-up.

---

## Test impact

After the full v3.5.0 merge: **3366 pass, 65 skipped, 1 pre-existing failure**. Of the skips, ~48 are explicit `@pytest.mark.skip` (divergence-driven, below); the rest are environmental (`pytest.importorskip("turbovec")` — `turbovecdb` is an optional backend not installed in the merge venv — and similar `skipif`s). The divergence skips, by site:

- **`test_mcp_server.py` (~21)** — upstream-only MCP internals: KG-cache scalars `_kg_by_path` / `_canonicalize_kg_path` (§7), removed `_collection_cache` / `_metadata_cache` scalars and the SQLite status fast-path's default-only assumptions (§3), and palace-less-import tests that trip the `PalaceNotDeclared` loud-fail (§3).
- **`test_miner.py` (~9)** + **`test_convo_miner*.py` (~4)** — tests that patch the monolithic `process_file` (§2), plus the v3.4.1 `#1535` `--limit` tests (§2 / §12).
- **`test_hybrid_candidate_union.py` + `test_searcher`/`test_sqlite_exact_backend` union tests (~8)** — `candidate_strategy="union"`, `effective_distance`, inline closet-boost, multi-backend lexical capability reporting (§1).
- **`test_sync.py` (~4)** — patch the removed `_metadata_cache` global (§3). (`test_metadata_cache_cleared_on_exception`, formerly a deselected/failing baseline, is now a clean `skip`.)
- **`test_save_hook_mines.py` (2)** — bash-hook transcript fallback that now lives in the Python module (§5).

The **1 remaining failure** is pre-existing (present at the pre-merge baseline, not introduced by any merge): `test_hook_chat_palace::test_every_mempalace_mine_call_targets_chat_palace` — it greps the bash hook for a literal `mempalace mine`, but the save logic lives in `mempalace.hooks_cli` (§5). Fixing it means updating the assertion to the Python code path.

None of the skips are "the code is broken." They're "the test asserts an implementation that no longer exists." Each has an inline `@pytest.mark.skip(reason=…)` describing what it asserted and what restoring it would need.

---

## What this means for upstream collaboration

Of the divergences:

- **§4 (lock reentrance)** is a candidate for an upstream PR with no architectural ask — a defensible bugfix even in a single-threaded mine, needed by anyone who later wants parallel mining.
- **§7 (KG lazy cache)** is mostly converged; mainly a naming question.
- **§6 (`.mempalaceignore`)** is small surface, additive, defensible upstream.
- **§3 (palace addressability)** and **§1 (closet/primary split)** are the load-bearing local changes; they're the reason this fork exists. Worth a design conversation before any backport.
- **§2 (parallel mine)** and **§5 (hooks architecture)** are pragmatic local choices that may or may not match upstream's roadmap.
- **§8 (backends)** and **§11 (HNSW quarantine)** already converged onto upstream — no longer tensions.
- **§9 (Cursor)** and **§10 (embedding)** are duplications/unifications, not backport candidates.

Net suggestion for the upstream conversation: lead with §4 and §6 (small, mergeable, valuable), then propose §3 and §1 as design RFCs. The rest follows from those.

---

## Deferred follow-ups (open work, tracked here so nothing is silently dropped)

None of these are regressions; each is a conscious "keep local / defer" from the v3.5.0 merge.

1. **§12 `--limit` semantics** — port upstream #1535 ("stop after N *new*") into the parallel mine consumer; today's local `files[:limit]` pre-slice under-mines a partially-mined directory. Highest-value follow-up.
2. **#1383 KG cache canonicalization** — upstream now keys the KG cache by `realpath`+`normcase` (collapses symlinked / case-variant palace paths); local `_kg_cache` still keys by `abspath`+`expanduser`. Port the canonicalization into the local dual-cache (§7).
3. **§3 `is_default` status fast-path gating** — v3.5.0's SQLite status/overview fast-path reads the global `_config.palace_path`, so it is gated to the default palace; a cross-palace `palace=` query falls through to the client path. Revisit if the fast-path should be made palace-parameterized.
4. **Turbovec MCP path — unverified end-to-end.** `mcp_server._get_collection` stays chroma-centric (§3); turbovec palaces are served through `palace.get_collection` (backend-routed) via the searcher / CLI-mine paths, not that MCP helper. The turbovec MCP status/search path could not be exercised here (`turbovecdb` not installed; `test_turbovec_backend.py` skips via `importorskip`). Since the chat palace runs on turbovec, verify direct MCP tools against a turbovec palace before relying on them.
