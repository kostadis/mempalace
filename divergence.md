# Divergence: `kostadis-dev` vs upstream `v3.3.5`

Context: `kostadis-dev` is a personal fork of MemPalace that started as a downstream branch of upstream `v3.3.4`. It accumulated 39 local commits before v3.3.5 was tagged upstream. PR #15 (`merge/v3.3.5-into-kostadis-dev`) brings v3.3.5 in. This document explains, for offline review and potential discussion with the upstream maintainers, where the two branches diverge and why.

The divergence is not the merge — most of v3.3.5's 41 commits applied cleanly. The divergence is **structural**: a handful of v3.3.5 features were dropped, restructured, or replaced because the local branch had already moved in a different direction on the same surface area. This document enumerates those tensions.

---

## Local features that have no upstream counterpart

These were built on `kostadis-dev` between v3.3.4 and the merge. None are upstream as of v3.3.5.

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

These are the load-bearing differences — places where the same surface area has incompatible designs. The merge resolved each by keeping the local design and adapting v3.3.5's bugfixes into it. Re-aligning with upstream would require an architectural conversation, not a code change.

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

---

## Test impact

After the merge, 2002 tests pass, 14 are skipped with documented reasons, 1 is deselected. The skips break down as:

- **`test_hybrid_candidate_union.py` (11)** — entire file; tests v3.3.5's `candidate_strategy="union"` (§1).
- **`test_searcher::test_effective_distance_clamped_to_valid_cosine_range` (1)** — v3.3.5's inline closet boost (§1).
- **`test_miner::test_mine_arbitrary_exception_prints_summary_and_reraises` (1)** — patches `process_file` (§2).
- **`test_save_hook_mines::test_mempal_dir_default_not_empty` (1)** — bash hook transcript fallback (§5).
- **`test_sync::test_metadata_cache_cleared_on_exception` (1, deselected)** — patches removed `_metadata_cache` global (§3).

None of these are "the code is broken" failures. They're "the test asserts an implementation that no longer exists" failures. Each skip has an inline `@pytest.mark.skip(reason=…)` annotation describing what the test asserted and what would be needed to restore it.

---

## What this means for upstream collaboration

Of the seven structural divergences:

- **§4 (lock reentrance)** is a candidate for an upstream PR with no architectural ask — it's a defensible bugfix even in a single-threaded mine, and it's needed by anyone who later wants parallel mining.
- **§7 (KG lazy cache)** is mostly converged; mainly a naming question.
- **§6 (`.mempalaceignore`)** is small surface, additive, defensible upstream.
- **§3 (palace addressability)** and **§1 (closet/primary split)** are the load-bearing local changes; they're the reason this fork exists in the first place. Worth a design conversation before any backport.
- **§2 (parallel mine)** and **§5 (hooks architecture)** are pragmatic local choices that may or may not match upstream's roadmap.

Net suggestion for the upstream conversation: lead with §4 and §6 (small, mergeable, valuable), then propose §3 and §1 as design RFCs. The rest follows from those.
