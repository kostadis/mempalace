# Embrace parallelism: miner first, then verify, then the rest

## Context

The user owns a DGX Spark running vLLM (embeddings) and Ollama (LLM refinement). Both happily handle concurrent requests. mempalace's mining pipeline is currently single-threaded: `_mine_impl` in `mempalace/miner.py:1110` iterates files serially, and each `process_file` (line 803) does `read → chunk → collection.upsert(documents=[...])`, where the upsert internally fires one synchronous HTTP request to vLLM and blocks the rest of the pipeline. Measured throughput: vLLM saturates at ~11,400 tok/s but sits idle most of the wallclock during a real mine.

The previous PR (`#6`, commit `79c09fe`) shipped remote embedding providers (Ollama, OpenAI-compat) and exposed the speed ceiling. This plan rewires mempalace to use that ceiling by introducing producer/consumer parallelism, while preserving the two non-negotiable constraints: ChromaDB HNSW is single-writer (`mempalace/backends/chroma.py:1104, 1158`), and mines must remain idempotent / KeyboardInterrupt-resumable.

**Delivery sequence (per user directive):**
0. **Initial PR — plan + LLM config-mirror.** Ships this design doc into `docs/design/`, the `.gitignore` hygiene fix for Spark-generated artifacts, and a small `config.py` + `cli.py` change to give remote LLMs the same config-persistence as remote embeddings got in PR #6. No parallelism code in this PR — it's the foundation the next one builds on.
1. **Phase 1 — Miner.** Separate PR. Ships `mempalace/miner.py` parallelism + the shared `mempalace/parallel.py` primitive.
2. **Phase 2 — Verify on the Spark.** Run the verification gate below. Gate-pass before moving on.
3. **Phase 3 — The rest.** `convo_miner`, `llm_refine`, `closet_llm`, then port the embedding-client tests and remove the temporary env shim.

---

## Initial PR — plan doc + .gitignore + LLM config-mirror

**Scope:** zero parallelism code. This PR is foundation only.

### Files

| File | Change |
| --- | --- |
| `docs/design/embrace-parallelism.md` | **NEW.** This plan, copied from `~/.claude/plans/i-want-you-to-luminous-hopcroft.md`. Mirrors the existing `docs/design/palace-isolation.md` location. |
| `.gitignore` | Already modified locally: ignore `mempalace.yaml` and `entities.json` (per-project files mempalace writes into consumer repos during Spark-era mining; issue #185). |
| `mempalace/config.py` | New properties: `llm_provider`, `llm_model`, `llm_endpoint`, `llm_api_key`. Mirror the env-first / file / default pattern from `embedding_provider` (`config.py:556-579`). |
| `mempalace/cli.py` | Update the `get_provider(...)` call at `cli.py:280-283` so the `model=` and `endpoint=` args default to `MempalaceConfig().llm_model` / `llm_endpoint` when the CLI flags are unset. Same for `api_key`. |
| `tests/test_config.py` (or wherever the `embedding_provider` test lives) | New tests: env > file > default precedence for each new key; verify the asymmetric default (`llm_provider="ollama"`, `llm_model="gemma4:e4b"`, `llm_endpoint=None`, `llm_api_key=None`) matches today's behavior so this is non-breaking. |

### Why this lands now

Remote LLM support **already exists end-to-end** in mempalace — `mempalace/llm_client.py:399-403` registers `ollama`, `openai-compat`, and `anthropic` providers, and `mempalace/cli.py:280` calls `get_provider(...)` with `--llm-endpoint`. The gap is *config persistence*: there's no `MEMPALACE_LLM_*` env / config-file mirror for the LLM the way PR #6 gave embeddings `MEMPALACE_EMBEDDING_*`. Today, "use remote LLM" requires passing `--llm-endpoint` on every command. Phase 3 (`llm_refine` parallelism) will need the persistence layer anyway — landing it now decouples the two concerns and keeps Phase 1 (miner) and Phase 3 (refine) focused purely on parallelism.

### Config properties to add (mirror `embedding_provider` shape verbatim)

```python
@property
def llm_provider(self) -> str:
    env_val = os.environ.get("MEMPALACE_LLM_PROVIDER")
    if env_val:
        return env_val.strip().lower()
    return str(self._file_config.get("llm_provider", "ollama")).strip().lower()

@property
def llm_model(self) -> str:
    env_val = os.environ.get("MEMPALACE_LLM_MODEL")
    if env_val:
        return env_val.strip()
    return str(self._file_config.get("llm_model", "gemma4:e4b")).strip()

@property
def llm_endpoint(self) -> Optional[str]:
    env_val = os.environ.get("MEMPALACE_LLM_ENDPOINT")
    if env_val:
        return env_val.strip()
    val = self._file_config.get("llm_endpoint")
    return str(val).strip() if val else None

@property
def llm_api_key(self) -> Optional[str]:
    # Never persisted to file — env-only, mirroring how OPENAI_API_KEY /
    # ANTHROPIC_API_KEY work in mempalace/llm_client.py:267, 347.
    env_val = os.environ.get("MEMPALACE_LLM_API_KEY")
    return env_val.strip() if env_val else None
```

**Defaults match today's behavior** (`cli.py:278` default `"gemma4:e4b"` on Ollama, no endpoint → `OllamaProvider` falls back to `localhost:11434`). Non-breaking.

### CLI wiring at `mempalace/cli.py:278-283`

Change from:

```python
provider_model = getattr(args, "llm_model", "gemma4:e4b") or "gemma4:e4b"
candidate = get_provider(
    "ollama",
    provider_model,
    endpoint=getattr(args, "llm_endpoint", None),
    ...
)
```

to:

```python
cfg = MempalaceConfig()
provider_name = getattr(args, "llm_provider", None) or cfg.llm_provider
provider_model = getattr(args, "llm_model", None) or cfg.llm_model
provider_endpoint = getattr(args, "llm_endpoint", None) or cfg.llm_endpoint
provider_api_key = getattr(args, "llm_api_key", None) or cfg.llm_api_key
candidate = get_provider(
    provider_name,
    provider_model,
    endpoint=provider_endpoint,
    api_key=provider_api_key,
    ...
)
```

This unlocks `mempalace mine` (and every other LLM-using command) honoring `~/.mempalace/config.json` and `MEMPALACE_LLM_*` env vars without per-invocation flags. **Critically**: also add a `--llm-provider` flag to the argparse setup at the same place `--llm-endpoint` is registered, since today the provider name is hardcoded to `"ollama"` in cli.py:280.

### Verification for this initial PR

```bash
python -m pytest tests/ -v --ignore=tests/benchmarks
```

Manual smoke test on the Spark:

```bash
# Before this PR: this requires --llm-endpoint on every command.
MEMPALACE_LLM_PROVIDER=openai-compat \
MEMPALACE_LLM_MODEL=qwen2.5-coder:14b \
MEMPALACE_LLM_ENDPOINT=http://192.168.1.147:8001 \
mempalace mine ~/src/CampaignGenerator --limit 5 --wing smoke_test

# Confirm the LLM call hit the Spark vLLM endpoint, not localhost Ollama.
```

If the smoke test passes and the test suite is green, this PR ships. Then the Phase 1 PR (parallel miner) opens against `kostadis-dev` with this PR's foundation already in place.

---

## Architecture

Producer/consumer split:

```
files → [Producer pool, N threads] ──[bounded queue]──→ [Consumer, 1 thread]
         read → chunk → ef(docs)                          collection.upsert(
         compute embeddings                                 documents=...,
         build deterministic IDs/metadata                   embeddings=...,
                                                            ids=...,
                                                            metadatas=...)
```

- Producers saturate the embedding endpoint (vLLM). Each producer holds no shared mutable state.
- The single consumer is the only thread that touches `collection` / `closets_col` after this refactor — HNSW's `num_threads=1` invariant holds.
- `collection.upsert(embeddings=...)` bypasses the EF inside chromadb (`mempalace/backends/chroma.py:697-703` is a pure pass-through; verified — no in-tree caller currently uses this code path).

---

## Phase 1 — Miner

### Files modified / created

| File | Change |
| --- | --- |
| `mempalace/parallel.py` | **NEW.** ~120-line producer/consumer harness. Reusable in Phase 3. |
| `mempalace/miner.py` | Split `process_file` at the EF seam; rewire `_mine_impl` loop onto `ParallelPipeline`; thread a shared EF reference through. |
| `mempalace/embedding_openai.py` | Add `urllib3.PoolManager` keep-alive path; gate on `MEMPALACE_HTTP_KEEPALIVE` env. |
| `mempalace/embedding_ollama.py` | Same keep-alive treatment. |
| `mempalace/config.py` | New `workers` property (env-first / file / default), asymmetric default. |
| `mempalace/cli.py` | New `--workers` flag on the `mine` subparser; plumbed through `cmd_mine`. |
| `tests/test_parallel_pipeline.py` | **NEW.** Unit tests for the harness. |
| `tests/test_miner_parallel.py` | **NEW.** Integration tests for the parallel miner. |
| `tests/test_embedding_openai_pool.py` | **NEW.** Pool-path tests. |
| `tests/test_embedding_ollama_pool.py` | **NEW.** Pool-path tests. |
| `tests/conftest.py` | Add autouse fixture setting `MEMPALACE_HTTP_KEEPALIVE=0` so existing `urlopen`-patching tests keep passing unchanged. |

### `mempalace/parallel.py` (new)

```python
@dataclass
class WorkerResult:
    payload: Any                # consumer-defined struct
    file_id: str                # for logging / progress
    drawer_count: int           # for progress counters
    extra: dict | None = None   # room name, etc.

class ParallelPipeline:
    def __init__(
        self,
        producer_fn,            # item -> WorkerResult | None
        consumer_fn,            # WorkerResult -> None (runs in single thread)
        *,
        workers: int,
        queue_size: int,
        on_progress=None,
        on_error=None,
    ): ...

    def run(self, items) -> PipelineStats: ...
```

Topology: one `threading.Thread` consumer + `ThreadPoolExecutor(max_workers=workers)` producers, `queue.Queue(maxsize=queue_size)`. `_SENTINEL = object()` for shutdown; `threading.Event` for Ctrl-C. A failed producer emits a `_FailedResult(file, exc)` queue entry rather than crashing the pipeline (skip-file semantics, see Error handling).

The single-writer guarantee is the **caller's** responsibility — the harness only promises that `consumer_fn` runs in exactly one thread.

### `mempalace/miner.py` rewire

Split `process_file` (currently lines 803-916) into three functions at the EF seam:

1. `_prepare_file(filepath, project_path, wing, rooms, agent, dry_run, collection) -> _PreparedFile | None`
   - All current logic up to chunking + deterministic ID construction.
   - Does NOT acquire `mine_lock` and does NOT touch the collection writer.
   - Returns `None` for files that should skip (already mined / too small / unreadable).

2. `_embed_prepared(prepared, ef) -> list[list[list[float]]]`
   - Calls `ef(documents)` batched per `DRAWER_UPSERT_BATCH_SIZE` (currently 1000, `mempalace/miner.py:72`).
   - The EF reference is created **once** in `_mine_impl` and passed through; do NOT call `get_embedding_function()` from inside producers — the `_EF_CACHE` dict at `mempalace/embedding.py:57` is mutated without a lock and races on first fill.

3. `_write_prepared(prepared, embeddings_batches, collection, closets_col, wing)`
   - Runs inside the consumer thread.
   - Acquires `mine_lock(source_file)` (preserves the existing cross-process guarantee at `mempalace/miner.py:839`), re-checks `file_already_mined`, deletes by `source_file`, then calls `collection.upsert(documents=..., ids=..., metadatas=..., embeddings=batch_embeddings)`. Builds + upserts the closet exactly as today.

Rewire `_mine_impl` (`mempalace/miner.py:1052`):

- Add `workers: int | None = None` to `mine()` (`miner.py:997`) and `_mine_impl()` signatures. `None` → `MempalaceConfig().workers`.
- Call `get_embedding_function()` once after `get_collection()` (around `miner.py:1097`) and bind the result locally.
- Replace the `for i, filepath in enumerate(files, 1):` loop with:
  ```python
  pipeline = ParallelPipeline(
      producer_fn=lambda fp: _producer(fp, project_path, wing, rooms, agent, dry_run, ef, collection),
      consumer_fn=lambda r: _consumer(r, collection, closets_col, wing),
      workers=workers,
      queue_size=max(2 * workers, 4),
      on_progress=_progress_callback,  # prints the existing "  + [..] +N" line
      on_error=_error_callback,        # increments files_failed
  )
  stats = pipeline.run(files)
  ```
- Wrap the call in the existing `try/except KeyboardInterrupt` (`miner.py:1163`) — the harness sets `_shutdown`, in-flight producers exit at next checkpoint, the consumer drains what it already has, and the main thread re-raises so the existing resume-summary path stays intact.
- `dry_run` short-circuits to the historic serial path (no point parallelizing `print()`).

### `mempalace/embedding_openai.py` and `mempalace/embedding_ollama.py`

Add a module-level `_HTTP_POOL: urllib3.PoolManager | None = None`. Lazy-init via `_get_http_pool()`:

```python
def _get_http_pool():
    global _HTTP_POOL
    if _HTTP_POOL is None:
        import urllib3
        workers = max(1, int(os.environ.get("MEMPALACE_WORKERS", "8")))
        _HTTP_POOL = urllib3.PoolManager(
            num_pools=4,
            maxsize=max(32, workers * 2),
            block=False,
        )
    return _HTTP_POOL
```

`maxsize` must exceed `workers` so producers don't serialize on connection acquisition. Read `MEMPALACE_WORKERS` directly (don't import `MempalaceConfig` — avoids the existing lazy-import cycle at `mempalace/embedding.py:163`).

`_post_json` gets one branch:

```python
if os.environ.get("MEMPALACE_HTTP_KEEPALIVE", "1") == "0":
    return _post_json_via_urlopen(...)   # historic path, kept verbatim
return _post_json_via_pool(...)
```

The pool path translates `urllib3.exceptions.HTTPError`, `MaxRetryError`, `TimeoutError` into the existing `OpenAIEmbeddingError` / `OllamaEmbeddingError` types so callers see identical exceptions. JSON marshaling stays in-tree (we use `urllib3` raw, not `requests`).

**Why the env-var escape hatch (user-chosen approach):** existing tests at `tests/test_embedding_openai.py:112` patch `mempalace.embedding_openai.urlopen` directly. Forcing all of them to rewrite for `urllib3.PoolManager.request` in this PR triples the diff and bleeds Phase 1 into the embedding-client area. The escape hatch lets Phase 1 ship with the parallelism + keep-alive change clean. The shim is temporary — Phase 3 ports the tests to the pool path and removes the env-var branch.

### `mempalace/config.py` (new property)

Insert between `embedding_endpoint` (ends ~line 631) and `topic_tunnel_min_count` (~line 633). Mirror the env-first / file / default pattern from `embedding_provider` (`mempalace/config.py:556-579`):

```python
@property
def workers(self) -> int:
    """Producer threads for the parallel miner.

    Defaults to 1 for embedding_provider=onnx (local, GIL-bound) and 8 for
    remote providers (ollama / openai-compat). Override via MEMPALACE_WORKERS
    or the `workers` config key. Clamped to [1, 64].
    """
    env_val = os.environ.get("MEMPALACE_WORKERS")
    if env_val:
        try:
            return max(1, min(64, int(env_val)))
        except ValueError:
            pass
    cfg_val = self._file_config.get("workers")
    if cfg_val is not None:
        try:
            return max(1, min(64, int(cfg_val)))
        except (TypeError, ValueError):
            pass
    return 1 if self.embedding_provider == "onnx" else 8
```

**Asymmetric default (user-chosen):** ONNX runs in-process under the GIL and gains little from threads — 1 keeps the laptop default behavior unchanged. Remote providers (the DGX Spark + vLLM path) get the parallel default automatically.

### `mempalace/cli.py` plumbing

`mempalace/cli.py:1084` — insert immediately after the existing `--limit` flag on `p_mine`:

```python
p_mine.add_argument(
    "--workers", type=int, default=None,
    help="Producer threads for parallel embedding "
         "(default: $MEMPALACE_WORKERS / config / 1 for onnx, 8 for remote). "
         "Use 1 for the historic serial mine.",
)
```

`mempalace/cli.py:533-545` — pass `workers=args.workers` into the `mine(...)` call. The `args.mode == "convos"` branch (`cli.py:521`) does NOT get `--workers` in Phase 1 — that's Phase 3.

---

## Phase 1 — Queue, error, shutdown semantics

**Bounded queue.** Each entry carries 768-dim float vectors for up to `DRAWER_UPSERT_BATCH_SIZE = 1000` chunks (≈10–15 MB worst case, 50–500 KB typical). Bound at `2 * workers` (default 16). With 8 workers and typical files this peaks at ~3 MB; pathological worst case ~120 MB — acceptable on a 128 GB Spark, surface via `MEMPALACE_MINE_QUEUE_SIZE` for tuning.

**Backpressure.** Producers `queue.put(entry, block=True)`. When ChromaDB falls behind on a large palace, producers naturally throttle. We never drop work.

**Error policy: skip-file with bounded retry.**
- `OpenAIEmbeddingError` / `OllamaEmbeddingError` in `_embed_prepared` → retry once with 250 ms backoff. Second failure → `_FailedResult(file, exc)` queue entry. Consumer logs `WARNING: skipped {file}: {exc}` and increments `files_failed`. Final summary reports `files_failed` alongside `files_skipped`.
- Mirrors current liberal per-file failure handling at `mempalace/miner.py:822` (OSError swallowed silently).

**Shutdown signaling.**
- Normal end: orchestrator joins producer pool → puts one `_SENTINEL` → joins consumer thread.
- Ctrl-C: `threading.Event` set; producers exit at next checkpoint; consumer drains what's already on the queue; main thread re-raises `KeyboardInterrupt` so the existing summary at `mempalace/miner.py:1163` fires.

**Idempotent resume after Ctrl-C** is preserved:
- Drawer IDs are SHA-deterministic (`mempalace/miner.py:869`) — already.
- A file whose producer-side embed completed but whose consumer-side upsert did not: `file_already_mined` returns False on resume, the file is re-embedded and re-upserted. A few hundred ms of duplicated GPU work per interrupted file. Acceptable.
- A file mid-consumer: `mine_lock` + the existing `delete(where={"source_file": ...})` means a partial drawer set looks stale to `file_already_mined` (mtime check) on resume → full re-mine of that file. Same as today.

---

## Existing functions / utilities to reuse

| Reuse | Location |
| --- | --- |
| `get_embedding_function()` — returns thread-safe callable EF | `mempalace/embedding.py:147-218` |
| `ChromaCollection.upsert(embeddings=...)` — pre-computed embeddings pass-through | `mempalace/backends/chroma.py:697-703` |
| `mine_lock(source_file)` — per-file cross-process lock | `mempalace/miner.py:839` |
| Drawer ID determinism (SHA-based) | `mempalace/miner.py:869` |
| `file_already_mined` — mtime-aware skip check | called at `mempalace/miner.py:817, 841` |
| `DRAWER_UPSERT_BATCH_SIZE = 1000` — keep current batch size | `mempalace/miner.py:72` |
| Existing argparse pattern (`--limit`) for new `--workers` flag | `mempalace/cli.py:1084` |
| Env-first / file / default config pattern (`embedding_provider`) | `mempalace/config.py:556-579` |
| `urllib3` (transitive dep via chromadb→requests) for `PoolManager` | uv.lock |
| HTTP test pattern: patch the urllib symbol on the module | `tests/test_embedding_openai.py:1-25, 112` |
| Concurrent test patterns: `ThreadPoolExecutor`, `threading.Barrier` | `tests/benchmarks/test_search_bench.py`, `tests/test_palace_indices.py`, `tests/test_closets.py` |

---

## Phase 2 — Verification gate on the Spark

Phase 1 must pass this gate before any Phase 3 work begins.

```bash
# Baseline: workers=1 must match the historic serial mine exactly.
MEMPALACE_EMBEDDING_PROVIDER=openai-compat \
MEMPALACE_EMBEDDING_ENDPOINT=http://192.168.1.147:8000 \
MEMPALACE_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5 \
time mempalace mine ~/src/CampaignGenerator --workers 1 --wing bench_serial

# Parallel mine, default 8 workers.
time mempalace mine ~/src/CampaignGenerator --workers 8 --wing bench_par8

# Push it.
time mempalace mine ~/src/CampaignGenerator --workers 16 --wing bench_par16

# Re-mine the workers=8 palace — must report 100% files skipped.
time mempalace mine ~/src/CampaignGenerator --workers 8 --wing bench_par8
```

In parallel with the parallel runs:
- `ssh spark "nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1"` — sustained ≥ 70% during steady state.
- vLLM server logs — `prompt_tokens_per_second` should approach the 11,400 tok/s ceiling.
- `ps -o rss= -p $(pgrep -f 'mempalace mine')` every second — peak RSS < 1 GB.

**Pass gate (all four required):**
1. `workers=8` mine completes with zero `_FailedResult` entries on a clean run.
2. `workers=8` palace returns top-5 search results overlapping ≥ 4 with the `workers=1` palace for a known-string query.
3. Wallclock improvement ≥ 4× over the `workers=1` baseline on a real corpus.
4. Re-mine reports `Files skipped (already filed): N` for every file, `Drawers filed: 0`.

If any check fails, fix in Phase 1 before Phase 3.

---

## Phase 3 — The rest (post-verification punch list)

| Site | File:line | Work |
| --- | --- | --- |
| Conversation miner | `mempalace/convo_miner.py:422` | Same producer/consumer pattern via `ParallelPipeline`. Add `--workers` to the `mine --mode convos` branch at `mempalace/cli.py:521`. |
| LLM entity refinement | `mempalace/llm_refine.py` | The current bottleneck per the user's notes. Fan-out Ollama chat calls via `ParallelPipeline`; single-writer for the refinement output. |
| Closet LLM scoring | `mempalace/closet_llm.py:251` | One LLM call per source file — straight `ParallelPipeline` fan-out. |
| Read-heavy fan-outs | `mempalace/dedup.py:184`, `mempalace/repair.py:121` | `collection.query` per source group / batch. Independent reads; thread-pool friendly. |
| Test seam cleanup | `tests/test_embedding_openai.py`, `tests/test_embedding_ollama.py`, `tests/conftest.py` | Port existing patches to `urllib3.PoolManager.request`. **Remove** the `MEMPALACE_HTTP_KEEPALIVE` env-var branch from both embedding clients. The escape hatch was Phase 1 scaffolding; it dies here. |

**Explicitly deferred** (revisit only if Phase 3 benchmarks demand it): CPU-bound regex fan-outs in `mempalace/entity_detector.py:475` and `mempalace/fact_checker.py:116` — GIL-bound, would need `ProcessPoolExecutor`. `mempalace/sweeper.py` — separate code path with its own cursor logic.

---

## Verification — Phase 1 tests

Run:

```bash
python -m pytest tests/ -v --ignore=tests/benchmarks
```

New tests that must pass:

**`tests/test_parallel_pipeline.py`**
- `test_pipeline_runs_serial_when_workers_is_1` — baseline equivalence.
- `test_pipeline_processes_all_items` — 100 items in, 100 callbacks fired.
- `test_pipeline_propagates_producer_exception_to_failed_result` — producer raises on item #5, pipeline emits `_FailedResult`, other items succeed.
- `test_pipeline_drains_queue_on_keyboard_interrupt` — `threading.Barrier` coordinates a Ctrl-C mid-stream; consumer drains taken items; main observes `KeyboardInterrupt`.
- `test_pipeline_queue_size_provides_backpressure` — slow consumer + fast producer + `queue_size=2`: producers block (asserted via a flag they set before `queue.put`).
- `test_pipeline_consumer_runs_in_single_thread` — all `consumer_fn` invocations share one `threading.current_thread().ident`. **This is the HNSW invariant.**

**`tests/test_miner_parallel.py`**
- `test_parallel_mine_idempotent_re_run` — two consecutive `workers=4` mines, second reports 100% skipped.
- `test_parallel_mine_matches_serial_output` — same fixture, two palaces (workers=1 vs workers=8); drawer ids and content sets identical.
- `test_parallel_mine_keyboardinterrupt_partial_progress_resumable` — interrupt mid-run, restart, all files end up filed.
- `test_parallel_mine_workers_plumbing` — `args.workers=4` reaches `_mine_impl` (mock `ParallelPipeline.__init__`).
- `test_parallel_mine_skips_file_on_embedding_error_does_not_abort` — EF raises for one file; mine completes; that file in `files_failed`; the rest filed.
- `test_workers_config_env_priority` — `MEMPALACE_WORKERS=12` overrides config-file, which overrides default 8 (or 1 for onnx).

**`tests/test_embedding_openai_pool.py`** / **`tests/test_embedding_ollama_pool.py`**
- `test_pool_reuses_single_manager_across_calls`
- `test_pool_max_size_honors_workers_env`
- `test_pool_max_retry_error_becomes_embedding_error`
- `test_legacy_urlopen_path_when_keepalive_disabled` — `MEMPALACE_HTTP_KEEPALIVE=0` → existing `urlopen` patch is exercised. Belt-and-braces for the test seam.

Existing tests expected to keep passing **unchanged** thanks to the `tests/conftest.py` autouse `MEMPALACE_HTTP_KEEPALIVE=0` fixture:
- `tests/test_embedding_openai.py` (~150 lines of urlopen patching)
- `tests/test_embedding_ollama.py` (same shape)
- `tests/test_miner.py:24` `test_project_mining` — tiny fixture, parallel path trivial; if any flake appears it's the EF cache race fixed by binding the EF once in `_mine_impl`.

---

## Risks and open unknowns

1. **`_EF_CACHE` dict race** at `mempalace/embedding.py:57` — addressed by binding the EF once in `_mine_impl` and passing the reference to every producer. The dict is never mutated during the parallel section.
2. **GIL contention on CPU-bound chunking/metadata work in producers.** Worst case: 8 producers serialize on the GIL and gain little. Mitigation: producers spend most wall time inside `ef(...)` waiting on the socket, which releases the GIL. Verify post-implementation with `py-spy top` during a parallel mine — if `chunk_text` shows > 30% wall time, surface it and revisit in a follow-up (ProcessPoolExecutor for CPU work, threads for IO).
3. **`urllib3` version drift** — `Retry.allowed_methods` was renamed in 2.0. The lockfile is on 2.6.3, so we're safe; but if we add a `Retry(...)` and a transitive downstream pins < 2.0 we'd break. Mitigation: skip the explicit `Retry` and rely on urllib3 defaults. Application-level retry in the error policy already covers what matters.
4. **Memory blowup on pathological files (thousands of chunks).** Single queue entry can hit ~15 MB. With `queue_size=16`, peak is ~240 MB — fine on the Spark but unfriendly on a laptop. Mitigation: if benchmarks show real pain, split a single file's batches across multiple queue entries with a `file_complete: bool` flag and an accumulator in the consumer. Don't pre-optimize.
5. **`describe_device()` log at `mempalace/miner.py:1087`** — called once, before the producer pool exists. No change needed; flag here so it doesn't accidentally migrate into the loop body.
6. **Hooks-side `_spawn_mine` PID file** at `mempalace/miner.py:1188+` — unchanged; the hook contract is process-level, not thread-level. Confirm no test in `tests/test_save_hook_mines.py` asserts on serial-mine progress-line ordering that would break under parallel emission.

---

## Critical files

- `mempalace/miner.py`
- `mempalace/parallel.py` (NEW)
- `mempalace/embedding_openai.py`
- `mempalace/embedding_ollama.py`
- `mempalace/config.py`
- `mempalace/cli.py`
- `tests/conftest.py`
- `tests/test_parallel_pipeline.py` (NEW)
- `tests/test_miner_parallel.py` (NEW)
- `tests/test_embedding_openai_pool.py` (NEW)
- `tests/test_embedding_ollama_pool.py` (NEW)
