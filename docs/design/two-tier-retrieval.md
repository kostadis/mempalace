# Two-tier retrieval: verbatim primary + LLM themes

## What this delivers

`search_within` returns two distinct lists per query, not one merged ranking. Drawer-direct (verbatim cosine) results go in `primary`. Closet-driven (LLM judgment) results go in `themes`. Closet content no longer influences drawer ranking.

The boost-not-gate principle the architecture already honored becomes its strong form: bad closets cannot corrupt the primary answer because they don't enter that ranking at all.

## Required reading before starting

Read these in order — they cover the rationale and the rule this implements:

1. `~/src/dgx/closets-as-themes.md` — the design rationale and the inversion this fixes.
2. `~/.claude/CLAUDE.md` (the LLM Pipeline Design Rule section) — why LLM scope decisions need a checkpoint.
3. `~/src/mempalace/CLAUDE.md` — the verbatim/100%-recall mission this honors.

Then skim `~/.claude/projects/-home-kroussos-src-mempalace/memory/project_campaign_dev_palace.md` from the section "Vocabulary-gap fix" onward — it has the empirical evidence (the dilution math, the 15-files-share-one-sentence finding) that motivated this work.

## Background — what's in the palace now

- ~906 source files, ~6,125 closet rows. Of those: 2,596 prose closets (`generated_by=llm:Qwen/Qwen2.5-*`), 3,516 taxonomy rows (`generated_by=taxonomy:v2`), 13 misc.
- `mempalace/closet_taxonomy.py` exists (commit `3378e86`) and the taxonomy rows in the palace came from it. The module stays; it's good source material for `themes`.
- `search_within` currently merges drawer hits + closet boost into one ranking (commit `781dc24`). This is the code that needs to change.
- The 4 validation queries from the prior session are documented in the memory file with before/after numbers. Use them to verify the new behavior.

## Architecture target

```python
search_within(query, palace_path, ...) -> {
    "query": str,
    "filters": {"wing": [...], "room": [...], "ids": int or None},
    "primary": [
        {
            "drawer_id": str,
            "text": str,
            "wing": str,
            "room": str,
            "source_file": str,
            "created_at": str,
            "similarity": float,   # 1 - distance, rounded
            "distance": float,
        },
        ...
    ],
    "themes": [
        {
            "source_file": str,
            "wing": str,
            "room": str,
            "closet_text": str,           # the sentence that matched
            "closet_distance": float,
            "similarity": float,
            "generated_by": str,          # "llm:..." or "taxonomy:v2"
            "drawer_ids": [str, ...],     # closet's drawer refs, for drilldown
        },
        ...
    ],
    "total_before_filter": int,
}
```

Two lists. No `effective_distance` on primary hits. No `closet_boost` field. No `matched_via` field (tier is implicit).

## Implementation plan

### 1. Rewrite `search_within` — `mempalace/searcher.py:577`

Three concrete changes inside this function:

a. **Drop the closet-boost subtraction.** Lines that compute `effective_dist = dist - boost` go away. Drawer hits are sorted by raw `distance`. Remove the `CLOSET_RANK_BOOSTS` list and `CLOSET_DISTANCE_CAP` from the active code path.

b. **Delete the closet-seed candidate injection.** The block I added at lines ~735-793 (the loop that pulls closet-tagged sources' drawers into `scored`) goes away. Those closet hits now go into `themes` instead of contaminating `primary`.

c. **Keep the closet collection query at lines ~647-669.** Instead of building `closet_boost_by_source` for the boost mechanism, build the `themes` list from those same results. Dedup to top closet row per `source_file` (first occurrence wins, since results are cosine-sorted).

The drawer-grep enrichment at lines ~798-849 was triggered by closet-boosted hits. In the new architecture, `primary` hits have no closet involvement, so this hydration logic becomes unused for `primary`. **Recommendation:** delete it from `primary`-hit handling. Consider reusing the same logic for `themes` hits in a follow-up — surfacing one verbatim drawer snippet per theme to help the user drill down — but **not in this PR**.

### 2. Update `search_memories` — `mempalace/searcher.py:872`

Thin wrapper. Pass through the new shape. Verify the `filters` block still has the historical single-value shape it promises.

### 3. Update `tool_search` and `tool_search_hierarchical` — `mempalace/mcp_server.py:707, 761`

`tool_search` (line 707) calls `search_memories` at line 735 and returns the dict. Likely just propagates the new shape; verify.

`tool_search_hierarchical` (line 761) calls `search_within` at lines 817 and 976. Both call sites need to forward `primary` and `themes` into the returned hierarchical result. The existing `path` (wings/rooms) stays unchanged — wing/room *routing* via closet indices is fine; that's not an answer-ranking decision.

### 4. Update CLI search — `mempalace/cli.py:595`

The `search` import in `cmd_search` writes to stdout. Print `primary` first as the headline answer block. Print `themes` below in a separate section labeled "Related (LLM-tagged themes — review before drilling in)". Use the existing visual style for consistency.

### 5. Tests — `tests/test_search_*.py` and `tests/benchmarks/test_palace_boost.py`

Existing tests assert on `results` and on `closet_boost` / `matched_via` fields. They need to:
- Rename assertions on `results` → `primary`.
- Drop assertions on `closet_boost`, `effective_distance`, `matched_via`.
- Add assertions on the new `themes` shape.

New tests required:
- `test_closet_content_does_not_rank_primary` — manually insert a closet row that would have ranked file A high. Verify that file A's drawer rank in `primary` is unchanged by the presence of the closet.
- `test_themes_dedup_per_source_file` — multiple closet rows for the same source produce one entry in `themes`.
- `test_themes_includes_taxonomy_rows` — a taxonomy:v2 row that matches the query appears in `themes` with the right `generated_by`.
- `test_themes_empty_when_no_closets` — a palace with zero closet rows returns `themes: []` (not an error).

Benchmark tests under `tests/benchmarks/test_palace_boost.py` rely on the boost mechanism; they need rewriting to either (a) test the new architecture's behavior or (b) be deleted as no-longer-relevant. Recommend keeping them but reframing as "verbatim cosine ranking is stable" tests.

## Backward compatibility — the key decision

The big choice: keep `results` as a field name vs rename to `primary`.

**Recommendation: rename `results` → `primary` and add `themes` as a new key.**

Why: the behavior is changing materially (rankings will differ). A renamed key forces callers to re-read the contract. Backward compat *without* behavior change is fine. Backward compat *with* silent behavior change is dangerous — callers that hardcode `results` would silently get different rankings than they used to.

Grep the codebase for `["results"]` and `.results` against searcher return values. Update every site. The CLAUDE.md "Backwards-compatibility hacks" rule says no aliasing — just change the contract.

## Verification gate

Before declaring done:

1. **The 4 validation queries.** Re-run from the memory file's documented set. Capture top-5 for `primary` and `themes` per query. Compare against the documented numbers in the memory file. Specifically check that `chroma.py` and `dialect.py` (which we know hit via drawer cosine) still surface in `primary`, and that `mempalace_client.py` and `rpg_retriever.py` (which we know have good taxonomy rows) surface in `themes`.

2. **Test suite.** All non-pre-existing tests pass. The known pre-existing failures (issue #8: 6 tests in `test_corpus_origin_integration.py`, `test_hnsw_capacity.py`, `test_save_hook_mines.py`) stay failing — don't fix them in this PR.

3. **Hand-check.** Run three real queries through `tool_search_hierarchical`. The `themes` results should look like recognizable conceptual neighbors of the query. The `primary` results should look like literal-content matches. Verify the two lists feel like different kinds of answers.

## Open questions to resolve during implementation

1. **Should `themes` include a similarity score visible to the user?** Probably yes — `themes` is speculative; showing the score helps calibrate. The MCP schema should document that `themes` scores aren't comparable to `primary` scores (different metrics).

2. **Should taxonomy rows be discounted vs prose rows in `themes` ranking?** Open. The taxonomy rows are generic; prose rows are file-specific. If a prose row matches at the same cosine, it's probably more useful. **Suggestion:** add +0.05 to taxonomy-tagged closet distances before sorting `themes`. Make it tunable via `MEMPALACE_THEME_TAXONOMY_PENALTY`. Default 0.05.

3. **`themes` size — match `n_results` or separate parameter?** Recommend a separate optional `n_themes` parameter defaulting to `n_results`. The MCP tool exposes both.

4. **`fallback=True` semantics in `tool_search_hierarchical`.** Currently signals "wing indices empty, fell back to flat." That stays. But the fallback path also needs to return `themes`. Make sure `_shortcut_flat_search` is updated.

## Out of scope for this PR

- Closet generation (closet_llm, closet_taxonomy). Themes consumes existing rows.
- `recursive_indexer.py`. Wing/room indices still drive hierarchical *routing* (not answer ranking); that's fine.
- New MCP tools or CLI commands. This is purely a shape/behavior change to existing entry points.
- Drawer hydration for `themes` (the "show me a verbatim snippet alongside the theme" feature). Worth doing later but not necessary for the two-tier shape to work.
- Cleaning up the existing taxonomy:v2 rows. They're good `themes` material as-is.

## Critical files (single-pane-of-glass list)

- `mempalace/searcher.py` — primary surgery
- `mempalace/mcp_server.py` — propagate the new shape
- `mempalace/cli.py` — print themes block
- `tests/test_*.py` — update assertions
- `tests/benchmarks/test_palace_boost.py` — reframe or delete
- `~/src/dgx/closets-as-themes.md` — rationale (do not modify)
