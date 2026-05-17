"""Two-tier retrieval tests: primary ranking is closet-immune; themes
surface the LLM-tagged conceptual layer separately.

These tests pin the load-bearing invariants of the two-tier shape:

* ``primary`` rank is determined by drawer cosine + BM25 alone. Bad
  closets cannot pull a wrong drawer to the top, and absent closets
  do not change ranks.
* ``themes`` is the LLM-judgment layer. It exposes closet rows that
  semantically match the query but never feeds back into ``primary``.
"""

from mempalace.palace import (
    get_closets_collection,
    get_collection,
    upsert_closet_lines,
)
from mempalace.searcher import search_memories


def _seed_drawers(palace_path):
    """Insert 4 short drawers with deterministic content."""
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=["D1", "D2", "D3", "D4"],
        documents=[
            "We switched the auth service to use JWT tokens with a 24h expiry.",
            "Database migration to PostgreSQL 15 completed last Tuesday.",
            "The frontend team is debating whether to adopt TanStack Query.",
            "Kafka consumer rebalance timeout set to 45 seconds after incident.",
        ],
        metadatas=[
            {"wing": "backend", "room": "auth", "source_file": "fixture_D1.md"},
            {"wing": "backend", "room": "db", "source_file": "fixture_D2.md"},
            {"wing": "frontend", "room": "state", "source_file": "fixture_D3.md"},
            {"wing": "backend", "room": "queue", "source_file": "fixture_D4.md"},
        ],
    )


def _seed_strong_closet_for(palace_path, drawer_id, source_file, topics):
    """Insert a closet whose content strongly overlaps the query keywords."""
    col = get_closets_collection(palace_path)
    lines = [f"{t}||→{drawer_id}" for t in topics]
    upsert_closet_lines(
        col,
        closet_id_base=f"closet_{drawer_id}",
        lines=lines,
        metadata={
            "wing": "backend",
            "room": "auth",
            "source_file": source_file,
            "generated_by": "test",
        },
    )


# ── primary is closet-immune ──────────────────────────────────────────────


class TestPrimaryIsClosetImmune:
    def test_no_closets_returns_pure_drawer_ranking(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        result = search_memories("Kafka rebalance timeout", palace, n_results=3)
        ids = [h["source_file"] for h in result["primary"]]
        assert ids, "primary should return drawer hits"
        assert "fixture_D4.md" in ids, "Kafka drawer must surface in primary"

    def test_misleading_closet_does_not_reshuffle_primary(self, tmp_path):
        """A closet pointing at a wrong source cannot pull that source's
        drawers into primary or push direct matches down. Primary's rank
        depends only on the drawers themselves."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # Baseline ranking before any closets exist.
        baseline = search_memories("Kafka consumer rebalance timeout", palace, n_results=5)
        baseline_ids = [h["drawer_id"] for h in baseline["primary"]]

        # Now seed a closet that claims D3 is about Kafka.
        _seed_strong_closet_for(
            palace,
            drawer_id="D3",
            source_file="fixture_D3.md",
            topics=["Kafka queue tuning", "consumer rebalance config"],
        )
        after = search_memories("Kafka consumer rebalance timeout", palace, n_results=5)
        after_ids = [h["drawer_id"] for h in after["primary"]]
        assert after_ids == baseline_ids, (
            "Primary ranking must be unchanged by closet content. "
            f"baseline={baseline_ids} after={after_ids}"
        )

    def test_primary_hits_have_no_closet_fields(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        _seed_strong_closet_for(
            palace,
            drawer_id="D1",
            source_file="fixture_D1.md",
            topics=["JWT auth tokens", "session expiry"],
        )
        result = search_memories("JWT authentication", palace, n_results=3)
        for h in result["primary"]:
            assert "closet_boost" not in h
            assert "matched_via" not in h
            assert "effective_distance" not in h


# ── themes layer ────────────────────────────────────────────────────────


class TestThemesLayer:
    def test_themes_present_when_closets_exist(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        _seed_strong_closet_for(
            palace,
            drawer_id="D1",
            source_file="fixture_D1.md",
            topics=["JWT auth tokens", "24h expiry", "authentication"],
        )
        result = search_memories("JWT authentication", palace, n_results=3)
        assert result["themes"], "themes block should be populated"
        top = result["themes"][0]
        assert top["source_file"] == "fixture_D1.md"
        assert "closet_text" in top
        assert "similarity" in top
        assert "generated_by" in top
        assert "drawer_ids" in top

    def test_themes_empty_when_no_closets(self, tmp_path):
        """A palace with zero closet rows returns ``themes: []`` (not an error).

        Required by docs/design/two-tier-retrieval.md."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        result = search_memories("TanStack Query", palace, n_results=2)
        assert result["primary"], "primary should still have hits"
        assert result["themes"] == []

    def test_themes_dedup_per_source_file(self, tmp_path):
        """Multiple closet rows for the same source produce one entry in themes.

        Required by docs/design/two-tier-retrieval.md."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # Seed three separate closet rows for the same source_file.
        col = get_closets_collection(palace)
        col.upsert(
            ids=["closet_dup_01", "closet_dup_02", "closet_dup_03"],
            documents=[
                "JWT authentication tokens|;|→D1",
                "session expiry policy|;|→D1",
                "auth service rewrite|;|→D1",
            ],
            metadatas=[
                {
                    "wing": "backend",
                    "room": "auth",
                    "source_file": "fixture_D1.md",
                    "generated_by": "llm:test",
                },
                {
                    "wing": "backend",
                    "room": "auth",
                    "source_file": "fixture_D1.md",
                    "generated_by": "llm:test",
                },
                {
                    "wing": "backend",
                    "room": "auth",
                    "source_file": "fixture_D1.md",
                    "generated_by": "llm:test",
                },
            ],
        )
        result = search_memories("JWT authentication", palace, n_results=5)
        same_source = [t for t in result["themes"] if t["source_file"] == "fixture_D1.md"]
        assert len(same_source) == 1, "themes must dedup per source_file — keep one row per file"

    def test_themes_includes_taxonomy_rows(self, tmp_path):
        """A taxonomy:v2 row that matches the query appears in themes with the
        right ``generated_by``.

        Required by docs/design/two-tier-retrieval.md."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        col = get_closets_collection(palace)
        col.upsert(
            ids=["closet_taxonomy_D2"],
            documents=[
                "the file is a wrapper that operates as boundary and external service|;|→D2"
            ],
            metadatas=[
                {
                    "wing": "backend",
                    "room": "db",
                    "source_file": "fixture_D2.md",
                    "generated_by": "taxonomy:v2",
                }
            ],
        )
        result = search_memories(
            "black-box integration of external service",
            palace,
            n_results=5,
        )
        tax_hits = [t for t in result["themes"] if t["generated_by"] == "taxonomy:v2"]
        assert tax_hits, "taxonomy:v2 rows must surface in themes"


# ── invariant: closet content does not affect primary ranking ────────────


class TestClosetContentDoesNotRankPrimary:
    def test_closet_does_not_promote_unrelated_drawer(self, tmp_path):
        """Manually insert a closet row that, under the old boost mechanism,
        would have ranked file A high. Verify that file A's drawer rank in
        ``primary`` is unchanged by the presence of the closet.

        Required by docs/design/two-tier-retrieval.md."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)

        query = "Postgres migration schedule"
        # Baseline: D2 (Postgres migration drawer) should rank above D3
        # (TanStack Query drawer).
        baseline = search_memories(query, palace, n_results=4)
        baseline_ids = [h["drawer_id"] for h in baseline["primary"]]
        assert "D2" in baseline_ids

        # Now inject a strong closet that points at D3 — under the old
        # rank-based boost (which capped at distance 1.5 and gave 0.40
        # off rank 0), this would have promoted D3 above D2. Under the
        # two-tier design, primary is closet-immune.
        _seed_strong_closet_for(
            palace,
            drawer_id="D3",
            source_file="fixture_D3.md",
            topics=[
                "Postgres migration",
                "database upgrade",
                "schema migration plan",
                "PostgreSQL 15 cutover",
            ],
        )
        after = search_memories(query, palace, n_results=4)
        after_ids = [h["drawer_id"] for h in after["primary"]]
        assert after_ids == baseline_ids, (
            "Primary ordering must not change when a closet is added. "
            f"baseline={baseline_ids} after={after_ids}"
        )
