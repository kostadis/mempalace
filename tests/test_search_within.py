"""Tests for the ``search_within`` scoped-search primitive.

``search_within`` is the leaf step of hierarchical descent: it accepts
multi-valued wing/room scopes (``wing IN {a, b, c}``) and an optional
drawer-ID post-filter, returning the same hybrid-retrieval hit shape as
``search_memories``. The older ``search_memories`` is now a thin shim
over it — these tests cover both paths.
"""

from mempalace.searcher import search_memories, search_within


class TestMultiWingScope:
    def test_single_wing_filter(self, palace_path, seeded_collection):
        result = search_within(
            "authentication",
            palace_path,
            wing_filters=["project"],
        )
        assert "error" not in result
        assert all(h["wing"] == "project" for h in result["primary"])

    def test_multi_wing_filter_includes_both(self, palace_path, seeded_collection):
        # The palace has two wings: "project" and "notes". A {a, b} scope
        # must return hits from either (not neither, not only one).
        result = search_within(
            "authentication planning",
            palace_path,
            wing_filters=["project", "notes"],
            n_results=10,
        )
        wings_seen = {h["wing"] for h in result["primary"]}
        assert "project" in wings_seen
        assert "notes" in wings_seen

    def test_empty_wing_filters_matches_nothing_restricting(self, palace_path, seeded_collection):
        # Empty list ≡ no filter (normalized to None inside search_within).
        unfiltered = search_within("auth", palace_path)
        scoped = search_within("auth", palace_path, wing_filters=[])
        assert len(unfiltered["primary"]) == len(scoped["primary"])

    def test_unknown_wing_returns_empty(self, palace_path, seeded_collection):
        result = search_within(
            "anything",
            palace_path,
            wing_filters=["nonexistent-wing"],
        )
        assert result["primary"] == []


class TestMultiRoomScope:
    def test_single_room_filter(self, palace_path, seeded_collection):
        result = search_within(
            "authentication",
            palace_path,
            room_filters=["backend"],
        )
        assert all(h["room"] == "backend" for h in result["primary"])

    def test_multi_room_filter(self, palace_path, seeded_collection):
        result = search_within(
            "auth fetch",
            palace_path,
            room_filters=["backend", "frontend"],
            n_results=10,
        )
        rooms_seen = {h["room"] for h in result["primary"]}
        assert rooms_seen.issubset({"backend", "frontend"})
        assert len(rooms_seen) >= 1

    def test_wing_and_room_filters_combine(self, palace_path, seeded_collection):
        result = search_within(
            "anything",
            palace_path,
            wing_filters=["project"],
            room_filters=["backend"],
            n_results=10,
        )
        for h in result["primary"]:
            assert h["wing"] == "project"
            assert h["room"] == "backend"


class TestIdsPostFilter:
    def test_ids_restrict_to_subset(self, palace_path, seeded_collection):
        # Restrict to the two backend drawers only.
        allowed = {"drawer_proj_backend_aaa", "drawer_proj_backend_bbb"}
        result = search_within(
            "auth database",
            palace_path,
            ids=list(allowed),
            n_results=10,
        )
        for h in result["primary"]:
            assert h["drawer_id"] in allowed

    def test_ids_empty_list_is_noop(self, palace_path, seeded_collection):
        # ids=[] (falsy) should behave like ids=None — no restriction.
        unfiltered = search_within("auth", palace_path, n_results=10)
        empty_ids = search_within("auth", palace_path, ids=[], n_results=10)
        assert len(empty_ids["primary"]) == len(unfiltered["primary"])

    def test_ids_matching_nothing_returns_empty(self, palace_path, seeded_collection):
        result = search_within(
            "auth",
            palace_path,
            ids=["nonexistent_drawer_xyz"],
            n_results=10,
        )
        assert result["primary"] == []


class TestResultShape:
    def test_hit_has_drawer_id(self, palace_path, seeded_collection):
        result = search_within("auth", palace_path)
        assert result["primary"]
        for h in result["primary"]:
            assert "drawer_id" in h
            assert h["drawer_id"].startswith("drawer_")

    def test_filters_block_reports_scopes(self, palace_path, seeded_collection):
        result = search_within(
            "auth",
            palace_path,
            wing_filters=["project"],
            room_filters=["backend"],
        )
        assert result["filters"]["wing_filters"] == ["project"]
        assert result["filters"]["room_filters"] == ["backend"]
        assert result["filters"]["ids"] is None

    def test_filters_block_none_when_unscoped(self, palace_path, seeded_collection):
        result = search_within("auth", palace_path)
        assert result["filters"]["wing_filters"] is None
        assert result["filters"]["room_filters"] is None
        assert result["filters"]["ids"] is None


class TestMissingPalace:
    def test_missing_palace_reports_error(self, tmp_dir):
        # No palace was created at this path.
        result = search_within("anything", tmp_dir + "/nope")
        assert "error" in result

    def test_missing_palace_via_search_memories(self, tmp_dir):
        result = search_memories("anything", tmp_dir + "/nope")
        assert "error" in result


class TestBackwardsCompatibility:
    def test_search_memories_preserves_old_filters_shape(self, palace_path, seeded_collection):
        result = search_memories("auth", palace_path, wing="project", room="backend")
        # Legacy single-value shape survives — scripts that peek at result["filters"]
        # don't have to learn the new key names.
        assert result["filters"] == {"wing": "project", "room": "backend"}

    def test_search_memories_and_within_return_same_hits(self, palace_path, seeded_collection):
        via_memories = search_memories("auth", palace_path, wing="project", n_results=5)
        via_within = search_within(
            "auth",
            palace_path,
            wing_filters=["project"],
            n_results=5,
        )
        mem_ids = [h["drawer_id"] for h in via_memories["primary"]]
        within_ids = [h["drawer_id"] for h in via_within["primary"]]
        assert mem_ids == within_ids
