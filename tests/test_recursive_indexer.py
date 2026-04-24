"""Tests for ``mempalace.recursive_indexer``.

The indexer rebuilds room + wing index documents from leaf closets by
arithmetic projection. It must be deterministic, idempotent, restartable,
and drift-free (rebuilding from leaves always yields the same document).
"""

import pytest

from mempalace import palace, recursive_indexer
from mempalace.recursive_indexer import (
    _room_index_id,
    _wing_index_id,
    build_room_index,
    build_wing_index,
    rebuild_all,
    rebuild_dirty,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def seeded_closets(palace_path):
    """Palace with drawers + closets spanning three rooms across two wings.

    project/backend:  two files, four closet lines each (auth + database)
    project/frontend: one file, three closet lines (react + fetch)
    notes/planning:   one file, two closet lines (sprint)

    The closet lines follow the AAAK pointer format
    ``topic|entities|→drawer_id``.
    """
    drawers = palace.get_collection(palace_path)
    closets = palace.get_closets_collection(palace_path)

    drawers.add(
        ids=["d_auth_1", "d_auth_2", "d_db_1", "d_react_1", "d_plan_1"],
        documents=["doc-auth-1", "doc-auth-2", "doc-db-1", "doc-react-1", "doc-plan-1"],
        metadatas=[
            {"wing": "project", "room": "backend", "source_file": "auth.py"},
            {"wing": "project", "room": "backend", "source_file": "auth.py"},
            {"wing": "project", "room": "backend", "source_file": "db.py"},
            {"wing": "project", "room": "frontend", "source_file": "App.tsx"},
            {"wing": "notes", "room": "planning", "source_file": "sprint.md"},
        ],
    )

    closets.add(
        ids=["c_auth_01", "c_auth_02", "c_db_01", "c_react_01", "c_plan_01"],
        documents=[
            # project/backend (auth)
            "jwt auth|Alice;JWT|→d_auth_1\n"
            "session management|Bob|→d_auth_2\n"
            "refresh token|Alice|→d_auth_1",
            # project/backend (auth second closet, overlapping topics)
            "jwt auth|Alice|→d_auth_1\n"
            "token expiry|Alice|→d_auth_2",
            # project/backend (db)
            "database migrations|Alembic|→d_db_1\n"
            "postgresql 15|Alembic;Postgres|→d_db_1\n"
            "pgbouncer|Postgres|→d_db_1",
            # project/frontend
            "tanstack query|React|→d_react_1\n"
            "fetch wrapper|React|→d_react_1\n"
            "server state|React|→d_react_1",
            # notes/planning
            "passkeys migration|Auth|→d_plan_1\n"
            "chromadb alternatives|Vector|→d_plan_1",
        ],
        metadatas=[
            {
                "wing": "project",
                "room": "backend",
                "source_file": "auth.py",
                "top_entities": "Alice;JWT;Bob",
            },
            {
                "wing": "project",
                "room": "backend",
                "source_file": "auth.py",
                "top_entities": "Alice",
            },
            {
                "wing": "project",
                "room": "backend",
                "source_file": "db.py",
                "top_entities": "Alembic;Postgres",
            },
            {
                "wing": "project",
                "room": "frontend",
                "source_file": "App.tsx",
                "top_entities": "React",
            },
            {
                "wing": "notes",
                "room": "planning",
                "source_file": "sprint.md",
                "top_entities": "Auth;Vector",
            },
        ],
    )
    return palace_path


# ── build_room_index ─────────────────────────────────────────────────────


class TestBuildRoomIndex:
    def test_writes_document_and_reports_counts(self, seeded_closets):
        result = build_room_index(seeded_closets, "project", "backend")
        assert result["status"] == "written"
        assert result["closet_count"] == 3
        # 3 + 2 + 3 = 8 non-empty lines across the three backend closets.
        assert result["line_count"] == 8
        # "jwt auth" appears twice, rest unique → 7 distinct topics.
        assert result["kept_lines"] == 7
        assert result["drawer_count"] >= 3

    def test_document_contains_projected_lines(self, seeded_closets):
        build_room_index(seeded_closets, "project", "backend")
        col = palace.get_room_indices_collection(seeded_closets)
        got = col.get(ids=[_room_index_id("project", "backend")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        assert len(docs) == 1
        text = docs[0]
        # Every kept line has the AAAK pointer format.
        for line in text.splitlines():
            assert "→" in line or "->" in line
        # Duplicate topic "jwt auth" collapses to one line representative.
        assert text.count("jwt auth") == 1

    def test_metadata_captures_entities(self, seeded_closets):
        build_room_index(seeded_closets, "project", "backend")
        col = palace.get_room_indices_collection(seeded_closets)
        got = col.get(
            ids=[_room_index_id("project", "backend")], include=["metadatas"]
        )
        metas = got.metadatas if hasattr(got, "metadatas") else got["metadatas"]
        meta = metas[0]
        assert meta["wing"] == "project"
        assert meta["room"] == "backend"
        assert meta["level"] == "room"
        assert meta["aaak_version"] == recursive_indexer.AAAK_VERSION
        assert "Alice" in meta["top_entities"]  # most-frequent across closets

    def test_empty_room_returns_empty_and_drops_doc(self, seeded_closets):
        # First build a doc, then delete every closet in that room, then rebuild.
        build_room_index(seeded_closets, "project", "frontend")
        col = palace.get_closets_collection(seeded_closets)
        col.delete(where={"$and": [{"wing": "project"}, {"room": "frontend"}]})

        result = build_room_index(seeded_closets, "project", "frontend")
        assert result["status"] == "empty"
        assert result["closet_count"] == 0

        idx_col = palace.get_room_indices_collection(seeded_closets)
        got = idx_col.get(ids=[_room_index_id("project", "frontend")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        assert docs == []

    def test_idempotent(self, seeded_closets):
        a = build_room_index(seeded_closets, "project", "backend")
        # Rebuild the same room — only built_at may differ; everything else must match.
        b = build_room_index(seeded_closets, "project", "backend")
        for key in ("closet_count", "line_count", "kept_lines", "drawer_count", "doc_id", "status"):
            assert a[key] == b[key]

        col = palace.get_room_indices_collection(seeded_closets)
        got = col.get(ids=[_room_index_id("project", "backend")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        text1 = docs[0]
        build_room_index(seeded_closets, "project", "backend")
        got = col.get(ids=[_room_index_id("project", "backend")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        text2 = docs[0]
        # Pure projection — byte-for-byte identical across rebuilds.
        assert text1 == text2

    def test_top_n_caps_output(self, seeded_closets):
        result = build_room_index(seeded_closets, "project", "backend", top_n=3)
        assert result["kept_lines"] == 3


# ── build_wing_index ─────────────────────────────────────────────────────


class TestBuildWingIndex:
    def test_requires_room_indices_built_first(self, seeded_closets):
        # No room indices written yet → wing aggregation has no input.
        result = build_wing_index(seeded_closets, "project")
        assert result["status"] == "empty"
        assert result["room_count"] == 0

    def test_aggregates_across_rooms(self, seeded_closets):
        build_room_index(seeded_closets, "project", "backend")
        build_room_index(seeded_closets, "project", "frontend")

        result = build_wing_index(seeded_closets, "project")
        assert result["status"] == "written"
        assert result["room_count"] == 2
        assert set(result["rooms"]) == {"backend", "frontend"}
        assert result["line_count"] > 0

        col = palace.get_wing_indices_collection(seeded_closets)
        got = col.get(ids=[_wing_index_id("project")], include=["documents", "metadatas"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        metas = got.metadatas if hasattr(got, "metadatas") else got["metadatas"]
        assert len(docs) == 1
        # Some line from each room should survive to the wing index.
        text = docs[0]
        assert any("jwt auth" in l for l in text.splitlines())
        assert any("tanstack" in l.lower() for l in text.splitlines())
        assert metas[0]["level"] == "wing"

    def test_wing_deleted_when_empty(self, seeded_closets):
        build_room_index(seeded_closets, "project", "backend")
        build_wing_index(seeded_closets, "project")

        # Simulate the wing losing every room.
        col = palace.get_room_indices_collection(seeded_closets)
        col.delete(where={"wing": "project"})
        result = build_wing_index(seeded_closets, "project")
        assert result["status"] == "empty"

        wc = palace.get_wing_indices_collection(seeded_closets)
        got = wc.get(ids=[_wing_index_id("project")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        assert docs == []

    def test_wing_entities_aggregated(self, seeded_closets):
        build_room_index(seeded_closets, "project", "backend")
        build_room_index(seeded_closets, "project", "frontend")
        build_wing_index(seeded_closets, "project")

        col = palace.get_wing_indices_collection(seeded_closets)
        got = col.get(ids=[_wing_index_id("project")], include=["metadatas"])
        metas = got.metadatas if hasattr(got, "metadatas") else got["metadatas"]
        ents = metas[0]["top_entities"]
        # At least one entity from each room makes it in.
        assert "Alice" in ents
        assert "React" in ents


# ── Orchestration: rebuild_dirty / rebuild_all ───────────────────────────


class TestRebuildDirty:
    def test_processes_all_flagged_rooms_and_wings(self, seeded_closets):
        palace.mark_room_dirty(seeded_closets, "project", "backend")
        palace.mark_room_dirty(seeded_closets, "project", "frontend")
        palace.mark_room_dirty(seeded_closets, "notes", "planning")

        report = rebuild_dirty(seeded_closets)
        assert report["room_count"] == 3
        # mark_room_dirty cascades into wing dirty flags → both wings rebuilt.
        assert report["wing_count"] == 2
        for r in report["rooms"]:
            assert r["status"] == "written"
        for w in report["wings"]:
            assert w["status"] == "written"

    def test_clears_flags_after_success(self, seeded_closets):
        palace.mark_room_dirty(seeded_closets, "project", "backend")
        rebuild_dirty(seeded_closets)
        assert list(palace.iter_dirty_rooms(seeded_closets)) == []
        assert list(palace.iter_dirty_wings(seeded_closets)) == []

    def test_empty_queue_is_noop(self, seeded_closets):
        report = rebuild_dirty(seeded_closets)
        assert report == {
            "rooms": [],
            "wings": [],
            "room_count": 0,
            "wing_count": 0,
            "elapsed_seconds": report["elapsed_seconds"],
        }

    def test_repeat_run_is_idempotent(self, seeded_closets):
        palace.mark_room_dirty(seeded_closets, "project", "backend")
        rebuild_dirty(seeded_closets)
        col = palace.get_room_indices_collection(seeded_closets)
        got = col.get(ids=[_room_index_id("project", "backend")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        text1 = docs[0]

        # Re-mark and re-run; the projected text is pure, so must be identical.
        palace.mark_room_dirty(seeded_closets, "project", "backend")
        rebuild_dirty(seeded_closets)
        got = col.get(ids=[_room_index_id("project", "backend")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        text2 = docs[0]
        assert text1 == text2


class TestRebuildAll:
    def test_walks_every_wing_room_pair(self, seeded_closets):
        report = rebuild_all(seeded_closets)
        # 3 wings/room pairs: (project, backend), (project, frontend), (notes, planning)
        assert report["room_count"] == 3
        assert report["wing_count"] == 2
        assert set(report["scope"].keys()) == {"project", "notes"}

    def test_produces_same_state_as_mark_all_dirty(self, seeded_closets):
        # rebuild_all should be functionally equivalent to flagging every
        # (wing, room) dirty and draining the queue.
        rebuild_all(seeded_closets)
        col = palace.get_room_indices_collection(seeded_closets)
        got = col.get(ids=[_room_index_id("project", "backend")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        text_all = docs[0]

        # Clear this specific room index; rebuild via dirty queue; compare bytes.
        col.delete(ids=[_room_index_id("project", "backend")])
        palace.mark_room_dirty(seeded_closets, "project", "backend")
        rebuild_dirty(seeded_closets)
        got = col.get(ids=[_room_index_id("project", "backend")], include=["documents"])
        docs = got.documents if hasattr(got, "documents") else got["documents"]
        text_dirty = docs[0]

        assert text_all == text_dirty

    def test_clears_any_pre_existing_dirty_flags(self, seeded_closets):
        palace.mark_room_dirty(seeded_closets, "project", "backend")
        palace.mark_wing_dirty(seeded_closets, "project")
        rebuild_all(seeded_closets)
        assert list(palace.iter_dirty_rooms(seeded_closets)) == []
        assert list(palace.iter_dirty_wings(seeded_closets)) == []


# ── Pure-function sanity ─────────────────────────────────────────────────


class TestIDHelpers:
    def test_room_id_is_deterministic(self):
        assert _room_index_id("w", "r") == _room_index_id("w", "r")

    def test_room_id_differs_for_different_inputs(self):
        assert _room_index_id("w1", "r") != _room_index_id("w2", "r")
        assert _room_index_id("w", "r1") != _room_index_id("w", "r2")

    def test_room_and_wing_ids_are_distinct(self):
        # No chance of a wing-index doc and a room-index doc colliding on ID
        # even if someone uses the same identifier as both a wing and a room.
        assert _room_index_id("x", "x") != _wing_index_id("x")

    def test_ids_are_prefixed(self):
        assert _room_index_id("w", "r").startswith("room_idx_")
        assert _wing_index_id("w").startswith("wing_idx_")
