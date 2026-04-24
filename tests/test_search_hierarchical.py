"""End-to-end tests for ``mempalace_search_hierarchical``.

Exercises the MCP tool handler on a palace seeded with drawers + closets
across multiple wings/rooms, after running the recursive indexer to
populate the room and wing indices.

The tool's contract:
  * ``max_depth=0``  → prune at wing level only, return path.wings
  * ``max_depth=1``  → prune wing + room, return path.rooms
  * ``max_depth=2``  → full descent; `results` holds drawer hits
  * falls back to ``search_within`` when indices are empty or an
    explicit wing/room filter is supplied (``fallback: true``)
"""

import pytest

from mempalace import mcp_server, palace, recursive_indexer


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def indexed_palace(palace_path):
    """Palace with drawers + closets across two wings, indices pre-built."""
    drawers = palace.get_collection(palace_path)
    closets = palace.get_closets_collection(palace_path)

    drawers.add(
        ids=["d_auth", "d_db", "d_react", "d_plan"],
        documents=[
            "JWT tokens expire in 24h. Refresh tokens live in HttpOnly cookies.",
            "Alembic migrations for PostgreSQL 15 with pgbouncer pooling.",
            "TanStack Query manages server state. Centralized fetch wrapper.",
            "Sprint planning notes: evaluate passkeys for auth by Q3.",
        ],
        metadatas=[
            {"wing": "project", "room": "backend", "source_file": "auth.py"},
            {"wing": "project", "room": "backend", "source_file": "db.py"},
            {"wing": "project", "room": "frontend", "source_file": "App.tsx"},
            {"wing": "notes", "room": "planning", "source_file": "sprint.md"},
        ],
    )
    closets.add(
        ids=["c_auth", "c_db", "c_react", "c_plan"],
        documents=[
            "jwt auth|JWT;tokens|→d_auth\n"
            "session management|JWT|→d_auth\n"
            "refresh token|JWT|→d_auth",
            "database migrations|Alembic|→d_db\n"
            "postgresql 15|Postgres|→d_db\n"
            "pgbouncer pooling|Postgres|→d_db",
            "tanstack query|React|→d_react\n"
            "fetch wrapper|React|→d_react\n"
            "server state|React|→d_react",
            "passkeys migration|Auth|→d_plan\n"
            "chromadb alternatives|Vector|→d_plan",
        ],
        metadatas=[
            {"wing": "project", "room": "backend", "source_file": "auth.py",
             "top_entities": "JWT;tokens"},
            {"wing": "project", "room": "backend", "source_file": "db.py",
             "top_entities": "Alembic;Postgres"},
            {"wing": "project", "room": "frontend", "source_file": "App.tsx",
             "top_entities": "React"},
            {"wing": "notes", "room": "planning", "source_file": "sprint.md",
             "top_entities": "Auth;Vector"},
        ],
    )

    recursive_indexer.rebuild_all(palace_path)
    return palace_path


@pytest.fixture
def empty_palace(tmp_dir):
    """Palace with drawers but no indices built — exercises the fallback path.

    Uses ``tmp_dir`` directly rather than ``palace_path`` so it never
    shares a ChromaDB directory with the ``indexed_palace`` fixture
    (which is pulled in by the autouse config patch).
    """
    import os as _os

    p = _os.path.join(tmp_dir, "empty_palace")
    _os.makedirs(p, exist_ok=True)
    drawers = palace.get_collection(p)
    drawers.add(
        ids=["d_only"],
        documents=["something about authentication"],
        metadatas=[{"wing": "project", "room": "backend", "source_file": "x.py"}],
    )
    return p


@pytest.fixture(autouse=True)
def _point_config_at_palace(indexed_palace, monkeypatch):
    """Make the MCP tool resolve the default palace to our fixture."""
    # The tool goes through _resolve_palace_arg(None) → _config.palace_path.
    # We override both so the handler lands on the indexed fixture.
    monkeypatch.setattr(mcp_server._config, "_file_config", {"palace_path": indexed_palace})


# ── max_depth variations ────────────────────────────────────────────────


class TestMaxDepthZero:
    def test_returns_wings_only(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical("authentication", max_depth=0)
        assert out["max_depth"] == 0
        assert out["fallback"] is False
        assert "results" not in out  # no drawer sweep at depth 0
        assert out["path"]["wings"]
        for w in out["path"]["wings"]:
            assert "wing" in w
            assert "distance" in w
            assert "similarity" in w

    def test_wing_hits_are_deduped(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical("auth", max_depth=0)
        wings = [w["wing"] for w in out["path"]["wings"]]
        assert len(wings) == len(set(wings))


class TestMaxDepthOne:
    def test_returns_wings_and_rooms(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical("authentication", max_depth=1)
        assert out["max_depth"] == 1
        assert "results" not in out  # still no drawers
        assert out["path"]["wings"]
        assert out["path"]["rooms"]
        for r in out["path"]["rooms"]:
            # Every room returned belongs to one of the selected wings.
            assert r["wing"] in {w["wing"] for w in out["path"]["wings"]}

    def test_rooms_include_backend_for_auth_query(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical("jwt auth tokens", max_depth=1)
        room_names = {r["room"] for r in out["path"]["rooms"]}
        assert "backend" in room_names


class TestMaxDepthTwo:
    def test_returns_drawer_hits(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical("jwt auth tokens", max_depth=2)
        assert out["max_depth"] == 2
        assert out["fallback"] is False
        assert "results" in out
        # We seeded an auth drawer; the top-level result should surface it.
        ids = [h["drawer_id"] for h in out["results"]]
        assert "d_auth" in ids

    def test_default_max_depth_is_two(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical("jwt auth tokens")
        assert out["max_depth"] == 2
        assert "results" in out

    def test_path_is_always_present(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical("jwt auth tokens", max_depth=2)
        assert "wings" in out["path"]
        assert "rooms" in out["path"]
        assert out["path"]["wings"]
        assert out["path"]["rooms"]

    def test_budget_caps_results(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical(
            "auth react sprint migration",
            max_depth=2,
            limit=10,
            budget=1,
        )
        assert len(out["results"]) <= 1


# ── Filters ─────────────────────────────────────────────────────────────


class TestExplicitFilter:
    def test_wing_filter_triggers_fallback(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical(
            "auth",
            wing_filter="project",
            max_depth=2,
        )
        # An explicit scope always short-circuits the hierarchical pruning.
        assert out["fallback"] is True
        assert "results" in out
        for h in out["results"]:
            assert h["wing"] == "project"

    def test_room_filter_triggers_fallback(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical(
            "auth",
            room_filter="backend",
        )
        assert out["fallback"] is True
        for h in out["results"]:
            assert h["room"] == "backend"

    def test_unknown_wing_filter_returns_no_results(self, indexed_palace):
        out = mcp_server.tool_search_hierarchical(
            "auth",
            wing_filter="does_not_exist",
        )
        assert out["fallback"] is True
        assert out["results"] == []


# ── Fallback behaviour ──────────────────────────────────────────────────


class TestFallback:
    def test_falls_back_when_indices_empty(self, empty_palace):
        # Pass the palace path explicitly to avoid the autouse fixture that
        # pins the default palace to the indexed one.
        out = mcp_server.tool_search_hierarchical(
            "authentication", palace=empty_palace
        )
        assert out["fallback"] is True
        assert "fallback_reason" in out
        # The flat search still hits the only drawer we seeded.
        assert out["results"]
        assert out["results"][0]["drawer_id"] == "d_only"


# ── Sanity: registered tool dispatches correctly ─────────────────────────


class TestRegistration:
    def test_tool_registered_with_expected_name(self):
        assert "mempalace_search_hierarchical" in mcp_server.TOOLS
        entry = mcp_server.TOOLS["mempalace_search_hierarchical"]
        assert callable(entry["handler"])
        assert "query" in entry["input_schema"]["properties"]

    def test_registered_handler_is_tool_search_hierarchical(self):
        entry = mcp_server.TOOLS["mempalace_search_hierarchical"]
        assert entry["handler"] is mcp_server.tool_search_hierarchical
