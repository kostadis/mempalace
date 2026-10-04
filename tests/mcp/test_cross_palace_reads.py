"""#45: every MCP read tool accepts ``palace=`` and reads that palace.

Two layers:
- routing: with ``palace=<other>`` each tool opens the *other* palace's
  collection / KG / tunnel+hallway files; without it, the default palace's
  (unchanged no-arg path);
- real data: two real chroma palaces holding different wings.
"""

import json
import os

import chromadb
import pytest
from _mcp_server_helpers import _patch_mcp_server


def _seed_chroma_palace(path, wing, room, doc_id, text):
    os.makedirs(path, exist_ok=True)
    col = chromadb.PersistentClient(path=path).get_or_create_collection(
        "mempalace_drawers", metadata={"hnsw:space": "cosine"}
    )
    col.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[
            {
                "wing": wing,
                "room": room,
                "hall": "hall_facts",
                "source_file": f"/src/{wing}.md",
                "chunk_index": 0,
                "added_by": "test",
                "filed_at": "2026-10-04T12:00:00",
            }
        ],
    )


@pytest.fixture
def two_palaces(monkeypatch, config, kg, palace_path, tmp_dir):
    """Default palace (fixture) holds wing alpha; the other holds wing beta."""
    _seed_chroma_palace(palace_path, "alpha", "room_a", "drawer_alpha", "aardvark notes")
    other = os.path.join(tmp_dir, "elsewhere", "palace_b")
    _seed_chroma_palace(other, "beta", "room_b", "drawer_beta", "zeppelin notes")
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace import mcp_server
    from mempalace.palace_graph import invalidate_graph_cache

    invalidate_graph_cache()
    return mcp_server, other


# ── routing: each tool opens the palace it was asked for ─────────────────

_COLLECTION_TOOLS = [
    ("tool_list_wings", {}),
    ("tool_list_rooms", {}),
    ("tool_get_taxonomy", {}),
    ("tool_get_drawer", {"drawer_id": "x"}),
    ("tool_list_drawers", {}),
    ("tool_check_duplicate", {"content": "zeppelin notes"}),
    ("tool_diary_read", {"agent_name": "bob"}),
    ("tool_graph_stats", {}),
    ("tool_follow_tunnels", {"wing": "beta", "room": "room_b"}),
]


@pytest.mark.parametrize("tool,kwargs", _COLLECTION_TOOLS)
def test_collection_tools_open_the_requested_palace(monkeypatch, two_palaces, tool, kwargs):
    mcp_server, other = two_palaces
    opened = []
    real = mcp_server._get_collection

    def spy(palace_path=None, create=False):
        opened.append(palace_path)
        return real(palace_path=palace_path, create=create)

    monkeypatch.setattr(mcp_server, "_get_collection", spy)
    getattr(mcp_server, tool)(**kwargs, palace=other)
    assert opened and all(p == other for p in opened), opened

    opened.clear()
    getattr(mcp_server, tool)(**kwargs)
    # Default palace: the unchanged no-arg path (or the sqlite fast path,
    # which opens no collection at all).
    assert all(p is None for p in opened), opened


@pytest.mark.parametrize("tool,kwargs", [("tool_kg_stats", {}), ("tool_kg_timeline", {})])
def test_kg_tools_use_the_requested_palace(monkeypatch, two_palaces, tool, kwargs):
    mcp_server, other = two_palaces
    seen = []
    monkeypatch.setattr(
        mcp_server, "_call_kg", lambda op, palace_path=None: seen.append(palace_path) or {}
    )
    try:
        getattr(mcp_server, tool)(**kwargs, palace=other)
    except (KeyError, TypeError):
        pass  # the stub returns {}; only the routing matters here
    assert seen[-1] == other
    try:
        getattr(mcp_server, tool)(**kwargs)
    except (KeyError, TypeError):
        pass
    assert seen[-1] is None


def test_unknown_palace_alias_is_an_error_not_a_default_read(two_palaces):
    mcp_server, _ = two_palaces
    for tool, kwargs in _COLLECTION_TOOLS + [("tool_list_tunnels", {}), ("tool_list_hallways", {})]:
        result = getattr(mcp_server, tool)(**kwargs, palace="no-such-alias")
        assert "error" in result, (tool, result)


# ── real data: answers come from the requested palace ────────────────────


def test_list_wings_reads_each_palace(two_palaces):
    mcp_server, other = two_palaces
    assert set(mcp_server.tool_list_wings(palace=other)["wings"]) == {"beta"}
    assert set(mcp_server.tool_list_wings()["wings"]) == {"alpha"}


def test_get_drawer_finds_a_drawer_only_in_its_palace(two_palaces):
    mcp_server, other = two_palaces
    assert "zeppelin" in mcp_server.tool_get_drawer("drawer_beta", palace=other)["content"]
    assert "error" in mcp_server.tool_get_drawer("drawer_beta")


def test_graph_cache_is_per_palace(two_palaces):
    """palace_graph's warm cache must not answer one palace with another's graph."""
    mcp_server, other = two_palaces
    from mempalace.palace_graph import build_graph
    from mempalace.config import MempalaceConfig

    default_nodes, _ = build_graph(config=mcp_server._config)
    other_nodes, _ = build_graph(config=MempalaceConfig(palace_path=other))
    assert set(default_nodes) == {"room_a"}
    assert set(other_nodes) == {"room_b"}


def test_list_tunnels_reads_the_requested_palaces_tunnel_file(two_palaces):
    mcp_server, other = two_palaces
    tunnel = {
        "id": "t1",
        "source": {"wing": "beta", "room": "room_b"},
        "target": {"wing": "gamma", "room": "room_c"},
        "label": "x",
    }
    with open(os.path.join(other, "tunnels.json"), "w") as f:
        json.dump([tunnel], f)
    assert [t["id"] for t in mcp_server.tool_list_tunnels(palace=other)] == ["t1"]
    assert mcp_server.tool_list_tunnels() == []
