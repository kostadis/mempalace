"""#48: tunnels and hallways are isolated per palace.

They used to live beside the palace directory (``dirname(palace)/tunnels.json``),
so every palace under ``~/.mempalace/palaces/`` shared one tunnels file and one
hallways file. They now live inside the palace, like the knowledge graph, and
the palace-replacing operations (repair rebuild, migrate) carry them over.
"""

import json
import logging
import os

import pytest

from mempalace.config import MempalaceConfig


def _cfg(palace):
    os.makedirs(palace, exist_ok=True)
    return MempalaceConfig(palace_path=str(palace))


@pytest.fixture
def siblings(tmp_path):
    root = tmp_path / "palaces"
    return _cfg(root / "chat"), _cfg(root / "campaign")


def test_sidecar_files_live_inside_the_palace(siblings):
    chat, _ = siblings
    assert chat.tunnel_file == os.path.join(chat.palace_path, "tunnels.json")
    assert chat.hallway_file == os.path.join(chat.palace_path, "hallways.json")


def test_tunnels_are_isolated_between_sibling_palaces(siblings):
    from mempalace.palace_graph import create_tunnel, list_tunnels

    chat, campaign = siblings
    create_tunnel("wing_a", "room_a", "wing_b", "room_b", label="chat only", config=chat)

    assert [t["label"] for t in list_tunnels(config=chat)] == ["chat only"]
    assert list_tunnels(config=campaign) == []


def test_hallways_are_isolated_between_sibling_palaces(siblings):
    from mempalace.hallways import _save_hallways, list_hallways

    chat, campaign = siblings
    _save_hallways([{"id": "h1", "wing": "wing_a", "entity_a": "x", "entity_b": "y"}], chat)

    assert [h["id"] for h in list_hallways(config=chat)] == ["h1"]
    assert list_hallways(config=campaign) == []


@pytest.mark.parametrize("name", ["tunnels.json", "hallways.json"])
def test_old_shared_sibling_file_is_ignored_with_a_warning(siblings, caplog, name):
    """No migration (user decision): the old shared file is left in place and named."""
    from mempalace.hallways import list_hallways
    from mempalace.palace_graph import list_tunnels

    chat, _ = siblings
    shared = os.path.join(os.path.dirname(chat.palace_path), name)
    with open(shared, "w") as f:
        json.dump([{"id": "old", "wing": "wing_a"}], f)

    reader = list_tunnels if name == "tunnels.json" else list_hallways
    with caplog.at_level(logging.WARNING):
        assert reader(config=chat) == []
    assert shared in caplog.text
    assert os.path.exists(shared)  # never moved or deleted


def test_carry_over_copies_every_palace_local_file(tmp_path):
    from mempalace.repair import carry_over_palace_sidecars

    src, dst = tmp_path / "old", tmp_path / "new"
    src.mkdir()
    for name in ("knowledge_graph.sqlite3", "tunnels.json", "hallways.json", "chroma.sqlite3"):
        (src / name).write_text(name)

    copied = carry_over_palace_sidecars(str(src), str(dst))

    assert sorted(copied) == ["hallways.json", "knowledge_graph.sqlite3", "tunnels.json"]
    assert (dst / "tunnels.json").read_text() == "tunnels.json"
    assert not (dst / "chroma.sqlite3").exists()  # the store itself is rebuilt, not copied


def test_migrate_keeps_palace_local_files(tmp_path, monkeypatch):
    """migrate swaps in a fresh palace dir; KG, tunnels and hallways must survive."""
    import chromadb

    from mempalace import migrate as migrate_mod

    palace = tmp_path / "palace"
    col = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        "mempalace_drawers", metadata={"hnsw:space": "cosine"}
    )
    col.add(ids=["d1"], documents=["hello"], metadatas=[{"wing": "w", "room": "r"}])
    for name in ("knowledge_graph.sqlite3", "tunnels.json", "hallways.json"):
        (palace / name).write_text(name)

    # Force the extract-and-swap path (a fresh palace is otherwise "already fine").
    monkeypatch.setattr(migrate_mod, "collection_write_roundtrip_works", lambda _col: False)
    assert migrate_mod.migrate(str(palace), confirm=True) is not False
    assert list(tmp_path.glob("palace.pre-migrate.*")), "migrate did not run its swap path"

    for name in ("knowledge_graph.sqlite3", "tunnels.json", "hallways.json"):
        assert (palace / name).read_text() == name, name
