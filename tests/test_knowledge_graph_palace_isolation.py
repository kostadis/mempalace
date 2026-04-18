"""Tests that KnowledgeGraph storage is palace-local.

Covers step 3 of docs/design/palace-isolation.md:
  - ``KnowledgeGraph(palace_path=...)`` stores under the palace dir
  - Two palaces produce two independent graphs
  - Legacy ``~/.mempalace/knowledge_graph.sqlite3`` is migrated into the
    default palace on first open — but never into a custom palace
  - ``db_path`` + ``palace_path`` conflict raises
"""

import os

import pytest

from mempalace.knowledge_graph import (
    KG_FILENAME,
    KnowledgeGraph,
)


def test_palace_path_builds_palace_local_kg(tmp_path):
    palace = tmp_path / "palaces" / "oota"
    kg = KnowledgeGraph(palace_path=str(palace))
    expected = os.path.join(str(palace), KG_FILENAME)
    assert os.path.abspath(kg.db_path) == os.path.abspath(expected)
    assert os.path.exists(kg.db_path)


def test_two_palaces_are_independent(tmp_path):
    palace_a = tmp_path / "palaces" / "a"
    palace_b = tmp_path / "palaces" / "b"
    kg_a = KnowledgeGraph(palace_path=str(palace_a))
    kg_b = KnowledgeGraph(palace_path=str(palace_b))

    kg_a.add_triple("Alice", "lives_in", "Portland")
    kg_b.add_triple("Bob", "lives_in", "Seattle")

    a_subjects = {r["subject"] for r in kg_a.query_entity("Alice")}
    b_subjects = {r["subject"] for r in kg_b.query_entity("Bob")}
    assert "Alice" in a_subjects
    assert "Bob" in b_subjects
    # Cross-palace leakage check
    assert kg_a.query_entity("Bob") == []
    assert kg_b.query_entity("Alice") == []


def test_palace_path_and_db_path_conflict(tmp_path):
    with pytest.raises(ValueError, match="either db_path or palace_path"):
        KnowledgeGraph(db_path=str(tmp_path / "x.db"), palace_path=str(tmp_path / "p"))


def _seed_legacy_kg(kg_mod, default_palace, subject="LEGACY_USER"):
    """Create a real SQLite KG at the legacy global path and seed a triple."""
    legacy = kg_mod.LEGACY_GLOBAL_KG_PATH
    os.makedirs(os.path.dirname(legacy), exist_ok=True)
    # Use the class with an explicit db_path pointing AT the legacy location
    # so we don't trip the migration path. This gives us a real SQLite DB
    # with a known row we can verify after migration.
    seed = kg_mod.KnowledgeGraph(db_path=legacy)
    seed.add_triple(subject, "exists_in", "legacy_graph")
    # Close the connection so os.replace() works on Windows-ish semantics
    if seed._connection is not None:
        seed._connection.close()
        seed._connection = None
    return legacy


def test_legacy_global_migrates_into_default_palace(tmp_path, monkeypatch):
    """Opening a KG against the default palace moves the legacy global DB in."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    import importlib

    import mempalace.config as config_mod
    import mempalace.knowledge_graph as kg_mod

    importlib.reload(config_mod)
    importlib.reload(kg_mod)

    legacy = _seed_legacy_kg(kg_mod, config_mod.DEFAULT_PALACE_PATH)
    default_palace = config_mod.DEFAULT_PALACE_PATH
    target = kg_mod._palace_kg_path(default_palace)

    kg = kg_mod.KnowledgeGraph(palace_path=default_palace)

    assert os.path.exists(target), "legacy KG should have moved into the palace dir"
    assert not os.path.exists(legacy), "legacy path cleared after move"
    # Seed row survived the migration
    hits = kg.query_entity("LEGACY_USER")
    assert any(h["predicate"] == "exists_in" for h in hits)


def test_legacy_global_not_migrated_into_custom_palace(tmp_path, monkeypatch):
    """Custom palaces must start empty — legacy file stays where it is."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    import importlib

    import mempalace.config as config_mod
    import mempalace.knowledge_graph as kg_mod

    importlib.reload(config_mod)
    importlib.reload(kg_mod)

    legacy = _seed_legacy_kg(kg_mod, config_mod.DEFAULT_PALACE_PATH)

    custom_palace = str(tmp_path / "custom_palace")
    kg = kg_mod.KnowledgeGraph(palace_path=custom_palace)

    # Custom palace is empty — no LEGACY_USER row here
    assert kg.query_entity("LEGACY_USER") == []
    # Legacy file untouched
    assert os.path.exists(legacy)
    assert os.path.exists(kg_mod._palace_kg_path(custom_palace))


def test_migration_is_idempotent(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    import importlib

    import mempalace.config as config_mod
    import mempalace.knowledge_graph as kg_mod

    importlib.reload(config_mod)
    importlib.reload(kg_mod)

    default_palace = config_mod.DEFAULT_PALACE_PATH
    kg = kg_mod.KnowledgeGraph(palace_path=default_palace)
    kg.add_triple("X", "is", "Y")

    # Second open — legacy file doesn't exist anymore, migration is a no-op,
    # and the existing graph data must survive.
    kg2 = kg_mod.KnowledgeGraph(palace_path=default_palace)
    results = kg2.query_entity("X")
    assert any(r["subject"] == "X" and r["object"] == "Y" for r in results)
