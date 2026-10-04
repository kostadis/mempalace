"""#51: the known-entities registry is per palace.

``~/.mempalace/known_entities.json`` used to be one registry for every palace:
names confirmed for one palace were tagged on another palace's drawers, and
topic tunnels were computed against every wing on the host. The registry now
lives inside each palace (``<palace>/known_entities.json``), like the KG,
tunnels and hallways.
"""

import json
import logging
import os

import pytest

from mempalace import miner
from mempalace.config import PALACE_SIDECAR_FILENAMES, MempalaceConfig


@pytest.fixture
def palaces(tmp_path):
    a, b = tmp_path / "palaces" / "chat", tmp_path / "palaces" / "campaign"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    return str(a), str(b)


def test_registry_lives_inside_the_palace(palaces):
    a, _ = palaces
    assert MempalaceConfig(palace_path=a).entity_registry_file == os.path.join(
        a, "known_entities.json"
    )
    assert "known_entities.json" in PALACE_SIDECAR_FILENAMES


def test_names_are_isolated_between_palaces(palaces):
    a, b = palaces
    miner.add_to_known_entities({"people": ["Aya"]}, palace_path=a)

    assert os.path.isfile(os.path.join(a, "known_entities.json"))
    assert "Aya" in miner._load_known_entities(palace_path=a)
    assert "Aya" not in miner._load_known_entities(palace_path=b)
    assert "Aya" in miner._extract_entities_for_metadata("Aya said hi", palace_path=a)
    assert "Aya" not in (miner._extract_entities_for_metadata("Aya said hi", palace_path=b) or "")


def test_topics_by_wing_are_isolated_between_palaces(palaces):
    a, b = palaces
    miner.add_to_known_entities({"topics": ["Angular"]}, wing="wing_one", palace_path=a)
    miner.add_to_known_entities({"topics": ["Angular"]}, wing="wing_two", palace_path=b)

    assert set(miner.get_topics_by_wing(palace_path=a)) == {"wing_one"}
    assert set(miner.get_topics_by_wing(palace_path=b)) == {"wing_two"}


def test_topic_tunnels_never_point_at_another_palaces_wing(palaces):
    from mempalace.palace_graph import list_tunnels

    a, b = palaces
    miner.add_to_known_entities({"topics": ["Angular"]}, wing="wing_one", palace_path=a)
    miner.add_to_known_entities({"topics": ["Angular"]}, wing="wing_two", palace_path=b)

    cfg_a = MempalaceConfig(palace_path=a)
    miner._compute_topic_tunnels_for_wing("wing_one", config=cfg_a)

    wings = {t[end]["wing"] for t in list_tunnels(config=cfg_a) for end in ("source", "target")}
    assert "wing_two" not in wings


def test_mine_tags_only_the_mined_palaces_names(palaces, tmp_path):
    a, b = palaces
    miner.add_to_known_entities({"people": ["Zebulon"]}, palace_path=a)
    project = tmp_path / "project"
    project.mkdir()
    # One lowercase mention: only the registry match (case-insensitive) can tag
    # it -- the capitalized-word heuristic needs >= 2 capitalized occurrences.
    (project / "notes.md").write_text(
        "the design review went well and everything shipped on time.\n"
        "zebulon approved the final plan after a short discussion.\n"
        "nothing else of note happened during the rest of the week.\n"
    )
    (project / "mempalace.yaml").write_text("wing: notes\nrooms:\n  - name: general\n")

    from mempalace.palace import get_collection

    for palace in (a, b):
        miner.mine(str(project), palace, wing_override="notes")

    def tagged(palace):
        metas = get_collection(palace, create=False).get(include=["metadatas"]).metadatas
        return any("Zebulon" in (m.get("entities") or "") for m in metas)

    assert tagged(a)
    assert not tagged(b)


def test_fact_checker_reads_its_palaces_registry(palaces):
    from mempalace.fact_checker import check_text

    a, b = palaces
    miner.add_to_known_entities({"people": ["Milla", "Mila"]}, palace_path=a)

    def similar_names(palace):
        issues = check_text("Chatted with Mila.", palace_path=palace)
        return [i for i in issues if i["type"] == "similar_name"]

    assert similar_names(a)
    assert not similar_names(b)


def test_old_global_registry_is_ignored_with_a_warning(palaces, tmp_path, monkeypatch, caplog):
    """No migration (user decision): the global file is left alone and named."""
    a, _ = palaces
    legacy = tmp_path / "home" / ".mempalace" / "known_entities.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"people": ["Legacy"]}))
    monkeypatch.setattr(miner, "_LEGACY_ENTITY_REGISTRY_PATH", str(legacy))
    miner._LEGACY_REGISTRY_WARNED.clear()

    with caplog.at_level(logging.WARNING):
        assert "Legacy" not in miner._load_known_entities(palace_path=a)
    assert str(legacy) in caplog.text
    assert legacy.exists()


def test_repair_and_migrate_carry_the_registry_over(tmp_path):
    from mempalace.repair import carry_over_palace_sidecars

    src, dst = tmp_path / "old", tmp_path / "new"
    src.mkdir()
    (src / "known_entities.json").write_text('{"people": ["Aya"]}')

    assert "known_entities.json" in carry_over_palace_sidecars(str(src), str(dst))
    assert (dst / "known_entities.json").read_text() == '{"people": ["Aya"]}'
