"""Tests for the one-shot ``~/.mempalace/palace/`` → ``~/.mempalace/palaces/chat/``
rename that ships with the palace-isolation feature.

Covers step 4 of docs/design/palace-isolation.md:
  - Legacy dir is renamed under the new ``palaces/chat`` layout on first
    ``MempalaceConfig()`` load after upgrade.
  - Pinned ``palace_path`` in config.json gets rewritten when it was
    pointing at the legacy default; custom paths are never touched.
  - A ``chat`` alias is added to ``palaces`` so ``--palace chat`` works
    immediately.
  - Migration is idempotent — running again on an already-migrated
    install is a no-op.
  - Destination already exists → migration refuses to clobber.
"""

import importlib
import json
import os


def _reload_config(monkeypatch, fake_home):
    """Re-import ``mempalace.config`` with ``$HOME`` pointing at ``fake_home``
    so the module-level ``os.path.expanduser`` constants pick up the fake.
    """
    monkeypatch.setenv("HOME", str(fake_home))
    import mempalace.config as config_mod

    importlib.reload(config_mod)
    return config_mod


def _write_config(cfg_dir, data):
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _read_config(cfg_dir):
    with open(os.path.join(cfg_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def test_renames_legacy_dir_into_palaces_chat(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    legacy = fake_home / ".mempalace" / "palace"
    legacy.mkdir(parents=True)
    # Leave a sentinel so we can prove the *data* moved, not just the name.
    (legacy / "marker.txt").write_text("canonical content")

    config_mod = _reload_config(monkeypatch, fake_home)
    config_mod.MempalaceConfig()

    new_path = fake_home / ".mempalace" / "palaces" / "chat"
    assert new_path.is_dir()
    assert (new_path / "marker.txt").read_text() == "canonical content"
    assert not legacy.exists()


def test_rewrites_pinned_palace_path_in_config_json(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".mempalace" / "palace").mkdir(parents=True)
    _write_config(
        str(fake_home / ".mempalace"),
        {"palace_path": "~/.mempalace/palace"},
    )

    config_mod = _reload_config(monkeypatch, fake_home)
    config_mod.MempalaceConfig()

    cfg = _read_config(str(fake_home / ".mempalace"))
    assert cfg["palace_path"] == config_mod.DEFAULT_PALACE_PATH
    assert cfg["palaces"]["chat"] == config_mod.DEFAULT_PALACE_PATH


def test_preserves_custom_palace_path(tmp_path, monkeypatch):
    """A user who pointed ``palace_path`` at a non-default dir shouldn't have
    their setting clobbered just because we moved the built-in default."""
    fake_home = tmp_path / "home"
    (fake_home / ".mempalace" / "palace").mkdir(parents=True)
    custom = str(tmp_path / "opt" / "my_palace")
    _write_config(str(fake_home / ".mempalace"), {"palace_path": custom})

    config_mod = _reload_config(monkeypatch, fake_home)
    config_mod.MempalaceConfig()

    cfg = _read_config(str(fake_home / ".mempalace"))
    assert cfg["palace_path"] == custom


def test_adds_chat_alias_without_rewriting_existing_aliases(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".mempalace" / "palace").mkdir(parents=True)
    _write_config(
        str(fake_home / ".mempalace"),
        {"palaces": {"oota": "~/.mempalace/palaces/oota"}},
    )

    config_mod = _reload_config(monkeypatch, fake_home)
    config_mod.MempalaceConfig()

    cfg = _read_config(str(fake_home / ".mempalace"))
    assert cfg["palaces"]["oota"] == "~/.mempalace/palaces/oota"
    assert cfg["palaces"]["chat"] == config_mod.DEFAULT_PALACE_PATH


def test_migration_refuses_to_clobber_existing_destination(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    legacy = fake_home / ".mempalace" / "palace"
    legacy.mkdir(parents=True)
    (legacy / "old.txt").write_text("legacy")

    # Simulate a user who already set up the new-style palace manually.
    new_path = fake_home / ".mempalace" / "palaces" / "chat"
    new_path.mkdir(parents=True)
    (new_path / "new.txt").write_text("new world")

    config_mod = _reload_config(monkeypatch, fake_home)
    config_mod.MempalaceConfig()

    # Neither side got touched.
    assert (legacy / "old.txt").read_text() == "legacy"
    assert (new_path / "new.txt").read_text() == "new world"


def test_migration_is_idempotent(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    legacy = fake_home / ".mempalace" / "palace"
    legacy.mkdir(parents=True)
    (legacy / "marker.txt").write_text("canonical content")

    config_mod = _reload_config(monkeypatch, fake_home)
    config_mod.MempalaceConfig()
    # Second init — legacy dir is gone, nothing to migrate.
    config_mod.MempalaceConfig()

    new_path = fake_home / ".mempalace" / "palaces" / "chat"
    assert (new_path / "marker.txt").read_text() == "canonical content"
    assert not legacy.exists()


def test_fresh_install_no_migration_side_effects(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    config_mod = _reload_config(monkeypatch, fake_home)
    config_mod.MempalaceConfig()

    # No legacy dir existed → no palaces/ scaffolding conjured into being.
    assert not (fake_home / ".mempalace" / "palaces" / "chat").exists()


def test_explicit_config_dir_skips_migration(tmp_path, monkeypatch):
    """Tests and library consumers pass ``config_dir=...`` to work against a
    sandboxed directory — they must not trigger the HOME-scoped rename as a
    side effect."""
    fake_home = tmp_path / "home"
    legacy = fake_home / ".mempalace" / "palace"
    legacy.mkdir(parents=True)
    (legacy / "marker.txt").write_text("canonical content")

    config_mod = _reload_config(monkeypatch, fake_home)
    sandbox = tmp_path / "sandbox"
    config_mod.MempalaceConfig(config_dir=str(sandbox))

    # HOME dir was untouched by the sandboxed init.
    assert (legacy / "marker.txt").read_text() == "canonical content"
    assert not (fake_home / ".mempalace" / "palaces" / "chat").exists()
