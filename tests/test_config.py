import os
import json
import tempfile

import pytest
from mempalace.config import MempalaceConfig, sanitize_kg_value, sanitize_name


def test_default_config():
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert "palace" in cfg.palace_path
    assert cfg.collection_name == "mempalace_drawers"


def test_config_from_file():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump({"palace_path": "/custom/palace"}, f)
    cfg = MempalaceConfig(config_dir=tmpdir)
    assert cfg.palace_path == "/custom/palace"


def test_env_override():
    os.environ["MEMPALACE_PALACE_PATH"] = "/env/palace"
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.palace_path == "/env/palace"
    del os.environ["MEMPALACE_PALACE_PATH"]


# --- resolve_palace (aliases + path expansion) ---


def _cfg_with_palaces(aliases):
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump({"palaces": aliases}, f)
    return MempalaceConfig(config_dir=tmpdir)


def test_resolve_palace_alias_absolute():
    cfg = _cfg_with_palaces({"oota": "/tmp/palaces/oota"})
    assert cfg.resolve_palace("oota") == "/tmp/palaces/oota"


def test_resolve_palace_alias_expands_tilde():
    cfg = _cfg_with_palaces({"chat": "~/.mempalace/palaces/chat"})
    resolved = cfg.resolve_palace("chat")
    assert resolved.endswith("/.mempalace/palaces/chat")
    assert "~" not in resolved


def test_resolve_palace_absolute_path_passthrough():
    cfg = _cfg_with_palaces({})
    assert cfg.resolve_palace("/var/data/palace") == "/var/data/palace"


def test_resolve_palace_tilde_path():
    cfg = _cfg_with_palaces({})
    resolved = cfg.resolve_palace("~/custom-palace")
    assert resolved.endswith("/custom-palace")
    assert "~" not in resolved


def test_resolve_palace_unknown_alias_raises():
    cfg = _cfg_with_palaces({"chat": "/tmp/chat"})
    with pytest.raises(ValueError, match="unknown palace alias"):
        cfg.resolve_palace("oota")


def test_resolve_palace_unknown_alias_lists_known():
    cfg = _cfg_with_palaces({"chat": "/tmp/chat", "oota": "/tmp/oota"})
    with pytest.raises(ValueError) as exc:
        cfg.resolve_palace("phandalin")
    msg = str(exc.value)
    assert "chat" in msg and "oota" in msg


def test_resolve_palace_rejects_empty():
    cfg = _cfg_with_palaces({})
    with pytest.raises(ValueError):
        cfg.resolve_palace("")


def test_resolve_palace_rejects_non_string():
    cfg = _cfg_with_palaces({})
    with pytest.raises(ValueError):
        cfg.resolve_palace(None)


def test_palaces_property_empty_when_unset():
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.palaces == {}


def test_palaces_property_ignores_non_dict():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump({"palaces": ["chat", "oota"]}, f)
    cfg = MempalaceConfig(config_dir=tmpdir)
    assert cfg.palaces == {}


# --- walk_up_palace + resolved_palace_path ---


def _write_yaml(path, data):
    import yaml as _yaml

    with open(path, "w") as f:
        _yaml.safe_dump(data, f)


def test_walk_up_finds_yaml_in_start_dir(tmp_path):
    project = tmp_path / "campaign"
    project.mkdir()
    _write_yaml(project / "mempalace.yaml", {"palace": "/palaces/oota", "wing": "narrative"})
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    assert cfg.walk_up_palace(start_dir=str(project)) == "/palaces/oota"


def test_walk_up_finds_yaml_in_ancestor(tmp_path):
    project = tmp_path / "campaign"
    nested = project / "docs" / "chapters"
    nested.mkdir(parents=True)
    _write_yaml(project / "mempalace.yaml", {"palace": "/palaces/oota"})
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    assert cfg.walk_up_palace(start_dir=str(nested)) == "/palaces/oota"


def test_walk_up_returns_none_when_no_yaml(tmp_path):
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    assert cfg.walk_up_palace(start_dir=str(tmp_path)) is None


def test_walk_up_skips_yaml_without_palace_key(tmp_path):
    project = tmp_path / "campaign"
    project.mkdir()
    _write_yaml(project / "mempalace.yaml", {"wing": "narrative"})
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    assert cfg.walk_up_palace(start_dir=str(project)) is None


def test_walk_up_resolves_alias(tmp_path):
    project = tmp_path / "campaign"
    project.mkdir()
    _write_yaml(project / "mempalace.yaml", {"palace": "oota"})
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    with open(cfg_dir / "config.json", "w") as f:
        json.dump({"palaces": {"oota": "/palaces/oota"}}, f)
    cfg = MempalaceConfig(config_dir=str(cfg_dir))
    assert cfg.walk_up_palace(start_dir=str(project)) == "/palaces/oota"


def test_walk_up_propagates_unknown_alias(tmp_path):
    project = tmp_path / "campaign"
    project.mkdir()
    _write_yaml(project / "mempalace.yaml", {"palace": "missing"})
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    with pytest.raises(ValueError, match="unknown palace alias"):
        cfg.walk_up_palace(start_dir=str(project))


def test_walk_up_survives_malformed_yaml(tmp_path):
    project = tmp_path / "campaign"
    project.mkdir()
    (project / "mempalace.yaml").write_text(": : not valid yaml :")
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    assert cfg.walk_up_palace(start_dir=str(project)) is None


def test_resolved_palace_path_env_wins(tmp_path, monkeypatch):
    project = tmp_path / "campaign"
    project.mkdir()
    _write_yaml(project / "mempalace.yaml", {"palace": "/palaces/oota"})
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", "/env/palace")
    monkeypatch.chdir(project)
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    assert cfg.resolved_palace_path() == "/env/palace"


def test_resolved_palace_path_walk_up_wins_over_default(tmp_path, monkeypatch):
    project = tmp_path / "campaign"
    project.mkdir()
    _write_yaml(project / "mempalace.yaml", {"palace": "/palaces/oota"})
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    monkeypatch.chdir(project)
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    assert cfg.resolved_palace_path() == "/palaces/oota"


def test_resolved_palace_path_default_palace_alias(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)  # no yaml here
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    with open(cfg_dir / "config.json", "w") as f:
        json.dump(
            {"palaces": {"chat": "/palaces/chat"}, "default_palace": "chat"},
            f,
        )
    cfg = MempalaceConfig(config_dir=str(cfg_dir))
    assert cfg.resolved_palace_path() == "/palaces/chat"


def test_resolved_palace_path_falls_back_to_palace_path(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    with open(cfg_dir / "config.json", "w") as f:
        json.dump({"palace_path": "/legacy/palace"}, f)
    cfg = MempalaceConfig(config_dir=str(cfg_dir))
    assert cfg.resolved_palace_path() == "/legacy/palace"


def test_init():
    tmpdir = tempfile.mkdtemp()
    cfg = MempalaceConfig(config_dir=tmpdir)
    cfg.init()
    assert os.path.exists(os.path.join(tmpdir, "config.json"))


# --- sanitize_name ---


def test_sanitize_name_ascii():
    assert sanitize_name("hello") == "hello"


def test_sanitize_name_latvian():
    assert sanitize_name("Jānis") == "Jānis"


def test_sanitize_name_cjk():
    assert sanitize_name("太郎") == "太郎"


def test_sanitize_name_cyrillic():
    assert sanitize_name("Алексей") == "Алексей"


def test_sanitize_name_rejects_leading_underscore():
    with pytest.raises(ValueError):
        sanitize_name("_foo")


def test_sanitize_name_rejects_path_traversal():
    with pytest.raises(ValueError):
        sanitize_name("../etc/passwd")


def test_sanitize_name_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_name("")


# --- sanitize_kg_value ---


def test_kg_value_accepts_commas():
    assert sanitize_kg_value("Alice, Bob, and Carol") == "Alice, Bob, and Carol"


def test_kg_value_accepts_colons():
    assert sanitize_kg_value("role: engineer") == "role: engineer"


def test_kg_value_accepts_parentheses():
    assert sanitize_kg_value("Python (programming)") == "Python (programming)"


def test_kg_value_accepts_slashes():
    assert sanitize_kg_value("owner/repo") == "owner/repo"


def test_kg_value_accepts_hash():
    assert sanitize_kg_value("issue #123") == "issue #123"


def test_kg_value_accepts_unicode():
    assert sanitize_kg_value("Jānis Bērziņš") == "Jānis Bērziņš"


def test_kg_value_strips_whitespace():
    assert sanitize_kg_value("  hello  ") == "hello"


def test_kg_value_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_kg_value("")


def test_kg_value_rejects_whitespace_only():
    with pytest.raises(ValueError):
        sanitize_kg_value("   ")


def test_kg_value_rejects_null_bytes():
    with pytest.raises(ValueError):
        sanitize_kg_value("hello\x00world")


def test_kg_value_rejects_over_length():
    with pytest.raises(ValueError):
        sanitize_kg_value("a" * 129)
