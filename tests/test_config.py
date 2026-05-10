import os
import json
import tempfile

import pytest
from mempalace.config import MempalaceConfig, normalize_wing_name, sanitize_kg_value, sanitize_name


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


def test_embedding_device_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.embedding_device == "auto"


def test_embedding_device_from_config_is_normalized(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"embedding_device": "  CUDA  "}, f)

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.embedding_device == "cuda"


def test_embedding_device_env_overrides_config(tmp_path, monkeypatch):
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"embedding_device": "cpu"}, f)
    monkeypatch.setenv("MEMPALACE_EMBEDDING_DEVICE", "  CoreML  ")

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.embedding_device == "coreml"


def test_env_override():
    raw = "/env/palace"
    os.environ["MEMPALACE_PALACE_PATH"] = raw
    try:
        cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
        # palace_path normalizes with abspath + expanduser to match the
        # --palace CLI code path. On Unix that's a no-op for "/env/palace";
        # on Windows abspath prepends the current drive letter.
        assert cfg.palace_path == os.path.abspath(os.path.expanduser(raw))
    finally:
        del os.environ["MEMPALACE_PALACE_PATH"]


def test_env_path_expanduser():
    # Tilde must be expanded to match the --palace CLI code path. We don't
    # assert "~" is absent from the final string because Windows 8.3 short
    # paths (e.g. C:\Users\RUNNER~1\...) legitimately contain tildes — the
    # equality check is authoritative.
    raw = os.path.join("~", "mempalace-test")
    os.environ["MEMPALACE_PALACE_PATH"] = raw
    try:
        cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
        assert cfg.palace_path == os.path.abspath(os.path.expanduser(raw))
        assert cfg.palace_path.endswith("mempalace-test")
    finally:
        del os.environ["MEMPALACE_PALACE_PATH"]


def test_env_path_abspath_collapses_traversal():
    # Build a raw path with a .. segment using the platform separator so
    # the assertion is portable (Windows uses \, POSIX uses /).
    raw = os.path.join(tempfile.gettempdir(), "palace", "..", "mempalace-test")
    expected = os.path.abspath(os.path.expanduser(raw))
    os.environ["MEMPALACE_PALACE_PATH"] = raw
    try:
        cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
        # .. segments must be collapsed, not preserved literally.
        assert ".." not in cfg.palace_path
        assert cfg.palace_path == expected
    finally:
        del os.environ["MEMPALACE_PALACE_PATH"]


def test_env_path_legacy_alias_normalized():
    # Legacy MEMPAL_PALACE_PATH gets the same normalization treatment as
    # MEMPALACE_PALACE_PATH. We don't assert "~" is absent from the final
    # string because Windows 8.3 short paths (e.g. C:\Users\RUNNER~1\...)
    # legitimately contain tildes — the equality check below is authoritative.
    os.environ.pop("MEMPALACE_PALACE_PATH", None)
    raw = os.path.join("~", "legacy-alias", "..", "mempalace-test")
    os.environ["MEMPAL_PALACE_PATH"] = raw
    try:
        cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
        assert ".." not in cfg.palace_path
        assert cfg.palace_path == os.path.abspath(os.path.expanduser(raw))
    finally:
        del os.environ["MEMPAL_PALACE_PATH"]


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


def test_resolved_palace_path_loud_fails_when_only_legacy_palace_path_set(tmp_path, monkeypatch):
    """Palace-isolation step 7: the legacy ``palace_path`` field is no longer
    a silent fallback. If the user has only that field — no env var, no
    walk-up yaml, no ``default_palace`` — raise PalaceNotDeclared so they
    have to opt into a default explicitly."""
    from mempalace.config import PalaceNotDeclared

    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    with open(cfg_dir / "config.json", "w") as f:
        json.dump({"palace_path": "/legacy/palace"}, f)
    cfg = MempalaceConfig(config_dir=str(cfg_dir))
    with pytest.raises(PalaceNotDeclared):
        cfg.resolved_palace_path()


def test_resolved_palace_path_loud_fails_with_no_declaration_at_all(tmp_path, monkeypatch):
    """Step 7 invariant: empty config + no walk-up + no env var raises."""
    from mempalace.config import PalaceNotDeclared

    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = MempalaceConfig(config_dir=str(tmp_path / "cfg"))
    with pytest.raises(PalaceNotDeclared, match="no palace declared"):
        cfg.resolved_palace_path()


def test_init_writes_default_palace_so_loud_fail_does_not_strand_fresh_users(tmp_path):
    """``mempalace init`` must seed ``default_palace`` so a fresh-install
    user can run CLI commands without immediately tripping the step-7
    loud-fail. Otherwise the recommended onboarding (``pip install`` →
    ``mempalace init``) would land on a broken state."""
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    cfg.init()
    with open(tmp_path / "config.json") as f:
        written = json.load(f)
    assert written.get("default_palace") == "chat"
    assert "chat" in written.get("palaces", {})


def test_legacy_palace_migration_seeds_default_palace(tmp_path, monkeypatch):
    """Pre-isolation users with a ``~/.mempalace/palace/`` dir + config without
    ``default_palace`` would trip the step-7 loud-fail on first upgrade.
    The migration seeds ``default_palace`` (from existing ``palace_path`` if
    set, else "chat") so they keep working."""
    from mempalace import config as cfg_mod

    home = tmp_path / "home"
    legacy = home / ".mempalace" / "palace"
    legacy.mkdir(parents=True)
    new_dest = home / ".mempalace" / "palaces" / "chat"
    config_file = home / ".mempalace" / "config.json"
    with open(config_file, "w") as f:
        json.dump({"palace_path": str(legacy)}, f)

    monkeypatch.setattr(cfg_mod, "LEGACY_PALACE_DIR", str(legacy))
    monkeypatch.setattr(cfg_mod, "DEFAULT_PALACE_PATH", str(new_dest))
    monkeypatch.setenv("HOME", str(home))

    cfg_mod._maybe_migrate_legacy_palace_dir()

    with open(config_file) as f:
        migrated = json.load(f)
    # default_palace must resolve to a still-valid location after the rename.
    # Since the user's pinned path matched the legacy dir we just moved, we
    # use the "chat" alias (which points at the new dest) rather than the
    # now-stale legacy path string.
    assert migrated["default_palace"] == "chat"
    assert migrated["palaces"]["chat"] == str(new_dest)
    assert migrated["palace_path"] == str(new_dest)
    assert new_dest.exists()
    assert not legacy.exists()


def test_legacy_palace_migration_defaults_to_chat_when_no_palace_path(tmp_path, monkeypatch):
    """If a pre-isolation config has no ``palace_path`` field at all, the
    migration should still seed ``default_palace`` — landing on the canonical
    "chat" alias since there is no user-pinned location to honor."""
    from mempalace import config as cfg_mod

    home = tmp_path / "home"
    legacy = home / ".mempalace" / "palace"
    legacy.mkdir(parents=True)
    new_dest = home / ".mempalace" / "palaces" / "chat"
    config_file = home / ".mempalace" / "config.json"
    with open(config_file, "w") as f:
        json.dump({"collection_name": "mempalace_drawers"}, f)

    monkeypatch.setattr(cfg_mod, "LEGACY_PALACE_DIR", str(legacy))
    monkeypatch.setattr(cfg_mod, "DEFAULT_PALACE_PATH", str(new_dest))
    monkeypatch.setenv("HOME", str(home))

    cfg_mod._maybe_migrate_legacy_palace_dir()

    with open(config_file) as f:
        migrated = json.load(f)
    assert migrated["default_palace"] == "chat"
    assert migrated["palaces"]["chat"] == str(new_dest)


def test_init():
    tmpdir = tempfile.mkdtemp()
    cfg = MempalaceConfig(config_dir=tmpdir)
    cfg.init()
    assert os.path.exists(os.path.join(tmpdir, "config.json"))


# --- normalize_wing_name ---


def test_normalize_wing_name_hyphen():
    assert normalize_wing_name("mempal-private") == "mempal_private"


def test_normalize_wing_name_space():
    assert normalize_wing_name("My Project") == "my_project"


def test_normalize_wing_name_already_clean():
    assert normalize_wing_name("memorymark") == "memorymark"


def test_normalize_wing_name_mixed():
    assert normalize_wing_name("My-Cool App") == "my_cool_app"


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


# --- llm_provider / llm_model / llm_endpoint / llm_api_key --------------------
#
# Mirror the env > file > default pattern that embedding_provider ships with.
# Defaults must match historical CLI behavior (ollama / gemma4:e4b / None)
# so this layer is non-breaking.


def _clear_llm_env(monkeypatch):
    for var in (
        "MEMPALACE_LLM_PROVIDER",
        "MEMPALACE_LLM_MODEL",
        "MEMPALACE_LLM_ENDPOINT",
        "MEMPALACE_LLM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_llm_provider_defaults_to_ollama(monkeypatch):
    _clear_llm_env(monkeypatch)
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.llm_provider == "ollama"


def test_llm_provider_from_config_is_normalized(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"llm_provider": "  Openai-Compat  "}, f)

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.llm_provider == "openai-compat"


def test_llm_provider_env_overrides_config(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"llm_provider": "ollama"}, f)
    monkeypatch.setenv("MEMPALACE_LLM_PROVIDER", "  Anthropic  ")

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.llm_provider == "anthropic"


def test_llm_model_defaults_to_gemma4(monkeypatch):
    _clear_llm_env(monkeypatch)
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.llm_model == "gemma4:e4b"


def test_llm_model_from_config(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"llm_model": "  qwen2.5-coder:14b  "}, f)

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.llm_model == "qwen2.5-coder:14b"


def test_llm_model_env_overrides_config(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"llm_model": "gemma4:e4b"}, f)
    monkeypatch.setenv("MEMPALACE_LLM_MODEL", "qwen2.5:14b")

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.llm_model == "qwen2.5:14b"


def test_llm_endpoint_defaults_to_none(monkeypatch):
    _clear_llm_env(monkeypatch)
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.llm_endpoint is None


def test_llm_endpoint_from_config_strips_trailing_slash(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"llm_endpoint": "http://192.168.1.147:8001/"}, f)

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.llm_endpoint == "http://192.168.1.147:8001"


def test_llm_endpoint_env_overrides_config(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"llm_endpoint": "http://localhost:11434"}, f)
    monkeypatch.setenv("MEMPALACE_LLM_ENDPOINT", "http://192.168.1.147:8001/")

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.llm_endpoint == "http://192.168.1.147:8001"


def test_llm_api_key_defaults_to_none(monkeypatch):
    _clear_llm_env(monkeypatch)
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.llm_api_key is None


def test_llm_api_key_from_env(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("MEMPALACE_LLM_API_KEY", "  sk-test-12345  ")

    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.llm_api_key == "sk-test-12345"


def test_llm_api_key_does_not_read_from_file(tmp_path, monkeypatch):
    """API keys must NEVER be loaded from config.json — env-only.

    Mirrors how ANTHROPIC_API_KEY / OPENAI_API_KEY are scoped in
    mempalace.llm_client. Persisting keys in a world-readable config file
    is a footgun we explicitly refuse.
    """
    _clear_llm_env(monkeypatch)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"llm_api_key": "sk-should-be-ignored"}, f)

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.llm_api_key is None
