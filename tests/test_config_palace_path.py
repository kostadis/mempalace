"""Tests for palace_path tilde expansion in MempalaceConfig."""

import json
import os
import tempfile

import pytest
from mempalace.config import MempalaceConfig


def test_palace_path_expands_tilde_from_config_file():
    """palace_path must expand ~ even when read from config.json, not env."""
    cfg = MempalaceConfig()
    # The session config seeds default_palace, which outranks palace_path (#44).
    cfg._file_config.pop("default_palace", None)
    cfg._file_config["palace_path"] = "~/.mempalace/palace"
    result = cfg.palace_path
    assert not result.startswith("~"), (
        f"palace_path returned unexpanded tilde: {result!r}. "
        "This causes mempalace mine to create a literal '~' directory "
        "relative to CWD instead of writing to the home directory."
    )
    assert result == os.path.expanduser("~/.mempalace/palace")


def test_palace_path_expands_tilde_nested():
    """Nested tilde paths (e.g. ~/custom/palace) are also expanded."""
    cfg = MempalaceConfig()
    # The session config seeds default_palace, which outranks palace_path (#44).
    cfg._file_config.pop("default_palace", None)
    cfg._file_config["palace_path"] = "~/custom/mempalace"
    result = cfg.palace_path
    assert not result.startswith("~")
    assert result == os.path.expanduser("~/custom/mempalace")


def test_palace_path_absolute_unchanged():
    """Absolute paths pass through without modification."""
    cfg = MempalaceConfig()
    # The session config seeds default_palace, which outranks palace_path (#44).
    cfg._file_config.pop("default_palace", None)
    cfg._file_config["palace_path"] = "/tmp/test_palace"
    assert cfg.palace_path == "/tmp/test_palace"


@pytest.mark.skip(
    reason="Local init() seeds palace_path/default_palace/palaces with the chat "
    "palace (palace isolation); persisting a --palace override as palace_path "
    "would silently change the MCP server's default palace. Kept local behavior."
)
def test_init_persists_constructor_override_not_default():
    """init() must persist the resolved palace_path, not the hardcoded default.

    `mempalace --palace <custom> init` passes palace_path via the constructor
    (mirrored from cli.py's env-var write for cmd_init). The persisted
    config.json must record that custom path so a later invocation with no
    --palace flag (e.g. `mempalace status`) still finds it.
    """
    config_dir = tempfile.mkdtemp()
    custom_palace = os.path.join(tempfile.mkdtemp(), "custom-palace")
    cfg = MempalaceConfig(config_dir=config_dir, palace_path=custom_palace)
    assert cfg.palace_path == custom_palace

    cfg.init()

    with open(os.path.join(config_dir, "config.json")) as f:
        saved = json.load(f)
    assert saved["palace_path"] == custom_palace

    # A later invocation with no override must read the persisted path back.
    later_cfg = MempalaceConfig(config_dir=config_dir)
    assert later_cfg.palace_path == custom_palace


@pytest.mark.skip(
    reason="Local init() seeds palace_path/default_palace/palaces with the chat "
    "palace (palace isolation); persisting a --palace override as palace_path "
    "would silently change the MCP server's default palace. Kept local behavior."
)
def test_init_persists_env_var_palace_path():
    """init() must persist a MEMPALACE_PALACE_PATH override, not the default.

    cmd_init sets this env var before constructing MempalaceConfig() when
    --palace is passed (cli.py:308); init() must write what it resolved to,
    not the module-level default.
    """
    config_dir = tempfile.mkdtemp()
    custom_palace = os.path.join(tempfile.mkdtemp(), "env-palace")
    os.environ["MEMPALACE_PALACE_PATH"] = custom_palace
    try:
        cfg = MempalaceConfig(config_dir=config_dir)
        cfg.init()
    finally:
        del os.environ["MEMPALACE_PALACE_PATH"]

    with open(os.path.join(config_dir, "config.json")) as f:
        saved = json.load(f)
    assert saved["palace_path"] == custom_palace

    # A later invocation with no --palace and no env var must still resolve
    # to the persisted custom path, not silently fall back to the default.
    later_cfg = MempalaceConfig(config_dir=config_dir)
    assert later_cfg.palace_path == custom_palace


# ── #44: palace_path honours default_palace ─────────────────────────────


def _cfg_with(tmp_path, **file_config):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    with open(cfg_dir / "config.json", "w") as f:
        json.dump(file_config, f)
    return MempalaceConfig(config_dir=str(cfg_dir))


@pytest.fixture
def _no_palace_env(monkeypatch):
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)


def test_default_palace_used_when_palace_path_absent(tmp_path, _no_palace_env):
    campaign = str(tmp_path / "campaign")
    cfg = _cfg_with(tmp_path, default_palace="campaign", palaces={"campaign": campaign})
    assert cfg.palace_path == campaign


def test_default_palace_wins_over_config_palace_path(tmp_path, _no_palace_env):
    """palace-isolation.md's chain has default_palace, not palace_path (#44)."""
    campaign = str(tmp_path / "campaign")
    cfg = _cfg_with(
        tmp_path,
        palace_path=str(tmp_path / "chat"),
        default_palace="campaign",
        palaces={"campaign": campaign},
    )
    assert cfg.palace_path == campaign


def test_default_palace_accepts_a_path(tmp_path, _no_palace_env):
    target = tmp_path / "somewhere"
    cfg = _cfg_with(tmp_path, default_palace=str(target))
    assert cfg.palace_path == str(target)


def test_env_and_override_still_win_over_default_palace(tmp_path, monkeypatch):
    env_palace = str(tmp_path / "from-env")
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        json.dumps({"default_palace": "c", "palaces": {"c": str(tmp_path / "c")}})
    )
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", env_palace)
    assert MempalaceConfig(config_dir=str(cfg_dir)).palace_path == env_palace
    override = str(tmp_path / "override")
    assert MempalaceConfig(config_dir=str(cfg_dir), palace_path=override).palace_path == override


def test_unknown_default_palace_alias_falls_through_with_warning(tmp_path, _no_palace_env, caplog):
    chat = str(tmp_path / "chat")
    cfg = _cfg_with(tmp_path, palace_path=chat, default_palace="nope", palaces={})
    with caplog.at_level("WARNING"):
        assert cfg.palace_path == chat
    assert "default_palace" in caplog.text
