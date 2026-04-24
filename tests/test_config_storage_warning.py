"""Tests for the DrvFs / 9P / CIFS storage warning in mempalace.config.

The palace cannot safely live on a filesystem without correct mmap /
flock / fsync semantics. On WSL2 this means DrvFs-backed Windows mounts
(``/mnt/c``, ``/mnt/d``, …) which /proc/mounts reports as fstype ``9p``
with an ``aname=drvfs;`` option. A dedicated VHD mounted at e.g.
``/mnt/data`` as ``ext4`` is fine and never warns.
"""

import os

import pytest

from mempalace import config as mp_config


@pytest.fixture(autouse=True)
def _reset_warned_cache():
    """Ensure each test starts from an empty one-shot warning cache."""
    mp_config._storage_warned.clear()
    yield
    mp_config._storage_warned.clear()


def test_ext4_palace_returns_no_warning(monkeypatch):
    monkeypatch.setattr(
        mp_config,
        "_find_mount_for",
        lambda path: ("/", "ext4", "rw,relatime"),
    )
    assert mp_config.check_palace_storage("/home/alice/.mempalace") == ""


def test_dedicated_vhd_mount_returns_no_warning(monkeypatch):
    # Simulates /mnt/data on a dedicated 80 GB VHD formatted as ext4.
    monkeypatch.setattr(
        mp_config,
        "_find_mount_for",
        lambda path: ("/mnt/data", "ext4", "rw,relatime"),
    )
    assert mp_config.check_palace_storage("/mnt/data/mempalace/palaces/chat") == ""


def test_drvfs_palace_warns(monkeypatch, caplog):
    # Real /proc/mounts line for /mnt/c on WSL2 uses fstype "9p" with
    # "aname=drvfs;" in the options — we catch both patterns.
    monkeypatch.setattr(
        mp_config,
        "_find_mount_for",
        lambda path: (
            "/mnt/c",
            "9p",
            "rw,noatime,aname=drvfs;path=C:\\;uid=1000;gid=1000",
        ),
    )
    with caplog.at_level("WARNING", logger="mempalace.config"):
        msg = mp_config.check_palace_storage("/mnt/c/Users/alice/.mempalace")
    assert msg
    assert "9p" in msg
    assert "/mnt/c" in msg
    assert any("9p" in rec.message for rec in caplog.records)


def test_cifs_palace_warns(monkeypatch):
    monkeypatch.setattr(
        mp_config,
        "_find_mount_for",
        lambda path: ("/mnt/share", "cifs", "rw,username=x"),
    )
    assert mp_config.check_palace_storage("/mnt/share/palace") != ""


def test_nfs_palace_warns(monkeypatch):
    monkeypatch.setattr(
        mp_config,
        "_find_mount_for",
        lambda path: ("/mnt/nfs", "nfs4", "rw"),
    )
    assert mp_config.check_palace_storage("/mnt/nfs/palace") != ""


def test_warning_is_emitted_once_per_path(monkeypatch):
    calls = {"n": 0}

    def _mount(_path):
        calls["n"] += 1
        return ("/mnt/c", "9p", "aname=drvfs;")

    monkeypatch.setattr(mp_config, "_find_mount_for", _mount)

    first = mp_config.check_palace_storage("/mnt/c/palace")
    second = mp_config.check_palace_storage("/mnt/c/palace")
    assert first != ""
    assert second == ""


def test_different_paths_warn_independently(monkeypatch):
    monkeypatch.setattr(
        mp_config,
        "_find_mount_for",
        lambda path: ("/mnt/c" if "/mnt/c" in path else "/mnt/d", "9p", "aname=drvfs;"),
    )
    assert mp_config.check_palace_storage("/mnt/c/palace") != ""
    assert mp_config.check_palace_storage("/mnt/d/palace") != ""


def test_missing_proc_mounts_returns_no_warning(monkeypatch):
    monkeypatch.setattr(mp_config, "_find_mount_for", lambda path: None)
    assert mp_config.check_palace_storage("/anywhere") == ""


def test_find_mount_picks_deepest_prefix(tmp_path, monkeypatch):
    # Two overlapping mounts at / and /mnt/data; the deeper one must win.
    fake_mounts = (
        "rootfs / rootfs rw 0 0\n"
        "/dev/sdd / ext4 rw,relatime 0 0\n"
        "/dev/sde /mnt/data ext4 rw,relatime 0 0\n"
    )
    mounts_file = tmp_path / "mounts"
    mounts_file.write_text(fake_mounts)

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/mounts":
            return real_open(mounts_file, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    got = mp_config._find_mount_for("/mnt/data/mempalace/x")
    assert got is not None
    mountpoint, fstype, _options = got
    assert mountpoint == "/mnt/data"
    assert fstype == "ext4"


def test_find_mount_returns_root_when_no_deeper_mount(tmp_path, monkeypatch):
    fake_mounts = "/dev/sdd / ext4 rw,relatime 0 0\n"
    mounts_file = tmp_path / "mounts"
    mounts_file.write_text(fake_mounts)

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/mounts":
            return real_open(mounts_file, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    got = mp_config._find_mount_for("/home/alice")
    assert got is not None
    mountpoint, fstype, _options = got
    assert mountpoint == "/"
    assert fstype == "ext4"


@pytest.mark.skipif(
    not os.path.isdir("/mnt/data") or not os.path.isfile("/proc/mounts"),
    reason="requires the /mnt/data VHD mount present (WSL2 + dedicated ext4 VHD)",
)
def test_live_mnt_data_is_quiet():
    """On the developer's actual box, /mnt/data is an ext4 VHD and must not warn."""
    assert mp_config.check_palace_storage("/mnt/data/mempalace") == ""
