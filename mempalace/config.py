"""
MemPalace configuration system.

Priority: env vars > config file > defaults.

The config directory is $MEMPALACE_CONFIG_DIR if set (explicit override,
used by tests and CI), otherwise ~/.mempalace. Upstream falls back to the
XDG Base Directory location; this fork pins ~/.mempalace because palace
isolation lives there (see _default_config_dir).
"""

import errno
import hashlib
import json
import logging
import math
import os
import stat
import re
import sys
import tempfile
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from .write_routing import (
    ResolvedWriteRoutingPolicy,
    RoutingPolicyCandidate,
    WriteRoutingError,
    WriteRoutingPolicy,
    resolve_write_routing_policy,
)

logger = logging.getLogger(__name__)


# ── Input validation ──────────────────────────────────────────────────────────
# Shared sanitizers for wing/room/entity names. Prevents path traversal,
# excessively long strings, and special characters that could cause issues
# in file paths, SQLite, or ChromaDB metadata.

MAX_NAME_LENGTH = 128
_SAFE_NAME_RE = re.compile(r"^(?:[^\W_]|[^\W_][\w .'-]{0,126}[^\W_])$")

# MCP clients (e.g. Claude Desktop, WorkBuddy) occasionally relay lone UTF-16
# surrogates (U+D800–U+DFFF) when proxying binary-in-Unicode or corrupted
# clipboard input. Python's ``str.encode('utf-8')`` raises on these, which
# crashes ChromaDB add/upsert with -32000. See issue #1235.
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def strip_lone_surrogates(text: str) -> str:
    """Replace lone UTF-16 surrogates with U+FFFD so the string is legal UTF-8 (#1235)."""
    return _LONE_SURROGATE_RE.sub("�", text)


# Tool output mined from real transcripts routinely embeds a NUL character
# (U+0000) — e.g. captured Bash output where a reader raced a background
# writer, or genuine binary/NUL-delimited command output. A document
# containing one is otherwise valid, well-formed text (unlike a lone
# surrogate, which is invalid UTF-8), but handing it to ChromaDB's
# SQLite/FTS5 layer can corrupt the FTS5 inverted index for the *whole*
# collection (``PRAGMA quick_check`` reports "malformed inverted index for
# FTS5 table"), not just fail to store that one document. Stripping it
# before it reaches the chromadb client is the same defense-in-depth this
# module already applies to lone surrogates (#1235) — sanitize input we
# don't control before it reaches a datastore we don't control.
def strip_nul_bytes(text: str) -> str:
    """Replace embedded NUL characters with U+FFFD before ChromaDB storage."""
    return text.replace("\x00", "�")


def normalize_wing_name(name: str) -> str:
    """Lower-case + collapse separators (`-`, ` `) to `_` for wing slugs.

    The same rule is applied by ``init`` when persisting `topics_by_wing`
    and when writing `mempalace.yaml`, so the miner's lookup matches at
    mine time regardless of the source dirname.

    Leading/trailing separators are stripped so a path-encoded dirname like
    ``-home-user-proj`` yields ``home_user_proj`` rather than a leading-
    underscore slug that ``sanitize_name`` (and thus the MCP write tools)
    would reject.
    """
    return name.lower().replace(" ", "_").replace("-", "_").strip("_")


def sanitize_name(value: str, field_name: str = "name") -> str:
    """Validate and sanitize a wing/room/entity name.

    Raises ValueError if the name is invalid.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    value = value.strip()

    if len(value) > MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length of {MAX_NAME_LENGTH} characters")

    # Block path traversal
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} contains invalid path characters")

    # Block null bytes
    if "\x00" in value:
        raise ValueError(f"{field_name} contains null bytes")

    # Enforce safe character set
    if not _SAFE_NAME_RE.match(value):
        raise ValueError(f"{field_name} contains invalid characters")

    return value


def sanitize_kg_value(value: str, field_name: str = "value") -> str:
    """Validate a knowledge-graph entity name (subject or object).

    More permissive than sanitize_name — allows punctuation like commas,
    colons, and parentheses that are common in natural-language KG values.
    Only blocks null bytes and over-length strings.

    Not used for wing/room names (which have filesystem constraints) or
    predicates (which should be simple relationship identifiers).
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    value = value.strip()

    if len(value) > MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length of {MAX_NAME_LENGTH} characters")

    if "\x00" in value:
        raise ValueError(f"{field_name} contains null bytes")

    return strip_lone_surrogates(value)


# ISO-8601 temporal validator for knowledge-graph temporal parameters
# (as_of, valid_from, valid_to, ended).
#
# The KG stores temporal values as TEXT. Lexicographic comparisons are only
# safe when datetime values use one canonical shape. Accept full dates for
# legacy compatibility and exact UTC datetimes for sub-day precision.
#
# Accepted:
#   YYYY-MM-DD
#   YYYY-MM-DDTHH:MM:SSZ
#   YYYY-MM-DDTHH:MM:SS+00:00  (normalized to ...Z)
#
# Rejected:
#   partial dates, naive datetimes, non-UTC timezone offsets, fractional
#   seconds, and SQLite-style space-separated datetimes.
_ISO_DATE_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")

_ISO_UTC_DATETIME_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:Z|\+00:00)$"
)


def _validate_iso_temporal_calendar(value: str) -> None:
    """Reject impossible calendar values after regex shape validation."""

    if _ISO_DATE_RE.match(value):
        date.fromisoformat(value)
        return

    if _ISO_UTC_DATETIME_RE.match(value):
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return

    raise ValueError


def sanitize_iso_temporal(value, field_name: str = "date"):
    """Validate an ISO-8601 date or canonical UTC datetime string.

    Accepts ``None`` and ``""`` as pass-through values.

    Accepted non-empty string forms:

    - ``YYYY-MM-DD``
    - ``YYYY-MM-DDTHH:MM:SSZ``
    - ``YYYY-MM-DDTHH:MM:SS+00:00`` normalized to ``...Z``

    Partial dates are rejected because KG queries compare TEXT temporal values.
    Non-canonical datetime forms are rejected because mixed temporal string
    formats can silently return wrong KG query results.
    """

    if value is None or value == "":
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    value = value.strip()

    try:
        _validate_iso_temporal_calendar(value)
    except ValueError:
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO-8601 date or UTC datetime "
            "(expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)"
        ) from None

    if value.endswith("+00:00"):
        value = f"{value[:-6]}Z"

    return value


def sanitize_iso_date(value, field_name: str = "date"):
    """Backward-compatible wrapper for ISO temporal validation.

    Historically this accepted only full dates. It now also accepts canonical
    UTC datetimes, but the old name is kept so existing imports continue to
    work.
    """

    return sanitize_iso_temporal(value, field_name)


def sanitize_content(value: str, max_length: int = 100_000) -> str:
    """Validate drawer/diary content length."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("content must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"content exceeds maximum length of {max_length} characters")
    if "\x00" in value:
        raise ValueError("content contains null bytes")
    return strip_lone_surrogates(value)


# Palace-local files that live inside a palace directory next to its vector
# store. Operations that replace the palace directory (repair rebuild,
# migrate) must carry these over (see repair.carry_over_palace_sidecars).
TUNNELS_FILENAME = "tunnels.json"
HALLWAYS_FILENAME = "hallways.json"
PALACE_SIDECAR_FILENAMES = (
    "knowledge_graph.sqlite3",
    "knowledge_graph.sqlite3-wal",
    "knowledge_graph.sqlite3-shm",
    TUNNELS_FILENAME,
    HALLWAYS_FILENAME,
)

LEGACY_PALACE_DIR = os.path.expanduser("~/.mempalace/palace")
DEFAULT_PALACE_PATH = os.path.expanduser("~/.mempalace/palaces/chat")


def _default_config_dir() -> Path:
    """Return the config directory: ``$MEMPALACE_CONFIG_DIR`` or ``~/.mempalace``.

    Upstream resolves this per the XDG Base Directory spec (#148), falling
    back to ``~/.config/mempalace`` when ``~/.mempalace`` holds no legacy
    marker file. This fork pins it to ``~/.mempalace``: palace isolation
    (``default_palace`` and aliases in ``config.json``, the chat palace at
    ``~/.mempalace/palaces/chat``, the hooks' chat-palace pinning) lives
    there, and a silent move to the XDG directory would split it.
    ``MEMPALACE_CONFIG_DIR`` remains the explicit override.
    """
    env_dir = os.environ.get("MEMPALACE_CONFIG_DIR")
    if env_dir and env_dir.strip():
        return Path(env_dir).expanduser()
    return Path.home() / ".mempalace"


DEFAULT_COLLECTION_NAME = "mempalace_drawers"
DEFAULT_BACKEND = "chroma"
DEFAULT_MILVUS_CONSISTENCY_LEVEL = "Strong"
_MILVUS_CONSISTENCY_LEVELS = {
    "strong": "Strong",
    "session": "Session",
    "bounded": "Bounded",
    "eventually": "Eventually",
}

# How many timestamped palace backups to retain before the oldest are
# pruned. Applies to the accumulating backups written by ``mempalace
# migrate`` and ``mempalace repair max-seq-id`` — see
# ``MempalaceConfig.max_backups``.
DEFAULT_MAX_BACKUPS = 10

# Weights for the hybrid re-rank blend (vector embedding-similarity vs BM25
# lexical) in ``searcher._hybrid_rank``. These were the function's hardcoded
# defaults; surfaced as config so users can retune the blend without patching
# site-packages (#2298).
DEFAULT_HYBRID_VECTOR_WEIGHT = 0.6
DEFAULT_HYBRID_BM25_WEIGHT = 0.4


# Filesystem types that break ChromaDB / SQLite mmap + flock + fsync semantics
# when the palace is stored on them. On WSL2 these are the DrvFs-backed
# Windows drive mounts (/mnt/c, /mnt/d, /mnt/g …) which /proc/mounts reports
# as fstype "9p" with an "aname=drvfs;" option.
#
# ext4 / xfs / btrfs / tmpfs (and any future native Linux fs) are fine.
_BAD_FSTYPES = {"9p", "drvfs", "cifs", "smbfs", "smb3", "nfs", "nfs4", "fuse.sshfs"}

# Process-local cache so we only warn once per (resolved) palace path.
_storage_warned: set = set()


def _find_mount_for(path: str):
    """Return (mountpoint, fstype, options) for the deepest mount prefix of ``path``.

    Returns ``None`` if /proc/mounts can't be read (non-Linux, chroot, etc.).
    """
    try:
        path_abs = os.path.abspath(path)
    except (OSError, ValueError):
        return None

    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            mounts = f.readlines()
    except OSError:
        return None

    best = None  # (mountpoint_len, mountpoint, fstype, options)
    for line in mounts:
        parts = line.split()
        if len(parts) < 4:
            continue
        mountpoint, fstype, options = parts[1], parts[2], parts[3]
        # Mount line uses octal escapes for embedded spaces (`\040`).
        mountpoint = mountpoint.encode().decode("unicode_escape", errors="replace")
        if mountpoint == path_abs or path_abs.startswith(mountpoint.rstrip("/") + "/"):
            mp_len = len(mountpoint.rstrip("/"))
            if best is None or mp_len > best[0]:
                best = (mp_len, mountpoint, fstype, options)

    if best is None:
        return None
    return (best[1], best[2], best[3])


def check_palace_storage(path: str) -> str:
    """Warn once when ``path`` resides on a filesystem that breaks ChromaDB/SQLite.

    Returns the warning message (also logged at WARNING level) on the first
    call per path; subsequent calls are no-ops and return an empty string.
    Returns empty string on native Linux filesystems (ext4/xfs/btrfs/tmpfs).

    Specifically targeted at WSL2 users who symlink or point their palace
    onto a DrvFs-backed mount (``/mnt/c/…``). A dedicated VHD mounted at
    e.g. ``/mnt/data`` as ext4 is fine and never warns.
    """
    try:
        resolved = os.path.realpath(os.path.expanduser(path))
    except (OSError, ValueError):
        return ""
    if resolved in _storage_warned:
        return ""

    mount_info = _find_mount_for(resolved)
    if mount_info is None:
        return ""
    mountpoint, fstype, options = mount_info

    is_bad = fstype in _BAD_FSTYPES or "aname=drvfs" in options
    if not is_bad:
        return ""

    _storage_warned.add(resolved)
    msg = (
        f"palace at {resolved} is on {fstype} mount {mountpoint!r} "
        f"(options: {options[:80]}{'…' if len(options) > 80 else ''}) — "
        "ChromaDB + SQLite rely on mmap/flock/fsync semantics that DrvFs/9P/CIFS "
        "do not provide correctly. Writes may corrupt indices under concurrent "
        "mining. Relocate the palace onto a native Linux filesystem "
        "(e.g. a dedicated VHD mounted at /mnt/data as ext4, or $HOME on the "
        "WSL2 root filesystem)."
    )
    logger.warning(msg)
    return msg


class PalaceNotDeclared(Exception):
    """Raised when no palace is declared in env, walk-up yaml, or config.

    Step 7 of the palace-isolation design: there is no implicit fallback
    to the chat palace from arbitrary directories. The user must say
    where they want their reads/writes to land, either via ``--palace``,
    a ``mempalace.yaml`` walk-up file, or a ``default_palace`` entry in
    ``~/.mempalace/config.json``.
    """


def _maybe_migrate_legacy_palace_dir():
    """One-shot rename of ``~/.mempalace/palace/`` → ``~/.mempalace/palaces/chat/``.

    Matches the palace-isolation design: pre-isolation users had a single
    ``~/.mempalace/palace/`` directory that held every wing — campaign canon
    and chat-hook mining mixed together. We promote it to the canonical
    chat palace so hook-written state keeps a home after the upgrade, and
    register a ``chat`` alias so ``--palace chat`` works immediately.

    Guards:
    - Skip if legacy dir doesn't exist (clean install / already migrated).
    - Skip if destination already exists (user set up a new-style palace
      manually; don't clobber).
    - Best-effort: filesystem failures are swallowed so a broken upgrade
      can't brick every subsequent ``MempalaceConfig()`` call.
    """
    if not os.path.isdir(LEGACY_PALACE_DIR):
        return
    if os.path.exists(DEFAULT_PALACE_PATH):
        return
    try:
        os.makedirs(os.path.dirname(DEFAULT_PALACE_PATH), exist_ok=True)
        os.rename(LEGACY_PALACE_DIR, DEFAULT_PALACE_PATH)
    except OSError:
        return

    config_file = os.path.join(os.path.expanduser("~/.mempalace"), "config.json")
    if not os.path.isfile(config_file):
        return
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(cfg, dict):
        return

    changed = False
    pinned = cfg.get("palace_path")
    if isinstance(pinned, str):
        if os.path.abspath(os.path.expanduser(pinned)) == LEGACY_PALACE_DIR:
            cfg["palace_path"] = DEFAULT_PALACE_PATH
            changed = True

    aliases = cfg.get("palaces")
    if not isinstance(aliases, dict):
        aliases = {}
        cfg["palaces"] = aliases
        changed = True
    if "chat" not in aliases:
        aliases["chat"] = DEFAULT_PALACE_PATH
        changed = True

    # Seed ``default_palace`` so step 7's loud-fail doesn't strand a
    # pre-isolation user the moment they upgrade. If they had a custom
    # ``palace_path`` we honor it; if it matched the legacy dir we just
    # renamed (or was unset) we land on the chat alias, which now points
    # at the new dest.
    if "default_palace" not in cfg:
        if (
            isinstance(pinned, str)
            and pinned.strip()
            and os.path.abspath(os.path.expanduser(pinned)) != LEGACY_PALACE_DIR
        ):
            cfg["default_palace"] = pinned
        else:
            cfg["default_palace"] = "chat"
        changed = True

    if changed:
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except OSError:
            pass


def normalize_milvus_consistency_level(value) -> str:
    raw = str(value).strip() if value else DEFAULT_MILVUS_CONSISTENCY_LEVEL
    normalized = _MILVUS_CONSISTENCY_LEVELS.get(raw.lower())
    if normalized:
        return normalized
    allowed = ", ".join(_MILVUS_CONSISTENCY_LEVELS.values())
    raise ValueError(f"milvus_consistency_level must be one of: {allowed}")


def sqlite_read_uri(db_path: str) -> str:
    """Return a read-only ``file:`` URI for ``sqlite3.connect(..., uri=True)``.

    A bare ``f"file:{db_path}?mode=ro"`` mis-parses paths containing spaces or
    other URI-reserved characters — common in real home directories (a Windows
    user folder like ``First Last``, many macOS paths). ``pathname2url``
    percent-encodes the path and normalizes separators so the database opens on
    every platform.
    """
    from urllib.request import pathname2url

    db_path = os.fspath(db_path)
    return f"file:{pathname2url(db_path)}?mode=ro"


def _is_wal_without_sidecars(db_path: str) -> bool:
    """True for a WAL database whose ``-wal``/``-shm`` sidecars are both absent.

    Byte 18 of the SQLite header is the file format write version: 1 for a
    rollback journal, 2 for WAL. Reading it costs one open and answers the
    question a connection cannot answer without already being usable.
    """
    if os.path.exists(f"{db_path}-shm") or os.path.exists(f"{db_path}-wal"):
        return False
    try:
        with open(db_path, "rb") as handle:
            header = handle.read(19)
    except OSError:
        # Unreadable for another reason; let the normal open report it.
        return False
    return len(header) == 19 and header[:16] == b"SQLite format 3\x00" and header[18] == 2


def connect_sqlite_read(db_path: str, *, timeout: "float | None" = None):
    """Open ``db_path`` for reading, and keep reading when ``mode=ro`` cannot.

    A WAL database whose ``-wal`` and ``-shm`` sidecars are absent cannot be
    read through a read-only connection on every SQLite build: Apple's system
    library, which CPython links on macOS, accepts the connect and then fails
    the first statement with ``SQLITE_CANTOPEN``, because a read-only
    connection may not create the shared-memory index that WAL needs. The
    sidecars are absent exactly when nothing holds the palace open, which is
    the ordinary state before this process opens chroma, so a healthy palace
    came back as unreadable and the integrity gate refused every tool (#2489).

    Only that case takes a normal open, which creates the sidecars the
    read-only connection may not and takes SQLite's usual locks. ``immutable=1``
    would also open, but it disables locking and can read a torn page set from
    under a live writer, so it is not a substitute. Everything else keeps the
    read-only connection it had.
    """
    import sqlite3

    kwargs = {} if timeout is None else {"timeout": timeout}
    if _is_wal_without_sidecars(db_path):
        return sqlite3.connect(os.fspath(db_path), **kwargs)
    return sqlite3.connect(sqlite_read_uri(db_path), uri=True, **kwargs)


@lru_cache(maxsize=1)
def get_configured_collection_name() -> str:
    """Return the configured drawer collection name without repeated config-file reads."""
    return MempalaceConfig().collection_name


# Single source of truth for chunking defaults. ``mempalace.miner``
# imports these so the legacy module-level ``CHUNK_SIZE`` /
# ``CHUNK_OVERLAP`` / ``MIN_CHUNK_SIZE`` constants stay in sync with
# ``MempalaceConfig.chunk_*``. Putting them here (not in miner.py) keeps
# the config layer self-contained and avoids circular imports.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 50

DEFAULT_TOPIC_WINGS = [
    "emotions",
    "consciousness",
    "memory",
    "technical",
    "identity",
    "family",
    "creative",
]

DEFAULT_HALL_KEYWORDS = {
    "emotions": [
        "scared",
        "afraid",
        "worried",
        "happy",
        "sad",
        "love",
        "hate",
        "feel",
        "cry",
        "tears",
    ],
    "consciousness": [
        "consciousness",
        "conscious",
        "aware",
        "real",
        "genuine",
        "soul",
        "exist",
        "alive",
    ],
    "memory": ["memory", "remember", "forget", "recall", "archive", "palace", "store"],
    "technical": [
        "code",
        "python",
        "script",
        "bug",
        "error",
        "function",
        "api",
        "database",
        "server",
    ],
    "identity": ["identity", "name", "who am i", "persona", "self"],
    "family": [
        "family",
        "kids",
        "children",
        "daughter",
        "son",
        "parent",
        "mother",
        "father",
    ],
    "creative": [
        "game",
        "gameplay",
        "player",
        "app",
        "design",
        "art",
        "music",
        "story",
    ],
}


def _write_target(path: Path) -> Path:
    """The file a write should replace, following a symlink to it.

    A config kept in a dotfiles checkout is reached through a link, and the
    setters wrote through it. Replacing the link itself would leave the real
    file holding what it held and send this setting, and every later one,
    somewhere the user is not looking, so the temporary file, the rename and
    the quarantine all happen at the target instead. ``realpath`` follows the
    whole chain rather than one link, which is the file the reader would have
    got.

    A link this call cannot ``lstat`` is not reported as "not a link": that
    reading would send the write through ``os.replace`` and put a regular file
    where the link was. The error is raised instead, since the caller has to
    write somewhere and there is no safe guess about where.
    """
    try:
        is_link = stat.S_ISLNK(os.lstat(str(path)).st_mode)
    except FileNotFoundError:
        return path
    if is_link:
        return Path(os.path.realpath(str(path)))
    return path


def _fsync_directory(directory: Path) -> None:
    """Make the rename itself durable.

    ``EntityRegistry.save`` spells out why: on ext4 the kernel can acknowledge
    a rename and, after a crash, come back to the temporary file present and
    the target still holding the old bytes. Windows cannot open a directory
    this way at all, and answering nothing there is the same as answering
    nothing on a filesystem that does not implement it.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _keep_unreadable_file(path: Path):
    """Move a file whose contents did not parse aside, keeping its bytes.

    Renaming needs write permission on the directory rather than on the file,
    and the caller is left with a free name to write. ``FileNotFoundError``
    from the rename is the one outcome that establishes there was nothing to
    keep; every other failure leaves the file where it is and is raised,
    because a file that could not be moved is not one to write over.

    A file this process could not read at all never reaches here: that state
    declines the write outright rather than renaming anything.

    Returns the path the old file now lives at, or ``None`` when there was
    nothing there.
    """
    path = _write_target(path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fd, target = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f"{path.name}.unreadable-{stamp}-",
    )
    os.close(fd)
    try:
        os.replace(str(path), target)
    except FileNotFoundError:
        _unlink_quietly(target)
        return None
    except OSError:
        _unlink_quietly(target)
        raise
    return target


def _make_config_dir(directory: Path) -> None:
    """Create the config directory, restricted to the owner when this call makes it.

    ``mkdir(mode=...)`` is masked by the umask, so the mode is set afterwards,
    and only on a directory this call created: the config written here holds
    the user's ``people_map``, while a directory that was already there has
    whatever the user gave it and is not this function's to change. That is
    narrower than ``init()``, which sets the mode on an existing directory too.

    Raises whatever stopped it, which is what ``develop`` did: three setters
    and ``save_people_map`` created the directory outside any ``try``, so a
    directory they could not make came out as ``PermissionError``. Only
    ``set_hook_setting`` did not create it at all and returned instead, which
    is what ``tool_hook_settings`` relies on, and that one still returns: it
    reaches this through the writer, where the error becomes a message.
    """
    existed = directory.is_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if not existed:
        try:
            directory.chmod(0o700)
        except (OSError, NotImplementedError):
            pass  # Windows has no Unix permission bits; init() tolerates this too.


def _unlink_quietly(path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _write_json_in_place(path: Path, payload) -> None:
    """Write without the rename, for a directory that refuses a new name.

    This is what the setters did before this module wrote atomically. It is
    kept for the one case where the atomic write cannot run at all, since a
    setting that is lost outright is worse than one written without
    crash-safety.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _atomic_write_json(path: Path, payload) -> None:
    """Serialize ``payload`` into ``path`` through a temporary file.

    The rename is what publishes the new contents, so an interrupted write
    leaves the previous file exactly where it was rather than emptied or half
    serialized, and the directory is synced afterwards so the rename itself
    survives a crash, the way ``EntityRegistry.save`` does.

    The temporary file carries this process's pid, so two processes writing
    the same config never share one. A run killed between the write and the
    rename leaves that file behind, and nothing here removes it. It is opened
    ``O_NOFOLLOW``, so a symlink dropped at that name is not written through,
    and anything else in the way of that name sends the write to a name the
    directory picks rather than to a write without the rename.

    A directory that will not take a name it chose itself is the one case that
    falls back to writing in place: a read-only config directory whose config
    is still writable is what reaches it, and the setters wrote into it
    without complaint before this.
    """
    path = _write_target(path)
    tmp = str(path.with_name(f"{path.name}.tmp-{os.getpid()}"))
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(tmp, flags, 0o600)
        if os.fstat(fd).st_nlink > 1:
            # A hard link at that name is not a symlink, so ``O_NOFOLLOW`` lets
            # it through, and truncating through it would empty a file nobody
            # named here. The truncate happens below, after this has ruled that
            # out, and the write goes to a name the directory chose instead.
            os.close(fd)
            raise OSError(errno.EMLINK, "temporary name has another link", tmp)
    except OSError:
        # This errno belongs to the name, not to the directory. An orphan
        # another user's run left at that name, a directory dropped there, and
        # a symlink planted there all answer the way a directory that takes no
        # new names answers, and giving up the rename on that reading is how
        # the atomic write turns itself off where it was needed. Ask the
        # directory for a name of its own instead: what it says about a name
        # it chooses is about the directory.
        try:
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                raise
            # The message follows the write rather than announcing it: a
            # directory that refuses the temporary file often refuses the
            # write too, and saying the file was written in place before
            # finding that out is how the caller's failure message ends up
            # contradicted.
            _write_json_in_place(path, payload)
            print(
                f"  ! {path.parent} would not take a temporary file, so {path.name} was "
                "written in place and an interrupted write can truncate it",
                file=sys.stderr,
            )
            return
    try:
        # The mode is set before anything is written rather than after: under a
        # umask that clears the owner's write bit, ``O_CREAT`` leaves the file
        # at 0400, and one left behind by a killed run is then a name its own
        # owner cannot open next time.
        try:
            os.chmod(tmp, 0o600)
        except (OSError, NotImplementedError):
            pass
        os.ftruncate(fd, 0)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        _unlink_quietly(tmp)
        raise
    try:
        os.replace(tmp, str(path))
    except BaseException as exc:
        if not isinstance(exc, OSError):
            # A signal between the write and the rename leaves the temporary
            # file behind for no reason: nothing has been published yet.
            _unlink_quietly(tmp)
            raise
        # Opening a temporary file that already exists needs the file, not the
        # directory, so a run that reused an orphan at the pid name never asked
        # the directory anything. The rename is where the directory answers,
        # and a read-only one answers here rather than above.
        if exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            _unlink_quietly(tmp)
            raise
        # The temporary file holds this write, complete and fsynced. Removing
        # it before writing in place would trade a finished copy for a write
        # that truncates first, so it is removed after that write returns, and
        # named if it could not be.
        try:
            _write_json_in_place(path, payload)
        except BaseException:
            # That write truncates before it serializes, so what was there is
            # gone whether or not this one finished.
            print(
                f"  ! {path.name} was not written and may have been truncated; "
                f"{tmp} holds what this call was asked to save",
                file=sys.stderr,
            )
            raise
        _unlink_quietly(tmp)
        if os.path.exists(tmp):
            # Removing it needs the directory too, which is what just refused.
            print(
                f"  ! {tmp} holds a copy of what was written and could not be "
                "removed; nothing here removes it later either",
                file=sys.stderr,
            )
        print(
            f"  ! {path.parent} would not take the rename, so {path.name} was "
            "written in place and an interrupted write can truncate it",
            file=sys.stderr,
        )
        return
    _fsync_directory(path.parent)


class MempalaceConfig:
    """Configuration manager for MemPalace.

    Load order: env vars > config file > defaults.
    """

    def __init__(self, config_dir=None, palace_path=None):
        """Initialize config.

        Args:
            config_dir: Override config directory (useful for testing).
                        Defaults to the XDG-aware base directory — see
                        _default_config_dir() for the resolution order.
            palace_path: Explicit palace data directory. This is primarily
                         used by CLI operations that received ``--palace``;
                         it takes precedence over environment and file config.
        """
        if config_dir is None:
            _maybe_migrate_legacy_palace_dir()
        self._config_dir = Path(config_dir).expanduser() if config_dir else _default_config_dir()
        self._config_file = self._config_dir / "config.json"
        self._people_map_file = self._config_dir / "people_map.json"
        self._palace_path_override = (
            os.path.abspath(os.path.expanduser(str(palace_path)))
            if palace_path is not None
            else None
        )
        self._file_config = {}
        # What this process established about the file on disk, which decides
        # what the first setter is allowed to do with it. The defaults below
        # apply either way, as they always did, but they stop being written
        # over a file nobody read.
        #   "absent"   nothing resolved under that name
        #   "read"     this is its content
        #   "unparsed" it was read and is not a JSON object: a truncated write
        #              or a hand-edit that lost a brace lands here, and the
        #              first setter renames it aside before writing
        #   "unread"   it is there and could not be read at all, a permission
        #              bit or a directory at that name, and the first setter
        #              declines rather than writing defaults into it
        self._file_config_state = "absent"
        self._file_config_error = None
        try:
            raw = self._config_file.read_bytes()
        except FileNotFoundError:
            raw = None
        except OSError as exc:
            # Present, or unreachable: not proven absent, so not ours to lose.
            # A permission bit, a directory at that name, a symlink loop and a
            # share that stopped answering all arrive here, and the setter's
            # message names which one rather than guessing at permissions.
            raw = None
            self._file_config_state = "unread"
            self._file_config_error = exc
        if raw is not None:
            try:
                # ``utf-8-sig`` rather than ``utf-8``: a BOM is what a
                # Windows editor leaves on an otherwise valid config, and
                # ``json.loads`` on bytes accepts one, so decoding by hand has
                # to accept it too.
                loaded = json.loads(raw.decode("utf-8-sig"))
            except ValueError:
                # JSONDecodeError and UnicodeDecodeError are both ValueErrors:
                # text that is not JSON, and bytes that are not UTF-8.
                loaded = None
            if isinstance(loaded, dict):
                self._file_config = loaded
                self._file_config_state = "read"
            else:
                self._file_config_state = "unparsed"

    @property
    def search_config_fingerprint(self) -> str:
        """Stable digest of the effective search configuration for this process.

        A long-running Hub keeps this configuration snapshot and may also keep
        a collection opened from it.  CLI search forwarding compares this
        digest with a freshly loaded config so a changed ``config.json`` falls
        back to the direct path instead of querying stale Hub state.

        Hash only resolved settings that affect the currently selected search
        backend.  This detects relevant file edits and Hub-start environment
        overrides without treating hook/UI settings or inactive-backend values
        as stale search state.  The digest keeps secrets out of the registry.
        """
        try:
            # Match the backend selection used by palace.get_collection(),
            # including artifact auto-detection for an existing palace.
            from .palace import resolve_backend_name

            backend = resolve_backend_name(self.palace_path)
            backend_resolution_error = None
        except Exception as exc:  # noqa: BLE001 - fingerprint must remain total
            # A mismatched or otherwise invalid palace still needs a stable
            # digest so the Hub gate can fall back to the direct path, where
            # normal backend opening reports the actionable error.
            backend = self.backend
            backend_resolution_error = f"{type(exc).__name__}: {exc}"
        embedding_model = self.embedding_model
        effective = {
            "backend": backend,
            "collection_name": self.collection_name,
            "embedding_model": embedding_model,
            "lang_explicit": self.lang_explicit,
        }
        if backend_resolution_error is not None:
            effective["backend_resolution_error"] = backend_resolution_error
        if embedding_model == "openai-compat":
            effective.update(
                embedding_api_key=self.embedding_api_key,
                embedding_api_model=self.embedding_api_model,
                embedding_api_url=self.embedding_api_url,
            )
        else:
            effective.update(
                embedding_device=self.embedding_device,
                embedding_threads=self.embedding_threads,
            )
        try:
            if backend == "qdrant":
                effective.update(
                    qdrant_api_key=self.qdrant_api_key,
                    qdrant_namespace=self.qdrant_namespace,
                    qdrant_timeout=self.qdrant_timeout,
                    qdrant_url=self.qdrant_url,
                )
            elif backend == "milvus":
                effective.update(
                    milvus_consistency_level=self.milvus_consistency_level,
                    milvus_db_name=self.milvus_db_name,
                    milvus_namespace=self.milvus_namespace,
                    milvus_token=self.milvus_token,
                    milvus_uri=self.milvus_uri,
                )
            elif backend == "pgvector":
                effective.update(
                    pgvector_dsn=self.pgvector_dsn,
                    pgvector_namespace=self.pgvector_namespace,
                )
        except (TypeError, ValueError) as exc:
            # Invalid active-backend settings still need a stable digest. The
            # backend will report their validation error if search reaches it;
            # fingerprinting must never turn an unrelated config edit into a
            # CLI crash before the direct-path fallback can be selected.
            effective["backend_config_error"] = f"{type(exc).__name__}: {exc}"
        payload = json.dumps(
            effective, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _persist_file_config(self):
        """Write ``_file_config`` to ``config.json``, keeping what it replaces.

        A file this process never read is renamed aside first, so a setter
        called on defaults cannot be the last word on settings nobody could
        read. The write itself goes through a temporary file, because
        truncating the config and serializing into it afterwards is what
        produced the unreadable files in the first place.
        """
        if self._file_config_state == "unread":
            # The file is there and this process could not open it. Its
            # contents are still whatever they are; the settings in memory are
            # this session's defaults, and writing them here is how a
            # permission bit or an absent volume turns into a lost config.
            print(
                f"  ! Not writing {self._config_file}: it exists and could not be read "
                f"({self._file_config_error}), so it is not overwritten. Move it aside "
                "or fix what is in its way.",
                file=sys.stderr,
            )
            return
        try:
            _make_config_dir(self._config_dir)
        except OSError as exc:
            print(
                f"  ! Could not create {self._config_dir} ({exc}), so "
                f"{self._config_file.name} was not written",
                file=sys.stderr,
            )
            return
        if self._file_config_state == "unparsed":
            try:
                kept = _keep_unreadable_file(self._config_file)
            except OSError as exc:
                print(
                    f"  ! Not writing {self._config_file}: it does not parse "
                    f"and could not be moved aside ({exc})",
                    file=sys.stderr,
                )
                return
            self._file_config_state = "read"
            if kept is not None:
                print(
                    f"  ! {self._config_file} does not parse; kept it as {kept} "
                    "and started a new one",
                    file=sys.stderr,
                )
        try:
            _atomic_write_json(self._config_file, self._file_config)
        except OSError as exc:
            print(f"  ! Could not write {self._config_file}: {exc}", file=sys.stderr)

    @property
    def config_dir(self):
        """Active config directory (XDG-aware).

        Public accessor for the resolved config-dir path so callers outside
        ``config.py`` can derive sibling locations (write-ahead-log dir,
        cache dir, etc.) without reaching into ``_config_dir``.
        """
        return self._config_dir

    @property
    def palace_path(self):
        """Path to the memory palace data directory.

        Precedence: constructor override > ``MEMPALACE_PALACE_PATH`` >
        ``default_palace`` (alias or path) > ``palace_path`` in config.json >
        the fallback below.

        Defaults to the chat palace inside the active config directory
        (``<config dir>/palaces/chat``; ``~/.mempalace/palaces/chat`` unless
        ``MEMPALACE_CONFIG_DIR`` overrides it). Upstream's fallback is
        ``<config dir>/palace``, which under ``~/.mempalace`` is the legacy
        single-palace directory the palace-isolation migration renames away.
        """
        if self._palace_path_override is not None:
            return self._palace_path_override
        env_val = os.environ.get("MEMPALACE_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
        if env_val:
            # Normalize: expand ~ and collapse .. to match the CLI --palace
            # code path (mcp_server.py:62) and prevent surprise redirection
            # when the env var contains unresolved components.
            return os.path.abspath(os.path.expanduser(env_val))
        # ``default_palace`` is the declared fallback in palace-isolation.md's
        # precedence chain; ``palace_path`` is not in that chain. Honour it
        # first so every reader (MCP server, KG, layers, daemon) agrees with
        # the CLI's resolved_palace_path() on which palace is the default
        # (#44). Seeded installs set both to the chat palace, so they see no
        # change. An unknown alias must not raise inside a property: warn and
        # fall through to palace_path.
        default_ref = self._file_config.get("default_palace")
        if isinstance(default_ref, str) and default_ref.strip():
            try:
                return self.resolve_palace(default_ref)
            except ValueError as exc:
                logger.warning(
                    "default_palace %r in %s does not resolve (%s); using palace_path instead",
                    default_ref,
                    self._config_file,
                    exc,
                )
        return os.path.expanduser(
            self._file_config.get("palace_path", str(self._config_dir / "palaces" / "chat"))
        )

    @property
    def palaces(self):
        """Alias map: short name → palace path.

        Read from the ``palaces`` dict in ``~/.mempalace/config.json``.
        Values may be absolute, ``~``-prefixed, or relative; ``resolve_palace``
        normalizes them.
        """
        aliases = self._file_config.get("palaces", {})
        return aliases if isinstance(aliases, dict) else {}

    def resolve_palace(self, value):
        """Resolve a palace reference (alias or path) to an absolute path.

        - Path-like input (starts with ``/``, ``~``, or ``.``) is expanded
          and returned as an absolute path.
        - Bare names are looked up in the ``palaces`` alias map; unknown
          aliases raise :class:`ValueError` listing the known aliases.

        Used by every caller that takes a user-supplied palace reference
        (CLI ``--palace``, MCP tool ``palace`` arg, ``mempalace.yaml``
        ``palace:`` field).
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("palace reference must be a non-empty string")
        value = value.strip()

        if value.startswith(("/", "~", ".")):
            return os.path.abspath(os.path.expanduser(value))

        aliases = self.palaces
        if value in aliases:
            return os.path.abspath(os.path.expanduser(aliases[value]))

        known = sorted(aliases) if aliases else []
        known_str = ", ".join(known) if known else "none"
        raise ValueError(f"unknown palace alias: {value!r} (known aliases: {known_str})")

    def walk_up_palace(self, start_dir=None):
        """Walk up from ``start_dir`` looking for ``mempalace.yaml`` declaring a palace.

        Returns the resolved absolute palace path (via ``resolve_palace``) if a
        yaml is found with a usable ``palace:`` key; otherwise ``None``.

        Malformed yaml, missing key, or non-string values cause the walk to
        continue up — the caller gets ``None`` only after reaching the
        filesystem root with no match. An unknown-alias ValueError from
        ``resolve_palace`` is propagated: a declared-but-unresolvable palace
        is a user error, not a walk-past condition.
        """
        try:
            start = Path(start_dir).resolve() if start_dir else Path.cwd().resolve()
        except (OSError, RuntimeError):
            return None

        current = start
        for _ in range(256):
            yaml_path = current / "mempalace.yaml"
            if yaml_path.is_file():
                try:
                    import yaml

                    with open(yaml_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except (OSError, yaml.YAMLError):
                    data = None
                if isinstance(data, dict):
                    value = data.get("palace")
                    if isinstance(value, str) and value.strip():
                        return self.resolve_palace(value)
            if current.parent == current:
                break
            current = current.parent
        return None

    def resolved_palace_path(self, start_dir=None):
        """Resolve the active palace path using the full precedence chain.

        Order:
        1. ``MEMPALACE_PALACE_PATH`` / ``MEMPAL_PALACE_PATH`` env var
        2. Walk-up ``mempalace.yaml`` ``palace:`` key from ``start_dir`` (or CWD)
        3. ``default_palace`` in ``~/.mempalace/config.json`` (alias or path)

        If none of those declare a palace, raises :class:`PalaceNotDeclared`.
        Step 7 of the palace-isolation design eliminates the silent
        fall-through to the chat palace: users must declare where reads
        and writes land. To restore the old "chat is the default
        everywhere" behavior, set ``"default_palace": "chat"`` in
        ``~/.mempalace/config.json``.
        """
        env_val = os.environ.get("MEMPALACE_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
        if env_val:
            return os.path.abspath(os.path.expanduser(env_val))

        walk_result = self.walk_up_palace(start_dir=start_dir)
        if walk_result is not None:
            return walk_result

        default_ref = self._file_config.get("default_palace")
        if isinstance(default_ref, str) and default_ref.strip():
            return self.resolve_palace(default_ref)

        raise PalaceNotDeclared(
            "no palace declared — run from a directory containing "
            "`mempalace.yaml` with a `palace:` key, pass `--palace <alias-or-path>`, "
            "or set `default_palace` in ~/.mempalace/config.json"
        )

    @property
    def tunnel_file(self):
        """Path to the tunnel file, inside the palace directory (#48).

        It used to sit beside the palace (``dirname(palace_path)``), so every
        palace under ``~/.mempalace/palaces/`` shared one file and tunnels
        leaked across palaces.
        """
        return os.path.join(self.palace_path, TUNNELS_FILENAME)

    @property
    def hallway_file(self):
        """Path to the hallway file, inside the palace directory (#48).

        Mirrors ``tunnel_file``. It is not stored in the vector store, so it
        survives a store rebuild; repair and migrate carry it over when they
        replace the palace directory. Earlier locations -- the hardcoded
        ``~/.mempalace/hallways.json`` and the palace's parent directory --
        were shared by every palace on the host.
        """
        return os.path.join(self.palace_path, HALLWAYS_FILENAME)

    @property
    def collection_name(self):
        """ChromaDB collection name."""
        return self._file_config.get("collection_name", DEFAULT_COLLECTION_NAME)

    @property
    def backend(self):
        """Storage backend name.

        Read from ``config.json`` first, then ``MEMPALACE_BACKEND``, then
        ``"chroma"`` for backwards compatibility with existing palaces.
        """
        cfg_val = self._file_config.get("backend")
        if cfg_val:
            return str(cfg_val).strip().lower()
        env_val = os.environ.get("MEMPALACE_BACKEND")
        if env_val:
            return env_val.strip().lower()
        return DEFAULT_BACKEND

    @property
    def qdrant_url(self):
        """Qdrant endpoint for the opt-in ``qdrant`` backend.

        Defaults to localhost so selecting Qdrant never silently sends memory
        to a remote service. Users can point at a LAN or cloud endpoint via
        config or ``MEMPALACE_QDRANT_URL`` when they deliberately choose that.
        """
        env_val = os.environ.get("MEMPALACE_QDRANT_URL")
        if env_val:
            return env_val.strip()
        return str(self._file_config.get("qdrant_url", "http://localhost:6333")).strip()

    @property
    def qdrant_api_key(self):
        """API key for the opt-in ``qdrant`` backend, if configured."""
        env_val = os.environ.get("MEMPALACE_QDRANT_API_KEY")
        if env_val:
            return env_val
        value = self._file_config.get("qdrant_api_key")
        return str(value) if value else None

    @property
    def qdrant_namespace(self):
        """Optional Qdrant collection namespace/prefix."""
        env_val = os.environ.get("MEMPALACE_QDRANT_NAMESPACE")
        if env_val:
            return env_val.strip()
        value = self._file_config.get("qdrant_namespace")
        return str(value).strip() if value else None

    @property
    def qdrant_timeout(self):
        """Qdrant HTTP timeout in seconds."""
        env_val = os.environ.get("MEMPALACE_QDRANT_TIMEOUT")
        raw = env_val if env_val is not None else self._file_config.get("qdrant_timeout", 10.0)
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            timeout = 10.0
        return timeout if timeout > 0 else 10.0

    @property
    def milvus_uri(self):
        """Milvus endpoint for the opt-in ``milvus`` backend.

        Defaults to ``None`` so selecting Milvus uses per-palace Milvus Lite at
        ``<palace>/milvus.db``. Set this only to deliberately use a shared
        Milvus server, Zilliz Cloud, or a custom local Lite file.
        """
        env_val = os.environ.get("MEMPALACE_MILVUS_URI")
        if env_val:
            return env_val.strip()
        value = self._file_config.get("milvus_uri")
        return str(value).strip() if value else None

    @property
    def milvus_token(self):
        """Token for the opt-in ``milvus`` backend, if configured."""
        env_val = os.environ.get("MEMPALACE_MILVUS_TOKEN")
        if env_val:
            return env_val
        value = self._file_config.get("milvus_token")
        return str(value) if value else None

    @property
    def milvus_db_name(self):
        """Optional Milvus database name for the opt-in ``milvus`` backend."""
        env_val = os.environ.get("MEMPALACE_MILVUS_DB_NAME")
        if env_val:
            return env_val.strip()
        value = self._file_config.get("milvus_db_name")
        return str(value).strip() if value else None

    @property
    def milvus_namespace(self):
        """Optional Milvus collection namespace/prefix."""
        env_val = os.environ.get("MEMPALACE_MILVUS_NAMESPACE")
        if env_val:
            return env_val.strip()
        value = self._file_config.get("milvus_namespace")
        return str(value).strip() if value else None

    @property
    def milvus_consistency_level(self):
        """Milvus read consistency level for the opt-in ``milvus`` backend."""
        env_val = os.environ.get("MEMPALACE_MILVUS_CONSISTENCY_LEVEL")
        if env_val:
            return normalize_milvus_consistency_level(env_val)
        value = self._file_config.get("milvus_consistency_level")
        return normalize_milvus_consistency_level(value)

    @property
    def pgvector_dsn(self):
        """Postgres DSN for the opt-in ``pgvector`` backend.

        Defaults to a localhost DSN so selecting pgvector never silently sends
        memory to a remote database. Point at a LAN or cloud Postgres via config
        or ``MEMPALACE_PGVECTOR_DSN`` only when deliberately chosen.
        """
        env_val = os.environ.get("MEMPALACE_PGVECTOR_DSN")
        if env_val:
            return env_val.strip()
        return str(
            self._file_config.get("pgvector_dsn", "postgresql://localhost:5432/mempalace")
        ).strip()

    @property
    def pgvector_namespace(self):
        """Optional pgvector table namespace/prefix for multi-tenant isolation."""
        env_val = os.environ.get("MEMPALACE_PGVECTOR_NAMESPACE")
        if env_val:
            return env_val.strip()
        value = self._file_config.get("pgvector_namespace")
        return str(value).strip() if value else None

    @property
    def pgvector_shared_namespace(self):
        """Optional shared table namespace for a pgvector palace spanning hosts.

        By default the pgvector backend derives each table name from a hash of
        the palace's *local* path, so several machines pointed at one Postgres
        silently write to separate tables instead of sharing memory. Set this to
        any name the whole fleet agrees on (letters, digits and ``_ - . / :``
        or spaces) and every node resolves the same tables.

        Leave unset — the default — for single-machine palaces; table naming is
        then exactly as before and no migration is needed. It is orthogonal to
        ``pgvector_namespace``, which stays the tenant-isolation dimension.
        """
        env_val = os.environ.get("MEMPALACE_PGVECTOR_SHARED_NAMESPACE")
        if env_val:
            return env_val.strip()
        value = self._file_config.get("pgvector_shared_namespace")
        return str(value).strip() if value else None

    @property
    def people_map(self):
        """Mapping of name variants to canonical names."""
        if self._people_map_file.exists():
            try:
                with open(self._people_map_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                pass
        return self._file_config.get("people_map", {})

    @property
    def hooks_auto_save(self):
        """Whether the stop/precompact hooks should block for auto-save.

        When False, hooks pass through without blocking — equivalent to
        disabling auto-save while keeping hook scripts installed.
        """
        env_val = os.environ.get("MEMPALACE_HOOKS_AUTO_SAVE")
        if env_val is not None:
            return env_val.lower() not in ("false", "0", "no")
        hooks = self._file_config.get("hooks", {})
        return hooks.get("auto_save", True)

    @property
    def topic_wings(self):
        """List of topic wing names."""
        return self._file_config.get("topic_wings", DEFAULT_TOPIC_WINGS)

    @property
    def hall_keywords(self):
        """Mapping of hall names to keyword lists."""
        return self._file_config.get("hall_keywords", DEFAULT_HALL_KEYWORDS)

    @staticmethod
    def _try_coerce_int(value, minimum=None):
        """Coerce a raw config value to int, or ``None`` if it cannot be a
        valid setting.

        bool, empty/garbage string, non-numeric, and below-``minimum``
        values all return ``None``. Shared by ``_coerce_config_int``
        (which substitutes a documented default) and
        ``min_chunk_size_explicit`` (which must distinguish "unusable"
        from "explicitly set" without crashing the convo path).
        """
        if isinstance(value, bool):
            return None
        try:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return None
            value = int(value)
        except (TypeError, ValueError, OverflowError):
            # OverflowError: JSON ``1e1000`` parses to float('inf'), and
            # ``int(inf)`` raises it — still just garbage config, not a crash.
            return None
        if minimum is not None and value < minimum:
            return None
        return value

    def _coerce_config_int(self, key: str, default: int, minimum=None) -> int:
        """Read an int config value, falling back to ``default`` on bad input.

        Hand-edited ``config.json`` is the most common source of garbage:
        a string, a bool, a negative number, or a JSON null. None of those
        should crash mining or hang ``chunk_text()`` — fall back silently
        to the documented default rather than letting a typo break ingest.
        """
        coerced = self._try_coerce_int(self._file_config.get(key, default), minimum)
        return default if coerced is None else coerced

    def _validated_chunk_config(self):
        """Return ``(chunk_size, chunk_overlap, min_chunk_size)`` post-validation.

        Enforces the invariants the miner relies on:
          * ``chunk_size >= 1``
          * ``0 <= chunk_overlap <= chunk_size // 2``. A larger overlap can
            loop the miner forever on short-line content (#2056)
          * ``min_chunk_size <= chunk_size`` — otherwise no chunk is ever
            large enough to file, and ingest silently produces 0 drawers

        Repairs (rather than raises) on violation so a single bad
        config.json key doesn't take ingest down.
        """
        chunk_size = self._coerce_config_int("chunk_size", DEFAULT_CHUNK_SIZE, minimum=1)
        chunk_overlap = self._coerce_config_int("chunk_overlap", DEFAULT_CHUNK_OVERLAP, minimum=0)
        min_chunk_size = self._coerce_config_int(
            "min_chunk_size", DEFAULT_MIN_CHUNK_SIZE, minimum=0
        )

        if chunk_overlap > chunk_size // 2:
            # Overlap past half the chunk size can hang miner.chunk_text's
            # windowing loop on short-line content (#2056): a boundary pull can
            # shrink a chunk below chunk_overlap, so
            # ``start = end - chunk_overlap`` stops advancing. Repair to the
            # default when it is still at most half, else clamp to the largest
            # safe overlap.
            chunk_overlap = min(DEFAULT_CHUNK_OVERLAP, chunk_size // 2)

        if min_chunk_size > chunk_size:
            min_chunk_size = (
                DEFAULT_MIN_CHUNK_SIZE if DEFAULT_MIN_CHUNK_SIZE <= chunk_size else chunk_size
            )

        return chunk_size, chunk_overlap, min_chunk_size

    @property
    def chunk_size(self) -> int:
        """Characters per drawer chunk (validated, ``>= 1``)."""
        return self._validated_chunk_config()[0]

    @property
    def chunk_overlap(self) -> int:
        """Overlap between adjacent chunks (validated, ``<= chunk_size // 2``)."""
        return self._validated_chunk_config()[1]

    @property
    def min_chunk_size(self) -> int:
        """Minimum chunk size — skip smaller chunks (validated, ``<= chunk_size``)."""
        return self._validated_chunk_config()[2]

    @property
    def min_chunk_size_explicit(self):
        """Validated ``min_chunk_size`` iff the user explicitly set it.

        Returns the coerced int when ``config.json`` defines a usable
        ``min_chunk_size`` (``>= 0`` and ``<= chunk_size``); ``None`` when
        the key is absent/null or the value is unusable. ``convo_miner``
        relies on the ``None`` sentinel to keep its lower 30-char floor
        (more permissive than the 50-char project default, so short
        exchanges are not dropped) for untuned users while still honoring
        an explicit override —
        replacing the raw, unvalidated ``_file_config`` reach that crashed
        convo ingest on a bad key (#1024 review).
        """
        raw = self._file_config.get("min_chunk_size")
        if raw is None:
            return None
        coerced = self._try_coerce_int(raw, minimum=0)
        if coerced is None or coerced > self.chunk_size:
            return None
        return coerced

    @property
    def entity_languages(self):
        """Languages whose entity-detection patterns should be applied.

        Reads from env var ``MEMPALACE_ENTITY_LANGUAGES`` (comma-separated)
        first, then the ``entity_languages`` field in ``config.json``,
        defaulting to ``["en"]``.
        """
        env_val = os.environ.get("MEMPALACE_ENTITY_LANGUAGES") or os.environ.get(
            "MEMPAL_ENTITY_LANGUAGES"
        )
        if env_val:
            return [s.strip() for s in env_val.split(",") if s.strip()] or ["en"]
        cfg = self._file_config.get("entity_languages")
        if isinstance(cfg, list) and cfg:
            return [str(s) for s in cfg]
        return ["en"]

    def set_entity_languages(self, languages):
        """Persist the entity-detection language list to ``config.json``."""
        normalized = [s.strip() for s in languages if s and s.strip()]
        if not normalized:
            normalized = ["en"]
        self._file_config["entity_languages"] = normalized
        # ``develop`` created the directory here, outside any ``try``, so this
        # setter raised when it could not be made. ``set_hook_setting`` did not
        # create it at all and returned instead, and that difference is kept:
        # ``tool_hook_settings`` does not wrap its call.
        _make_config_dir(self._config_dir)
        self._persist_file_config()
        return normalized

    @property
    def embedding_device(self):
        """Hardware device for the ONNX embedding model.

        Values: ``"auto"`` (default), ``"cpu"``, ``"cuda"``, ``"coreml"``,
        ``"dml"``. Read from env ``MEMPALACE_EMBEDDING_DEVICE`` first, then
        ``embedding_device`` in ``config.json``, then ``"auto"``.

        ``auto`` resolves to the first available accelerator at runtime via
        :mod:`mempalace.embedding`; requesting an unavailable accelerator
        logs a warning and falls back to CPU.
        """
        env_val = os.environ.get("MEMPALACE_EMBEDDING_DEVICE")
        if env_val:
            return env_val.strip().lower()
        return str(self._file_config.get("embedding_device", "auto")).strip().lower()

    @property
    def embedding_provider(self):
        """Which embedding backend to use.

        Values:
          * ``"onnx"`` (default) — local ONNX MiniLM, 384-dim.
          * ``"ollama"`` — Ollama ``/api/embed`` HTTP shape; model and dim
            chosen by :attr:`embedding_model`.
          * ``"openai-compat"`` — OpenAI ``/v1/embeddings`` shape (vLLM,
            LM Studio, Together, OpenAI itself, etc.); model and dim
            chosen by :attr:`embedding_model`.

        Switching providers is a destructive operation: existing palaces hold
        vectors at the old provider's dimensionality, and the EF identity is
        persisted on each collection. After flipping this setting, wipe the
        palace and re-mine.

        Read priority: env ``MEMPALACE_EMBEDDING_PROVIDER`` first, then the
        ``embedding_provider`` key in ``config.json``, then ``"onnx"``.
        """
        env_val = os.environ.get("MEMPALACE_EMBEDDING_PROVIDER")
        if env_val:
            return env_val.strip().lower()
        return str(self._file_config.get("embedding_provider", "onnx")).strip().lower()

    @property
    def embedding_model(self):
        """Embedding model identifier — meaning depends on :attr:`embedding_provider`.

        * ``provider == "onnx"``: selects the local ONNX model — ``"minilm"``
          (all-MiniLM-L6-v2, English-only) or ``"embeddinggemma"``
          (multilingual, 100+ languages). Compared case-insensitively in
          :func:`mempalace.embedding.get_embedding_function`. New installs get
          ``embeddinggemma`` written by onboarding (:meth:`set_embedding_model`);
          the back-compat default is ``"minilm"``.
        * ``provider == "ollama"`` / ``"openai-compat"``: the remote model name
          (e.g. ``"nomic-embed-text"``), passed verbatim to the endpoint. See
          ``embedding_api_url`` / ``embedding_api_model`` / ``embedding_api_key``
          for the ``openai-compat`` endpoint's own settings.

        Read priority: env ``MEMPALACE_EMBEDDING_MODEL`` first, then the
        ``embedding_model`` key in ``config.json``, then a provider-specific
        default. Switching models on an existing palace requires re-embedding
        (``mempalace repair rebuild-index``) — different vector space.
        """
        env_val = os.environ.get("MEMPALACE_EMBEDDING_MODEL")
        if env_val:
            return env_val.strip().lower()
        cfg_val = self._file_config.get("embedding_model")
        if cfg_val:
            return str(cfg_val).strip().lower()
        provider = self.embedding_provider
        if provider == "ollama":
            return "nomic-embed-text"
        if provider == "openai-compat":
            return "nomic-ai/nomic-embed-text-v1.5"
        return "minilm"

    @property
    def embedding_endpoint(self):
        """Endpoint URL for remote embedding providers.

        Ignored when :attr:`embedding_provider` is ``"onnx"``.

        Default for Ollama: ``"http://localhost:11434"``. Set to a Tailscale
        / LAN address (e.g. ``http://192.168.1.147:11434``) to offload
        embedding to a GPU box like a DGX Spark while keeping the rest of
        MemPalace local. Read priority: env ``MEMPALACE_EMBEDDING_ENDPOINT``
        first, then the ``embedding_endpoint`` key in ``config.json``, then
        the provider-specific default.
        """
        env_val = os.environ.get("MEMPALACE_EMBEDDING_ENDPOINT")
        if env_val:
            return env_val.strip().rstrip("/")
        cfg_val = self._file_config.get("embedding_endpoint")
        if cfg_val:
            return str(cfg_val).strip().rstrip("/")
        provider = self.embedding_provider
        if provider == "ollama":
            return "http://localhost:11434"
        if provider == "openai-compat":
            return "http://localhost:8000"
        return ""

    @property
    def llm_provider(self):
        """Which LLM backend to use for refinement / classification / closet scoring.

        Values:
          * ``"ollama"`` (default) — Ollama ``/api/chat`` HTTP shape; model
            chosen by :attr:`llm_model`.
          * ``"openai-compat"`` — OpenAI ``/v1/chat/completions`` shape (vLLM,
            LM Studio, Together, OpenAI itself, etc.).
          * ``"anthropic"`` — direct Anthropic API; requires
            ``ANTHROPIC_API_KEY`` env (or ``MEMPALACE_LLM_API_KEY``).

        Mirrors :attr:`embedding_provider` so a user can persist a Spark / LAN
        LLM endpoint in config rather than passing ``--llm-endpoint`` on every
        invocation. Read priority: env ``MEMPALACE_LLM_PROVIDER`` first, then
        the ``llm_provider`` key in ``config.json``, then ``"ollama"``.
        """
        env_val = os.environ.get("MEMPALACE_LLM_PROVIDER")
        if env_val:
            return env_val.strip().lower()
        return str(self._file_config.get("llm_provider", "ollama")).strip().lower()

    @property
    def llm_model(self):
        """Model name for the LLM provider.

        Default ``"gemma4:e4b"`` matches the historical CLI default at
        ``mempalace.cli`` so behavior is unchanged when the key is unset.
        Read priority: env ``MEMPALACE_LLM_MODEL`` first, then the
        ``llm_model`` key in ``config.json``, then ``"gemma4:e4b"``.
        """
        env_val = os.environ.get("MEMPALACE_LLM_MODEL")
        if env_val:
            return env_val.strip()
        cfg_val = self._file_config.get("llm_model")
        if cfg_val:
            return str(cfg_val).strip()
        return "gemma4:e4b"

    @property
    def llm_endpoint(self):
        """Endpoint URL for the LLM provider, or ``None`` for provider default.

        ``None`` lets each provider apply its own default
        (Ollama → ``http://localhost:11434``; Anthropic → ``api.anthropic.com``).
        Set to a Tailscale / LAN address (e.g. ``http://192.168.1.147:8001``)
        to point LLM refinement at a Spark vLLM container while keeping the
        rest of MemPalace local. Read priority: env ``MEMPALACE_LLM_ENDPOINT``
        first, then the ``llm_endpoint`` key in ``config.json``, then ``None``.
        """
        env_val = os.environ.get("MEMPALACE_LLM_ENDPOINT")
        if env_val:
            return env_val.strip().rstrip("/")
        cfg_val = self._file_config.get("llm_endpoint")
        if cfg_val:
            return str(cfg_val).strip().rstrip("/")
        return None

    @property
    def llm_api_key(self):
        """API key for the LLM provider, env-only (never persisted to file).

        Mirrors how ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` are consumed in
        :mod:`mempalace.llm_client`. Read from ``MEMPALACE_LLM_API_KEY``; falls
        back to ``None`` so each provider can pick up its native env var.
        """
        env_val = os.environ.get("MEMPALACE_LLM_API_KEY")
        return env_val.strip() if env_val else None

    @property
    def workers(self):
        """Producer thread count for the parallel miner.

        Asymmetric default:
          * ``embedding_provider == "onnx"`` → ``1``. ONNX runs in-process
            under the GIL; thread fan-out has no benefit and risks GIL
            contention on the chunking/metadata work.
          * Remote providers (``ollama`` / ``openai-compat``) → ``8``. The
            EF call releases the GIL while waiting on the socket, so N
            producers saturate the remote endpoint.

        Override via ``MEMPALACE_WORKERS`` env var or the ``workers`` config
        key. Clamped to ``[1, 64]``. ``workers=1`` reproduces the historic
        serial mine.
        """
        env_val = os.environ.get("MEMPALACE_WORKERS")
        if env_val:
            try:
                return max(1, min(64, int(env_val)))
            except ValueError:
                pass
        cfg_val = self._file_config.get("workers")
        if cfg_val is not None:
            try:
                return max(1, min(64, int(cfg_val)))
            except (TypeError, ValueError):
                pass
        return 1 if self.embedding_provider == "onnx" else 8

    @property
    def embedding_threads(self) -> int:
        """Cap on the embedder's ONNX Runtime intra-op thread pool (#1068).

        ChromaDB's ONNX embedder builds its ``InferenceSession`` with no thread
        cap, so the intra-op pool defaults to the physical core count and a
        background ``mine`` pins every core — stacked Stop-hook fires turn into
        thermal events. ``OMP_NUM_THREADS`` is inert here (ORT owns its own
        pool), so the cap is applied via ``SessionOptions`` in
        :mod:`mempalace.embedding`.

        Read from env ``MEMPALACE_EMBEDDING_THREADS`` first, then
        ``embedding_threads`` in ``config.json``. Semantics:

        - unset / ``"auto"`` → half the logical CPUs (min 1), so a background
          mine leaves the machine usable out of the box.
        - a positive integer → exactly that many intra-op threads.
        - ``0`` or negative → uncapped: ORT's default (physical core count),
          for users who want maximum indexing throughput.
        """
        raw = os.environ.get("MEMPALACE_EMBEDDING_THREADS")
        if raw is None:
            raw = self._file_config.get("embedding_threads")
        if raw is None or str(raw).strip().lower() in ("", "auto"):
            return max(1, (os.cpu_count() or 2) // 2)
        try:
            val = int(str(raw).strip())
        except (TypeError, ValueError):
            return max(1, (os.cpu_count() or 2) // 2)
        return val if val > 0 else 0

    @property
    def embeddinggemma_batch_size(self) -> int:
        """Documents per ``session.run()`` for the EmbeddingGemma ONNX model (#2330).

        The sub-batching added for #1770 bounds a run by document COUNT, not
        allocation: attention buffers scale with ``batch * padded_len ** 2``, so
        a batch of long documents can still exceed available memory at the
        module default (``mempalace.embedding._EMBEDDINGGEMMA_BATCH_SIZE``, 32).
        Read from env ``MEMPALACE_EMBEDDINGGEMMA_BATCH_SIZE`` first, then
        ``embeddinggemma_batch_size`` in ``config.json``, then the module
        default. Unset, non-numeric, or non-positive values fall back to the
        default rather than raising; ``EmbeddinggemmaONNX.__init__`` still
        raises on an explicitly-passed non-positive ``batch_size``.
        """
        from .embedding import _EMBEDDINGGEMMA_BATCH_SIZE

        raw = os.environ.get("MEMPALACE_EMBEDDINGGEMMA_BATCH_SIZE")
        if raw is None:
            raw = self._file_config.get("embeddinggemma_batch_size")
        if raw is None:
            return _EMBEDDINGGEMMA_BATCH_SIZE
        try:
            val = int(str(raw).strip())
        except (TypeError, ValueError):
            return _EMBEDDINGGEMMA_BATCH_SIZE
        return val if val > 0 else _EMBEDDINGGEMMA_BATCH_SIZE

    @property
    def hybrid_rank_vector_weight(self) -> float:
        """Weight of the vector (embedding-similarity) signal in the hybrid
        re-rank (``searcher._hybrid_rank``).

        Read from env ``MEMPALACE_HYBRID_VECTOR_WEIGHT`` first, then
        ``hybrid_rank_vector_weight`` in ``config.json``, then the built-in
        default (``0.6``). Unset, non-numeric, infinite, or negative values
        fall back to the default rather than raising — a hand-edited
        ``config.json`` shouldn't take retrieval down. The default reproduces
        the previously-hardcoded weight, so leaving it unset is a
        byte-identical blend (#2298).
        """
        return self._resolve_float_setting(
            "MEMPALACE_HYBRID_VECTOR_WEIGHT",
            "hybrid_rank_vector_weight",
            DEFAULT_HYBRID_VECTOR_WEIGHT,
        )

    @property
    def hybrid_rank_bm25_weight(self) -> float:
        """Weight of the BM25 (lexical) signal in the hybrid re-rank
        (``searcher._hybrid_rank``).

        Read from env ``MEMPALACE_HYBRID_BM25_WEIGHT`` first, then
        ``hybrid_rank_bm25_weight`` in ``config.json``, then the built-in
        default (``0.4``). Unset, non-numeric, infinite, or negative values
        fall back to the default rather than raising (#2298).
        """
        return self._resolve_float_setting(
            "MEMPALACE_HYBRID_BM25_WEIGHT",
            "hybrid_rank_bm25_weight",
            DEFAULT_HYBRID_BM25_WEIGHT,
        )

    def set_embedding_model(self, model: str) -> None:
        """Persist the embedding-model choice to ``config.json``.

        Onboarding calls this once on first run. Accepts ``"minilm"`` or
        ``"embeddinggemma"``; other values are normalized to lowercase and
        passed through (``embedding.get_embedding_function`` falls back to
        minilm for unrecognized values).
        """
        self._file_config["embedding_model"] = str(model).strip().lower()
        # ``develop`` created the directory here, outside any ``try``, so this
        # setter raised when it could not be made. ``set_hook_setting`` did not
        # create it at all and returned instead, and that difference is kept:
        # ``tool_hook_settings`` does not wrap its call.
        _make_config_dir(self._config_dir)
        self._persist_file_config()

    def set_backend(self, backend: str) -> None:
        """Persist the storage backend choice to ``config.json``."""
        backend = str(backend).strip().lower()
        from .backends import get_backend_class

        get_backend_class(backend)
        self._file_config["backend"] = backend
        # ``develop`` created the directory here, outside any ``try``, so this
        # setter raised when it could not be made. ``set_hook_setting`` did not
        # create it at all and returned instead, and that difference is kept:
        # ``tool_hook_settings`` does not wrap its call.
        _make_config_dir(self._config_dir)
        self._persist_file_config()

    def _resolve_str_setting(self, env_var: str, config_key: str):
        """Resolve a string setting: env var > ``config.json`` > ``None``.

        Whitespace-only values are treated as unset, so a blank env var or a
        hand-edited empty config key doesn't mask the value below it. Unlike
        ``embedding_model`` the result is not lower-cased — URLs, model ids,
        and API keys are case-sensitive.
        """
        env_val = os.environ.get(env_var)
        if env_val and env_val.strip():
            return env_val.strip()
        cfg_val = self._file_config.get(config_key)
        if isinstance(cfg_val, str) and cfg_val.strip():
            return cfg_val.strip()
        return None

    def _resolve_float_setting(self, env_var: str, config_key: str, default: float) -> float:
        """Resolve a float setting: env var > ``config.json`` > ``default``.

        Whitespace-only values are treated as unset, so a blank env var or a
        hand-edited empty config key doesn't mask the value below it. A set
        value that is not a finite, non-negative number falls back to
        ``default`` rather than raising — a stray type or non-numeric string
        in ``config.json`` shouldn't take retrieval down (mirrors
        ``embedding_threads`` / ``embeddinggemma_batch_size``).
        """
        raw = os.environ.get(env_var)
        if raw is None or not str(raw).strip():
            raw = self._file_config.get(config_key)
        if raw is None:
            return default
        try:
            val = float(str(raw).strip())
        except (TypeError, ValueError):
            return default
        if not math.isfinite(val) or val < 0:
            return default
        return val

    @property
    def embedding_api_url(self):
        """Base URL of the OpenAI-compatible ``/v1/embeddings`` endpoint.

        Used only when ``embedding_model == "openai-compat"``. Resolved from
        env ``MEMPALACE_EMBEDDING_API_URL`` first, then ``embedding_api_url``
        in ``config.json``; ``None`` when unset. Accepts a bare host, a
        ``…/v1`` base, or a full endpoint URL.
        """
        return self._resolve_str_setting("MEMPALACE_EMBEDDING_API_URL", "embedding_api_url")

    @property
    def embedding_api_model(self):
        """Server-side model id for the ``openai-compat`` embeddings endpoint.

        Resolved from env ``MEMPALACE_EMBEDDING_API_MODEL`` first, then
        ``embedding_api_model`` in ``config.json``; ``None`` when unset.
        """
        return self._resolve_str_setting("MEMPALACE_EMBEDDING_API_MODEL", "embedding_api_model")

    @property
    def embedding_api_key(self):
        """Optional bearer token / API key for the embeddings endpoint.

        Resolved from env ``MEMPALACE_EMBEDDING_API_KEY`` first, then
        ``embedding_api_key`` in ``config.json``; ``None`` when unset (for
        local endpoints that need no auth).
        """
        return self._resolve_str_setting("MEMPALACE_EMBEDDING_API_KEY", "embedding_api_key")

    @property
    def topic_tunnel_min_count(self):
        """Minimum number of overlapping confirmed topics required to create
        a cross-wing tunnel between two wings.

        Default is ``1`` — any single shared topic produces a tunnel. Bump
        to ``2+`` if your projects share lots of common-tech labels (Python,
        Docker, Git) and you want only meaningfully overlapping wings to
        link. Reads ``MEMPALACE_TOPIC_TUNNEL_MIN_COUNT`` env first, then the
        config-file value, then ``1``.
        """
        env_val = os.environ.get("MEMPALACE_TOPIC_TUNNEL_MIN_COUNT")
        if env_val:
            try:
                parsed = int(env_val)
                if parsed >= 1:
                    return parsed
            except ValueError:
                pass
        cfg_val = self._file_config.get("topic_tunnel_min_count")
        try:
            parsed = int(cfg_val) if cfg_val is not None else 1
        except (TypeError, ValueError):
            parsed = 1
        return max(1, parsed)

    @property
    def max_backups(self) -> int:
        """Number of timestamped palace backups to retain before pruning.

        Applies to the accumulating, timestamped backups created by
        ``mempalace migrate`` (``<palace>.pre-migrate.<timestamp>``) and
        ``mempalace repair max-seq-id``
        (``chroma.sqlite3.max-seq-id-backup-<timestamp>``). Each of those
        commands writes a fresh full-size copy every run and historically
        never deleted the old ones, so on a machine that mines or repairs on
        a schedule the backup set could silently grow until it filled the
        disk. After each backup is written, copies beyond this count (oldest
        first) are removed.

        Reads ``MEMPALACE_MAX_BACKUPS`` env first, then ``max_backups`` in
        ``config.json``, then the default of ``10``. A value of ``0`` disables
        pruning and keeps every backup (use when an external retention policy
        manages cleanup). Negative or non-numeric values fall back to the
        default rather than crashing migrate/repair.
        """
        env_val = os.environ.get("MEMPALACE_MAX_BACKUPS")
        if env_val is not None:
            coerced = self._try_coerce_int(env_val, minimum=0)
            if coerced is not None:
                return coerced
        coerced = self._try_coerce_int(
            self._file_config.get("max_backups", DEFAULT_MAX_BACKUPS), minimum=0
        )
        return DEFAULT_MAX_BACKUPS if coerced is None else coerced

    @property
    def lang_explicit(self):
        """Primary language code when explicitly configured, else ``None``.

        Resolution order: ``MEMPALACE_LANG`` / ``MEMPAL_LANG`` env var, then
        ``config.json["lang"]``. Returns ``None`` if neither is set. Use this
        when a caller needs to know whether the user has opted in to locale
        behaviour (e.g. to avoid silently changing search scoring for palaces
        that have never set a language).
        """
        env_val = os.environ.get("MEMPALACE_LANG") or os.environ.get("MEMPAL_LANG")
        if env_val and env_val.strip():
            return env_val.strip()
        cfg = self._file_config.get("lang")
        if isinstance(cfg, str) and cfg.strip():
            return cfg.strip()
        return None

    @property
    def lang(self):
        """Primary language code for localized output and display.

        Resolution order: ``lang_explicit`` (env or config.json), first entry
        of ``entity_languages``, then ``"en"``. Always returns a non-empty
        string so callers that need a language for display purposes never
        have to handle ``None``. Code paths that must not silently change
        behaviour for unconfigured palaces should read ``lang_explicit``
        instead.
        """
        explicit = self.lang_explicit
        if explicit:
            return explicit
        entity_langs = self.entity_languages
        if entity_langs:
            return entity_langs[0]
        return "en"

    @property
    def hook_silent_save(self):
        """Whether the stop hook saves directly (True) or blocks for MCP calls (False)."""
        return self._file_config.get("hooks", {}).get("silent_save", True)

    @property
    def hook_desktop_toast(self):
        """Whether the stop hook shows a desktop notification via notify-send."""
        return self._file_config.get("hooks", {}).get("desktop_toast", False)

    def resolve_write_routing(self, scope: str) -> ResolvedWriteRoutingPolicy:
        """Resolve the configured write policy for ``hooks`` or ``cli``.

        Precedence is:

        1. scope-specific environment variable;
        2. global environment variable;
        3. legacy hook environment variable;
        4. scope-specific config value;
        5. global config value;
        6. legacy hook config value;
        7. ``direct``.

        This foundation does not change current hook or CLI behavior. The
        policy-aware consumers are introduced by follow-up PRs.
        """

        normalized_scope = str(scope).strip().lower()
        env_names = {
            "hooks": "MEMPALACE_HOOK_WRITE_ROUTING",
            "cli": "MEMPALACE_CLI_WRITE_ROUTING",
        }

        if normalized_scope not in env_names:
            raise WriteRoutingError("write routing scope must be 'hooks' or 'cli'")

        routing_config = self._file_config.get("write_routing", {})
        if routing_config is None:
            routing_config = {}

        if not isinstance(routing_config, dict):
            raise WriteRoutingError("config write_routing must be an object")

        candidates = [
            RoutingPolicyCandidate(
                env_names[normalized_scope],
                os.environ.get(env_names[normalized_scope]),
            ),
            RoutingPolicyCandidate(
                "MEMPALACE_WRITE_ROUTING",
                os.environ.get("MEMPALACE_WRITE_ROUTING"),
            ),
        ]

        if normalized_scope == "hooks":
            candidates.append(
                RoutingPolicyCandidate(
                    "MEMPALACE_HOOKS_DAEMON (legacy)",
                    os.environ.get("MEMPALACE_HOOKS_DAEMON"),
                    legacy_boolean=True,
                )
            )

        candidates.extend(
            [
                RoutingPolicyCandidate(
                    f"config write_routing.{normalized_scope}",
                    routing_config.get(normalized_scope),
                ),
                RoutingPolicyCandidate(
                    "config write_routing.default",
                    routing_config.get("default"),
                ),
            ]
        )

        if normalized_scope == "hooks":
            hooks_config = self._file_config.get("hooks", {})
            if hooks_config is None:
                hooks_config = {}

            if not isinstance(hooks_config, dict):
                raise WriteRoutingError("config hooks must be an object")

            candidates.append(
                RoutingPolicyCandidate(
                    "config hooks.daemon (legacy)",
                    hooks_config.get("daemon"),
                    legacy_boolean=True,
                )
            )

        return resolve_write_routing_policy(candidates)

    @property
    def hook_write_routing(self) -> WriteRoutingPolicy:
        """Resolved future routing policy for hook-triggered writes."""

        return self.resolve_write_routing("hooks").policy

    @property
    def cli_write_routing(self) -> WriteRoutingPolicy:
        """Resolved future routing policy for routine CLI writes."""

        return self.resolve_write_routing("cli").policy

    @property
    def hook_use_daemon(self):
        """Whether hooks should submit save/mine work to the opt-in daemon."""
        env_val = os.environ.get("MEMPALACE_HOOKS_DAEMON")
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes", "on")
        value = self._file_config.get("hooks", {}).get("daemon", False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return value == 1

    def set_hook_setting(self, key: str, value: bool):
        """Update a hook setting and write config to disk."""
        if "hooks" not in self._file_config:
            self._file_config["hooks"] = {}
        self._file_config["hooks"][key] = value
        self._persist_file_config()

    def init(self):
        """Create config directory and write default config.json if it doesn't exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        # Restrict directory permissions to owner only (Unix)
        try:
            self._config_dir.chmod(0o700)
        except (OSError, NotImplementedError):
            pass  # Windows doesn't support Unix permissions
        if not self._config_file.exists():
            # Chunking parameters (chunk_size, chunk_overlap, min_chunk_size)
            # are intentionally NOT written here — convo_miner.py distinguishes
            # "user has tuned this" from "user is on defaults" by checking
            # ``_file_config.get("min_chunk_size") is None``. Writing the
            # miner.py defaults (50) into config.json breaks that detection
            # and silently overrides convo_miner's stricter 30-char floor,
            # dropping legitimate short conversation exchanges. Module-level
            # defaults already apply correctly when these keys are absent.
            default_config = {
                "palace_path": DEFAULT_PALACE_PATH,
                "default_palace": "chat",
                "palaces": {"chat": DEFAULT_PALACE_PATH},
                "collection_name": DEFAULT_COLLECTION_NAME,
                "topic_wings": DEFAULT_TOPIC_WINGS,
                "hall_keywords": DEFAULT_HALL_KEYWORDS,
            }
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            # Restrict config file to owner read/write only
            try:
                self._config_file.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
        return self._config_file

    def save_people_map(self, people_map):
        """Write people_map.json to config directory.

        Args:
            people_map: Dict mapping name variants to canonical names.
        """
        _make_config_dir(self._config_dir)
        _atomic_write_json(self._people_map_file, people_map)
        return self._people_map_file
