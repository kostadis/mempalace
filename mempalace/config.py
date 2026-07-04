"""
MemPalace configuration system.

Priority: env vars > config file (~/.mempalace/config.json) > defaults
"""

import json
import logging
import os
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

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


LEGACY_PALACE_DIR = os.path.expanduser("~/.mempalace/palace")
DEFAULT_PALACE_PATH = os.path.expanduser("~/.mempalace/palaces/chat")
DEFAULT_COLLECTION_NAME = "mempalace_drawers"
DEFAULT_BACKEND = "chroma"

# How many timestamped palace backups to retain before the oldest are
# pruned. Applies to the accumulating backups written by ``mempalace
# migrate`` and ``mempalace repair max-seq-id`` — see
# ``MempalaceConfig.max_backups``.
DEFAULT_MAX_BACKUPS = 10


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


class MempalaceConfig:
    """Configuration manager for MemPalace.

    Load order: env vars > config file > defaults.
    """

    def __init__(self, config_dir=None):
        """Initialize config.

        Args:
            config_dir: Override config directory (useful for testing).
                        Defaults to ~/.mempalace.
        """
        if config_dir is None:
            _maybe_migrate_legacy_palace_dir()
        self._config_dir = (
            Path(config_dir) if config_dir else Path(os.path.expanduser("~/.mempalace"))
        )
        self._config_file = self._config_dir / "config.json"
        self._people_map_file = self._config_dir / "people_map.json"
        self._file_config = {}

        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._file_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._file_config = {}

    @property
    def palace_path(self):
        """Path to the memory palace data directory."""
        env_val = os.environ.get("MEMPALACE_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
        if env_val:
            # Normalize: expand ~ and collapse .. to match the CLI --palace
            # code path (mcp_server.py:62) and prevent surprise redirection
            # when the env var contains unresolved components.
            return os.path.abspath(os.path.expanduser(env_val))
        return self._file_config.get("palace_path", DEFAULT_PALACE_PATH)

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
        """Path to the tunnel file, sibling of palace_path."""
        return os.path.join(os.path.dirname(self.palace_path), "tunnels.json")

    @property
    def hallway_file(self):
        """Path to the hallway file, sibling of palace_path.

        Mirrors ``tunnel_file`` so within-wing hallway state is scoped to the
        configured palace and survives palace rebuilds (it does not live in
        ChromaDB which can be recreated). Prior to this property the path was
        hardcoded under ``~/.mempalace/hallways.json`` and multiple palaces on
        one host silently shared one file (see ``hallways._legacy_hallway_file``).
        """
        return os.path.join(os.path.dirname(self.palace_path), "hallways.json")

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
    def people_map(self):
        """Mapping of name variants to canonical names."""
        if self._people_map_file.exists():
            try:
                with open(self._people_map_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
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
          * ``0 <= chunk_overlap < chunk_size`` — equality would loop forever
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

        if chunk_overlap >= chunk_size:
            chunk_overlap = (
                DEFAULT_CHUNK_OVERLAP
                if DEFAULT_CHUNK_OVERLAP < chunk_size
                else max(0, chunk_size - 1)
            )

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
        """Overlap between adjacent chunks (validated, ``< chunk_size``)."""
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
        self._config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._file_config, f, indent=2, ensure_ascii=False)
        except OSError:
            pass
        try:
            self._config_file.chmod(0o600)
        except (OSError, NotImplementedError):
            pass
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
          (e.g. ``"nomic-embed-text"``), passed verbatim to the endpoint.

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

    def set_embedding_model(self, model: str) -> None:
        """Persist the embedding-model choice to ``config.json``.

        Onboarding calls this once on first run. Accepts ``"minilm"`` or
        ``"embeddinggemma"``; other values are normalized to lowercase and
        passed through (``embedding.get_embedding_function`` falls back to
        minilm for unrecognized values).
        """
        self._file_config["embedding_model"] = str(model).strip().lower()
        self._config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._file_config, f, indent=2, ensure_ascii=False)
        except OSError:
            pass
        try:
            self._config_file.chmod(0o600)
        except (OSError, NotImplementedError):
            pass

    def set_backend(self, backend: str) -> None:
        """Persist the storage backend choice to ``config.json``."""
        backend = str(backend).strip().lower()
        from .backends import get_backend_class

        get_backend_class(backend)
        self._file_config["backend"] = backend
        self._config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._file_config, f, indent=2, ensure_ascii=False)
        except OSError:
            pass
        try:
            self._config_file.chmod(0o600)
        except (OSError, NotImplementedError):
            pass

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
    def hook_silent_save(self):
        """Whether the stop hook saves directly (True) or blocks for MCP calls (False)."""
        return self._file_config.get("hooks", {}).get("silent_save", True)

    @property
    def hook_desktop_toast(self):
        """Whether the stop hook shows a desktop notification via notify-send."""
        return self._file_config.get("hooks", {}).get("desktop_toast", False)

    def set_hook_setting(self, key: str, value: bool):
        """Update a hook setting and write config to disk."""
        if "hooks" not in self._file_config:
            self._file_config["hooks"] = {}
        self._file_config["hooks"][key] = value
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._file_config, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

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
            with open(self._config_file, "w") as f:
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
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._people_map_file, "w") as f:
            json.dump(people_map, f, indent=2)
        try:
            self._people_map_file.chmod(0o600)
        except (OSError, NotImplementedError):
            pass
        return self._people_map_file
