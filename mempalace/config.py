"""
MemPalace configuration system.

Priority: env vars > config file (~/.mempalace/config.json) > defaults
"""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Input validation ──────────────────────────────────────────────────────────
# Shared sanitizers for wing/room/entity names. Prevents path traversal,
# excessively long strings, and special characters that could cause issues
# in file paths, SQLite, or ChromaDB metadata.

MAX_NAME_LENGTH = 128
_SAFE_NAME_RE = re.compile(r"^(?:[^\W_]|[^\W_][\w .'-]{0,126}[^\W_])$")


def normalize_wing_name(name: str) -> str:
    """Lower-case + collapse separators (`-`, ` `) to `_` for wing slugs.

    The same rule is applied by ``init`` when persisting `topics_by_wing`
    and when writing `mempalace.yaml`, so the miner's lookup matches at
    mine time regardless of the source dirname.
    """
    return name.lower().replace(" ", "_").replace("-", "_")


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

    return value


def sanitize_content(value: str, max_length: int = 100_000) -> str:
    """Validate drawer/diary content length."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("content must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"content exceeds maximum length of {max_length} characters")
    if "\x00" in value:
        raise ValueError("content contains null bytes")
    return value


LEGACY_PALACE_DIR = os.path.expanduser("~/.mempalace/palace")
DEFAULT_PALACE_PATH = os.path.expanduser("~/.mempalace/palaces/chat")
DEFAULT_COLLECTION_NAME = "mempalace_drawers"


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
    "family": ["family", "kids", "children", "daughter", "son", "parent", "mother", "father"],
    "creative": ["game", "gameplay", "player", "app", "design", "art", "music", "story"],
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
    def collection_name(self):
        """ChromaDB collection name."""
        return self._file_config.get("collection_name", DEFAULT_COLLECTION_NAME)

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
    def topic_wings(self):
        """List of topic wing names."""
        return self._file_config.get("topic_wings", DEFAULT_TOPIC_WINGS)

    @property
    def hall_keywords(self):
        """Mapping of hall names to keyword lists."""
        return self._file_config.get("hall_keywords", DEFAULT_HALL_KEYWORDS)

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
