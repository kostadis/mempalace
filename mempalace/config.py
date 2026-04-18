"""
MemPalace configuration system.

Priority: env vars > config file (~/.mempalace/config.json) > defaults
"""

import json
import os
import re
from pathlib import Path


# ── Input validation ──────────────────────────────────────────────────────────
# Shared sanitizers for wing/room/entity names. Prevents path traversal,
# excessively long strings, and special characters that could cause issues
# in file paths, SQLite, or ChromaDB metadata.

MAX_NAME_LENGTH = 128
_SAFE_NAME_RE = re.compile(r"^(?:[^\W_]|[^\W_][\w .'-]{0,126}[^\W_])$")


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


DEFAULT_PALACE_PATH = os.path.expanduser("~/.mempalace/palace")
DEFAULT_COLLECTION_NAME = "mempalace_drawers"

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
            return env_val
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
        4. Existing ``palace_path`` fallback (``palace_path`` file field or
           ``DEFAULT_PALACE_PATH``)

        Step 7 of the palace-isolation design replaces step 4 with a loud
        error; until then the legacy default keeps existing users working.
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

        return os.path.abspath(os.path.expanduser(self.palace_path))

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
