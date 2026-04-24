"""
palace.py — Shared palace operations.

Consolidates collection access patterns used by both miners and the MCP server.
"""

import contextlib
import hashlib
import os
import re

from .backends.chroma import ChromaBackend
from .config import check_palace_storage

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
}

_DEFAULT_BACKEND = ChromaBackend()

# Schema version for drawer normalization. Bump when the normalization
# pipeline changes in a way that existing drawers should be rebuilt to pick up
# (e.g., new noise-stripping rules). `file_already_mined` treats drawers with
# a missing or stale `normalize_version` as "not mined", so the next mine pass
# silently rebuilds them — users don't need to manually erase + re-mine.
#
# v2 (2026-04): introduced strip_noise() for Claude Code JSONL; previous
#               drawers stored system tags / hook chrome verbatim.
NORMALIZE_VERSION = 2


def get_collection(
    palace_path: str,
    collection_name: str = "mempalace_drawers",
    create: bool = True,
):
    """Get the palace collection through the backend layer."""
    # One-shot warning if the palace path sits on a DrvFs / 9P / CIFS /
    # NFS mount that breaks ChromaDB + SQLite mmap/flock/fsync semantics.
    # No-op on native Linux filesystems (ext4/xfs/btrfs/tmpfs), which is the
    # common case ($HOME on the WSL2 root VHD or a dedicated VHD at /mnt/data).
    check_palace_storage(palace_path)
    return _DEFAULT_BACKEND.get_collection(
        palace_path,
        collection_name=collection_name,
        create=create,
    )


def get_closets_collection(palace_path: str, create: bool = True):
    """Get the closets collection — the searchable index layer."""
    return get_collection(palace_path, collection_name="mempalace_closets", create=create)


def get_room_indices_collection(palace_path: str, create: bool = True):
    """Get the room-level index collection.

    Holds one document per (wing, room) — a deterministic, rank-bucketed
    projection of that room's leaf closets. Written by ``recursive_indexer``;
    consumed by ``tool_search_hierarchical`` as the middle pruning layer.
    """
    return get_collection(palace_path, collection_name="mempalace_room_indices", create=create)


def get_wing_indices_collection(palace_path: str, create: bool = True):
    """Get the wing-level index collection.

    Holds one document per wing — a union + top-N projection over that
    wing's room indices. The outermost pruning layer for hierarchical
    retrieval; a ``max_depth=0`` query can stop here without touching
    room indices or leaf closets.
    """
    return get_collection(palace_path, collection_name="mempalace_wing_indices", create=create)


# === Dirty-flag plumbing =====================================================
#
# Leaf drawers are mined constantly; room/wing indices should not be
# rebuilt on every single drawer write — that would make mining O(n²)
# across a palace. Instead, writers call ``mark_room_dirty`` /
# ``mark_wing_dirty`` to record that some room / wing needs re-aggregation,
# and ``recursive_indexer`` picks up the flags under ``mine_lock`` and
# batches the work.
#
# State lives in ``<palace_path>/.dirty/rooms/`` and ``.dirty/wings/`` as
# small JSON files keyed by a sha-16 of the wing|room identifier. One file
# per dirty item, idempotent write, unlinked on clear. File-system based
# so it survives process crashes and is cross-process safe.


_DIRTY_DIRNAME = ".dirty"


def _dirty_dir(palace_path: str, kind: str) -> str:
    assert kind in ("rooms", "wings")
    return os.path.join(palace_path, _DIRTY_DIRNAME, kind)


def _dirty_key(wing: str, room: str = None) -> str:
    raw = wing if room is None else f"{wing}|{room}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _write_dirty_file(dir_path: str, key: str, payload: dict) -> None:
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{key}.json")
    # Atomic write: temp + rename. Avoids torn reads if indexer glob()s
    # mid-write on another thread.
    import json
    import tempfile

    fd, tmp_path = tempfile.mkstemp(prefix=key + ".", suffix=".tmp", dir=dir_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def mark_room_dirty(palace_path: str, wing: str, room: str) -> None:
    """Record that ``(wing, room)``'s index is stale and must be rebuilt.

    Idempotent. Safe to call under a write-heavy mining loop — writes are
    atomic via tempfile + rename. Automatically marks the containing wing
    dirty too, since any room-level change invalidates the wing roll-up.
    """
    _write_dirty_file(
        _dirty_dir(palace_path, "rooms"),
        _dirty_key(wing, room),
        {"wing": wing, "room": room},
    )
    mark_wing_dirty(palace_path, wing)


def mark_wing_dirty(palace_path: str, wing: str) -> None:
    """Record that ``wing``'s index is stale and must be rebuilt. Idempotent."""
    _write_dirty_file(
        _dirty_dir(palace_path, "wings"),
        _dirty_key(wing),
        {"wing": wing},
    )


def _iter_dirty_entries(dir_path: str):
    import json

    if not os.path.isdir(dir_path):
        return
    try:
        entries = os.listdir(dir_path)
    except OSError:
        return
    for entry in sorted(entries):
        if not entry.endswith(".json"):
            continue
        full = os.path.join(dir_path, entry)
        try:
            with open(full, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        yield (entry[:-5], payload, full)  # (key, payload, full_path)


def iter_dirty_rooms(palace_path: str):
    """Yield ``(wing, room, key)`` for every dirty room. Stable sort."""
    for key, payload, _ in _iter_dirty_entries(_dirty_dir(palace_path, "rooms")):
        wing = payload.get("wing")
        room = payload.get("room")
        if isinstance(wing, str) and isinstance(room, str):
            yield (wing, room, key)


def iter_dirty_wings(palace_path: str):
    """Yield ``(wing, key)`` for every dirty wing. Stable sort."""
    for key, payload, _ in _iter_dirty_entries(_dirty_dir(palace_path, "wings")):
        wing = payload.get("wing")
        if isinstance(wing, str):
            yield (wing, key)


def clear_room_dirty(palace_path: str, wing: str, room: str) -> bool:
    """Remove the dirty flag for ``(wing, room)``. Returns True if one was removed."""
    path = os.path.join(_dirty_dir(palace_path, "rooms"), f"{_dirty_key(wing, room)}.json")
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def clear_wing_dirty(palace_path: str, wing: str) -> bool:
    """Remove the dirty flag for ``wing``. Returns True if one was removed."""
    path = os.path.join(_dirty_dir(palace_path, "wings"), f"{_dirty_key(wing)}.json")
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


CLOSET_CHAR_LIMIT = 1500  # fill closet until ~1500 chars, then start a new one
CLOSET_EXTRACT_WINDOW = 5000  # how many chars of source content to scan for entities/topics

# Common capitalized words that look like proper nouns but are usually
# sentence-starters or filler. Filtered out of entity extraction.
_ENTITY_STOPLIST = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "When",
        "Where",
        "What",
        "Why",
        "Who",
        "Which",
        "How",
        "After",
        "Before",
        "Then",
        "Now",
        "Here",
        "There",
        "And",
        "But",
        "Or",
        "Yet",
        "So",
        "If",
        "Else",
        "Yes",
        "No",
        "Maybe",
        "Okay",
        "User",
        "Assistant",
        "System",
        "Tool",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
)


_CANDIDATE_RX_CACHE = None


def _candidate_entity_words(text: str) -> list:
    """Find entity candidate words using i18n-aware patterns.

    Uses the same candidate_patterns as entity_detector (loaded from locale
    JSON files via get_entity_patterns), so non-Latin names (Cyrillic,
    accented Latin, etc.) are detected alongside ASCII names.
    """
    global _CANDIDATE_RX_CACHE
    if _CANDIDATE_RX_CACHE is None:
        from .config import MempalaceConfig
        from .i18n import get_entity_patterns

        patterns = get_entity_patterns(MempalaceConfig().entity_languages)
        rxs = []
        for pat in patterns["candidate_patterns"]:
            try:
                rxs.append(re.compile(pat))
            except re.error:
                continue
        _CANDIDATE_RX_CACHE = rxs
    words = []
    for rx in _CANDIDATE_RX_CACHE:
        words.extend(rx.findall(text))
    return words


def build_closet_lines(source_file, drawer_ids, content, wing, room):
    """Build compact closet pointer lines from drawer content.

    Returns a LIST of lines (not joined). Each line is one complete topic
    pointer — never split across closets.

    Format: topic|entities|→drawer_ids
    """
    import re
    from pathlib import Path

    drawer_ref = ",".join(drawer_ids[:3])
    window = content[:CLOSET_EXTRACT_WINDOW]

    # Extract proper nouns (2+ occurrences). Uses i18n-aware patterns so
    # non-Latin names (Cyrillic, accented Latin, etc.) are also detected.
    words = _candidate_entity_words(window)
    word_freq = {}
    for w in words:
        if w in _ENTITY_STOPLIST:
            continue
        word_freq[w] = word_freq.get(w, 0) + 1
    entities = sorted(
        [w for w, c in word_freq.items() if c >= 2],
        key=lambda w: -word_freq[w],
    )[:5]
    entity_str = ";".join(entities) if entities else ""

    # Extract key phrases — action verbs + context
    topics = []
    for pattern in [
        r"(?:built|fixed|wrote|added|pushed|tested|created|decided|migrated|reviewed|deployed|configured|removed|updated)\s+[\w\s]{3,40}",
    ]:
        topics.extend(re.findall(pattern, window, re.IGNORECASE))
    # Also grab section headers if present
    for header in re.findall(r"^#{1,3}\s+(.{5,60})$", window, re.MULTILINE):
        topics.append(header.strip())
    # Dedupe preserving order
    topics = list(dict.fromkeys(t.strip().lower() for t in topics))[:12]

    # Extract quotes
    quotes = re.findall(r'"([^"]{15,150})"', window)

    # Build pointer lines — each one is atomic, never split
    lines = []
    for topic in topics:
        lines.append(f"{topic}|{entity_str}|→{drawer_ref}")
    for quote in quotes[:3]:
        lines.append(f'"{quote}"|{entity_str}|→{drawer_ref}')

    # Always have at least one line
    if not lines:
        name = Path(source_file).stem[:40]
        lines.append(f"{wing}/{room}/{name}|{entity_str}|→{drawer_ref}")

    return lines


def purge_file_closets(closets_col, source_file: str) -> None:
    """Delete every closet associated with ``source_file``.

    Call this before ``upsert_closet_lines`` on a re-mine so stale topics
    from a prior schema/version don't survive in the closet collection.
    Mirrors the drawer-purge step in process_file().
    """
    try:
        closets_col.delete(where={"source_file": source_file})
    except Exception:
        pass


def upsert_closet_lines(closets_col, closet_id_base, lines, metadata):
    """Write topic lines to closets, packed greedily without splitting a line.

    Closets are deterministically numbered (``..._01``, ``..._02``, …) and
    each ``upsert`` fully overwrites the prior content at that ID. Callers
    are expected to ``purge_file_closets`` first when re-mining a source
    file so stale-numbered closets from larger prior runs don't leak.

    Returns the number of closets written.
    """
    closet_num = 1
    current_lines: list = []
    current_chars = 0
    closets_written = 0

    def _flush():
        nonlocal closets_written
        if not current_lines:
            return
        closet_id = f"{closet_id_base}_{closet_num:02d}"
        text = "\n".join(current_lines)
        closets_col.upsert(documents=[text], ids=[closet_id], metadatas=[metadata])
        closets_written += 1

    for line in lines:
        line_len = len(line)
        # Would this line fit whole in the current closet?
        if current_chars > 0 and current_chars + line_len + 1 > CLOSET_CHAR_LIMIT:
            _flush()
            closet_num += 1
            current_lines = []
            current_chars = 0

        current_lines.append(line)
        current_chars += line_len + 1  # +1 for newline

    _flush()
    return closets_written


@contextlib.contextmanager
def mine_lock(source_file: str):
    """Cross-platform file lock for mine operations.

    Prevents multiple agents from mining the same file simultaneously,
    which causes duplicate drawers when the delete+insert cycle interleaves.
    """
    lock_dir = os.path.join(os.path.expanduser("~"), ".mempalace", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(
        lock_dir, hashlib.sha256(source_file.encode()).hexdigest()[:16] + ".lock"
    )

    lf = open(lock_path, "w")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception:
            pass
        lf.close()


def file_already_mined(collection, source_file: str, check_mtime: bool = False) -> bool:
    """Check if a file has already been filed in the palace.

    Returns False (so the file gets re-mined) when:
      - no drawers exist for this source_file
      - the stored `normalize_version` is missing or older than the current
        schema (triggers silent rebuild after a normalization upgrade)
      - `check_mtime=True` and the file's mtime differs from the stored one

    When check_mtime=True (used by project miner), also re-mines on content
    change. When check_mtime=False (used by convo miner), transcripts are
    assumed immutable, so only the version gate triggers a rebuild.
    """
    try:
        results = collection.get(where={"source_file": source_file}, limit=1)
        if not results.get("ids"):
            return False
        stored_meta = results.get("metadatas", [{}])[0] or {}
        # Pre-v2 drawers have no version field — treat them as stale.
        stored_version = stored_meta.get("normalize_version", 1)
        if stored_version < NORMALIZE_VERSION:
            return False
        if check_mtime:
            stored_mtime = stored_meta.get("source_mtime")
            if stored_mtime is None:
                return False
            current_mtime = os.path.getmtime(source_file)
            return abs(float(stored_mtime) - current_mtime) < 0.001
        return True
    except Exception:
        return False
