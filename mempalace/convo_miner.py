#!/usr/bin/env python3
"""
convo_miner.py — Mine conversations into the palace.

Ingests chat exports (Claude Code, ChatGPT, Slack, plain text transcripts).
Normalizes format, chunks by exchange pair (Q+A = one unit), files to palace.

Same palace as project mining. Different ingest strategy.
"""

import errno
import hashlib
import os
import sys
import json
import logging
from dataclasses import dataclass
import stat
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

from .backends import PalaceNotFoundError
from .collision_scan import assert_no_collisions
from .config import MempalaceConfig
from .ids import (
    ID_RECIPE,
    make_convo_drawer_id,
    make_convo_sentinel_id,
    make_exchange_drawer_id,
)
from .normalize import normalize
from .entities import entities_metadata
from .palace import (
    NORMALIZE_VERSION,
    SKIP_DIRS,
    _metadata_matches_extract_mode,
    _validate_palace_fts5_after_mine,
    file_already_mined,
    get_collection,
    mine_lock,
    mine_palace_lock,
    prefetch_content_hashes,
    prefetch_mined_set,
)
from .parallel import ParallelPipeline, WorkerResult

logger = logging.getLogger("mempalace_mcp")


# Cached hall keywords — avoids re-reading config per drawer
_HALL_KEYWORDS_CACHE = None


def _detect_hall_cached(content: str) -> str:
    """Route content to a hall using cached keywords. Same logic as miner.detect_hall."""
    global _HALL_KEYWORDS_CACHE
    if _HALL_KEYWORDS_CACHE is None:
        from .config import MempalaceConfig

        _HALL_KEYWORDS_CACHE = MempalaceConfig().hall_keywords
    content_lower = content[:3000].lower()
    scores = {}
    for hall, keywords in _HALL_KEYWORDS_CACHE.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[hall] = score
    return max(scores, key=scores.get) if scores else "general"


def file_conversation_exchange(
    collection,
    *,
    wing: str,
    room: str,
    text: str,
    source_file: str,
    agent: str,
    authored_at: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Optional[str]:
    """File one verbatim conversation exchange as a single drawer.

    Canonical write path for live agent integrations (e.g. Hermes) and
    their backfills — both must route here so routing, normalization,
    and metadata conventions stay identical between live and historical
    ingest. Builds the same metadata the convo miner writes so hallway
    traversal, entity search, and since/before date filters see
    integration drawers exactly like mined ones.

    ``wing`` and ``room`` are validated with the same ``sanitize_name``
    rules the MCP write tools apply, but a failed name falls back
    (``wing_general`` / ``conversations``) instead of erroring: this
    path files *live* turns, and dropping a turn over a config typo
    would break the verbatim / 100%-recall promise. The fallback is
    logged at warning level so the misconfiguration is visible.

    ``extra_metadata`` lets callers append integration-specific fields
    (e.g. ``source`` / ``session_id``); keys that collide with the
    canonical fields are ignored, so it cannot be used to overwrite or
    drop them. Returns the drawer id, or None when ``text`` is empty
    after stripping.
    """
    from .config import sanitize_name

    text = (text or "").strip()
    if not text:
        return None
    try:
        wing = sanitize_name(wing, "wing")
    except ValueError:
        logger.warning(
            "file_conversation_exchange: invalid wing %r — filing under wing_general", wing
        )
        wing = "wing_general"
    try:
        room = sanitize_name(room, "room")
    except ValueError:
        logger.warning(
            "file_conversation_exchange: invalid room %r — filing under conversations", room
        )
        room = "conversations"
    filed_at = datetime.now().isoformat()
    drawer_id = make_exchange_drawer_id(wing, room, source_file, filed_at, text)
    metadata = {
        "wing": wing,
        "room": room,
        "hall": _detect_hall_cached(text),
        "source_file": source_file,
        "chunk_index": 0,
        "added_by": agent,
        "filed_at": filed_at,
        "entities": entities_metadata(text),
        "authored_at": authored_at if authored_at is not None else filed_at,
        "ingest_mode": "convos",
        "extract_mode": "exchange",
        "normalize_version": NORMALIZE_VERSION,
        "id_recipe": ID_RECIPE,
    }
    if extra_metadata:
        for key, value in extra_metadata.items():
            metadata.setdefault(key, value)
    collection.upsert(ids=[drawer_id], documents=[text], metadatas=[metadata])
    return drawer_id


# File types that might contain conversations
CONVO_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

# Directories inside conversation sources that never hold conversations.
# ``tool-results``: Claude Code pages large tool outputs to
# ``<session>/tool-results/*.txt`` inside ``~/.claude/projects/<slug>/``.
# They are raw machine dumps referenced from the transcript JSONL — mining
# them stores megabytes of command output as "memories" (field measurement:
# 12.8k drawers from tool-results files on one palace; a single file
# produced 3.6k). Extends the generic SKIP_DIRS set for the convo scanner
# only — project mining semantics are unchanged.
CONVO_SKIP_DIRS = SKIP_DIRS | {"tool-results"}

MIN_CHUNK_SIZE = 30
CHUNK_SIZE = (
    800  # chars per drawer — align with miner.py; keeps drawers under nomic 2048-token limit
)
DRAWER_UPSERT_BATCH_SIZE = 1000
_LINE_GROUP_SIZE = 25  # lines per fallback group when no paragraph breaks
_LINE_FALLBACK_MIN_NEWLINES = 20  # trigger line-group fallback above this newline count
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB — skip files larger than this.
# Matches miner.py at 500 MB. Long Claude Code sessions, multi-year
# ChatGPT exports, and lifetime Slack dumps routinely exceed 10 MB; the
# cap at that level silently dropped them with `continue`. Per-drawer
# size is bounded by CHUNK_SIZE, but larger source files still produce
# more drawers and therefore more embedding/storage work — and content
# is normalized and loaded fully into memory before chunking, so memory
# use also scales with source size.


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
        return True
    except (OSError, ValueError):
        return False


def _is_regular_source_file(filepath: Path, root: Path) -> bool:
    if not _path_within_root(filepath, root):
        return False
    # O_NONBLOCK keeps the S_ISREG verdict below reachable: a blocking open
    # of a FIFO waits in the kernel for a writer, so a named pipe called
    # ``session.jsonl`` would hang this check instead of failing it. See the
    # matching comment in ``miner._read_text_no_follow``, including why the
    # EAGAIN branch re-checks the type and then opens without the flag.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = -1
    try:
        try:
            fd = os.open(filepath, flags)
        except OSError as exc:
            if exc.errno != errno.EAGAIN or not stat.S_ISREG(os.lstat(filepath).st_mode):
                raise
            fd = os.open(filepath, flags & ~getattr(os, "O_NONBLOCK", 0))
        st = os.fstat(fd)
        return stat.S_ISREG(st.st_mode) and st.st_size <= MAX_FILE_SIZE
    except OSError:
        return False
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass


def _register_file(
    collection,
    source_file: str,
    wing: str,
    agent: str,
    extract_mode: str,
    content_hash: Optional[str] = None,
):
    """Write a sentinel so file_already_mined() returns True for 0-chunk files.

    Without this, files that normalize to nothing or produce zero chunks are
    re-read and re-processed on every mine run because nothing was written to
    ChromaDB on the first pass. The sentinel is scoped to ``extract_mode`` so
    an empty file mined in exchange-mode does not mask a later general-mode
    mine of the same transcript (and vice versa).

    Stamps source_mtime like every real drawer does, so a file that later
    grows past the min-chunk-size floor (e.g. a short session that gets
    extended) is correctly detected as changed on the next mine instead of
    being skipped forever by this sentinel.

    Also used to register a file recognized as a content-duplicate of an
    already-mined transcript under a different path (see
    ``prefetch_content_hashes``) — stamping it here means the next run skips
    it via the cheap mtime check instead of re-normalizing and re-hashing it.
    """
    try:
        source_mtime = os.path.getmtime(source_file)
    except OSError:
        source_mtime = None
    sentinel_id = make_convo_sentinel_id(source_file, extract_mode)
    meta = {
        "wing": wing,
        "room": "_registry",
        "source_file": source_file,
        "added_by": agent,
        "filed_at": datetime.now().isoformat(),
        "ingest_mode": "registry",
        "extract_mode": extract_mode,
        "normalize_version": NORMALIZE_VERSION,
        "id_recipe": ID_RECIPE,
    }
    if source_mtime is not None:
        meta["source_mtime"] = source_mtime
    if content_hash is not None:
        meta["content_hash"] = content_hash
    collection.upsert(
        documents=[f"[registry] {source_file}"],
        ids=[sentinel_id],
        metadatas=[meta],
    )


def _source_file_delete_ids(collection, source_file: str, extract_mode: str) -> list:
    """Collect drawer IDs for one source file and extraction mode.

    Legacy conversation drawers did not carry extract_mode; treat those as
    exchange-mode rows so schema rebuilds can still clean them up without
    deleting newer general-mode drawers for the same transcript.
    """
    ids: list = []
    offset = 0
    while True:
        batch = collection.get(
            where={"source_file": source_file},
            limit=1000,
            offset=offset,
            include=["metadatas"],
        )
        batch_ids = batch.get("ids") or []
        metadatas = batch.get("metadatas") or []
        for drawer_id, meta in zip(batch_ids, metadatas):
            if _metadata_matches_extract_mode(meta or {}, extract_mode):
                ids.append(drawer_id)
        if not batch_ids:
            break
        offset += len(batch_ids)
    return ids


def _is_ai_tool_path(path: Path) -> bool:
    """Return True when `path` lives inside a known AI-tool storage dir.

    Detected paths (exact-segment match — substrings like `.gemini-backup`
    or `.codex-archive` do NOT match):
      - any segment ``.codex`` (Codex CLI sessions / archives)
      - any segment ``.gemini`` (Gemini CLI sessions under ~/.gemini/tmp/...)
      - the consecutive segment pair ``.claude/projects`` (Claude Code).
        ``.claude`` alone is NOT matched — that is the settings/config dir,
        not a conversation source.

    Used by ``_resolve_wing`` to default the destination wing to
    ``wing_api`` when the user hasn't passed an explicit ``--wing``.
    """
    try:
        parts = path.resolve().parts
    except (OSError, RuntimeError):
        return False

    if ".codex" in parts:
        return True
    if ".gemini" in parts:
        return True
    for i in range(len(parts) - 1):
        if parts[i] == ".claude" and parts[i + 1] == "projects":
            return True
    return False


def _is_unchanged_since_last_mine(source_file: str, mined_mtimes: dict) -> bool:
    """True iff source_file was mined at the current schema AND its on-disk
    mtime still matches what was stored -- the mtime-aware replacement for
    "we've seen this source_file before" (transcripts are not immutable).

    False (re-mine) whenever the file isn't in mined_mtimes at all, its
    stored mtime is None (never recorded -- pre-mtime-tracking drawer, or
    getmtime failed when it was written), or getmtime fails right now
    (treat as changed rather than silently trusting stale data).
    """
    if source_file not in mined_mtimes:
        return False
    stored_mtime = mined_mtimes[source_file]
    if stored_mtime is None:
        return False
    try:
        current_mtime = os.path.getmtime(source_file)
    except OSError:
        return False
    return abs(stored_mtime - current_mtime) < 0.001


def _resolve_wing(convo_path: Path, wing: Optional[str]) -> str:
    """Determine the destination wing for ``mine_convos``.

    Precedence (first match wins):

      1. Explicit ``wing`` argument from the user — always wins, even on
         an AI-tool path. Empty string is treated as "no wing".
      2. AI-tool path detection — defaults to ``wing_api`` so Claude
         Code / Codex / Gemini conversations group under a single wing
         dedicated to API-sourced content.
      3. Basename fallback — sanitized via ``config.normalize_wing_name``.
    """
    from .config import normalize_wing_name

    if wing:
        return wing
    if _is_ai_tool_path(convo_path):
        return "wing_api"
    return normalize_wing_name(convo_path.name)


# =============================================================================
# CHUNKING — exchange pairs for conversations
# =============================================================================


def chunk_exchanges(
    content: str,
    chunk_size: int = None,
    min_chunk_size: int = None,
) -> list:
    """
    Chunk by exchange pair: one > turn + AI response = one unit.
    Falls back to paragraph chunking if no > markers.

    Optional params override module-level defaults when provided.

    Raises ``ValueError`` if ``chunk_size`` is not a positive integer or
    ``min_chunk_size`` is negative. A non-positive ``chunk_size`` would
    cause ``_chunk_by_exchange`` below to loop forever — ``content[:0]``
    is empty, ``content[0:]`` is the whole string, and the remainder
    never shrinks.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if min_chunk_size is None:
        min_chunk_size = MIN_CHUNK_SIZE

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if min_chunk_size < 0:
        raise ValueError(f"min_chunk_size must be >= 0, got {min_chunk_size}")

    lines = content.split("\n")
    quote_lines = sum(1 for line in lines if line.strip().startswith(">"))

    if quote_lines >= 3:
        return _chunk_by_exchange(lines, chunk_size, min_chunk_size)
    else:
        return _chunk_by_paragraph(content, chunk_size, min_chunk_size)


def _chunk_by_exchange(lines: list, chunk_size: int, min_chunk_size: int) -> list:
    """One user turn (>) + the AI response that follows = one or more chunks.

    The full AI response is preserved verbatim.  When the combined
    user-turn + response exceeds chunk_size the response is split across
    consecutive drawers so nothing is silently discarded.
    """
    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_turn = line.strip()
            i += 1

            ai_lines = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith(">") or next_line.strip().startswith("---"):
                    break
                # Preserve the line as-is — blank lines and indentation carry meaning
                # (paragraph breaks, list/code structure) and must survive verbatim.
                ai_lines.append(next_line)
                i += 1

            # Join on newline (not space) so line structure, blank lines, and
            # indentation reach the drawer unchanged. Trim only trailing blank
            # lines produced by the loop stopping at the next `>` turn.
            ai_response = "\n".join(ai_lines).rstrip("\n")
            content = f"{user_turn}\n{ai_response}" if ai_response else user_turn

            _emit_bounded(chunks, content, chunk_size, min_chunk_size)
        else:
            i += 1

    return chunks


def _emit_bounded(
    chunks: list,
    content: str,
    chunk_size: int,
    min_chunk_size: int,
) -> None:
    """Append ``content`` as one or more drawers, none exceeding ``chunk_size``.

    The ``min_chunk_size`` floor gates the WHOLE call (drops the input if
    its stripped length is at or below the floor, treated as noise). Once
    the input passes the floor, every slice is emitted verbatim so a
    small trailing remainder is preserved instead of silently dropped.
    The index-based loop avoids the O(N^2) repeated-substring allocation
    of a ``while content: content = content[chunk_size:]`` shape.
    """
    if len(content.strip()) <= min_chunk_size:
        return
    for i in range(0, len(content), chunk_size):
        chunks.append({"content": content[i : i + chunk_size], "chunk_index": len(chunks)})


def _chunk_by_paragraph(content: str, chunk_size: int, min_chunk_size: int) -> list:
    """Fallback: chunk by paragraph breaks."""
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no paragraph breaks and long content, chunk by line groups
    if len(paragraphs) <= 1 and content.count("\n") > _LINE_FALLBACK_MIN_NEWLINES:
        lines = content.split("\n")
        for i in range(0, len(lines), _LINE_GROUP_SIZE):
            group = "\n".join(lines[i : i + _LINE_GROUP_SIZE]).strip()
            _emit_bounded(chunks, group, chunk_size, min_chunk_size)
        return chunks

    for para in paragraphs:
        _emit_bounded(chunks, para, chunk_size, min_chunk_size)

    return chunks


# =============================================================================
# ROOM DETECTION — topic-based for conversations
# =============================================================================

TOPIC_KEYWORDS = {
    "technical": [
        "code",
        "python",
        "function",
        "bug",
        "error",
        "api",
        "database",
        "server",
        "deploy",
        "git",
        "test",
        "debug",
        "refactor",
    ],
    "architecture": [
        "architecture",
        "design",
        "pattern",
        "structure",
        "schema",
        "interface",
        "module",
        "component",
        "service",
        "layer",
    ],
    "planning": [
        "plan",
        "roadmap",
        "milestone",
        "deadline",
        "priority",
        "sprint",
        "backlog",
        "scope",
        "requirement",
        "spec",
    ],
    "decisions": [
        "decided",
        "chose",
        "picked",
        "switched",
        "migrated",
        "replaced",
        "trade-off",
        "alternative",
        "option",
        "approach",
    ],
    "problems": [
        "problem",
        "issue",
        "broken",
        "failed",
        "crash",
        "stuck",
        "workaround",
        "fix",
        "solved",
        "resolved",
    ],
}


def detect_convo_room(content: str) -> str:
    """Score conversation content against topic keywords."""
    content_lower = content[:3000].lower()
    scores = {}
    for room, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[room] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


# =============================================================================
# PALACE OPERATIONS
# =============================================================================


# =============================================================================
# SCAN FOR CONVERSATION FILES
# =============================================================================


def scan_convos(convo_dir: str, include_subagents: bool = False) -> list:
    """Find all potential conversation files.

    Skips symlinks and oversized files. Each skipped symlink is logged to
    ``sys.stderr`` with a ``  SKIP: <relative-path> (symlink)`` line so the
    caller can tell why an apparent conversation directory yielded no files.

    By default, directories named ``subagents`` are skipped: Claude Code
    records Explore/Plan/Grep subagent transcripts there, and on typical
    workspaces they outnumber main session files by one to two orders of
    magnitude. Pass ``include_subagents=True`` to mine them anyway.

    The match is case-insensitive on the directory name only (``subagents``
    or ``Subagents``), so directories like ``mysubagents`` or
    ``subagentsbackup`` are not affected.
    """
    # A direct conversation file is a valid source. For a file, feed only
    # its basename through the existing directory validation loop.
    requested_path = Path(convo_dir).expanduser()
    single_file = requested_path.is_file()
    convo_path = (requested_path.parent if single_file else requested_path).resolve()
    scan_entries = (
        [(str(convo_path), [], [requested_path.name])] if single_file else os.walk(convo_path)
    )
    files = []
    for root, dirs, filenames in scan_entries:
        dirs[:] = [
            d
            for d in dirs
            if d not in CONVO_SKIP_DIRS and (include_subagents or d.lower() != "subagents")
        ]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() in CONVO_EXTENSIONS:
                # Skip symlinks and oversized files
                if filepath.is_symlink():
                    rel = filepath.relative_to(convo_path).as_posix()
                    try:
                        print(f"  SKIP: {rel} (symlink)", file=sys.stderr)
                    except OSError:
                        pass
                    continue
                # Skip files exceeding size limit, or those whose stat() raises
                # (permission denied, racing delete, broken symlink that
                # survived the earlier is_symlink check). Both branches log
                # to stderr to match the SKIP: (symlink) line above; silent
                # drops at this gate were the original #923 complaint.
                try:
                    file_stat = filepath.stat()
                    # Drop non-regular entries (FIFO, socket, device node)
                    # before any reader touches them — see the matching
                    # gate in ``miner.scan_project``.
                    if not stat.S_ISREG(file_stat.st_mode):
                        print(
                            f"  SKIP: {filepath.name} (not a regular file)",
                            file=sys.stderr,
                        )
                        continue
                    file_size = file_stat.st_size
                    if file_size > MAX_FILE_SIZE:
                        print(
                            f"  SKIP: {filepath.name} ({file_size / (1024 * 1024):.1f} MB)"
                            f" exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit",
                            file=sys.stderr,
                        )
                        continue
                except OSError as exc:
                    # Prefer ``exc.strerror`` so the path isn't duplicated in
                    # the output (see the matching comment in
                    # ``miner.scan_project``).
                    print(
                        f"  SKIP: {filepath.name} (stat error: {exc.strerror or exc})",
                        file=sys.stderr,
                    )
                    continue
                if not _is_regular_source_file(filepath, convo_path):
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MINE CONVERSATIONS
# =============================================================================


def _extract_authored_at(filepath):
    """Most-recent message timestamp in a transcript, used as the drawer's authored date.

    Both Claude Code and Codex JSONL transcripts carry a top-level ISO-8601
    ``timestamp`` on each line. We take the max so ``authored_at`` reflects when the
    content was actually written, independent of when it was mined (``filed_at``).
    This restores chronology: a session from days ago keeps its real date even when
    re-mined today, instead of every drawer collapsing to ingest time. Returns None
    for formats without per-line timestamps (e.g. plain ``.md``).
    """
    path = Path(filepath)
    if path.suffix != ".jsonl":
        return None
    latest = None
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts = json.loads(line).get("timestamp")
                except (ValueError, TypeError, AttributeError):
                    continue
                # ISO-8601 timestamps are strings; guard against a non-string
                # ``timestamp`` so a malformed line can't raise TypeError on compare.
                if isinstance(ts, str) and (latest is None or ts > latest):
                    latest = ts
    except OSError:
        return None
    return latest


@dataclass
class _ConvoBatch:
    """One sub-batch of a transcript's chunks, sized to DRAWER_UPSERT_BATCH_SIZE."""

    documents: list
    ids: list
    metadatas: list
    rooms: list  # per-chunk room name — varies in extract_mode="general"


@dataclass
class _PreparedConvo:
    """Output of _prepare_convo: ready for embed + single-writer upsert."""

    source_file: str
    chunks: list  # raw chunk_exchanges or extract_memories output
    room: Optional[str]  # None when extract_mode="general" (per-chunk rooms)
    extract_mode: str
    batches: list  # list of _ConvoBatch


def _prepare_convo(
    filepath: Path,
    wing: str,
    agent: str,
    extract_mode: str,
    dry_run: bool,
    collection,
    chunk_size: int = None,
    min_chunk_size: int = None,
    content_hashes: Optional[dict] = None,
):
    """Read + normalize + chunk + build per-batch documents/ids/metadata.

    Returns one of:
      * ``None`` — file is already mined (pure skip).
      * ``("register", source_file, content_hash, is_duplicate)`` — readable
        but yields no chunks, or a content-hash duplicate of an
        already-mined transcript under a different path; the consumer must
        write the sentinel via _register_file (passing content_hash
        through) so it's marked seen on the next run. ``content_hash`` is
        ``None`` when normalize() failed before a hash could be computed.
        ``is_duplicate`` distinguishes a real content-hash skip (counted
        under files_skipped) from a readable-but-empty file (counted as
        processed, matching the serial mine's pre-existing behavior).
      * ``_PreparedConvo(...)`` — ready for embed + write.

    Idempotency and drawer IDs are scoped to ``extract_mode`` so exchange-
    and general-mode drawers for the same transcript coexist instead of
    overwriting each other. The rebuild contract (purge-before-insert so
    stale drawers never survive) fires on either a normalize-version bump
    or a changed/grown source file (mtime differs from what's stored) —
    transcripts are not assumed immutable, since a Claude Code session
    keeps appending to its own file while active and /compact or /clear
    can rewrite one in place.

    Safe to call from a producer thread; does NOT acquire mine_lock and
    does NOT touch the collection writer.
    """
    effective_chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    effective_min = min_chunk_size if min_chunk_size is not None else MIN_CHUNK_SIZE

    source_file = str(filepath)
    if not dry_run and file_already_mined(
        collection, source_file, check_mtime=True, extract_mode=extract_mode
    ):
        return None

    try:
        content = normalize(str(filepath))
    except (OSError, ValueError):
        return ("register", source_file, None, False)

    if not content or len(content.strip()) < effective_min:
        return ("register", source_file, None, False)

    # Content-hash dedup: the same conversation re-exported under a new
    # filename (fresh export bundle, regenerated slug) must not create a
    # duplicate set of drawers. Hashed on the whole normalized file, so
    # this only catches a duplicate whole file — a bundle that re-exports
    # several conversations together, one of them new, needs per-conversation
    # hashing (palace.prefetch_content_hashes already supports the
    # comma-joined multi-hash shape for that); wiring per-conversation
    # splitting here is a separate, not-yet-ported follow-up.
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if content_hashes:
        original = content_hashes.get((wing, content_hash))
        if original and original != source_file:
            print(f"    [dup] {filepath.name} -- duplicate of {Path(original).name}, skipping")
            return ("register", source_file, content_hash, True)

    if extract_mode == "general":
        from .general_extractor import extract_memories

        chunks = extract_memories(content, chunk_size=effective_chunk_size)
    else:
        chunks = chunk_exchanges(
            content,
            chunk_size=effective_chunk_size,
            min_chunk_size=effective_min,
        )

    if not chunks:
        return ("register", source_file, content_hash, False)

    if extract_mode != "general":
        room = detect_convo_room(content)
    else:
        room = None  # per-chunk rooms below

    # Single filed_at per source so all drawers from this transcript
    # share an ingest timestamp. authored_at reflects when the content was
    # actually written (from the transcript's own timestamps), independent
    # of when it was mined — falls back to filed_at for formats without
    # per-line timestamps.
    filed_at = datetime.now().isoformat()
    authored_at = _extract_authored_at(filepath)
    try:
        source_mtime = os.path.getmtime(source_file)
    except OSError:
        source_mtime = None
    # Every drawer of this pass carries chunk_total so file_already_mined /
    # prefetch_mined_set can tell a complete multi-batch mine from one that
    # crashed mid-file (#2183). Without it a stable mtime + any surviving
    # drawer permanently skips the file and the missing exchanges never
    # come back.
    chunk_total = len(chunks)
    batches: list = []
    for batch_start in range(0, len(chunks), DRAWER_UPSERT_BATCH_SIZE):
        batch_docs: list = []
        batch_ids: list = []
        batch_metas: list = []
        batch_rooms: list = []
        for chunk in chunks[batch_start : batch_start + DRAWER_UPSERT_BATCH_SIZE]:
            chunk_room = chunk.get("memory_type", room) if extract_mode == "general" else room
            drawer_id = make_convo_drawer_id(
                wing, chunk_room, source_file, extract_mode, chunk["chunk_index"]
            )
            batch_docs.append(chunk["content"])
            batch_ids.append(drawer_id)
            meta = {
                "wing": wing,
                "room": chunk_room,
                "hall": _detect_hall_cached(chunk["content"]),
                "source_file": source_file,
                "chunk_index": chunk["chunk_index"],
                "added_by": agent,
                "filed_at": filed_at,
                "entities": entities_metadata(chunk["content"]),
                "authored_at": authored_at if authored_at is not None else filed_at,
                "ingest_mode": "convos",
                "extract_mode": extract_mode,
                "normalize_version": NORMALIZE_VERSION,
                "id_recipe": ID_RECIPE,
                "chunk_total": chunk_total,
                "content_hash": content_hash,
            }
            if source_mtime is not None:
                meta["source_mtime"] = source_mtime
            batch_metas.append(meta)
            batch_rooms.append(chunk_room)
        batches.append(
            _ConvoBatch(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas,
                rooms=batch_rooms,
            )
        )

    return _PreparedConvo(
        source_file=source_file,
        chunks=chunks,
        room=room,
        extract_mode=extract_mode,
        batches=batches,
    )


def _embed_prepared_convo(prepared: "_PreparedConvo", ef) -> list:
    """Call the EF on each batch. Thread-safe: ef is a stateless callable.

    Returns one embeddings list per batch, parallel to prepared.batches.
    """
    return [list(ef(batch.documents)) for batch in prepared.batches]


def _write_prepared_convo(
    prepared: "_PreparedConvo",
    embeddings_batches: list,
    collection,
    wing: str,
):
    """Single-writer phase: under mine_lock, upsert with pre-computed embeddings.

    Returns ``(drawers_added, room_counts_delta, skipped)``. ``skipped=True``
    means another agent just finished this file between the prepare and
    write phases (the post-lock file_already_mined re-check fired).
    """
    source_file = prepared.source_file
    room_counts_delta: dict = defaultdict(int)
    drawers_added = 0
    with mine_lock(source_file):
        # Re-check after lock — another agent may have just finished this file
        # at the current schema/mtime. A stale hit here returns False, so we
        # still fall through to the purge+rebuild path below.
        if file_already_mined(
            collection, source_file, check_mtime=True, extract_mode=prepared.extract_mode
        ):
            return 0, room_counts_delta, True

        # Purge stale drawers first. Fires both on a normalize-schema bump
        # (file_already_mined() returned False for pre-v2 drawers) and on a
        # changed/grown transcript (mtime differs) — clean them out so the
        # source doesn't end up with mixed old/new drawers. The delete is
        # scoped to this extract_mode so a general-mode re-mine doesn't wipe
        # the transcript's exchange-mode drawers (and vice versa) — the two
        # modes coexist for the same source file.
        #
        # A failed purge must abort this file's mine attempt rather than
        # fall through to upsert: proceeding on top of an unpurged (or
        # partially purged) set produces duplicate/stale drawers under
        # mixed schema versions, with no operator-visible signal beyond a
        # debug log (#105 — convo_miner's own instance of the same swallow
        # already fixed for miner.py at #23). Returning here leaves the old
        # drawers' stored mtime untouched, so the next mine still sees a
        # mismatch and retries.
        try:
            delete_ids = _source_file_delete_ids(collection, source_file, prepared.extract_mode)
            if delete_ids:
                collection.delete(ids=delete_ids)
        except Exception as exc:
            print(
                f"  ! [skip] stale-drawer purge failed for {source_file!r} "
                f"({exc!r}); leaving existing drawers untouched, will retry "
                f"on the next mine",
                file=sys.stderr,
            )
            logger.debug("Stale-drawer purge failed for %s", source_file, exc_info=True)
            return 0, room_counts_delta, True

        for batch, batch_embeddings in zip(prepared.batches, embeddings_batches):
            assert_no_collisions(list(zip(batch.ids, batch.metadatas)), collection)
            try:
                collection.upsert(
                    documents=batch.documents,
                    ids=batch.ids,
                    metadatas=batch.metadatas,
                    embeddings=batch_embeddings,
                )
                drawers_added += len(batch.documents)
                if prepared.extract_mode == "general":
                    for r in batch.rooms:
                        room_counts_delta[r] += 1
            except Exception as e:
                if "already exists" not in str(e).lower():
                    # A successful earlier batch has the source's current
                    # mtime and chunk_total. Leaving those drawers behind
                    # would make the next run treat the incomplete set as
                    # fully filed (#2183 / #2122).
                    try:
                        cleanup_ids = _source_file_delete_ids(
                            collection, source_file, prepared.extract_mode
                        )
                        if cleanup_ids:
                            collection.delete(ids=cleanup_ids)
                    except Exception:
                        logger.warning(
                            "Failed to clean partial convo drawers after upsert error for %s",
                            source_file,
                            exc_info=True,
                        )
                    raise
    return drawers_added, room_counts_delta, False


def mine_convos(
    convo_dir: str,
    palace_path: str,
    wing: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
    workers: Optional[int] = None,
    include_subagents: bool = False,
):
    """Mine a directory of conversation files into the palace.

    extract_mode:
        "exchange" — default exchange-pair chunking (Q+A = one unit)
        "general"  — general extractor: decisions, preferences, milestones, problems, emotions
    include_subagents:
        False (default) — skip Claude Code ``subagents/`` directories
        True            — also mine subagent transcripts

    workers: producer-thread fan-out for the parallel embed path. ``None``
    (default) resolves to :attr:`MempalaceConfig.workers` — 1 onnx /
    8 remote. ``1`` reproduces the historic fully-serial mine.

    The real work is in :func:`_mine_convos_impl`; this wrapper holds the
    per-palace flock around it so two concurrent ``mempalace mine --mode
    convos`` invocations against the same palace can't pile up (mirrors
    :func:`mempalace.miner.mine`). ``MineAlreadyRunning`` propagates to the
    caller. Dry-run skips the lock — it never writes.
    """
    if dry_run:
        return _mine_convos_impl(
            convo_dir,
            palace_path,
            wing=wing,
            agent=agent,
            limit=limit,
            dry_run=dry_run,
            extract_mode=extract_mode,
            workers=workers,
            include_subagents=include_subagents,
        )

    with mine_palace_lock(palace_path):
        return _mine_convos_impl(
            convo_dir,
            palace_path,
            wing=wing,
            agent=agent,
            limit=limit,
            dry_run=dry_run,
            extract_mode=extract_mode,
            workers=workers,
            include_subagents=include_subagents,
        )


def _compute_hallways_for_wing_safe(wing, collection, drawers_filed, config=None):
    """Auto-populate the associative graph from the entities just mined.

    Best-effort: hallway computation must never fail an otherwise-good mine, and is
    skipped when nothing new was filed.
    """
    if drawers_filed <= 0:
        return
    try:
        from .hallways import compute_hallways_for_wing

        compute_hallways_for_wing(wing, col=collection, config=config)
    except Exception as exc:
        print(f"  (hallways skipped: {exc})")


def _open_convo_collection(
    palace_path: str,
    *,
    dry_run: bool,
):
    """Open the conversation collection without creating it during dry-run."""
    if not dry_run:
        return get_collection(palace_path)

    try:
        return get_collection(
            palace_path,
            create=False,
            read_only=True,
        )
    except PalaceNotFoundError:
        # A missing palace or uninitialized collection represents empty
        # prior state to a dry-run. Do not create either one.
        return None


def _mine_convos_impl(  # noqa: C901 — parallel-pipeline orchestrator: dry-run/parallel branch plus per-file extract-mode routing are intrinsically branchy
    convo_dir: str,
    palace_path: str,
    wing: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
    workers: Optional[int] = None,
    include_subagents: bool = False,
):
    palace_config = MempalaceConfig(palace_path=palace_path)
    cfg_chunk_size = palace_config.chunk_size
    # Only override convo_miner's MIN_CHUNK_SIZE when the user set
    # min_chunk_size explicitly — None keeps convo's lower 30-char floor
    # (more permissive than the 50-char project default).
    explicit_min = palace_config.min_chunk_size_explicit
    cfg_min_chunk_size = explicit_min if explicit_min is not None else MIN_CHUNK_SIZE

    convo_path = Path(convo_dir).expanduser().resolve()
    wing = _resolve_wing(convo_path, wing)

    files = scan_convos(convo_dir, include_subagents=include_subagents)

    # Resolve workers from arg → config (asymmetric default: 1 onnx, 8 remote).
    if workers is None:
        workers = palace_config.workers

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine -- Conversations")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Source:  {convo_path}")
    limit_suffix = f" (limit: {limit} new)" if limit > 0 else ""
    print(f"  Files:   {len(files)}{limit_suffix}")
    print(f"  Palace:  {palace_path}")
    if not dry_run:
        print(f"  Workers: {workers}")
    if dry_run:
        print("  DRY RUN -- nothing will be filed")
    print(f"{'-' * 55}\n")

    collection = _open_convo_collection(
        palace_path,
        dry_run=dry_run,
    )

    # Bulk pre-fetch already-mined source_file -> stored mtime in one
    # paginated pass instead of `len(files)` separate WHERE-source_file
    # queries. On a 150k-drawer palace each per-file query costs ~2s, so a
    # 2000-file sweep used to spend >1h just deciding to skip.
    # prefetch_mined_set() does the same decisions in a single scan; the
    # producer below becomes an O(1) dict lookup + a cheap local mtime
    # comparison instead of a DB round trip per file.
    mined_mtimes: dict = (
        prefetch_mined_set(collection, extract_mode=extract_mode) if collection is not None else {}
    )
    # Same bulk pre-fetch for content-hash dedup (a re-export of the same
    # conversation under a new filename) — see _prepare_convo.
    content_hashes: dict = (
        prefetch_content_hashes(collection, extract_mode=extract_mode)
        if collection is not None
        else {}
    )
    total_drawers = 0
    files_skipped = 0
    files_failed = 0
    room_counts = defaultdict(int)

    if dry_run:
        # Dry-run keeps the serial path — no point parallelizing prints,
        # and there's no collection writer to coordinate around anyway.
        for i, filepath in enumerate(files, 1):
            # Same in-memory skip the real producer uses (below) — a dry
            # run must preview what a real mine would actually do, not
            # reprocess every file as if it were new work every time.
            if _is_unchanged_since_last_mine(str(filepath), mined_mtimes):
                files_skipped += 1
                continue

            try:
                content = normalize(str(filepath))
            except (OSError, ValueError):
                continue

            if not content or len(content.strip()) < cfg_min_chunk_size:
                continue

            # Same content-hash dedup as _prepare_convo (below) — a re-export
            # under a new filename must preview as a skip, not as new work.
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hashes:
                original = content_hashes.get((wing, content_hash))
                if original and original != str(filepath):
                    print(
                        f"    [dup] {filepath.name} -- duplicate of {Path(original).name}, skipping"
                    )
                    files_skipped += 1
                    continue

            if extract_mode == "general":
                from .general_extractor import extract_memories

                chunks = extract_memories(content, chunk_size=cfg_chunk_size)
            else:
                chunks = chunk_exchanges(
                    content,
                    chunk_size=cfg_chunk_size,
                    min_chunk_size=cfg_min_chunk_size,
                )

            if not chunks:
                continue

            if extract_mode == "general":
                from collections import Counter

                type_counts = Counter(c.get("memory_type", "general") for c in chunks)
                types_str = ", ".join(f"{t}:{n}" for t, n in type_counts.most_common())
                print(f"    [DRY RUN] {filepath.name} → {len(chunks)} memories ({types_str})")
                for c in chunks:
                    room_counts[c.get("memory_type", "general")] += 1
            else:
                room = detect_convo_room(content)
                print(f"    [DRY RUN] {filepath.name} → room:{room} ({len(chunks)} drawers)")
                room_counts[room] += 1
            total_drawers += len(chunks)
    else:
        # Bind the EF once so every producer thread shares the cached instance.
        from .embedding import get_embedding_function

        ef = get_embedding_function()

        # Producer: prepare convo file (read + chunk + build IDs/metadata),
        # then call the EF to compute embeddings per batch. Result payload
        # tags: "skip" (already mined), "register" (no chunks → sentinel),
        # "drawers" (full upsert payload).
        def producer(item):
            idx, filepath = item
            # Fast in-memory skip via the bulk prefetch — avoids the
            # per-file file_already_mined() DB round trip _prepare_convo
            # would otherwise do for every already-current file.
            if _is_unchanged_since_last_mine(str(filepath), mined_mtimes):
                return WorkerResult(
                    payload=("skip",),
                    item_id=filepath.name,
                    extra={"idx": idx, "total": len(files)},
                )
            result = _prepare_convo(
                filepath,
                wing,
                agent,
                extract_mode,
                False,
                collection,
                chunk_size=cfg_chunk_size,
                min_chunk_size=cfg_min_chunk_size,
                content_hashes=content_hashes,
            )
            if result is None:
                return WorkerResult(
                    payload=("skip",),
                    item_id=filepath.name,
                    extra={"idx": idx, "total": len(files)},
                )
            if isinstance(result, tuple) and result[0] == "register":
                _, reg_source_file, reg_content_hash, reg_is_duplicate = result
                return WorkerResult(
                    payload=("register", reg_source_file, reg_content_hash, reg_is_duplicate),
                    item_id=filepath.name,
                    extra={"idx": idx, "total": len(files)},
                )
            embeddings = _embed_prepared_convo(result, ef)
            return WorkerResult(
                payload=("drawers", result, embeddings),
                item_id=filepath.name,
                extra={"idx": idx, "total": len(files), "room": result.room},
            )

        # Consumer: single-thread writer for all chromadb operations.
        # Preserves HNSW's num_threads=1 invariant.
        def consumer(result):
            nonlocal total_drawers, files_skipped
            kind = result.payload[0]
            idx = result.extra["idx"]
            total = result.extra["total"]

            if kind == "skip":
                files_skipped += 1
                return

            if kind == "register":
                # A readable file that yields no chunks is registered (sentinel
                # written) but is NOT an "already filed" skip — it was processed
                # this run. Counting it under files_skipped mislabels it and
                # inflates the skip line (e.g. re-mining a transcript in a
                # different extract_mode). Matches the serial mine, which
                # registers-and-continues without incrementing the skip count.
                # A content-hash duplicate is the opposite: nothing new was
                # filed, so it IS a skip.
                _, source_file, content_hash, is_duplicate = result.payload
                _register_file(
                    collection, source_file, wing, agent, extract_mode, content_hash=content_hash
                )
                if is_duplicate:
                    files_skipped += 1
                return

            # drawers
            _, prepared, embeddings = result.payload
            drawers_added, room_delta, was_skipped = _write_prepared_convo(
                prepared, embeddings, collection, wing
            )
            if was_skipped:
                files_skipped += 1
                return
            if drawers_added == 0:
                files_skipped += 1
                return

            total_drawers += drawers_added
            if prepared.extract_mode == "general":
                for r, n in room_delta.items():
                    room_counts[r] += n
            else:
                room_counts[prepared.room] += 1
            print(f"  + [{idx:4}/{total}] {result.item_id[:50]:50} +{drawers_added}")

        def on_error(failed):
            nonlocal files_failed
            files_failed += 1
            print(
                f"  ! [{failed.item_id[:50]:50}] FAILED: "
                f"{type(failed.exception).__name__}: {failed.exception}",
                file=sys.stderr,
            )

        pipeline = ParallelPipeline(
            producer_fn=producer,
            consumer_fn=consumer,
            workers=workers,
            queue_size=max(workers * 2, 4),
            on_error=on_error,
        )
        pipeline.run(list(enumerate(files, 1)))

    if not dry_run:
        # Compute hallways before the FTS5 validation: the latter opens a direct sqlite
        # connection to the Chroma DB, which can invalidate the live collection handle on
        # some Chroma builds and make the hallway fetch fail.
        _compute_hallways_for_wing_safe(wing, collection, total_drawers, config=palace_config)
        # End-of-mine FTS5 integrity gate (#1537), mirroring miner._mine_impl.
        # Runs only after real writes; raises MineValidationError on corruption.
        _validate_palace_fts5_after_mine(palace_path)

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {len(files) - files_skipped - files_failed}")
    print(f"  Files skipped (already filed): {files_skipped}")
    if files_failed:
        print(f"  Files failed: {files_failed}")
    print(f"  Drawers filed: {total_drawers}")
    if room_counts:
        print("\n  By room:")
        for room, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {room:20} {count} files")
    print('\n  Next: mempalace search "what you\'re looking for"')
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convo_miner.py <convo_dir> [--palace PATH] [--limit N] [--dry-run]")
        sys.exit(1)
    from .config import MempalaceConfig

    mine_convos(sys.argv[1], palace_path=MempalaceConfig().palace_path)
