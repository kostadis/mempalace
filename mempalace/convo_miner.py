#!/usr/bin/env python3
"""
convo_miner.py — Mine conversations into the palace.

Ingests chat exports (Claude Code, ChatGPT, Slack, plain text transcripts).
Normalizes format, chunks by exchange pair (Q+A = one unit), files to palace.

Same palace as project mining. Different ingest strategy.
"""

import os
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

from .config import MempalaceConfig
from .normalize import normalize
from .palace import (
    NORMALIZE_VERSION,
    SKIP_DIRS,
    file_already_mined,
    get_collection,
    mine_lock,
)
from .parallel import ParallelPipeline, WorkerResult


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


# File types that might contain conversations
CONVO_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

MIN_CHUNK_SIZE = 30
CHUNK_SIZE = 2000  # chars per drawer — approx 500 tokens, safely under 2048 limit
DRAWER_UPSERT_BATCH_SIZE = 10
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB — skip files larger than this.
# Matches miner.py at 500 MB. Long Claude Code sessions, multi-year
# ChatGPT exports, and lifetime Slack dumps routinely exceed 10 MB; the
# cap at that level silently dropped them with `continue`. Per-drawer
# size is bounded by CHUNK_SIZE, but larger source files still produce
# more drawers and therefore more embedding/storage work — and content
# is normalized and loaded fully into memory before chunking, so memory
# use also scales with source size.


def _register_file(collection, source_file: str, wing: str, agent: str):
    """Write a sentinel so file_already_mined() returns True for 0-chunk files.

    Without this, files that normalize to nothing or produce zero chunks are
    re-read and re-processed on every mine run because nothing was written to
    ChromaDB on the first pass.
    """
    sentinel_id = f"_reg_{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"
    collection.upsert(
        documents=[f"[registry] {source_file}"],
        ids=[sentinel_id],
        metadatas=[
            {
                "wing": wing,
                "room": "_registry",
                "source_file": source_file,
                "added_by": agent,
                "filed_at": datetime.now().isoformat(),
                "ingest_mode": "registry",
                "normalize_version": NORMALIZE_VERSION,
            }
        ],
    )


# =============================================================================
# CHUNKING — exchange pairs for conversations
# =============================================================================


def chunk_exchanges(content: str) -> list:
    """
    Chunk by exchange pair: one > turn + AI response = one unit.
    Falls back to paragraph chunking if no > markers.
    """
    if not content.strip():
        return []
    lines = content.split("\n")
    quote_lines = sum(1 for line in lines if line.strip().startswith(">"))

    if quote_lines >= 1:
        return _chunk_by_exchange(lines)
    else:
        # If there are few quote lines, it's likely not a standard exchange-pair format.
        # Use paragraph chunking as a fallback.
        return _chunk_by_paragraph(content)


def _chunk_by_exchange(lines: list) -> list:
    """One user turn (>) + the AI response that follows = one or more chunks.
    
    The full AI response is preserved verbatim.  When the combined
    user-turn + response exceeds CHUNK_SIZE the response is split across
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
                if next_line.strip():
                    ai_lines.append(next_line.strip())
                i += 1

            ai_response = " ".join(ai_lines)
            content = f"{user_turn}\n{ai_response}" if ai_response else user_turn

            # Split into multiple drawers when the exchange exceeds CHUNK_SIZE
            if len(content) > CHUNK_SIZE:
                # First chunk: user turn + as much response as fits
                first_part = content[:CHUNK_SIZE]
                chunks.append({"content": first_part, "chunk_index": len(chunks)})
                
                # Remaining response in CHUNK_SIZE-sized continuation drawers
                remainder = content[CHUNK_SIZE:]
                while remainder:
                    part = remainder[:CHUNK_SIZE]
                    remainder = remainder[CHUNK_SIZE:]
                    chunks.append({"content": part, "chunk_index": len(chunks)})
            else:
                chunks.append(
                    {
                        "content": content,
                        "chunk_index": len(chunks),
                    }
                )
            # Safety check: ensure no chunk somehow escaped the limit
            assert len(chunks[-1]["content"]) <= CHUNK_SIZE, f"Chunk size {len(chunks[-1]['content'])} exceeds CHUNK_SIZE {CHUNK_SIZE}"


        else:
            i += 1

    return chunks



def _chunk_by_paragraph(content: str) -> list:
    """Fallback: chunk by paragraph breaks, with sub-chunking for large paragraphs."""
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no paragraph breaks and long content, chunk by line groups
    if len(paragraphs) <= 1 and content.count("\n") > 20:
        lines = content.split("\n")
        for i in range(0, len(lines), 25):
            group = "\n".join(lines[i : i + 25]).strip()
            if len(group) > MIN_CHUNK_SIZE:
                # Sub-chunk the line group if it's still too big
                remainder = group
                while remainder:
                    part = remainder[:CHUNK_SIZE]
                    remainder = remainder[CHUNK_SIZE:]
                    chunks.append({"content": part, "chunk_index": len(chunks)})
        return chunks

    for para in paragraphs:
        if len(para) < MIN_CHUNK_SIZE:
            continue

        if len(para) <= CHUNK_SIZE:
            chunks.append({"content": para, "chunk_index": len(chunks)})
        else:
            # Sub-chunk the large paragraph
            remainder = para
            while remainder:
                part = remainder[:CHUNK_SIZE]
                remainder = remainder[CHUNK_SIZE:]
                chunks.append({"content": part, "chunk_index": len(chunks)})

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


def scan_convos(convo_dir: str) -> list:
    """Find all potential conversation files."""
    convo_path = Path(convo_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(convo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() in CONVO_EXTENSIONS:
                # Skip symlinks and oversized files
                if filepath.is_symlink():
                    continue
                try:
                    if filepath.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MINE CONVERSATIONS
# =============================================================================


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
):
    """Read + normalize + chunk + build per-batch documents/ids/metadata.

    Returns one of:
      * ``None`` — file is already mined (pure skip).
      * ``("register", source_file)`` — readable but yields no chunks; the
        consumer must write the sentinel via _register_file so it's marked
        seen on the next run.
      * ``_PreparedConvo(...)`` — ready for embed + write.

    Safe to call from a producer thread; does NOT acquire mine_lock and
    does NOT touch the collection writer.
    """
    source_file = str(filepath)
    if not dry_run and file_already_mined(collection, source_file):
        return None

    try:
        content = normalize(str(filepath))
    except (OSError, ValueError):
        return ("register", source_file)

    if not content or len(content.strip()) < MIN_CHUNK_SIZE:
        return ("register", source_file)

    if extract_mode == "general":
        from .general_extractor import extract_memories

        chunks = extract_memories(content)
    else:
        chunks = chunk_exchanges(content)

    if not chunks:
        return ("register", source_file)

    if extract_mode != "general":
        room = detect_convo_room(content)
    else:
        room = None  # per-chunk rooms below

    # Single filed_at per source so all drawers from this transcript
    # share an ingest timestamp.
    filed_at = datetime.now().isoformat()
    batches: list = []
    for batch_start in range(0, len(chunks), DRAWER_UPSERT_BATCH_SIZE):
        batch_docs: list = []
        batch_ids: list = []
        batch_metas: list = []
        batch_rooms: list = []
        for chunk in chunks[batch_start : batch_start + DRAWER_UPSERT_BATCH_SIZE]:
            chunk_room = chunk.get("memory_type", room) if extract_mode == "general" else room
            drawer_id = f"drawer_{wing}_{chunk_room}_{hashlib.sha256((source_file + str(chunk['chunk_index'])).encode()).hexdigest()[:24]}"
            batch_docs.append(chunk["content"])
            batch_ids.append(drawer_id)
            batch_metas.append(
                {
                    "wing": wing,
                    "room": chunk_room,
                    "hall": _detect_hall_cached(chunk["content"]),
                    "source_file": source_file,
                    "chunk_index": chunk["chunk_index"],
                    "added_by": agent,
                    "filed_at": filed_at,
                    "ingest_mode": "convos",
                    "extract_mode": extract_mode,
                    "normalize_version": NORMALIZE_VERSION,
                }
            )
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
        if file_already_mined(collection, source_file):
            return 0, room_counts_delta, True

        # Purge stale drawers first — pre-v2 drawers wouldn't have triggered
        # file_already_mined, so we clean them out before re-insert.
        try:
            collection.delete(where={"source_file": source_file})
        except Exception:
            pass

        for batch, batch_embeddings in zip(prepared.batches, embeddings_batches):
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
):
    """Mine a directory of conversation files into the palace.

    extract_mode:
        "exchange" — default exchange-pair chunking (Q+A = one unit)
        "general"  — general extractor: decisions, preferences, milestones, problems, emotions

    workers: producer-thread fan-out for the parallel embed path. ``None``
    (default) resolves to :attr:`MempalaceConfig.workers` — 1 onnx /
    8 remote. ``1`` reproduces the historic fully-serial mine.
    """

    convo_path = Path(convo_dir).expanduser().resolve()
    if not wing:
        from .config import normalize_wing_name

        wing = normalize_wing_name(convo_path.name)

    files = scan_convos(convo_dir)
    if limit > 0:
        files = files[:limit]

    # Resolve workers from arg → config (asymmetric default: 1 onnx, 8 remote).
    if workers is None:
        workers = MempalaceConfig().workers

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine — Conversations")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Source:  {convo_path}")
    print(f"  Files:   {len(files)}")
    print(f"  Palace:  {palace_path}")
    if not dry_run:
        print(f"  Workers: {workers}")
    if dry_run:
        print("  DRY RUN — nothing will be filed")
    print(f"{'-' * 55}\n")

    collection = get_collection(palace_path) if not dry_run else None

    total_drawers = 0
    files_skipped = 0
    files_failed = 0
    room_counts = defaultdict(int)

    if dry_run:
        # Dry-run keeps the serial path — no point parallelizing prints,
        # and there's no collection writer to coordinate around anyway.
        for i, filepath in enumerate(files, 1):
            source_file = str(filepath)

            try:
                content = normalize(str(filepath))
            except (OSError, ValueError):
                continue

            if not content or len(content.strip()) < MIN_CHUNK_SIZE:
                continue

            if extract_mode == "general":
                from .general_extractor import extract_memories

                chunks = extract_memories(content)
            else:
                chunks = chunk_exchanges(content)

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
            result = _prepare_convo(
                filepath, wing, agent, extract_mode, False, collection
            )
            if result is None:
                return WorkerResult(
                    payload=("skip",),
                    item_id=filepath.name,
                    extra={"idx": idx, "total": len(files)},
                )
            if isinstance(result, tuple) and result[0] == "register":
                return WorkerResult(
                    payload=("register", result[1]),
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
                source_file = result.payload[1]
                _register_file(collection, source_file, wing, agent)
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
