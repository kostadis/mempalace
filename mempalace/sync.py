"""
sync.py — Gitignore-aware drawer prune (#1252).

Removes drawers whose source files are now gitignored, deleted, or moved
out of the project. Reuses the same GitignoreMatcher infrastructure that
the miner uses on the way in, so the same rules that block ingest also
drive the corresponding cleanup.

Usage:
    from mempalace.sync import sync_palace
    report = sync_palace(palace_path, project_dirs=["/repo"], dry_run=True)
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, TypedDict

from .config import normalize_wing_name
from .miner import is_ignored as is_gitignored, load_ignore_matcher as load_gitignore_matcher
from .palace import (
    MineAlreadyRunning,
    get_closets_collection,
    get_collection,
    mine_palace_lock,
)


logger = logging.getLogger(__name__)
_BATCH = 1000


class SyncReport(TypedDict):
    scanned: int
    kept: int
    gitignored: int
    missing: int
    no_source: int
    out_of_scope: int
    removed_drawers: int
    removed_closets: int
    dry_run: bool
    by_source: dict[str, int]


_WING_CONFIG_NAMES = ("mempalace.yaml", "mempalace.yml", "mempal.yaml", "mempal.yml")


def _resolve_project_root(source_file: Path, project_roots: list) -> Optional[Path]:
    """Return the longest project_root that source_file lives under.

    Assumes ``project_roots`` is sorted by path-length descending so the
    first match is the longest (deepest) prefix.
    """
    for root in project_roots:
        try:
            source_file.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def _find_wing_source_root(
    source_file: Path,
    project_root: Path,
    wing_root_cache: dict,
    wing: Optional[str] = None,
) -> Path:
    """Return the nearest per-wing mine root for ``source_file`` (within
    ``project_root``).

    Stops at the nearest ``mempalace.yaml``, at the first ancestor whose
    basename matches ``wing`` (auto-detect parity with ``load_config``), or
    at ``project_root``.

    The miner walks each wing's tree from its own mine directory, so that
    directory — not a higher ancestor — is the matcher scope for the wing's
    drawers. A root-level ``.mempalaceignore`` typically lists each sub-
    wing's source dir so the ROOT wing's mine skips them; those patterns are
    not in scope for sub-wing drawers, which were deliberately mined from
    inside the listed dirs.

    Walks are cached per (directory, wing) so every chunk under a subtree
    reuses the answer.
    """
    cache_wing = wing or ""
    visited: list = []
    candidate = source_file.parent
    while True:
        cache_key = (candidate, cache_wing)
        if cache_key in wing_root_cache:
            answer = wing_root_cache[cache_key]
            break
        visited.append(candidate)
        if any((candidate / name).is_file() for name in _WING_CONFIG_NAMES):
            answer = candidate
            break
        if wing and normalize_wing_name(candidate.name) == wing:
            answer = candidate
            break
        if candidate == project_root or candidate.parent == candidate:
            answer = project_root
            break
        candidate = candidate.parent
    for d in visited:
        wing_root_cache[(d, cache_wing)] = answer
    return answer


def _resolve_matcher_root(
    source_file: Path,
    project_root: Path,
    wing: Optional[str],
    wing_root_cache: dict,
) -> Path:
    """Pick the per-wing ignore-matcher scope for ``source_file``."""
    return _find_wing_source_root(source_file, project_root, wing_root_cache, wing)


def _ancestor_matchers(source_file: Path, root: Path, matcher_cache: dict) -> list:
    """Build the ancestor-chain matcher list, root → file's parent.

    Callers are expected to invoke this only after `_resolve_project_root`
    confirms `source_file` lives under `root`. The defensive try/except
    keeps the function safe if a future caller skips that check.
    """
    matchers: list = []
    try:
        parts = source_file.relative_to(root).parts
    except ValueError:
        return matchers
    cursor = root
    matcher = load_gitignore_matcher(cursor, matcher_cache)
    if matcher is not None:
        matchers.append(matcher)
    for part in parts[:-1]:
        cursor = cursor / part
        matcher = load_gitignore_matcher(cursor, matcher_cache)
        if matcher is not None:
            matchers.append(matcher)
    return matchers


def _is_registry_row(meta: dict, drawer_id: str) -> bool:
    """Convo miner sentinels track 'have I seen this transcript' — preserve them.

    Deleting a `_reg_*` sentinel makes the next mine pass re-chunk and re-embed
    the entire transcript even though its content has not changed.
    """
    if (meta or {}).get("room") == "_registry":
        return True
    if (meta or {}).get("ingest_mode") == "registry":
        return True
    if drawer_id and drawer_id.startswith("_reg_"):
        return True
    return False


def _classify_drawer(
    meta: dict,
    matcher_cache: dict,
    project_roots: list,
    drawer_id: str = "",
    wing_root_cache: Optional[dict] = None,
) -> str:
    """Classify a drawer by its source_file metadata.

    Returns one of: kept, gitignored, missing, no_source, out_of_scope.

    ``wing_root_cache`` (optional, recommended for production callers) narrows
    the matcher scope to the drawer's per-wing mine root — the nearest
    ``mempalace.yaml`` or auto-detect dirname match — so a root-level
    ``.mempalaceignore`` that excludes sub-wing source dirs from the ROOT wing's
    mine does not flag legitimate sub-wing drawers as gitignored. Without it
    the function falls back to the user-supplied project_root, which is only
    safe for single-wing palaces.
    """
    # Defensive: main loop filters registry rows; this guards direct callers.
    if _is_registry_row(meta, drawer_id):
        return "kept"

    source_file = (meta or {}).get("source_file")
    if not source_file:
        return "no_source"

    src = Path(source_file)
    if not src.is_absolute():
        return "no_source"
    src = src.resolve(strict=False)

    root = _resolve_project_root(src, project_roots)
    if root is None:
        return "out_of_scope"

    if not src.exists():
        return "missing"

    if wing_root_cache is not None:
        matcher_root = _resolve_matcher_root(
            src,
            root,
            (meta or {}).get("wing"),
            wing_root_cache,
        )
    else:
        matcher_root = root
    matchers = _ancestor_matchers(src, matcher_root, matcher_cache)
    if matchers and is_gitignored(src, matchers, is_dir=False):
        return "gitignored"

    return "kept"


def _iter_drawer_metadata(col, wing: Optional[str]):
    """Yield (id, metadata) tuples from the drawers collection in batches."""
    offset = 0
    where = {"wing": wing} if wing else None
    while True:
        kwargs = {"include": ["metadatas"], "limit": _BATCH, "offset": offset}
        if where:
            kwargs["where"] = where
        batch = col.get(**kwargs)
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or []
        if not ids:
            return
        for drawer_id, meta in zip(ids, metas):
            yield drawer_id, meta
        if len(ids) < _BATCH:
            return
        offset += len(ids)


def _auto_detect_project_roots(col, wing: Optional[str]) -> list:
    """Walk drawer metadata once collecting candidate project roots.

    A path is a project root if any ancestor up to filesystem root holds
    a `.git` directory or a `.gitignore` file. The deepest such ancestor
    wins, so nested-but-still-tracked subprojects are honoured.
    `Path.parents` iterates deepest-first, so the first hit IS deepest.

    Dedupes on ``source_file`` string so a 200-chunk file costs one disk
    walk, not 200.
    """
    roots: set = set()
    seen_sources: set = set()
    for _, meta in _iter_drawer_metadata(col, wing):
        source_file = (meta or {}).get("source_file")
        if not source_file or source_file in seen_sources:
            continue
        seen_sources.add(source_file)
        src = Path(source_file)
        if not src.is_absolute():
            continue
        for parent in src.parents:
            if (parent / ".git").exists() or (parent / ".gitignore").is_file():
                roots.add(parent.resolve(strict=False))
                break
    return sorted(roots, key=lambda p: (-len(str(p)), str(p)))


def _normalize_project_dirs(project_dirs) -> list:
    """Resolve and sort project dirs so deepest-prefix wins on first match."""
    resolved = [Path(p).resolve(strict=False) for p in project_dirs]
    return sorted(resolved, key=lambda p: (-len(str(p)), str(p)))


def _delete_in_batches(col, ids: list, batch_size: int, wal_log: Optional[Callable]):
    """Delete drawer IDs in batches, optionally logging each batch to WAL."""
    deleted = 0
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        col.delete(ids=chunk)
        deleted += len(chunk)
        if wal_log is not None:
            wal_log(
                "sync_prune",
                {"first_id": chunk[0]},
                {"removed_count": len(chunk)},
            )
    return deleted


def sync_palace(
    palace_path: str,
    project_dirs: Optional[list] = None,
    wing: Optional[str] = None,
    dry_run: bool = True,
    batch_size: int = _BATCH,
    wal_log: Optional[Callable] = None,
) -> SyncReport:
    """Prune drawers whose source files are gitignored, missing, or moved.

    Returns a SyncReport with bucket counts. Dry-run by default; pass
    dry_run=False to actually delete drawers and matching closets.

    Holds ``mine_palace_lock`` for the whole call so the classify pass and
    the apply branch see the same drawer snapshot. Raises
    ``MineAlreadyRunning`` if another mine is in progress on this palace.

    On apply (``dry_run=False``), at least one of ``wing`` or
    ``project_dirs`` must be set so a caller cannot accidentally prune
    every wing in a multi-project palace via auto-detected roots.
    """
    if not dry_run and not wing and not project_dirs:
        raise ValueError(
            "sync apply requires explicit wing= or project_dirs= so it cannot "
            "auto-prune every wing in a multi-project palace; pass --wing or "
            "a project directory"
        )
    if project_dirs is not None and not project_dirs:
        raise ValueError(
            "project_dirs was provided but is empty; pass at least one project "
            "root or pass project_dirs=None to auto-detect from drawer metadata"
        )

    counts = {
        "scanned": 0,
        "kept": 0,
        "gitignored": 0,
        "missing": 0,
        "no_source": 0,
        "out_of_scope": 0,
    }
    by_source: dict = defaultdict(int)
    removable_ids: list = []
    removable_sources: set = set()

    with mine_palace_lock(palace_path):
        col = get_collection(palace_path, create=False)

        if project_dirs is not None:
            roots = _normalize_project_dirs(project_dirs)
        else:
            roots = _auto_detect_project_roots(col, wing)

        matcher_cache: dict = {}
        # Wing-root lookups are cached per (directory, wing) so every drawer
        # under one wing reuses the answer instead of re-walking the tree.
        wing_root_cache: dict = {}
        # Same source_file → same verdict holds because mine_palace_lock
        # blocks concurrent writers and the loop is synchronous.
        classification_cache: dict = {}

        for drawer_id, meta in _iter_drawer_metadata(col, wing):
            counts["scanned"] += 1
            meta = meta or {}
            source_file = meta.get("source_file")

            if _is_registry_row(meta, drawer_id):
                bucket = "kept"
            elif source_file and source_file in classification_cache:
                bucket = classification_cache[source_file]
            else:
                bucket = _classify_drawer(
                    meta,
                    matcher_cache,
                    roots,
                    drawer_id,
                    wing_root_cache,
                )
                if source_file:
                    classification_cache[source_file] = bucket

            counts[bucket] += 1
            if bucket in ("gitignored", "missing"):
                removable_ids.append(drawer_id)
                if source_file:
                    removable_sources.add(source_file)
                    by_source[source_file] += 1

        report: SyncReport = {
            **counts,
            "removed_drawers": 0,
            "removed_closets": 0,
            "dry_run": dry_run,
            "by_source": dict(by_source),
        }

        if dry_run or not removable_ids:
            return report

        report["removed_drawers"] = _delete_in_batches(col, removable_ids, batch_size, wal_log)

        closets_col = None
        try:
            closets_col = get_closets_collection(palace_path, create=False)
        except Exception as exc:
            logger.warning("Closet purge skipped (collection unavailable): %s", exc)

        closets_removed = 0
        if closets_col is not None and removable_sources:
            closet_ids = (
                closets_col.get(
                    where={"source_file": {"$in": list(removable_sources)}},
                    include=[],
                ).get("ids")
                or []
            )
            if closet_ids:
                closets_col.delete(ids=closet_ids)
                closets_removed = len(closet_ids)
        report["removed_closets"] = closets_removed
    return report


__all__ = [
    "MineAlreadyRunning",
    "SyncReport",
    "sync_palace",
]
