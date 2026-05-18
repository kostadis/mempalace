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
from .miner import (
    detect_ignore_filename,
    is_ignored as apply_ignore_matchers,
    load_ignore_matcher,
)
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
    wing_name: Optional[str],
    cache: dict,
) -> Path:
    """Find the wing's mine root — the dir originally passed to ``mempalace mine``.

    The miner's ignore walk starts at the mine root and goes downward, so
    every ignore file the miner saw lives at-or-below it. Mirroring that
    scope at sync time keeps a drawer from being flagged by ancestor
    ignore files the miner never consulted (the original Phandalin bug:
    a root ``.mempalaceignore`` listing sub-wing dirs flagged every
    sub-wing drawer).

    Walks ``source_file``'s parents up to ``project_root`` (inclusive):

    - **Basename match wins immediately.** If ``normalize_wing_name(dir.name)``
      equals the drawer's wing, this is the wing's mine root. Auto-detected
      wings inherit their name from the source-dir basename via the same
      normalization, so this match identifies the original mine dir
      directly.
    - **Otherwise, the innermost ``mempalace.yaml`` ancestor wins.** Yaml-
      configured wings whose ``wing:`` field differs from the dir basename
      (e.g. dir ``chapters/`` with ``wing: narrative``) get found here.
    - **Otherwise, fall back to ``project_root``.** Preserves the historical
      single-wing behaviour when neither signal pins down a wing-specific
      mine root.

    A yaml encountered higher up than a basename match is *not* preferred:
    the miner reads the yaml at the dir it was invoked on, not at ancestors.
    A misleading ancestor yaml (e.g. ``/repo/mempalace.yaml`` while the
    user mined ``/repo/sub/``) must not override the basename signal.

    Cache key is ``(dir, wing_name)``: the basename comparison is wing-
    specific so two wings sharing a path subtree need separate entries.
    """
    cache_key = (source_file.parent, wing_name)
    if cache_key in cache:
        return cache[cache_key]

    normalized_wing = normalize_wing_name(wing_name) if wing_name else None
    candidate = source_file.parent
    first_yaml: Optional[Path] = None
    answer: Path = project_root

    while True:
        if normalized_wing and normalize_wing_name(candidate.name) == normalized_wing:
            answer = candidate
            break
        if first_yaml is None and any((candidate / name).is_file() for name in _WING_CONFIG_NAMES):
            first_yaml = candidate
        if candidate == project_root or candidate.parent == candidate:
            answer = first_yaml if first_yaml is not None else project_root
            break
        candidate = candidate.parent

    cache[cache_key] = answer
    return answer


def _ancestor_matchers(
    source_file: Path,
    root: Path,
    matcher_cache: dict,
    filename: str,
) -> list:
    """Build the ancestor-chain matcher list, root → file's parent.

    Callers are expected to invoke this only after `_resolve_project_root`
    confirms `source_file` lives under `root`. The defensive try/except
    keeps the function safe if a future caller skips that check.

    ``filename`` selects which ignore file to read at each level (the
    project-scoped choice from :func:`detect_ignore_filename`). The
    historical per-dir fallback between ``.mempalaceignore`` and
    ``.gitignore`` is gone — projects pick one mode.
    """
    matchers: list = []
    try:
        parts = source_file.relative_to(root).parts
    except ValueError:
        return matchers
    cursor = root
    matcher = load_ignore_matcher(cursor, matcher_cache, filename=filename)
    if matcher is not None:
        matchers.append(matcher)
    for part in parts[:-1]:
        cursor = cursor / part
        matcher = load_ignore_matcher(cursor, matcher_cache, filename=filename)
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
    project_mode_map: dict,
    drawer_id: str = "",
    wing_root_cache: Optional[dict] = None,
) -> str:
    """Classify a drawer by its source_file metadata.

    Returns one of: kept, gitignored, missing, no_source, out_of_scope.

    ``project_mode_map`` maps each project_root to the ignore filename
    that applies (``.mempalaceignore`` or ``.gitignore``). Decided once
    per project_root by :func:`detect_ignore_filename` so the per-dir
    fallback between the two file types is gone — pick one per project.

    ``wing_root_cache`` (optional, recommended for production callers)
    narrows the matcher walk to the drawer's per-wing source root via
    :func:`_find_wing_source_root` so ancestor ignore files the wing's
    miner never saw don't flag legitimate sub-wing drawers as gitignored.
    Without it the function falls back to the user-supplied project_root,
    which is the historical (buggy-for-multi-wing-palaces) behaviour and
    only safe for single-wing palaces.
    """
    # Defensive: main loop filters registry rows; this guards direct callers.
    if _is_registry_row(meta, drawer_id):
        return "kept"

    meta = meta or {}
    source_file = meta.get("source_file")
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
        matcher_root = _find_wing_source_root(src, root, meta.get("wing"), wing_root_cache)
    else:
        matcher_root = root
    ignore_filename = project_mode_map[root]
    matchers = _ancestor_matchers(src, matcher_root, matcher_cache, ignore_filename)
    if matchers and apply_ignore_matchers(src, matchers, is_dir=False):
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

        # Each project_root picks its ignore-file mode independently — a
        # palace can sync against several project_dirs with different
        # conventions, but within one project_root the choice is fixed.
        project_mode_map: dict = {root: detect_ignore_filename(root) for root in roots}

        matcher_cache: dict = {}
        # Wing source-root lookups (yaml + basename heuristic) are cached
        # by (dir, wing_name) so every drawer under one wing reuses the
        # walk result.
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
                    project_mode_map,
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
