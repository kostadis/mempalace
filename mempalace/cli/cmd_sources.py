# Loaded into mempalace.cli via exec (see __init__.py). Not a standalone module.
if __name__ != "mempalace.cli":
    raise ImportError(f"{__name__} is an implementation fragment; import mempalace.cli")


def _collect_sources(col, wing):
    """Aggregate drawer metadata into ``{(wing, source_file): [mtime|None, drawers]}``.

    Pages through every drawer's metadata rather than using ``facet_counts``:
    the facet path returns counts only (no ``source_mtime``) and truncates at
    its ``limit``, so it cannot give a complete, mtime-bearing answer.
    Drawers with no ``source_file`` (diary / explicit adds) are not file
    sources and are left out.
    """
    from ..sync import _iter_drawer_metadata

    found = {}
    for _drawer_id, meta in _iter_drawer_metadata(col, wing):
        meta = meta or {}
        src = meta.get("source_file")
        if not isinstance(src, str) or not src:
            continue
        key = (str(meta.get("wing") or ""), src)
        mtime = meta.get("source_mtime")
        try:
            mtime = None if mtime is None or isinstance(mtime, bool) else float(mtime)
        except (TypeError, ValueError):
            mtime = None
        entry = found.setdefault(key, [None, 0])
        entry[1] += 1
        if mtime is not None and (entry[0] is None or mtime > entry[0]):
            entry[0] = mtime
    return found


def cmd_sources(args):
    """List every ``(wing, source_file)`` with drawer count. Read-only."""
    import json

    palace_path = _resolve_cli_palace(args)
    try:
        if not os.path.isdir(palace_path):
            raise FileNotFoundError(f"no palace at {palace_path}")
        from ..palace import get_collection

        col = get_collection(palace_path, create=False, read_only=True)
        found = _collect_sources(col, args.wing)
    except Exception as exc:  # noqa: BLE001 - any failure means "unknown", never "empty"
        print(f"mempalace: cannot read sources from {palace_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    rows = [
        {"wing": w, "source_file": s, "source_mtime": m, "drawers": n}
        for (w, s), (m, n) in sorted(found.items())
    ]
    if getattr(args, "as_json", False):
        print(json.dumps(rows))
        return
    if not rows:
        print("  No sources.")
        return
    for r in rows:
        print(f"  {r['wing']}  {r['source_file']}  ({r['drawers']})")
