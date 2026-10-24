# Loaded into mempalace.cli via exec (see __init__.py). Not a standalone module.
if __name__ != "mempalace.cli":
    raise ImportError(f"{__name__} is an implementation fragment; import mempalace.cli")


def _read_sources_from(spec):
    """Read ``source_file`` values (``-`` is stdin); relative paths resolve against cwd."""
    try:
        if spec == "-":
            text = sys.stdin.read()
        else:
            with open(os.path.expanduser(spec), encoding="utf-8", errors="surrogateescape") as fh:
                text = fh.read()
    except OSError as exc:
        print(
            f"mempalace: cannot read --sources-from {spec!r}: {exc.strerror or exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    out = []
    seen = set()
    for line in text.split("\n"):
        line = line.strip("\r\n")
        if not line.strip():
            continue
        src = os.path.abspath(os.path.expanduser(line))
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out


def _source_where(source, wing):
    if wing:
        return {"$and": [{"source_file": source}, {"wing": wing}]}
    return {"source_file": source}


def _count_source_drawers(col, source, wing):
    """Exact number of drawers for ``source`` (within ``wing`` if given), paged."""
    where = _source_where(source, wing)
    total = 0
    offset = 0
    page = 1000
    while True:
        ids = col.get(where=where, include=[], limit=page, offset=offset).get("ids") or []
        total += len(ids)
        if len(ids) < page:
            return total
        offset += len(ids)


def cmd_forget(args):
    """Delete the drawers of the listed source files. Dry run unless ``--apply``."""
    sources = _read_sources_from(args.sources_from)
    palace_path = _resolve_cli_palace(args)
    apply = bool(args.apply)
    wing = args.wing

    from ..palace import (
        MineAlreadyRunning,
        get_closets_collection,
        get_collection,
        mine_palace_lock,
    )

    per_source = {}
    try:
        if not os.path.isdir(palace_path):
            raise FileNotFoundError(f"no palace at {palace_path}")
        lock = mine_palace_lock(palace_path) if apply else contextlib.nullcontext()
        with lock:
            col = get_collection(palace_path, create=False, read_only=not apply)
            for src in sources:
                n = _count_source_drawers(col, src, wing)
                if n:
                    per_source[src] = n
            if apply and per_source:
                from ..wal import _wal_log

                for src, n in per_source.items():
                    _wal_log("forget_source", {"source_file": src, "wing": wing, "match_count": n})
                    col.delete(where=_source_where(src, wing))
                    # Drop the closet index for this source only once no drawer of
                    # it remains in any wing; another wing's drawers keep theirs.
                    if _count_source_drawers(col, src, None) == 0:
                        try:
                            closets = get_closets_collection(palace_path, create=False)
                            closets.delete(where={"source_file": src})
                        except Exception:  # noqa: BLE001 - best-effort, as sync does
                            pass
    except MineAlreadyRunning as exc:
        print(f"mempalace: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - any failure must be non-zero
        print(f"mempalace: forget failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Sources: {len(per_source)}")
    print(f"Drawers: {sum(per_source.values())}")
    for src in sorted(per_source):
        print(f"  {src}  ({per_source[src]})")
    if not apply and per_source:
        print("\n  Dry run: nothing deleted. Re-run with --apply to delete.", file=sys.stderr)
