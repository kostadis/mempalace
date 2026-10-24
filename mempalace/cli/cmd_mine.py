# Loaded into mempalace.cli via exec (see __init__.py). Not a standalone module.
if __name__ != "mempalace.cli":
    raise ImportError(f"{__name__} is an implementation fragment; import mempalace.cli")


def _read_files_from(spec):
    """Read the newline-separated path list for ``--files-from`` (``-`` is stdin)."""
    try:
        if spec == "-":
            text = sys.stdin.read()
        else:
            with open(os.path.expanduser(spec), encoding="utf-8", errors="surrogateescape") as fh:
                text = fh.read()
    except OSError as exc:
        print(
            f"mempalace: cannot read --files-from {spec!r}: {exc.strerror or exc}", file=sys.stderr
        )
        sys.exit(2)
    return [line.strip("\r\n") for line in text.split("\n") if line.strip()]


def _prepare_files_from(args, include_ignored, mode, source_adapter):
    """Validate ``--files-from`` and return the list of Paths to hand to ``miner.mine``.

    Usage errors exit 2. Every listed path that cannot be mined is reported as
    ``  SKIP: <rel> (<reason>)`` on stderr (the ``scan_project`` format) and is
    not returned, so nothing is dropped silently. Returned paths are absolute
    and lexically under the resolved DIR, as ``scan_project`` yields them.
    """
    import stat

    from ..miner import READABLE_EXTENSIONS

    if include_ignored or getattr(args, "no_mempalaceignore", False):
        print(
            "mempalace: --files-from cannot be combined with --include-ignored, "
            "--no-gitignore or --no-mempalaceignore (a listed file list bypasses scanning "
            "and ignore files entirely)",
            file=sys.stderr,
        )
        sys.exit(2)
    if mode != "projects" or source_adapter:
        print("mempalace: --files-from is only supported in projects mode", file=sys.stderr)
        sys.exit(2)

    base = Path(os.path.expanduser(args.dir)).resolve()
    if not base.is_dir():
        print(f"mempalace: not a directory: {args.dir}", file=sys.stderr)
        sys.exit(2)

    def _skip(label, reason):
        try:
            print(f"  SKIP: {label} ({reason})", file=sys.stderr)
        except OSError:
            pass

    kept = []
    seen = set()
    for raw in _read_files_from(args.files_from):
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute():
            p = base / p
        # Resolve the directory part only: resolving the final component would
        # turn a symlink into its target and hide it from the symlink check.
        p = Path(os.path.normpath(str(p)))
        candidate = p.parent.resolve() / p.name
        try:
            rel = candidate.relative_to(base).as_posix()
        except ValueError:
            _skip(raw, "outside the mined directory")
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_symlink():
            _skip(rel, "symlink")
            continue
        try:
            st = candidate.stat()
        except FileNotFoundError:
            _skip(rel, "missing")
            continue
        except OSError as exc:
            _skip(rel, f"stat error: {exc.strerror or exc}")
            continue
        if not stat.S_ISREG(st.st_mode):
            _skip(rel, "not a regular file")
            continue
        if candidate.suffix.lower() not in READABLE_EXTENSIONS:
            _skip(rel, f"unsupported extension {candidate.suffix or '(none)'}")
            continue
        kept.append(candidate)
    return kept


def cmd_mine(args):
    palace_path = _resolve_cli_palace(args)
    mode = getattr(args, "mode", None) or "projects"
    source_adapter = getattr(args, "source", None)
    include_ignored = []
    for raw in args.include_ignored or []:
        include_ignored.extend(part.strip() for part in raw.split(",") if part.strip())

    payload = {
        "source": os.path.abspath(os.path.expanduser(args.dir)),
        "mode": mode,
        "wing": args.wing,
        "agent": args.agent,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "extract": args.extract,
        # Wire key stays "no_gitignore"; the local parser dest is no_mempalaceignore.
        "no_gitignore": getattr(args, "no_mempalaceignore", False),
        "include_ignored": include_ignored,
        "max_chunks_per_file": getattr(
            args,
            "max_chunks_per_file",
            None,
        ),
        "redetect_origin": getattr(
            args,
            "redetect_origin",
            False,
        ),
    }
    if source_adapter:
        payload["source_adapter"] = source_adapter

    files_from_list = None
    if getattr(args, "files_from", None) is not None:
        files_from_list = _prepare_files_from(args, include_ignored, mode, source_adapter)

    routing = _resolve_cli_write_routing_or_exit(
        args,
        "mine",
    )
    if routing.use_daemon and files_from_list is not None:
        print(
            "mempalace: --files-from cannot be submitted to the write daemon; "
            "run it with direct write routing",
            file=sys.stderr,
        )
        sys.exit(2)
    if routing.use_daemon:
        _submit_daemon_cli_job(
            "mine",
            payload,
            args,
            background=bool(getattr(args, "background", False)),
            auto_start=routing.decision.auto_start_daemon,
        )
        return

    from ..palace import MineAlreadyRunning, MineValidationError

    if source_adapter:
        try:
            drawers_written = mine_source_adapter(
                source_name=source_adapter,
                source_path=args.dir,
                palace_path=palace_path,
                dry_run=args.dry_run,
            )
        except (UnknownSourceAdapterError, UnsupportedSourceAdapterProtocolError) as exc:
            print(f"mempalace: {exc}", file=sys.stderr)
            sys.exit(2)
        except MineAlreadyRunning as exc:
            print(f"mempalace: {exc}", file=sys.stderr)
            sys.exit(1)
        suffix = " would be written" if args.dry_run else " written"
        print(f"  Source adapter {source_adapter!r}: {drawers_written} drawer(s){suffix}.")
        return

    # A live HTTP hub for this palace holds the MCP writer lease, so a
    # direct mine here would be refused. Hand the job to the hub instead —
    # this is how the save hooks keep capturing transcripts on a machine
    # that runs `mempalace serve`.
    # A --files-from mine is never forwarded: the mempalace_mine MCP tool has no
    # parameter for an explicit file list, so forwarding would silently mine the
    # whole directory instead. Like --include-ignored it takes the direct path.
    if (
        files_from_list is None
        and _mine_args_forwardable(args, include_ignored)
        and _forward_mine_to_hub(args, palace_path)
    ):
        return

    # --redetect-origin re-runs corpus_origin on the current corpus state
    # and overwrites <palace>/.mempalace/origin.json before mining proceeds.
    # Heuristic-only by design — full LLM detection lives on `mempalace init`.
    if getattr(args, "redetect_origin", False):
        _run_pass_zero(
            project_dir=args.dir,
            palace_dir=palace_path,
            llm_provider=None,
        )

    try:
        if mode == "convos":
            from ..convo_miner import mine_convos

            mine_convos(
                convo_dir=args.dir,
                palace_path=palace_path,
                wing=args.wing,
                agent=args.agent,
                limit=args.limit,
                dry_run=args.dry_run,
                extract_mode=args.extract,
                workers=getattr(args, "workers", None),
                include_subagents=getattr(args, "include_subagents", False),
            )
        elif mode == "extract":
            from ..format_miner import mine_formats

            mine_formats(
                format_dir=args.dir,
                palace_path=palace_path,
                wing=args.wing,
                agent=args.agent,
                limit=args.limit,
                dry_run=args.dry_run,
            )
        else:
            from ..miner import mine

            # Only pass ``files`` for --files-from; None would mean "scan" anyway.
            extra = {} if files_from_list is None else {"files": files_from_list}
            mine(
                project_dir=args.dir,
                palace_path=palace_path,
                wing_override=args.wing,
                agent=args.agent,
                limit=args.limit,
                dry_run=args.dry_run,
                respect_ignore=not getattr(args, "no_mempalaceignore", False),
                include_ignored=include_ignored,
                workers=getattr(args, "workers", None),
                max_chunks_per_file=getattr(args, "max_chunks_per_file", None),
                **extra,
            )
    except MineAlreadyRunning as exc:
        # A live MCP server or another mine is already writing to this
        # palace. Surface the holder identity so the operator knows what
        # to wait for (or stop), and exit non-zero so wrappers like
        # nohup / scripts can detect the contention.
        print(f"mempalace: {exc}", file=sys.stderr)
        sys.exit(1)
    except MineValidationError as exc:
        # PRAGMA quick_check on chroma.sqlite3 returned errors at end of mine.
        # The corruption may pre-date the mine; we surface it here so automation
        # cannot proceed against a half-broken palace. Reuse cmd_repair's
        # recovery banner so the operator sees one consistent message regardless
        # of which command surfaces it.
        from ..repair import print_sqlite_integrity_abort

        print_sqlite_integrity_abort(exc.palace_path, exc.errors)
        print(
            "\n  PRAGMA quick_check after this mine reported errors (the corruption\n"
            "  may pre-date the mine itself). Drawers may still be intact for direct\n"
            "  lookup; wing-filtered or full-text search will fail until the FTS5\n"
            "  index is rebuilt. `mempalace repair --yes` rebuilds the FTS5 virtual\n"
            "  table automatically (step 6 of the recovery above).",
            file=sys.stderr,
        )
        sys.exit(1)


class UnknownSourceAdapterError(ValueError):
    """Raised when an explicit ``--source`` name is absent from the registry."""


class UnsupportedSourceAdapterProtocolError(ValueError):
    """Raised when an adapter requires runner semantics not implemented yet."""


class _DryRunCollectionProxy:
    """Empty collection facade that records, but never persists, writes.

    Source adapters are deliberately allowed to access ``drawer_collection``
    directly.  A dry run must not open the real backend: even read-only-looking
    opens can create or repair backend artifacts (for example SQLite WAL files).
    """

    def __init__(self):
        self.operations = []

    def add(self, **kwargs):
        self.operations.append(("add", kwargs))

    def upsert(self, **kwargs):
        self.operations.append(("upsert", kwargs))

    def delete(self, **kwargs):
        self.operations.append(("delete", kwargs))

    def update(self, **kwargs):
        self.operations.append(("update", kwargs))

    def query(self, **kwargs):
        from ..backends import QueryResult

        query_input = kwargs.get("query_texts", kwargs.get("query_embeddings"))
        num_queries = len(query_input) if isinstance(query_input, (list, tuple)) else 1
        include = kwargs.get("include") or []
        return QueryResult.empty(
            num_queries=num_queries,
            embeddings_requested="embeddings" in include,
        )

    def get(self, **kwargs):
        from ..backends import GetResult

        return GetResult.empty()

    def count(self):
        return 0


class _DryRunKnowledgeGraphProxy:
    """Recording no-op facade for the KG mutation surface published to adapters."""

    def __init__(self):
        self.operations = []

    def add_entity(self, *args, **kwargs):
        self.operations.append(("add_entity", args, kwargs))

    def add_triple(self, *args, **kwargs):
        self.operations.append(("add_triple", args, kwargs))

    def invalidate(self, *args, **kwargs):
        self.operations.append(("invalidate", args, kwargs))

    def supersede(self, *args, **kwargs):
        self.operations.append(("supersede", args, kwargs))


def mine_source_adapter(
    *,
    source_name: str,
    source_path: str,
    palace_path: str,
    dry_run: bool = False,
) -> int:
    """Run an explicitly selected RFC 002 source adapter through ``PalaceContext``.

    This deliberately sits alongside, rather than inside, the legacy mode
    miners.  Until those miners are migrated to first-party adapters, no-flag
    and ``--mode`` calls must retain their established dispatch paths.
    """
    from ..knowledge_graph import KnowledgeGraph
    from ..palace import get_collection, mine_palace_lock
    from ..sources import (
        DrawerRecord,
        PalaceContext,
        SourceRef,
        SourceItemMetadata,
        get_adapter,
        resolve_adapter_for_source,
    )

    adapter_name = resolve_adapter_for_source(explicit=source_name)
    try:
        adapter = get_adapter(adapter_name)
    except KeyError as exc:
        raise UnknownSourceAdapterError(
            f"unknown source adapter {adapter_name!r}; install its adapter package or "
            "check the adapter name with `mempalace mine --help`"
        ) from exc

    if "supports_incremental" in adapter.capabilities:
        raise UnsupportedSourceAdapterProtocolError(
            f"source adapter {adapter_name!r} requires incremental ingestion, which "
            "mempalace mine does not support yet"
        )

    # A dry run must never open a collection: backend opens can create or
    # repair storage even when requested as read-only.  Non-dry runs hold one
    # writer lease from handle creation through adapter iteration, including
    # direct KG mutations by adapters.
    lock = mine_palace_lock(palace_path) if not dry_run else contextlib.nullcontext()
    with lock:
        knowledge_graph = None
        try:
            if dry_run:
                drawer_collection = _DryRunCollectionProxy()
                knowledge_graph = _DryRunKnowledgeGraphProxy()
            else:
                drawer_collection = get_collection(palace_path)
                knowledge_graph = KnowledgeGraph(
                    db_path=os.path.join(palace_path, "knowledge_graph.sqlite3")
                )
            context = PalaceContext(
                drawer_collection=drawer_collection,
                knowledge_graph=knowledge_graph,
                palace_path=palace_path,
                config=MempalaceConfig(palace_path=palace_path),
                adapter_name=adapter.name,
                adapter_version=adapter.adapter_version,
            )
            drawers_written = 0
            for result in adapter.ingest(
                source=SourceRef(local_path=source_path),
                palace=context,
            ):
                if isinstance(result, SourceItemMetadata):
                    # Non-incremental adapters may report a cursor or version
                    # while still doing a complete re-extract.  Incremental
                    # adapters are rejected before ingest above, so accepting
                    # this avoids a late partial-ingest failure.
                    warnings.warn(
                        f"Source adapter {adapter_name!r} yielded non-incremental item "
                        "metadata; ignoring it during complete ingest",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                if isinstance(result, DrawerRecord):
                    drawers_written += 1
                    context.upsert_drawer(result)
                    continue
                raise TypeError(
                    f"source adapter {adapter_name!r} yielded unsupported result type "
                    f"{type(result).__name__}"
                )
            return drawers_written
        finally:
            if knowledge_graph is not None and hasattr(knowledge_graph, "close"):
                knowledge_graph.close()


def cmd_sweep(args):
    """Sweep a transcript file or directory.

    The sweeper deduplicates against its own prior writes via
    deterministic drawer IDs + a timestamp cursor. It does NOT currently
    coordinate with the file-level miners (miner.py / convo_miner.py) —
    those produce char-chunked drawers without compatible message
    metadata, so running both miners may store overlapping content under
    different IDs.
    """
    from ..sweeper import sweep, sweep_directory

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    target = os.path.expanduser(args.target)

    routing = _resolve_cli_write_routing_or_exit(
        args,
        "sweep",
    )
    if routing.use_daemon:
        _submit_daemon_cli_job(
            "sweep",
            {"target": target},
            args,
            background=bool(getattr(args, "background", False)),
            auto_start=routing.decision.auto_start_daemon,
        )
        return
    if os.path.isfile(target):
        result = sweep(target, palace_path)
        print(
            f"  Swept {target}: +{result['drawers_added']} new, "
            f"{result['drawers_already_present']} already present, "
            f"{result['drawers_skipped']} skipped (< cursor)."
        )
    elif os.path.isdir(target):
        result = sweep_directory(target, palace_path)
        print(
            f"  Swept {result['files_succeeded']}/{result['files_attempted']} "
            f"files from {target}: +{result['drawers_added']} new, "
            f"{result['drawers_already_present']} already present, "
            f"{result['drawers_skipped']} skipped (< cursor)."
        )
        failures = result.get("failures") or []
        if failures:
            print(
                f"  WARNING: {len(failures)} file(s) failed to sweep - see stderr / logs for details.",
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        print(f"  ERROR: Not a file or directory: {target}", file=sys.stderr)
        sys.exit(1)
