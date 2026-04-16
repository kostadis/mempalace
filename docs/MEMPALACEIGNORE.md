# .mempalaceignore

## What it is

`.mempalaceignore` is a per-directory ignore file that controls which files and directories MemPalace skips during mining. It uses the same pattern syntax as `.gitignore`.

## Why it exists

MemPalace previously read `.gitignore` to decide what to skip when mining a project. This couples mining behavior to git's version-control concerns, which are not always the same. For example, you might want git to track a file but exclude it from your memory palace, or vice versa.

`.mempalaceignore` gives you independent control over what gets mined.

## How it works

When the miner scans a directory, it looks for ignore files in this order:

1. `.mempalaceignore` in the current directory
2. `.gitignore` in the current directory (fallback)

If a `.mempalaceignore` file exists, it is used and `.gitignore` is **not read** for that directory. If no `.mempalaceignore` exists, the miner falls back to `.gitignore`. If neither exists, nothing is ignored for that directory.

This is per-directory: a project root can have `.mempalaceignore` while a subdirectory falls back to `.gitignore`, or vice versa.

## Pattern syntax

The syntax is identical to `.gitignore`:

```
# Comments start with #
*.log           # Ignore all .log files
generated/      # Ignore the generated directory
!generated/keep.py  # But keep this specific file
/local-only.txt     # Anchored to this directory only
```

- `*` matches anything except `/`
- `**` matches any number of directories
- `?` matches a single character
- Trailing `/` means directory only
- Leading `!` negates (un-ignores) a pattern
- Leading `/` anchors to the directory containing the ignore file
- `#` lines are comments
- Blank lines are ignored

## CLI flags

```bash
# Normal mining (reads .mempalaceignore, falls back to .gitignore)
mempalace mine ~/projects/myapp

# Disable all ignore file processing
mempalace mine ~/projects/myapp --no-mempalaceignore

# Legacy flag (still works, same behavior)
mempalace mine ~/projects/myapp --no-gitignore

# Force-include specific paths even if ignored
mempalace mine ~/projects/myapp --include-ignored docs,generated/keep.py
```

## What this does NOT change

The `_ensure_mempalace_files_gitignored()` function in `cli.py` still writes to `.gitignore` when you run `mempalace init`. That is correct — its job is to prevent git from committing `mempalace.yaml` and `entities.json`, which is a git concern, not a mining concern.

## Code locations

- **`mempalace/miner.py`** — `IgnoreMatcher` class, `load_ignore_matcher()`, `is_ignored()`, `scan_project()`
- **`mempalace/cli.py`** — `--no-mempalaceignore` / `--no-gitignore` flag definition, `cmd_mine()` passes `respect_ignore` to the miner
- **`tests/test_miner.py`** — all ignore-related test cases
- **`tests/test_cli.py`** — CLI argument forwarding tests

## Migrating an existing project

No migration needed. If you do nothing, the miner reads your existing `.gitignore` files and everything works as before. To customize mining separately from git:

1. Create a `.mempalaceignore` in your project root
2. Add the patterns you want (copy from `.gitignore` and modify as needed)
3. The miner will use `.mempalaceignore` and stop reading `.gitignore` for that directory
