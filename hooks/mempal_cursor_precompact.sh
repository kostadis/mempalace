#!/bin/bash
# MemPalace Cursor PreCompact Hook — thin wrapper calling the Python CLI.
#
# Cursor's preCompact hook is observational — unlike Claude Code's
# PreCompact, Cursor cannot block compaction from a hook. The Python
# harness still runs the synchronous mine + diary write so memories
# land before context shrinks, and surfaces a short user_message in
# the UI confirming the checkpoint.
run_mempalace_hook() {
  if command -v mempalace >/dev/null 2>&1; then
    mempalace hook run "$@"
    return $?
  fi

  if command -v python3 >/dev/null 2>&1 && python3 -c "import mempalace" >/dev/null 2>&1; then
    python3 -m mempalace hook run "$@"
    return $?
  fi

  if command -v python >/dev/null 2>&1 && python -c "import mempalace" >/dev/null 2>&1; then
    python -m mempalace hook run "$@"
    return $?
  fi

  echo "MemPalace hook error: could not find a runnable mempalace command or module" >&2
  return 1
}

run_mempalace_hook --hook precompact --harness cursor
