#!/bin/bash
# MemPalace Cursor SessionStart Hook — thin wrapper calling the Python CLI.
#
# Currently a no-op pass-through that just initializes the hook state
# directory and logs the session id. Future Cursor-only enhancements
# (e.g. additional_context injection from mempalace.layers.MemoryStack)
# would land in mempalace.hooks_cli.hook_session_start.
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

run_mempalace_hook --hook session-start --harness cursor
