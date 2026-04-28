#!/bin/bash
# MemPalace Cursor Stop Hook — thin wrapper calling the Python CLI.
# All logic lives in mempalace.hooks_cli for cross-harness extensibility.
#
# === INSTALL ===
# Either drop the project-level template at <repo>/.cursor/hooks.json,
# or hand-edit ~/.cursor/hooks.json. See hooks/README.md for the full
# Cursor install instructions.
#
# This wrapper is intentionally identical in shape to the Claude Code
# and Codex wrappers — only the --harness flag differs. Cursor's stop
# payload uses conversation_id (not session_id) and lacks
# stop_hook_active; the Python harness handles that mapping.
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

run_mempalace_hook --hook stop --harness cursor
