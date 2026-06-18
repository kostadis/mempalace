# Plan: Adding MemPalace Hooks to Opencode

## 1. Research & Requirements Gathering
- [x] **Identify opencode's hook mechanism:** Search the `opencode` codebase for existing hook implementations or configuration structures (e.g., `settings.json`, `config.py`, or any `Stop`/`PreCompact` lifecycle events).
- [x] **Analyze event payloads:** Determine the shape of the data `opencode` passes to hooks (e.g., session IDs, message counts, or file paths) to ensure compatibility with `mempal_save_hook.sh` and `mempal_precompact_hook.sh`.
- [x] **Determine hook execution mode:** Verify if `opencode` supports "command" type hooks that can return JSON to block or prompt the agent (similar to Claude Code's `{"decision": "block", ...}`).

## 2. Implementation
- [x] **Define Hook Configuration Schema:** Add support for a `hooks` section in `opencode`'s configuration (e.g., `opencode.json`).
- [x] **Implement Hook Runner:** 
    - Create a core engine to execute registered shell commands.
    - Implement timeout handling and error logging.
    - Implement the "blocking" logic: if a hook returns a specific JSON structure, the agent must pause and perform the requested action (saving context).
- [x] **Integrate Lifecycle Triggers:**
    - **Stop Hook:** Trigger the runner whenever an agent interaction completes.
    - **PreCompact Hook:** Trigger the runner immediately before the agent's context window is compressed.

## 3. Integration & Testing
- [x] **Create Opencode-specific wrappers:** If `opencode` payloads differ significantly from Claude Code, create thin wrappers for `mempal_save_hook.sh` to map `opencode` variables to `mempalace` expectations.
- [x] **Verification:**
    - Run a session and verify that `mempalace mine` is triggered in the background.
    - Verify that the agent is correctly prompted to save verbatim content when the `SAVE_INTERVAL` is reached.
    - Verify that context compaction triggers an "emergency" save.
- [x] **Documentation:** Update `opencode` docs to explain how users can install and configure MemPalace hooks.
