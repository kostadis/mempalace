# MemPalace Opencode Plugin

This plugin integrates [MemPalace](https://github.com/anomalyco/mempalace) with OpenCode, providing automated verbatim memory preservation during your AI coding sessions.

## Features

- **Auto-Mining on Compaction**: Automatically runs `mempalace mine` before OpenCode compacts its context, ensuring your most recent work is indexed and preserved.
- **Periodic Checkpoints**: Triggers a background save every 15 messages to prevent data loss.
- **Context Awareness**: Injects a confirmation message into the compaction prompt so the AI knows its recent activity has been secured.
- **User Notifications**: Uses OpenCode's app logging to notify you when periodic saves occur.

## Installation

### Prerequisites

- [Bun](https://bun.sh/) must be installed on your system.
- [MemPalace](https://github.com/anomalyco/mempalace) must be installed and available in your `PATH`.

### Setup

1. **Ensure dependencies are installed**:
   The plugin requires `shescape`. OpenCode handles this automatically via Bun, but you can manually ensure it's installed by running:
   ```bash
   cd ~/.config/opencode && bun install
   ```

2. **Plugin Placement**:
   The plugin file `mempalace.ts` should be located in:
   - Global: `~/.config/opencode/plugins/mempalace.ts`
   - Project-level: `.opencode/plugins/mempalace.ts`

3. **Restart OpenCode**:
   Plugins are loaded at startup. Restart your OpenCode session to activate the plugin.

## Configuration

The plugin currently uses a hardcoded `SAVE_INTERVAL` of 15 messages and a `BLOCKING_SAVE` mode. To customize these, you can modify `~/.config/opencode/plugins/mempalace.ts`.

## Verification

To verify the plugin is working:

1. **Initialization**: Check the console/logs for `MemPalace Plugin initialized!`.
2. **Compaction**: When OpenCode compacts context, verify that `mempalace mine` is executed and a `[MemPalace System]` message appears in the context.
3. **Periodic Save**: After 15 messages, verify the log message: `MemPalace: Periodic save interval reached...`.
