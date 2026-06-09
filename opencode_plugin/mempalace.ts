import type { Plugin } from "@opencode-ai/plugin"
import { mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"

export const MemPalacePlugin: Plugin = async ({ project, client, $, directory, worktree }) => {
  console.log("MemPalace Plugin initialized!")

  // Configuration (could be moved to opencode.json later)
  const SAVE_INTERVAL = 15

  // Resolve the mempalace CLI by absolute path rather than relying on PATH.
  // OpenCode plugins run under the opencode process, whose PATH may not include
  // the canonical venv's bin dir, so a bare `mempalace` lookup fails with ENOENT.
  // Mirrors the hooks fix (commit 79455ad). Override with MEMPALACE_BIN if needed.
  const MEMPALACE_BIN =
    process.env.MEMPALACE_BIN || "/home/kroussos/.venvs/main/bin/mempalace"

  // Target the live chat palace on the turbovec backend — the same one the
  // Claude Code MCP uses. Without MEMPALACE_BACKEND=turbovec, `mine` writes to
  // the default (chroma) backend, i.e. the frozen backup, not the active index.
  // Embedding config is intentionally left to ~/.mempalace/config.json
  // (nomic-embed-text @ :11434), which is what the chat palace was built with.
  const MEMPALACE_PALACE =
    process.env.MEMPALACE_PALACE || "/home/kroussos/.mempalace/palaces/chat"
  const mineEnv = { ...process.env, MEMPALACE_BACKEND: "turbovec" }

  // Where exported OpenCode transcripts are staged before mining. Unlike the
  // Claude Code hooks (which are handed a TRANSCRIPT_PATH on disk), OpenCode
  // keeps no plaintext transcript for us to point `mine` at — so we pull the
  // conversation via the SDK (client.session.messages) and write it here in
  // Claude Code JSONL format, which mempalace's `--mode convos` miner parses.
  // One stable file per session => re-mining overwrites in place (idempotent),
  // and mining only that session's subdir keeps each pass scoped to one convo.
  const EXPORT_ROOT =
    process.env.MEMPALACE_EXPORT_DIR ||
    "/home/kroussos/.mempalace/opencode_export"

  // In-memory state for this session
  let messageCount = 0

  // Pull the live session from OpenCode and write it as Claude Code JSONL.
  // Returns the directory to point `mine` at, or null if there was nothing
  // worth mining (no text yet, or the fetch failed).
  const exportSession = async (sessionID: string): Promise<string | null> => {
    if (!sessionID) return null

    const result = await client.session.messages({ path: { id: sessionID } })
    // hey-api client returns { data, error }; tolerate a bare array too.
    const rows: any[] = (result as any)?.data ?? (result as any) ?? []
    if (!Array.isArray(rows) || rows.length === 0) return null

    const lines: string[] = []
    for (const row of rows) {
      const role = row?.info?.role
      if (role !== "user" && role !== "assistant") continue
      const text = (row?.parts ?? [])
        .filter(
          (p: any) =>
            p?.type === "text" && p.text && !p.ignored && !p.synthetic,
        )
        .map((p: any) => p.text)
        .join("\n")
        .trim()
      if (!text) continue
      lines.push(
        JSON.stringify({
          type: role,
          message: { content: [{ type: "text", text }] },
        }),
      )
    }
    if (lines.length === 0) return null

    const sessionDir = join(EXPORT_ROOT, sessionID)
    await mkdir(sessionDir, { recursive: true })
    await writeFile(join(sessionDir, "transcript.jsonl"), lines.join("\n") + "\n")
    return sessionDir
  }

  // Export + mine the given session into the chat palace.
  const mineSession = async (sessionID: string): Promise<boolean> => {
    const dir = await exportSession(sessionID)
    if (!dir) {
      console.log("MemPalace: nothing to mine yet for session", sessionID)
      return false
    }
    await $`${MEMPALACE_BIN} --palace ${MEMPALACE_PALACE} mine ${dir} --mode convos --agent opencode`
      .env(mineEnv)
      .text()
    return true
  }

  return {
    "experimental.session.compacting": async (input, output) => {
      console.log("MemPalace: Running compaction pre-check (mining)...")
      try {
        const mined = await mineSession(input.sessionID)
        if (mined) {
          output.context.push(`
[MemPalace System]
Recent activity has been successfully mined and stored verbatim in the MemPalace.
`)
          console.log("MemPalace: Mining complete.")
        }
      } catch (error) {
        console.error("MemPalace: Mining failed during compaction:", error)
      }
    },

    "session.updated": async ({ event }) => {
      messageCount++

      if (messageCount >= SAVE_INTERVAL) {
        const sessionID = (event as any)?.properties?.info?.id
        console.log(
          `MemPalace: Reached ${SAVE_INTERVAL} messages. Saving session ${sessionID}...`,
        )

        try {
          const mined = await mineSession(sessionID)
          await client.app.log({
            body: {
              service: "mempalace",
              level: "info",
              message: mined
                ? "MemPalace: Periodic save completed — conversation mined to the Palace."
                : "MemPalace: Periodic save skipped — no new conversation text to mine.",
            },
          })
          messageCount = 0
        } catch (error) {
          console.error("MemPalace: Save failed:", error)
        }
      }
    },
  }
}
