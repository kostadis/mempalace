import type { Plugin } from "@opencode-ai/plugin"

export const MemPalacePlugin: Plugin = async ({ project, client, $, directory, worktree }) => {
  console.log("MemPalace Plugin initialized!")

  // Configuration (could be moved to opencode.json later)
  const SAVE_INTERVAL = 15
  const BLOCKING_SAVE = true 

  // In-memory state for this session
  let messageCount = 0

  return {
    "experimental.session.compacting": async (input, output) => {
      console.log("MemPalace: Running compaction pre-check (mining)...")
      try {
        await $`mempalace mine`.text()
        output.context.push(`
[MemPalace System]
Recent activity has been successfully mined and stored verbatim in the MemPalace.
`)
        console.log("MemPalace: Mining complete.")
      } catch (error) {
        console.error("MemPalace: Mining failed during compaction:", error)
      }
    },

    "session.updated": async ({ event }) => {
      messageCount++

      if (messageCount >= SAVE_INTERVAL) {
        console.log(`MemPalace: Reached ${SAVE_INTERVAL} messages. Handling save...`)
        
        try {
          if (BLOCKING_SAVE) {
            // Instead of a silent background mine, we use the client to prompt the user.
            // We use a log/toast to signal the AI should perform a save.
            await client.app.log({
              body: {
                service: "mempalace",
                level: "info",
                message: "MemPalace: Periodic save interval reached. Please ensure your latest context is saved to the Palace.",
              },
            })
            
            // In a real 'blocking' scenario, we might use a tool call or a specialized 
            // UI prompt if the Opencode SDK provides it. For now, we trigger the mine 
            // and notify the user to verify.
            await $`mempalace mine`.text()
          } else {
            await $`mempalace mine`.text()
            await client.app.log({
              body: {
                service: "mempalace",
                level: "info",
                message: "MemPalace: Periodic background save completed.",
              },
            })
          }
          
          messageCount = 0
        } catch (error) {
          console.error("MemPalace: Save failed:", error)
        }
      }
    },
  }
}
