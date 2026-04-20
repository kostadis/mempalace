# Palace Isolation — Design

Status: implemented (v3.3.0)
Branch: `feat/palace-isolation`

## Problem

MemPalace today stores every wing in a single shared ChromaDB at
`~/.mempalace/palace/`. Wings namespace content *logically* but not
*physically*. Consequences:

- Cross-wing searches (no `--wing` filter) surface content from every
  campaign and every chat session at once.
- Claude Code hooks mine conversation transcripts into the same DB that
  holds curated D&D campaign content, violating the trust hierarchy the
  campaign workflow depends on (see `campaigns/MEMPALACE_HOWTO.md`).
- `$MP status` dumps all wings across all use cases together.

The user runs MemPalace for two distinct purposes that must not mix:

1. **Curated campaign state** — NPC dossiers, chapter splits, distill
   extractions. Every entry is human-reviewed canon.
2. **Chat session state** — Claude Code conversation transcripts, mined
   automatically by hooks. Raw, unreviewed, high-volume.

## Invariants

Two rules drive the entire design:

1. **Hook writes are chat-palace-only.** Automatic, uncurated writes
   never touch a campaign palace. The chat palace path is hardcoded in
   the hook scripts — hooks do not walk up, do not read
   `mempalace.yaml`, do not accept runtime palace selection.
2. **Campaign palace is the default for reads in its workspace.** The
   CLI walks up from `$CWD` to find the nearest `mempalace.yaml`
   declaring a palace. Chat is reachable only via explicit
   `--palace chat`.

Everything below follows from these two rules.

## Model

A **palace** is one physical directory containing:
- ChromaDB files (drawers + embeddings)
- `knowledge_graph.sqlite3`
- Entity registry

Layout:

```
~/.mempalace/
├── config.json
└── palaces/
    ├── chat/         # hooks ALWAYS write here; searched via --palace chat
    ├── oota/         # written by $MP mine ~/campaigns/oota; default in that dir
    └── phandalin/
```

Wings still live inside palaces exactly as today.

## Selection mechanism

### Per-workspace declaration

Extend the existing `mempalace.yaml` with one optional field:

```yaml
palace: oota                         # alias resolved via config.json, OR
palace: ~/.mempalace/palaces/oota    # absolute path
wing: narrative                      # existing
rooms: [...]                         # existing
```

### Palace aliases

`~/.mempalace/config.json` maps names to paths:

```json
{
  "palaces": {
    "chat": "~/.mempalace/palaces/chat",
    "oota": "~/.mempalace/palaces/oota",
    "phandalin": "~/.mempalace/palaces/phandalin"
  }
}
```

Aliases work everywhere a palace path is accepted (`--palace`,
`mempalace.yaml` `palace:` field).

### Config precedence for explicit CLI operations

```
1. --palace flag                         (explicit, highest)
2. MEMPALACE_PALACE_PATH env var
3. walk-up mempalace.yaml `palace:`      ← NEW
4. ~/.mempalace/config.json default_palace (optional)
5. error: "no palace declared"           (no silent fallback)
```

No silent fallback to chat. If walk-up misses and no `--palace` given,
fail loud. Users who want a fallback set `default_palace` in
`config.json`.

### Hook path resolution (separate!)

Hooks do **not** use the precedence above. They hardcode:

```bash
MEMPAL_CHAT_PALACE="${MEMPAL_CHAT_PALACE:-$HOME/.mempalace/palaces/chat}"
python3 -m mempalace mine "$DIR" --palace "$MEMPAL_CHAT_PALACE"
```

The `MEMPAL_CHAT_PALACE` env var exists only so power users can relocate
their chat palace; default is fine for everyone else.

## Read flows

### CLI

| Where you are | Command | Palace searched |
|--|--|--|
| `~/campaigns/oota` | `$MP search "drow"` | oota (walk-up) |
| `~/campaigns/oota` | `$MP search "drow" --palace chat` | chat |
| `~/src/random` (no yaml) | `$MP search "x"` | error: declare a palace |
| anywhere | `$MP search "x" --palace chat` | chat |

### MCP (inside Claude Code)

Each workspace's MCP registration is launched with `--palace` pointing
at that workspace's palace:

```bash
claude mcp add mempalace -- python -m mempalace.mcp_server --palace oota
```

Default tool calls hit the workspace palace. To cross-search chat from
inside a campaign workspace, add an optional `palace` arg to the search
tool:

```
mcp__mempalace__search(query="drow alliance", palace="chat")
```

This requires `mcp_server.py` to hold a small lazy cache of backend
connections keyed by palace path. The singleton becomes a map.

## Write flows

| Who writes | Target | Mechanism |
|--|--|--|
| `$MP mine ~/campaigns/oota` | oota palace | walk-up |
| `$MP mine <dir> --palace <p>` | p | explicit |
| `mempal_save_hook.sh` | chat palace | hardcoded path |
| `mempal_precompact_hook.sh` | chat palace | hardcoded path |
| `convo_miner` via CLI | wherever palace resolves to | standard precedence |

The **hook lines** are the only ones that carry the isolation invariant
in code. Everything else is policy enforced by the user's own
`$MP mine` discipline.

## Code touchpoints

| File | Change |
|--|--|
| `mempalace/config.py:85,145-172` | Walk-up discovery in `palace_path`; palace-alias resolution; new `default_palace` in file config; raise `PalaceNotDeclared` on miss |
| `mempalace/mcp_server.py:79-103` | Lazy backend map keyed by palace path; optional `palace` arg on search tool; walk-up fallback when no `--palace` given |
| `mempalace/knowledge_graph.py:47` | Always co-locate KG under palace path; drop global default |
| `mempalace/entity_registry.py` | Co-locate under palace path |
| `mempalace/room_detector_local.py:282` | Parse new `palace:` yaml key (CLI only; hooks ignore) |
| `hooks/mempal_save_hook.sh:156` | `--palace "$MEMPAL_CHAT_PALACE"` |
| `hooks/mempal_precompact_hook.sh:68` | Same |
| `mempalace/cli.py` | No change — inherits via `MempalaceConfig()` |
| One-time migration | Rename/symlink `~/.mempalace/palace/` → `~/.mempalace/palaces/chat/` on first `config.py` load after upgrade |

## Backwards compatibility

- Existing users with data at `~/.mempalace/palace/` — migrated to
  `~/.mempalace/palaces/chat/` via rename on first run. No data loss.
- Existing `mempalace.yaml` files without `palace:` — still work; CLI
  operations fall through to `default_palace` or error.
- Hooks upgrade transparently — they just start writing to a differently
  named path.

## v1 scope

1. Walk-up discovery + alias resolution in `config.py`.
2. `palace:` yaml field parsing.
3. KG + entity registry always palace-local.
4. Hooks hardcode `--palace "$MEMPAL_CHAT_PALACE"`.
5. MCP lazy backend map + optional `palace` arg on search/status tools.
6. One-time rename migration.
7. User-facing doc (`docs/PALACE_ISOLATION.md`).

## Out of scope for v1

- `$MP palace list` / `$MP palace use` CLI subcommands.
- Cross-palace federated search in a single query.
- Automatic palace-split tooling (take an existing palace and fork it
  into two).
- Per-palace quotas or size caps.

## Open questions

- **Does `$MP status` stay palace-scoped** (show only the active palace,
  require `--palace chat` for chat stats)? Default: yes, scoped.
- **Should `default_palace` in `config.json` exist at all**, or is
  "no palace declared → error" the whole story? Leaning: include it for
  users who want a fallback, but document that loud-fail is the
  recommended posture.
- **Does the MCP search tool's `palace` arg accept aliases**? Yes —
  shares resolver with CLI `--palace`.
