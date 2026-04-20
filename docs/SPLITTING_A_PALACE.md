# Splitting a Campaign Out of the Chat Palace

How to migrate an existing wing of mined content (e.g., a D&D campaign) out of the shared `chat` palace and into its own dedicated palace under the palace-isolation feature (v3.3.0+).

This runbook was first walked end-to-end on the `phandalin` wing on 2026-04-19 — see commit history for the live execution.

## When you need this

You should split a wing into its own palace when **all** of these are true:

- The wing represents a self-contained body of work (one campaign, one project, one client) that you don't want surfacing in cross-wing searches against unrelated material.
- The content was originally mined into the chat palace before you set up palace isolation for that workspace (i.e., the workspace's `mempalace.yaml` had no `palace:` key at mine time).
- The source content has since been edited/regenerated, so the chat copy is stale and re-mining into a clean dedicated palace is desirable.

If you're starting a new campaign from scratch, skip this runbook — just declare `palace: <alias>` in `mempalace.yaml` from day one and the content lands in the right place automatically.

## Prerequisites

- mempalace v3.3.0+ installed (palace-isolation feature live)
- A workspace directory with a `mempalace.yaml` file
- The `chat` palace exists at `~/.mempalace/palaces/chat/`
- You are willing to do a fresh re-mine (the new palace starts empty)

## Step 0 — Inventory: figure out what's in chat for this wing

Before changing anything, count the drawers and check which source files are still on disk vs. missing/renamed. This tells you how much of the chat copy is dead weight.

Replace `WING_NAME` and `WORKSPACE_DIR` below.

```python
import chromadb, os
from collections import Counter

WING = 'WING_NAME'                          # e.g. 'phandalin', 'abyss'
WORKSPACE = '/home/USER/campaigns/SOMETHING' # e.g. ~/campaigns/Phandalin

client = chromadb.PersistentClient(path=os.path.expanduser('~/.mempalace/palaces/chat'))
col = client.get_collection('mempalace_drawers')
res = col.get(where={'wing': WING}, include=['metadatas'])

print(f'total drawers: {len(res["ids"])}')
print('by room:', dict(Counter(m['room'] for m in res['metadatas'])))

files = Counter(m['source_file'] for m in res['metadatas'])
on_disk = missing = 0
for src in files:
    abs_path = src if src.startswith('/') else os.path.join(WORKSPACE, src)
    if os.path.exists(abs_path): on_disk += 1
    else: missing += 1
print(f'source files: {len(files)} unique  ({on_disk} on disk, {missing} missing)')
print(f'filed: {min(m["filed_at"] for m in res["metadatas"])} → {max(m["filed_at"] for m in res["metadatas"])}')
```

Save the output somewhere (e.g., `WORKSPACE/notes/mempalace_pre_split_inventory.md`) so you have a record of what the chat copy contained before you wipe it.

## Step 1 — Define the new palace (no DB writes)

Two text edits.

### 1a. `~/.mempalace/config.json`

Add the new alias under `palaces`. Example for `abyss`:

```json
{
  "default_palace": "chat",
  "palaces": {
    "chat": "/home/USER/.mempalace/palaces/chat",
    "phandalin": "/home/USER/.mempalace/palaces/phandalin",
    "abyss": "/home/USER/.mempalace/palaces/abyss"
  },
  ...
}
```

If `default_palace` isn't already set, add `"default_palace": "chat"` too — without it, commands run from directories with no walk-up yaml will loud-fail (palace-isolation step 7).

### 1b. `WORKSPACE/mempalace.yaml`

Prepend `palace: <alias>` as the first key:

```yaml
palace: abyss
wing: abyss
rooms:
  - ...
```

After this, every `mempalace` command run from anywhere under `WORKSPACE/` walks up, finds this yaml, and resolves to the new palace automatically.

### 1c. Verify

From inside the workspace:

```bash
cd WORKSPACE
python3 -c "from mempalace.config import MempalaceConfig; print(MempalaceConfig().resolved_palace_path())"
# expect: /home/USER/.mempalace/palaces/<alias>
```

From outside it:

```bash
cd /tmp
python3 -c "from mempalace.config import MempalaceConfig; print(MempalaceConfig().resolved_palace_path())"
# expect: /home/USER/.mempalace/palaces/chat   (default_palace fallback)
```

**Reversible:** revert the two text files. No DB changes yet.

## Step 2 — Mine the workspace into the new palace

```bash
cd WORKSPACE
mempalace mine .
```

**Important:** the walk-up uses `$CWD`, not the directory argument. Always `cd` into the workspace first; running `mempalace mine ~/campaigns/X` from elsewhere will resolve to whatever palace `$CWD` points at.

ChromaDB creates the new palace dir on first write. Expect this to take a few minutes for hundreds of files. Watch the progress bar.

**Reversible:** `rm -rf ~/.mempalace/palaces/<alias>` and start over. Chat palace untouched.

## Step 3 — Verify

```bash
cd WORKSPACE
mempalace status
# expect: drawer count > 0, only the new wing, all expected rooms

mempalace search "<thing you know is in the regenerated content>"
# expect: hits from current source files

mempalace search "<thing that was in OLD content but you removed/renamed>"
# expect: zero hits (or, if the rename was a consolidation, hits on the new canonical file)

mempalace --palace chat status | grep -A1 "WING: <wing>"
# expect: chat still has the OLD drawer counts for this wing (unchanged)
```

**Stop here unless you're confident the new palace looks right.** Step 4 is destructive.

## Step 4 — Wipe the stale wing from chat (destructive)

Only after step 3 verification. Deletes drawers in chat where `wing == WING_NAME`. Other wings in chat are preserved.

```python
import chromadb, os

WING = 'WING_NAME'
client = chromadb.PersistentClient(path=os.path.expanduser('~/.mempalace/palaces/chat'))
col = client.get_collection('mempalace_drawers')

before_total = col.count()
ids = col.get(where={'wing': WING})['ids']
print(f'about to delete {len(ids)} drawers from chat (chat total before: {before_total})')

# chromadb has payload limits; batch the delete
BATCH = 500
for i in range(0, len(ids), BATCH):
    col.delete(ids=ids[i:i+BATCH])

print(f'chat total after: {col.count()}, {WING}-wing in chat: {len(col.get(where={"wing": WING})["ids"])}')
```

Expected: `before - len(ids) == after`, and the wing-in-chat count is `0`.

**Not reversible** without re-mining from source.

## Step 5 (optional) — Clean up the chat knowledge graph

If the wing's mining run was the only thing populating the chat KG (you can check by listing entity names and dates — see `~/.mempalace/palaces/chat/knowledge_graph.sqlite3`), you can wipe the chat KG safely:

```python
import sqlite3, os
con = sqlite3.connect(os.path.expanduser('~/.mempalace/palaces/chat/knowledge_graph.sqlite3'))
con.execute('delete from triples')   # delete triples first (FK to entities)
con.execute('delete from entities')
con.commit()
con.execute('vacuum')
```

If the chat KG has entries from other wings, filter by entity name / extracted_at date instead of wiping wholesale. Inspect first:

```python
import sqlite3, os
from collections import Counter
con = sqlite3.connect(os.path.expanduser('~/.mempalace/palaces/chat/knowledge_graph.sqlite3'))
print('entities:', [r[0] for r in con.execute('select name from entities order by name')])
print('triple dates:', dict(Counter(r[0][:10] for r in con.execute('select extracted_at from triples'))))
```

`mempalace mine` does **not** populate the KG by default — KG entries come from separate operations (closet regeneration, fact-check, LLM extraction). If you want a KG for the new palace, run those operations against it after mining.

## Aftercare

- The `chat` palace still has its `chroma.sqlite3` file at the same path; the deletes don't compact ChromaDB on disk. Run `mempalace migrate` if you want to reclaim space.
- If you had any `~/.mempalace/config.json` `palace_path` legacy key pointing at chat, leave it — the new code reads `default_palace` first; `palace_path` is just legacy fallback.
- Hooks always write to the chat palace regardless of `$CWD`. They will not start filing to the new campaign palace, and that's correct — automatic mining of conversation transcripts shouldn't pollute curated campaign content.

## Quick reference: the inventory query (for any wing in any palace)

```bash
python3 -c "
import chromadb, os
from collections import Counter
PALACE = '$HOME/.mempalace/palaces/chat'  # change as needed
WING = 'abyss'                             # change as needed
c = chromadb.PersistentClient(path=os.path.expanduser(PALACE))
col = c.get_collection('mempalace_drawers')
res = col.get(where={'wing': WING}, include=['metadatas'])
print(f'{len(res[\"ids\"])} drawers; rooms:', dict(Counter(m['room'] for m in res['metadatas'])))
"
```
