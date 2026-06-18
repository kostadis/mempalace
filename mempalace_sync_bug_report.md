# mempalace sync — three bugs (v3.3.5)

Filed 2026-05-17 from a Phandalin palace rebuild attempt. Three issues found while trying to use `mempalace sync` to prune a single orphan drawer.

---

## Bug 1 (critical, data-loss potential): `sync` interprets `.mempalaceignore` patterns globally instead of per-wing

### Symptom

`mempalace sync --dry-run` against any sub-wing reports that **every drawer in that wing is "gitignored" and would be deleted**, even though the source files exist, are tracked in git, and are not in any `.gitignore`. Running `--apply` would wipe the wing.

### Reproduction

Palace layout:

```
Phandalin/
├── .mempalaceignore          ← excludes sub-wing source dirs from root mine
├── docs/
│   ├── chapters/             ← narrative wing (own mempalace.yaml)
│   │   └── mempalace.yaml
│   ├── distill_extractions/  ← distill_extractions wing (own mempalace.yaml)
│   │   └── mempalace.yaml
│   └── ...                   ← phandalin (root) wing
└── ...
```

`.mempalaceignore` (excerpt):

```
docs/chapters/
docs/distill_extractions/
docs/state_extractions/
...
```

These patterns are correct: they tell the **root wing** mine to skip those subdirs because they belong to other wings.

After a clean three-wing rebuild:

```
$ mempalace --palace /home/kroussos/.mempalace/palaces/phandalin status
  WING: distill_extractions
    ROOM: general               1048 drawers
  WING: narrative
    ROOM: chapters              1199 drawers
    ROOM: general                  1 drawers
  WING: phandalin
    ROOM: npcs                  1993 drawers
    ROOM: arcs                   321 drawers
    ROOM: world                  156 drawers
    ROOM: dead                   111 drawers
    ROOM: mechanics               80 drawers
```

Then run sync against each wing, dry-run only:

```
$ mempalace --palace /home/kroussos/.mempalace/palaces/phandalin sync \
    --wing narrative --dry-run /home/kroussos/campaigns/Phandalin
  Scanned:        1200
  Kept:           0
  Gitignored:     1199  (would remove)
  Missing:        1  (would remove)
  No source:      0  (kept)
  Out of scope:   0  (kept)

  Top sources to remove:
    .../docs/chapters/chapter_38_the_intervention.md  (107)
    .../docs/chapters/chapter_34_the_one_hit_point_principle.md  (94)
    .../docs/chapters/chapter_39_the_charge_of_the_light_brigade.md  (83)
    .../docs/chapters/chapter_36_a_dragon_defeated_a_bard_tempted_a_barbarian_s_vengeance.md  (57)
    .../docs/chapters/chapter_41_unraveling_the_storm_god_s_secrets.md  (54)
```

```
$ mempalace --palace .../phandalin sync \
    --wing distill_extractions --dry-run /home/kroussos/campaigns/Phandalin
  Scanned:        1048
  Kept:           0
  Gitignored:     1048  (would remove)
  Missing:        0  (would remove)
```

```
$ mempalace --palace .../phandalin sync \
    --wing phandalin --dry-run /home/kroussos/campaigns/Phandalin
  Scanned:        2661
  Kept:           2661                ← root wing is fine
  Gitignored:     0  (would remove)
  Missing:        0  (would remove)
```

Independent verification that the files are not actually gitignored:

```
$ git check-ignore docs/chapters/chapter_41_unraveling_the_storm_god_s_secrets.md
(no output — file is NOT ignored)

$ git ls-files docs/chapters/*.md | head
docs/chapters/chapter_01_*.md         ← tracked
docs/chapters/chapter_02_*.md         ← tracked
...
```

### Root cause (suspected)

`sync` reads `.mempalaceignore` patterns and applies them as a global "drawer source must NOT match any pattern" check, irrespective of which wing the drawer belongs to. But `.mempalaceignore` patterns are semantically scoped to the root wing — they exist to tell the root mine "skip these subdirs because other wings own them."

For sub-wings whose source directory **is** listed in `.mempalaceignore` (the common case: every sub-wing's root dir is excluded from the root mine), every legitimate drawer matches the ignore pattern and gets flagged for deletion.

### Expected behavior

One of:

- **(A) Scope ignore patterns to the wing being synced.** When syncing wing `W`, only apply `.mempalaceignore` rules whose path lives **outside** wing `W`'s source root. Patterns that point INTO wing `W`'s source root (or describe wing `W`'s root itself) should not flag wing `W`'s drawers.
- **(B) Use the per-wing `mempalace.yaml` as the source-of-truth** for what belongs in the wing, and ignore `.mempalaceignore` entirely during sync. Sync against wing `W` would walk wing `W`'s configured source dir and any per-wing ignore file, not the root-level `.mempalaceignore`.
- **(C) Require an explicit per-wing ignore file** for sub-wings; do not inherit from parents.

(A) is the smallest behavioral change; (B) is the cleanest semantic model.

### Severity

**Critical.** If a user runs `--apply` once on a fresh rebuild, every sub-wing is wiped. The only signal to stop is that the "Kept" count is suspiciously 0 — which is easy to miss if you trusted the command name. The dry-run did stop me, but a less wary user would discover the bug post-`--apply` with a 0-drawer palace.

### Suggested test cases

1. Multi-wing palace with `.mempalaceignore` listing each sub-wing's source dir → `sync --dry-run --wing <each-wing>` should report `Kept == drawers`, `Gitignored == 0`.
2. Genuinely gitignored sub-wing source file → should be flagged.
3. Deleted source file → should be flagged as `Missing`.

---

## Bug 2 (annoyance): `--palace <named>` doesn't resolve in `sync` subcommand

### Symptom

The named-palace form works for every subcommand except `sync`:

```
$ mempalace --palace phandalin status
... (works)

$ mempalace --palace phandalin mine docs/chapters
... (works)

$ mempalace --palace phandalin search "Adabra"
... (works)

$ mempalace --palace phandalin sync --wing narrative --dry-run .
  No palace found at phandalin
```

Workaround: pass an absolute path.

```
$ mempalace --palace /home/kroussos/.mempalace/palaces/phandalin sync --wing narrative --dry-run .
... (works)
```

### Suspected cause

`sync` is resolving `--palace` against a raw filesystem path instead of going through the named-palaces map in `~/.mempalace/config.json`. The named-palaces map clearly has the entry:

```json
"palaces": {
  "chat": "/home/kroussos/.mempalace/palaces/chat",
  "phandalin": "/home/kroussos/.mempalace/palaces/phandalin",
  "abyss": "/home/kroussos/.mempalace/palaces/abyss",
  "campaign-dev": "/home/kroussos/.mempalace/palaces/campaign-dev"
}
```

…and every other subcommand reads from it.

### Severity

Low. Workaround is trivial. But the inconsistency violates principle of least surprise — users learn `--palace <name>` for status/mine/search and expect it everywhere.

---

## Bug 3 (separate concern, may not be sync's fault): HNSW segments are being quarantined within minutes of writes

### Symptom

After a clean rebuild that completed at 2026-05-17 17:30 UTC:

```
$ mempalace --palace phandalin status
Quarantined corrupt HNSW segment .../724af8ea-2541-45d2-9711-24a2eabc1a00
  (sqlite 357s newer than HNSW and integrity check failed);
  renamed to .../724af8ea-2541-45d2-9711-24a2eabc1a00.drift-20260517-223210
```

Within ~6 minutes of the last write, an HNSW segment was found stale (sqlite 357 seconds newer than HNSW) and integrity check failed. The same symptom — under the same `sqlite newer than HNSW and integrity check failed` message — was what necessitated the rebuild in the first place (two segments quarantined pre-rebuild).

### Possible root causes

- HNSW segment writes are not being durably flushed on commit; on next process startup the integrity check sees stale state and quarantines.
- `status` is performing a write (e.g., a touch on metadata) that races with an unfinished HNSW background flush.
- The 3.0→3.1 chromadb migration path (`mempalace migrate`) didn't run, but isn't surfaced as a warning here.

### Severity

High over time. Each `status` call risks quarantining another segment. The user is being slowly bricked.

### Diagnostic data to collect for this one

- Whether `mempalace migrate` would help — there's a subcommand for the 3.0.0 → 3.1.0 chromadb upgrade.
- Whether other palaces on the same host (e.g. `abyss`, `chat`) are also accumulating `.drift-*` segments. If yes → environmental. If no → palace-specific.
- HNSW writer flush settings / whether chromadb persistence is being properly closed on process exit.

---

## Environment

- MemPalace **3.3.5** (`mempalace --version`)
- Embedding: openai-compat, model `nomic-ai/nomic-embed-text-v1.5`, endpoint `http://192.168.1.147:8000`
- Linux 6.6.87.2-microsoft-standard-WSL2 (Ubuntu under WSL2)
- Palace path: `/home/kroussos/.mempalace/palaces/phandalin`
- Backups retained for forensics if you want to inspect the corrupted-state palace:
  - `/home/kroussos/.mempalace/palaces/phandalin.bak.20260517-172458` (corrupted state at start of rebuild)
  - `/home/kroussos/.mempalace/palaces/phandalin.bak.20260504-213838` (older healthy snapshot)
