# How we evaluated a candidate storage backend (process notes)

A short write-up of the method we used to assess whether **turbovecdb** can
replace **ChromaDB** as a MemPalace storage backend. Shareable as a description
of the approach, not the findings.

## The problem

MemPalace has a pluggable storage-backend seam (RFC 001, `backends/base.py`).
The question was not "does a candidate implement the interface?" — that's easy to
check — but "does MemPalace actually depend on more than the interface promises?"
Vector stores accumulate implementation-specific behavior that callers quietly
rely on, and that coupling is invisible until you go looking for it.

## Three steps

### 1. Map what MemPalace truly requires of a backend

We read the ChromaDB backend and every module that touches it (`searcher.py`,
`repair.py`, `migrate.py`, `embedding.py`) and separated the dependencies into
two tiers:

- **Tier A — the formal contract.** The backend-agnostic interface any backend
  must implement.
- **Tier B — leaked implementation assumptions.** Behaviors that are *not* in the
  contract but that MemPalace's correctness silently relies on — distance-metric
  math, a recall fallback that reads ChromaDB's private SQLite schema, HNSW
  corruption/divergence detection, write-concurrency limits, embedder-identity
  rules, file-lock lifecycle.

Tier B is the real cost of swapping backends, and it's the part that doesn't show
up in an interface diff. Output: **`backend-requirements.md`** — a reusable
rubric with blank per-candidate columns.

### 2. Read the candidate's own architecture

We read turbovecdb's `ARCHITECTURE.md` to understand its design on its own terms
before judging it — specifically its load-bearing principle (*SQLite is the
source of truth; the vector index is a rebuildable cache*).

### 3. Score the candidate against the rubric — including "this gap doesn't exist"

We walked each Tier B requirement and classified it:

- **GAP** — a real gap the candidate must close.
- **PARTIAL** — mostly met, one piece missing.
- **DISSOLVED** — the requirement only exists because of ChromaDB's design and
  *does not apply* to the candidate's architecture.
- **CALLER** — the coupling is on MemPalace's side; the candidate has nothing to
  close until MemPalace refactors.

The DISSOLVED category was the important move: many "requirements" turned out to
be ChromaDB-corruption-recovery machinery that a different architecture makes
unnecessary. Recording them as *deliberately absent* (with the reason) is more
useful than silently dropping them. Output: **`turbovecdb/docs/mempalace-backend-gaps.md`**.

## Key principle

> Evaluate against *requirements*, not against the incumbent's *implementation*.

ChromaDB's repair/quarantine/divergence code is a solution to ChromaDB's failure
modes — not a feature list every backend must reproduce. Restating each
ChromaDB-specific behavior as the underlying requirement let us ask "does this
candidate even have this problem?" and answer "no, by design" where true.

## Artifacts produced

| File | Repo | Purpose |
|---|---|---|
| `docs/design/backend-requirements.md` | mempalace | Reusable rubric: Tier A contract + Tier B leaks, blank per-candidate columns |
| `docs/mempalace-backend-gaps.md` | turbovecdb | turbovecdb scored against the rubric (GAP / PARTIAL / DISSOLVED / CALLER) |
| `docs/design/backend-evaluation-process.md` | mempalace | This file — the method |

## Caveats on the output

- The turbovecdb scoring was done from architecture docs only, not source; a few
  verdicts are marked *(confirm in code)*.
- We deliberately stopped at structural gap-finding — no quality/performance
  judgment of the candidate, and no recommendation to switch.
