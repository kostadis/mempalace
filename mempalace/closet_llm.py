"""
closet_llm.py — Generate closets via a user-configured LLM for richer indexing.

The regex-based closet extraction catches action verbs, headers, and proper
nouns — but misses implicit topics, foreign-language content, and contextual
references. An LLM reads everything and produces better closets.

This module is **OPTIONAL and opt-in**. Regex closets are always created by
the miner; this path regenerates them afterward using whatever LLM the user
chooses. Core memory operations remain API-free by design (see CLAUDE.md,
"Local-first, zero API").

## Bring-your-own-LLM configuration

The endpoint is any OpenAI-compatible Chat Completions URL:

    LLM_ENDPOINT=http://localhost:11434/v1   # Ollama
    LLM_ENDPOINT=http://localhost:8000/v1    # vLLM, llama.cpp
    LLM_ENDPOINT=https://api.openai.com/v1
    LLM_ENDPOINT=https://openrouter.ai/api/v1
    LLM_ENDPOINT=https://api.anthropic.com/v1  # when proxied through a compat layer

Set:
    LLM_ENDPOINT — base URL (required)
    LLM_KEY      — bearer token (optional; local inference usually doesn't need it)
    LLM_MODEL    — model name (required), e.g. "gpt-4o-mini", "llama3:8b", "qwen2.5:7b"

Or pass flags on the CLI (flags win over env):

    python -m mempalace.closet_llm \\
        --palace ~/.mempalace/palace \\
        --endpoint http://localhost:11434/v1 \\
        --model llama3:8b

No vendor lock-in. No hidden dependency on any specific provider. Zero deps
added to pyproject — uses stdlib urllib.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional

from .config import MempalaceConfig
from .parallel import ParallelPipeline, WorkerResult
from .palace import (
    NORMALIZE_VERSION,
    get_closets_collection,
    get_collection,
    mine_lock,
    purge_file_closets,
    upsert_closet_lines,
)

MAX_CONTENT_CHARS = 30000
MAX_OUTPUT_TOKENS = 1500
# Long enough to ride out vLLM/Ollama queue backlog under high concurrency.
# At workers=8-16 and a real palace (~900 sources), the server queue can hold
# 10+ items behind a big prefill; previous 60s gave up before vLLM got to the
# request and recorded a false "LLM failed". Override via env if needed.
HTTP_TIMEOUT_S = int(os.environ.get("MEMPALACE_LLM_HTTP_TIMEOUT_S", "600"))

PROMPT_TEMPLATE = """You are helping a user search their notes. For the
SPECIFIC content shown below, write 8-15 SEARCH QUERIES — the kind of
natural-language questions or statements a person would type into a
search bar when looking for THIS content later.

Each search query is a normal English sentence or phrase with WORDS
SEPARATED BY SPACES. It is NOT a tag, NOT an identifier, NOT a slug.

Source: {source_file}
Wing: {wing} | Room: {room}

CONTENT:
{content}

---

Output ONE JSON OBJECT (starts with `{{`, ends with `}}`). Not a list,
not an array — a single object with EXACTLY these three top-level fields.

The "index_sentences" array MUST contain EXACTLY 10 entries. Not 3, not
5, not 8 — exactly 10. Count them as you write. If you have fewer than
10, you haven't finished. If you have more than 10, drop the weakest ones.

{{
  "index_sentences": [
    "<entry 1 — a sentence about something specific in the content above>",
    "<entry 2 — about a different specific aspect>",
    "<entry 3 — naming a function, concept, or named thing from the content>",
    "<entry 4 — a question someone would search for about this file>",
    "<entry 5 — about an input, output, or side effect>",
    "<entry 6 — about an error case or edge case the content handles>",
    "<entry 7 — about the file's relationship to other modules or files>",
    "<entry 8 — about a config, constant, or default value>",
    "<entry 9 — about who uses this, when, or why>",
    "<entry 10 — a paraphrase of the file's main purpose in plain words>"
  ],
  "quotes": ["[Speaker] verbatim quote", "[Speaker] another quote"],
  "summary": "<2-3 sentences describing what this specific content is about>"
}}

RULE 1 — CONTENT MUST BE SPECIFIC TO THIS FILE
Every index_sentences entry must reference something that actually
appears in the CONTENT above. Names of functions, variables, sections,
people, concepts that are PRESENT in the content. Do NOT write
sentences that would apply equally to any random file.

If the content is a Python file about, say, parsing YAML, your sentences
should mention YAML, the function names, what they do. Not "this module
does things in a pipeline."

RULE 1B — ENTRY COUNT IS MANDATORY: EXACTLY 10
Produce EXACTLY 10 entries in index_sentences. Not 4, not 6, not 8 —
ten. A response with fewer than 10 is invalid and will be rejected.

Strategies to reach 10 entries when the content seems thin (use them):
  1. One entry per function, section, or class name in the content.
  2. One entry per input type / output type / data shape.
  3. One entry per error case, edge case, or failure mode handled.
  4. One entry per config option, constant, or default value.
  5. One entry per external dependency the content uses.
  6. One entry per side effect (writes to disk, sends HTTP, etc.).
  7. One entry per concept or idea in a doc.
  8. One entry per named noun (NPC, command, file, person).
  9. One entry per question a user could ask about it.
 10. A paraphrase of the file's main purpose.

Even a 20-line shell script has 10 aspects to index — the shebang, the
shell choice, the env vars, the commands, the order, the side effects,
the inputs, the assumptions, the failure modes, the purpose. Find them.

If you produced fewer than 10, you stopped too early. Keep going.

RULE 2 — EVERY SENTENCE CONTAINS SPACES BETWEEN ITS WORDS
This is the most important formatting rule. Each index_sentences entry
must contain at least FIVE space-separated words. If you write a "tag"
that has hyphens, underscores, dots, or no separator at all instead of
spaces, you have violated this rule.

Self-check for each entry before emitting it: does it contain 5 or more
words separated by actual space characters? Count them. If fewer than 5,
rewrite it as a real sentence.

REAL FAILURES FROM PRIOR RUNS — DO NOT REPEAT:

  ✗ "rpg-retriever-retrieves-campaign-prose-from-mem-palace"
     (sentence with hyphens — 7 hyphens, 0 spaces — WRONG)
  ✗ "client-sends-json-rpc-requests-to-mempalace-mcp-subprocess"
     (same problem — WRONG)
  ✗ "CampaignCatalogBuildsIndexFromJSONFiles"
     (CamelCase sentence — no spaces — WRONG)
  ✗ "reads_session_doc_scene_NN_slash_md_files"
     (underscores instead of spaces — WRONG)
  ✗ "python-enhance-recap-file-takes-recap-compare-campaign-documents"
     (especially bad, long fake-tag — WRONG)

WRITE THE SAME IDEAS AS SENTENCES:

  ✓ "The rpg retriever retrieves campaign prose from mem palace."
  ✓ "The client sends JSON RPC requests to the mempalace MCP subprocess."
  ✓ "The campaign catalog builds an index from JSON files."
  ✓ "Reads session document scene markdown files."
  ✓ "The enhance recap script takes a recap and compares it with campaign documents."

EXCEPTION: hyphens, dots, slashes, and underscores ARE allowed inside
actual code identifiers and filenames that appear in the content (e.g.
"rpg_retriever.py", "AWQ-quantised", "google/gemma-2-9b-it"). You may
quote those verbatim. The rule is about JOINING regular English words,
not about quoting identifiers that already exist in the code.

RULE 3 — NO GENERIC SENTENCES
The example placeholders above use angle brackets `<like this>` because
those are PLACEHOLDERS, not actual sentences to copy. NEVER include
literal angle-bracket text in your output. NEVER copy any of the
"correct example" sentences verbatim if they don't describe THIS file's
content.

Self-check before emitting each entry: would this sentence make sense
for a totally different file's content? If yes, it's too generic —
rewrite it to be specific.

- Quotes: 2-5 entries. EXACT verbatim from the content. Attribute with
  [Speaker] prefix if identifiable.
- Summary: 2-3 plain English sentences. WHO, WHAT, WHY — about THIS content.
- Write in the same language as the content.
- Output valid JSON only. No code fences, no commentary.
"""


class LLMConfig:
    """Resolved LLM connection config. CLI flags > env vars."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.endpoint = (endpoint or os.environ.get("LLM_ENDPOINT", "")).rstrip("/")
        self.key = key or os.environ.get("LLM_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        if self.endpoint:
            # Privacy-by-architecture: reject file:// and other non-HTTP schemes
            # so a misconfigured endpoint cannot exfiltrate local files.
            scheme = urllib.parse.urlparse(self.endpoint).scheme.lower()
            if scheme not in ("http", "https"):
                raise ValueError(
                    f"LLM_ENDPOINT must use http:// or https:// (got scheme {scheme!r})"
                )

    def missing(self) -> list:
        missing = []
        if not self.endpoint:
            missing.append("LLM_ENDPOINT (or --endpoint)")
        if not self.model:
            missing.append("LLM_MODEL (or --model)")
        # key is optional — local inference servers (Ollama, vLLM) often don't require one
        return missing


def _call_llm(cfg: LLMConfig, source_file: str, wing: str, room: str, content: str):
    """Single LLM call via OpenAI-compatible /chat/completions.

    Returns (parsed_json_dict_or_None, usage_dict_or_None).
    """
    try:
        from mempalace.i18n import t

        lang_instruction = t("aaak.instruction")
    except Exception:
        lang_instruction = ""

    prompt = PROMPT_TEMPLATE.format(
        source_file=source_file[:100],
        wing=wing,
        room=room,
        content=content[:MAX_CONTENT_CHARS],
    )
    if lang_instruction and "english" not in lang_instruction.lower():
        prompt += f"\n\nLanguage instruction: {lang_instruction}"

    body = json.dumps(
        {
            "model": cfg.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if cfg.key:
        headers["Authorization"] = f"Bearer {cfg.key}"

    url = f"{cfg.endpoint}/chat/completions"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)

            text = payload["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            parsed = json.loads(text)
            return parsed, payload.get("usage")
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            return None, None
        except urllib.error.HTTPError as e:
            # 429 / 503 = retry with backoff
            if e.code in (429, 503) and attempt < 2:
                time.sleep(2**attempt)
                continue
            return None, None
        except Exception as e:
            if "rate" in str(e).lower() and attempt < 2:
                time.sleep(2**attempt)
                continue
            return None, None
    return None, None


def _parsed_to_closet_lines(parsed, drawer_ids, entities_str):
    """Convert LLM's JSON output to closet pointer lines.

    Each closet line is a single natural-language sentence in the first
    column. The sentence is what gets embedded + BM25-indexed for search,
    so it MUST read as prose, not as a tag chain. The LLM prompt asks for
    ``index_sentences`` for this reason; we also accept a legacy ``topics``
    field for backward compatibility with closets generated before the
    prose-format change.

    Small LLMs sometimes return a bare list instead of the expected JSON
    object — be liberal about shapes:

    * Object with ``index_sentences`` / ``topics`` → use that.
    * Bare list of strings → treat as ``index_sentences`` directly.
    * Bare list of objects each containing ``index_sentences`` → merge.

    Any unrecognised shape falls through to an empty result rather than
    crashing the producer thread.
    """
    lines = []
    drawer_ref = ",".join(drawer_ids[:3])

    sentences: list = []
    quotes: list = []
    summary: str = ""

    if isinstance(parsed, dict):
        sentences = list(parsed.get("index_sentences") or parsed.get("topics") or [])
        quotes = list(parsed.get("quotes", []) or [])
        summary = str(parsed.get("summary", "") or "")
    elif isinstance(parsed, list):
        # Two sub-cases: list of strings or list of objects.
        if parsed and isinstance(parsed[0], str):
            sentences = parsed
        elif parsed and isinstance(parsed[0], dict):
            for entry in parsed:
                sentences.extend(entry.get("index_sentences") or entry.get("topics") or [])
                quotes.extend(entry.get("quotes", []) or [])
                if not summary:
                    summary = str(entry.get("summary", "") or "")
    for sentence in sentences[:15]:
        s = str(sentence).strip()
        if not s:
            continue
        # Cap line length to keep one closet row reasonable for embed/BM25.
        lines.append(f"{s[:280]}|{entities_str}|→{drawer_ref}")
    for quote in quotes[:5]:
        q = str(quote).strip()
        if not q:
            continue
        lines.append(f"{q[:280]}|{entities_str}|→{drawer_ref}")
    if summary:
        lines.append(f"{summary.strip()[:280]}|{entities_str}|→{drawer_ref}")

    return lines


def regenerate_closets(
    palace_path,
    wing=None,
    sample=0,
    dry_run=False,
    cfg: Optional[LLMConfig] = None,
    workers: Optional[int] = None,
):
    """Regenerate closets using a configured LLM for richer topic extraction.

    Reads existing drawers, sends content to the configured endpoint,
    replaces regex closets with LLM-generated ones. Regex closets remain
    as the fallback whenever the call fails.
    """
    if cfg is None:
        cfg = LLMConfig()
    missing = cfg.missing()
    if missing:
        print("Error: missing configuration: " + ", ".join(missing))
        print("Set env vars LLM_ENDPOINT / LLM_MODEL (and optionally LLM_KEY),")
        print("or pass --endpoint / --model / --key on the CLI.")
        return {"error": "missing-config", "missing": missing}

    drawers_col = get_collection(palace_path, create=False)
    closets_col = get_closets_collection(palace_path)

    total = drawers_col.count()
    if total == 0:
        print("No drawers in palace.")
        return {"processed": 0}

    # Page through drawers — chromadb's SQLite backend errors with "too many
    # SQL variables" when limit exceeds ~32K (the SQLITE_MAX_VARIABLE_NUMBER
    # parameter limit), so we can't load everything in one call on large
    # palaces. 10K per page keeps us well under the limit.
    by_source: dict = {}
    PAGE = 10000
    offset = 0
    while offset < total:
        page = drawers_col.get(limit=PAGE, offset=offset, include=["documents", "metadatas"])
        ids = page["ids"]
        if not ids:
            break
        for doc_id, doc, meta in zip(ids, page["documents"], page["metadatas"]):
            meta = meta or {}
            source = meta.get("source_file", "unknown")
            w = meta.get("wing", "")
            if wing and w != wing:
                continue
            if source not in by_source:
                by_source[source] = {"drawer_ids": [], "content": [], "meta": meta}
            by_source[source]["drawer_ids"].append(doc_id)
            by_source[source]["content"].append(doc)
        offset += len(ids)

    sources = list(by_source.keys())
    if sample > 0:
        sources = sources[:sample]

    print(
        f"Regenerating closets for {len(sources)} source files via {cfg.endpoint} ({cfg.model})..."
    )
    if dry_run:
        print("DRY RUN — no changes will be written")

    processed = 0
    failed = 0
    total_input = 0
    total_output = 0

    if dry_run:
        # Dry-run keeps the serial path — no point parallelizing prints.
        for i, source in enumerate(sources, 1):
            data = by_source[source]
            content = "\n\n".join(data["content"])
            print(f"  [{i}/{len(sources)}] {os.path.basename(source)} ({len(content)} chars)")
        print(f"\nDone. {processed} regenerated, {failed} failed.")
        return {"processed": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0}

    # Resolve worker count: explicit arg → config → asymmetric default
    # (1 for onnx, 8 for remote). LLM token generation is latency-bound
    # per request so N concurrent calls translate directly to speedup.
    if workers is None:
        workers = MempalaceConfig().workers

    # Producer: call _call_llm for one source. Independent across sources;
    # LLMConfig is read-only after construction so it's safe to share.
    def producer(item):
        idx, source = item
        data = by_source[source]
        content = "\n\n".join(data["content"])
        meta = data["meta"]
        w = meta.get("wing", "")
        r = meta.get("room", "")
        entities = meta.get("entities", "")

        parsed, usage = _call_llm(cfg, source, w, r, content)
        return WorkerResult(
            payload={
                "parsed": parsed,
                "usage": usage,
                "drawer_ids": data["drawer_ids"],
                "wing": w,
                "room": r,
                "entities": entities,
                "source": source,
            },
            item_id=os.path.basename(source),
            extra={"idx": idx, "total": len(sources)},
        )

    # Consumer: single-thread. Holds the only references to closets_col
    # — preserves the HNSW single-writer invariant via the existing
    # mine_lock(source) plus serial dispatch from this thread.
    def consumer(result):
        nonlocal processed, failed, total_input, total_output
        p = result.payload
        idx = result.extra["idx"]
        total = result.extra["total"]
        if not p["parsed"]:
            failed += 1
            print(f"  [{idx}/{total}] ✗ {result.item_id} — LLM failed")
            return

        if p["usage"]:
            total_input += p["usage"].get("prompt_tokens", 0)
            total_output += p["usage"].get("completion_tokens", 0)

        lines = _parsed_to_closet_lines(p["parsed"], p["drawer_ids"], p["entities"])
        # Use os.path.basename so Windows-style paths survive unchanged;
        # the naive split('/') would leave a bare path component on Windows
        # and collide across different files under different drives.
        closet_id_base = f"closet_{p['wing']}_{p['room']}_{os.path.basename(p['source'])[:30]}"

        # Serialize with concurrent mine operations on the same source —
        # otherwise a regex closet rebuild mid-regenerate races with our
        # purge+upsert cycle and leaves mixed regex/LLM lines.
        with mine_lock(p["source"]):
            purge_file_closets(closets_col, p["source"])
            upsert_closet_lines(
                closets_col,
                closet_id_base,
                lines,
                {
                    "wing": p["wing"],
                    "room": p["room"],
                    "source_file": p["source"],
                    "generated_by": f"llm:{cfg.model}",
                    "filed_at": datetime.now().isoformat(),
                    "entities": p["entities"],
                    # Stamp so the miner's stale-drawer gate doesn't treat
                    # LLM closets as leftovers and rebuild over them next run.
                    "normalize_version": NORMALIZE_VERSION,
                },
            )

        processed += 1
        n_topics = len(p["parsed"].get("topics", []))
        print(f"  [{idx}/{total}] ✓ {result.item_id} — {n_topics} topics")

    def on_error(failed_result):
        nonlocal failed
        failed += 1
        print(
            f"  ✗ {failed_result.item_id} — {type(failed_result.exception).__name__}: "
            f"{failed_result.exception}"
        )

    pipeline = ParallelPipeline(
        producer_fn=producer,
        consumer_fn=consumer,
        workers=workers,
        queue_size=max(workers * 2, 4),
        on_error=on_error,
    )
    pipeline.run(list(enumerate(sources, 1)))

    print(f"\nDone. {processed} regenerated, {failed} failed.")
    if total_input or total_output:
        print(f"Tokens: {total_input:,} in + {total_output:,} out (cost depends on provider)")

    return {
        "processed": processed,
        "failed": failed,
        "input_tokens": total_input,
        "output_tokens": total_output,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate closets via a user-configured LLM (OpenAI-compatible API)"
    )
    from .config import DEFAULT_PALACE_PATH

    parser.add_argument(
        "--palace",
        default=DEFAULT_PALACE_PATH,
        help="Path to the palace",
    )
    parser.add_argument("--wing", default=None, help="Limit to one wing")
    parser.add_argument("--sample", type=int, default=0, help="Only process first N source files")
    parser.add_argument("--dry-run", action="store_true", help="List work without calling the LLM")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="LLM base URL (overrides $LLM_ENDPOINT), e.g. http://localhost:11434/v1",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="LLM bearer token (overrides $LLM_KEY). Optional for local inference.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help='LLM model name (overrides $LLM_MODEL), e.g. "gpt-4o-mini" or "llama3:8b"',
    )
    args = parser.parse_args()

    cfg = LLMConfig(endpoint=args.endpoint, key=args.key, model=args.model)
    regenerate_closets(
        args.palace, wing=args.wing, sample=args.sample, dry_run=args.dry_run, cfg=cfg
    )
