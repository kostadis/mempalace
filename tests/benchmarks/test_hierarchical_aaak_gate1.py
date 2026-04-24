"""Gate 1 benchmark: hierarchical AAAK over a generic multi-wing palace.

Loads ``hierarchical_aaak.json``, materializes the fixture into a fresh
palace, builds the room + wing indices, and compares flat search
(baseline) against ``tool_search_hierarchical`` across ~15 queries. Emits
metrics for each pass criterion:

    * recall@10 delta    — hierarchical must be within 5% of baseline
    * precision@10 delta — hierarchical must be ≥ baseline
    * drawers-scored     — hierarchical must be ≥ 2× cheaper than baseline
    * indices size on disk — must fit under 50 MB
    * human-readable paths — 10/10 query paths must be interpretable

Metrics are also recorded via ``record_metric("hierarchical_aaak", ...)``
so the session-level benchmark report captures them.
"""

import json
import os
import re
from pathlib import Path

import pytest

from mempalace import mcp_server, palace as palace_mod, recursive_indexer
from mempalace.searcher import search_memories
from tests.benchmarks.report import record_metric


FIXTURE_PATH = Path(__file__).parent / "hierarchical_aaak.json"


# ── Fixture loading / palace materialization ─────────────────────────────


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def benchmark_palace(tmp_path_factory, fixture):
    palace_path = str(tmp_path_factory.mktemp("gate1_palace"))

    drawers_col = palace_mod.get_collection(palace_path)
    closets_col = palace_mod.get_closets_collection(palace_path)

    # ── Seeded (ground-truth) drawers + their closets ────────────────
    drawers = fixture["fixture"]["drawers"]
    drawers_col.add(
        ids=[d["id"] for d in drawers],
        documents=[d["text"] for d in drawers],
        metadatas=[
            {
                "wing": d["wing"],
                "room": d["room"],
                "source_file": d["source_file"],
                "chunk_index": 0,
                "added_by": "benchmark",
            }
            for d in drawers
        ],
    )
    closets_col.add(
        ids=[f"closet_{d['id']}" for d in drawers],
        documents=["\n".join(d["closet_lines"]) for d in drawers],
        metadatas=[
            {
                "wing": d["wing"],
                "room": d["room"],
                "source_file": d["source_file"],
                "top_entities": _top_entities_from_closets(d["closet_lines"]),
            }
            for d in drawers
        ],
    )

    # ── Distractor drawers so pruning has work to do ─────────────────
    #
    # The seeded drawers above are the ground truth — queries target
    # them by design. For the drawer-reduction metric to mean anything
    # we need rooms that aren't dominated by the query-relevant drawer,
    # so the baseline scans many more drawers than the hierarchical
    # candidate set. Fillers have content orthogonal to every seeded
    # query; they should never appear in the top-k results.
    filler_per_room = int(fixture["fixture"].get("filler_per_room", 0))
    filler_drawer_count = 0
    if filler_per_room > 0:
        filler_docs = []
        filler_ids = []
        filler_metas = []
        for w in fixture["fixture"]["wings"]:
            wing = w["name"]
            for room in w["rooms"]:
                for i in range(filler_per_room):
                    did = f"filler_{wing}_{room}_{i:03d}"
                    filler_docs.append(_filler_text(wing, room, i))
                    filler_ids.append(did)
                    filler_metas.append(
                        {
                            "wing": wing,
                            "room": room,
                            "source_file": f"_filler_{wing}_{room}_{i}.md",
                            "chunk_index": 0,
                            "added_by": "benchmark-filler",
                        }
                    )
        drawers_col.add(ids=filler_ids, documents=filler_docs, metadatas=filler_metas)
        filler_drawer_count = len(filler_ids)

    # Size the palace BEFORE indices — used to compute the index overhead.
    size_before = _dir_size_bytes(palace_path)

    recursive_indexer.rebuild_all(palace_path)

    size_after = _dir_size_bytes(palace_path)

    return {
        "path": palace_path,
        "seed_drawers": len(drawers),
        "filler_drawers": filler_drawer_count,
        "total_drawers": len(drawers) + filler_drawer_count,
        "size_before_bytes": size_before,
        "size_after_bytes": size_after,
        "index_overhead_bytes": size_after - size_before,
    }


_FILLER_TOPICS = [
    "meeting agenda with routine status items",
    "quarterly budget review summary notes",
    "generic code cleanup tracker line",
    "vacation calendar reminder for next week",
    "travel expense receipt ledger entry",
    "onboarding checklist item for new hires",
    "office supply reorder request note",
    "weekly one-on-one manager talking points",
    "bookshelf reading list unrelated to palace topics",
    "daily standup blocker follow-ups from yesterday",
]


def _filler_text(wing: str, room: str, i: int) -> str:
    """Lorem-adjacent content that never matches any benchmark query."""
    base = _FILLER_TOPICS[i % len(_FILLER_TOPICS)]
    return (
        f"[filler {wing}/{room} #{i}] {base}. "
        f"This drawer exists to exercise the hierarchical pruning path. "
        f"It should never appear in the top-k for any seeded query."
    )


# ── Utilities ────────────────────────────────────────────────────────────


_CLOSET_LINE_RE = re.compile(r"→|->")


def _top_entities_from_closets(lines):
    """Heuristic rollup of entities across a drawer's closet lines."""
    ents = []
    for line in lines:
        # topic|entities|→ids → second field
        parts = _CLOSET_LINE_RE.split(line, maxsplit=1)
        if not parts:
            continue
        head = parts[0]
        fields = head.split("|")
        if len(fields) >= 2:
            for e in fields[-1].split(";"):
                e = e.strip()
                if e and e not in ents:
                    ents.append(e)
    return ";".join(ents[:10])


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _recall_at_k(expected: set, returned: list, k: int = 10) -> float:
    if not expected:
        return 1.0
    top = set(returned[:k])
    return len(expected & top) / len(expected)


def _precision_at_k(expected: set, returned: list, k: int = 10) -> float:
    top = returned[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in expected) / len(top)


def _is_readable_path(path_block: dict) -> bool:
    """A path is readable when wings and rooms have real names + an
    AAAK preview with at least one alphabetic token. Pure heuristic
    but sufficient for Gate 1's "can a human interpret the path without
    opening a drawer?" check.
    """
    wings = path_block.get("wings") or []
    rooms = path_block.get("rooms") or []
    if not wings:
        return False
    top_wing = wings[0]
    if not top_wing.get("wing") or not top_wing.get("preview"):
        return False
    wing_tokens = re.findall(r"[A-Za-z]{3,}", top_wing["preview"])
    if not wing_tokens:
        return False
    if rooms:
        top_room = rooms[0]
        if not top_room.get("room") or not top_room.get("preview"):
            return False
        room_tokens = re.findall(r"[A-Za-z]{3,}", top_room["preview"])
        return bool(room_tokens)
    return True


# ── The benchmark ────────────────────────────────────────────────────────


@pytest.mark.benchmark
def test_gate1_hierarchical_aaak(benchmark_palace, fixture):
    palace_path = benchmark_palace["path"]
    total_drawers = benchmark_palace["total_drawers"]
    queries = fixture["queries"]
    thresholds = fixture["gate_thresholds"]

    baseline_recalls = []
    baseline_precisions = []
    hier_recalls = []
    hier_precisions = []
    hier_drawers_scored = []
    readable_paths = 0
    per_query_details = []

    for q in queries:
        expected = set(q["expected_drawer_ids"])

        # Baseline: flat search scores every drawer in the palace (conceptually).
        base = search_memories(q["query"], palace_path=palace_path, n_results=10)
        base_ids = [h["drawer_id"] for h in base.get("results", [])]
        baseline_recalls.append(_recall_at_k(expected, base_ids))
        baseline_precisions.append(_precision_at_k(expected, base_ids))

        # Hierarchical: passes through wing + room pruning before scoring drawers.
        hier = mcp_server.tool_search_hierarchical(
            q["query"], palace=palace_path, limit=10, max_depth=2
        )
        hier_ids = [h["drawer_id"] for h in hier.get("results", [])]
        hier_recalls.append(_recall_at_k(expected, hier_ids))
        hier_precisions.append(_precision_at_k(expected, hier_ids))

        # Drawers scored ≈ total drawers in the rooms selected by pruning.
        # (Chroma uses ANN over the restricted set; the candidate pool is
        # our per-room drawer count summed.)
        room_counts = 0
        for r in hier.get("path", {}).get("rooms", []):
            room_counts += int(r.get("drawer_count", 0) or 0)
        if room_counts == 0:  # fallback path — whole palace in play
            room_counts = total_drawers
        hier_drawers_scored.append(room_counts)

        if _is_readable_path(hier.get("path", {})):
            readable_paths += 1

        per_query_details.append(
            {
                "id": q["id"],
                "query": q["query"],
                "expected": sorted(expected),
                "baseline_top5": base_ids[:5],
                "hier_top5": hier_ids[:5],
                "hier_path_wings": [w["wing"] for w in hier.get("path", {}).get("wings", [])][:3],
                "hier_path_rooms": [
                    f"{r['wing']}/{r['room']}"
                    for r in hier.get("path", {}).get("rooms", [])
                ][:3],
            }
        )

    # Aggregate metrics
    baseline_recall = _mean(baseline_recalls)
    baseline_precision = _mean(baseline_precisions)
    hier_recall = _mean(hier_recalls)
    hier_precision = _mean(hier_precisions)
    recall_delta = baseline_recall - hier_recall
    precision_delta = hier_precision - baseline_precision  # positive when hier ≥ baseline

    baseline_drawers_scored = total_drawers  # flat search covers everything
    avg_hier_drawers_scored = _mean(hier_drawers_scored)
    drawer_reduction_ratio = (
        baseline_drawers_scored / avg_hier_drawers_scored
        if avg_hier_drawers_scored > 0
        else float("inf")
    )

    indices_mb = benchmark_palace["index_overhead_bytes"] / (1024 * 1024)

    # Record everything for the session report.
    record_metric("hierarchical_aaak", "queries", len(queries))
    record_metric("hierarchical_aaak", "total_drawers", total_drawers)
    record_metric("hierarchical_aaak", "baseline_recall_at_10", round(baseline_recall, 3))
    record_metric("hierarchical_aaak", "hier_recall_at_10", round(hier_recall, 3))
    record_metric("hierarchical_aaak", "recall_delta", round(recall_delta, 3))
    record_metric("hierarchical_aaak", "baseline_precision_at_10", round(baseline_precision, 3))
    record_metric("hierarchical_aaak", "hier_precision_at_10", round(hier_precision, 3))
    record_metric("hierarchical_aaak", "precision_delta", round(precision_delta, 3))
    record_metric("hierarchical_aaak", "baseline_drawers_scored", baseline_drawers_scored)
    record_metric(
        "hierarchical_aaak", "hier_avg_drawers_scored", round(avg_hier_drawers_scored, 2)
    )
    record_metric(
        "hierarchical_aaak", "drawer_reduction_ratio", round(drawer_reduction_ratio, 2)
    )
    record_metric("hierarchical_aaak", "indices_mb", round(indices_mb, 3))
    record_metric("hierarchical_aaak", "human_readable_paths", readable_paths)

    # Echo to stderr so a bare run shows the report without opening the file.
    report_lines = [
        "\n── Gate 1 — hierarchical AAAK benchmark ──",
        f"  queries                  : {len(queries)}",
        f"  total drawers in palace  : {total_drawers}",
        f"  baseline recall@10       : {baseline_recall:.3f}",
        f"  hier     recall@10       : {hier_recall:.3f}  (delta {-recall_delta:+.3f})",
        f"  baseline precision@10    : {baseline_precision:.3f}",
        f"  hier     precision@10    : {hier_precision:.3f}  (delta {precision_delta:+.3f})",
        f"  baseline drawers scored  : {baseline_drawers_scored}",
        f"  hier avg drawers scored  : {avg_hier_drawers_scored:.1f}",
        f"  drawer-reduction ratio   : {drawer_reduction_ratio:.2f}×",
        f"  indices overhead on disk : {indices_mb:.3f} MB",
        f"  human-readable paths     : {readable_paths} / {len(queries)}",
        "",
    ]
    print("\n".join(report_lines))
    for row in per_query_details:
        print(
            "  "
            + f"{row['id']:<3} "
            + f"{row['query'][:38]:<38} "
            + f"→ {','.join(row['hier_path_rooms']) or '?':<30} "
            + f"hits={row['hier_top5']}"
        )

    # Gate 1 assertions
    assert recall_delta <= thresholds["recall_at_10_delta_max"], (
        f"recall dropped by {recall_delta:.3f}, max allowed "
        f"{thresholds['recall_at_10_delta_max']}"
    )
    assert precision_delta >= thresholds["precision_at_10_delta_max"], (
        f"precision dropped by {-precision_delta:.3f}; must be ≥ baseline"
    )
    assert drawer_reduction_ratio >= thresholds["drawers_scored_reduction_min"], (
        f"drawer-scored reduction only {drawer_reduction_ratio:.2f}×, "
        f"need ≥ {thresholds['drawers_scored_reduction_min']}×"
    )
    assert indices_mb <= thresholds["indices_size_mb_max"], (
        f"indices on disk {indices_mb:.2f} MB > {thresholds['indices_size_mb_max']} MB limit"
    )
    assert readable_paths >= thresholds["human_readable_paths_min"], (
        f"only {readable_paths} interpretable paths, need "
        f"≥ {thresholds['human_readable_paths_min']}"
    )


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0
