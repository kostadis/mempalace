"""Integration tests for the parallel mining path in mempalace.miner.

These tests exercise ``_mine_impl`` end-to-end against a real ChromaDB
PersistentClient on a temp dir, with a stubbed embedding function (so we
don't need vLLM / Ollama on the test runner). The single-writer invariant
and idempotency contracts are the load-bearing pieces.
"""

from __future__ import annotations

import threading
from pathlib import Path

import chromadb
import pytest

from mempalace import embedding, miner
from mempalace.parallel import ParallelPipeline


class _StubEF:
    """ChromaDB-protocol-shaped embedding function.

    ChromaDB validates ``ef.__class__.__call__``'s signature against the
    EmbeddingFunction protocol ``(self, input)``. A plain function has
    ``function.__call__(*args, **kwargs)`` and fails that check, so we need
    a real class. Returns a fixed dim-3 zero vector per text — sufficient
    for any test that only cares about drawer accounting, not retrieval
    quality.
    """

    @staticmethod
    def name() -> str:
        return "stub-test-ef"

    def __call__(self, input):
        return [[0.0, 0.0, 0.0] for _ in input]


def _stub_ef():
    return _StubEF()


@pytest.fixture(autouse=True)
def _patch_ef(monkeypatch):
    """Force every mine in this module to use the stub EF.

    Patching at ``mempalace.embedding`` because that's the module
    ``miner._mine_impl`` imports it from at call time.
    """
    monkeypatch.setattr(embedding, "get_embedding_function", _stub_ef)


def _make_project(tmp_path: Path, n_files: int = 5, content_per_file: str = None) -> Path:
    """Write a minimal scannable project tree."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "mempalace.yaml").write_text(
        "wing: test_wing\nrooms:\n  - name: general\n    description: All\n",
        encoding="utf-8",
    )
    for i in range(n_files):
        body = content_per_file or (f"file {i} content " * 50)
        (project / f"f{i}.py").write_text(body, encoding="utf-8")
    return project


def _make_palace(tmp_path: Path) -> Path:
    palace = tmp_path / "palace"
    palace.mkdir()
    return palace


def test_parallel_mine_files_a_corpus(tmp_path):
    """workers=4 mines all files and persists drawers to chromadb."""
    project = _make_project(tmp_path, n_files=5)
    palace = _make_palace(tmp_path)

    miner.mine(
        project_dir=str(project),
        palace_path=str(palace),
        wing_override="test_wing",
        workers=4,
    )

    client = chromadb.PersistentClient(path=str(palace))
    col = client.get_collection("mempalace_drawers")
    assert col.count() > 0


def test_parallel_mine_idempotent_re_run(tmp_path):
    """A second mine on the same corpus must report 100% files skipped."""
    project = _make_project(tmp_path, n_files=5)
    palace = _make_palace(tmp_path)

    miner.mine(
        project_dir=str(project),
        palace_path=str(palace),
        wing_override="test_wing",
        workers=4,
    )

    client = chromadb.PersistentClient(path=str(palace))
    col = client.get_collection("mempalace_drawers")
    first_count = col.count()
    assert first_count > 0

    # Re-mine — every file should be detected as already-mined.
    miner.mine(
        project_dir=str(project),
        palace_path=str(palace),
        wing_override="test_wing",
        workers=4,
    )

    second_count = col.count()
    assert second_count == first_count, (
        "Re-mine must not duplicate drawers when the corpus is unchanged"
    )


def test_parallel_mine_matches_serial_output(tmp_path):
    """workers=1 and workers=8 produce identical drawer sets for the same input."""
    project = _make_project(tmp_path, n_files=5)
    palace_serial = tmp_path / "palace_serial"
    palace_parallel = tmp_path / "palace_parallel"

    miner.mine(
        project_dir=str(project),
        palace_path=str(palace_serial),
        wing_override="test_wing",
        workers=1,
    )
    miner.mine(
        project_dir=str(project),
        palace_path=str(palace_parallel),
        wing_override="test_wing",
        workers=8,
    )

    client_a = chromadb.PersistentClient(path=str(palace_serial))
    client_b = chromadb.PersistentClient(path=str(palace_parallel))
    col_a = client_a.get_collection("mempalace_drawers")
    col_b = client_b.get_collection("mempalace_drawers")

    assert col_a.count() == col_b.count(), "Drawer count must not depend on workers"

    # Drawer IDs are SHA-deterministic — the two palaces must hold the
    # same id set (order-independent).
    ids_a = set(col_a.get()["ids"])
    ids_b = set(col_b.get()["ids"])
    assert ids_a == ids_b, "Drawer IDs must be identical regardless of worker count"


def test_parallel_mine_workers_argument_overrides_config(tmp_path, monkeypatch):
    """Explicit workers= arg wins over MempalaceConfig().workers default."""
    project = _make_project(tmp_path, n_files=3)
    palace = _make_palace(tmp_path)

    captured = {}
    real_pipeline = ParallelPipeline

    class SpyPipeline(real_pipeline):
        def __init__(self, *args, **kwargs):
            captured["workers"] = kwargs.get("workers")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(miner, "ParallelPipeline", SpyPipeline)

    miner.mine(
        project_dir=str(project),
        palace_path=str(palace),
        wing_override="test_wing",
        workers=12,
    )

    assert captured["workers"] == 12


def test_parallel_mine_consumer_runs_in_single_thread(tmp_path, monkeypatch):
    """The HNSW invariant: only one thread ever calls collection.upsert.

    This is the single most important behavioral guarantee — concurrent
    upserts against the same chromadb collection corrupt the HNSW index.
    """
    project = _make_project(tmp_path, n_files=20)
    palace = _make_palace(tmp_path)

    upsert_thread_ids: list[int] = []
    upsert_lock = threading.Lock()

    # Wrap chromadb collection's upsert to record which thread invokes it.
    from mempalace.backends import chroma as chroma_backend

    real_upsert = chroma_backend.ChromaCollection.upsert

    def spy_upsert(self, *args, **kwargs):
        with upsert_lock:
            upsert_thread_ids.append(threading.get_ident())
        return real_upsert(self, *args, **kwargs)

    monkeypatch.setattr(chroma_backend.ChromaCollection, "upsert", spy_upsert)

    miner.mine(
        project_dir=str(project),
        palace_path=str(palace),
        wing_override="test_wing",
        workers=8,
    )

    assert len(upsert_thread_ids) > 0, "expected at least one upsert call"
    assert len(set(upsert_thread_ids)) == 1, (
        f"upsert ran on {len(set(upsert_thread_ids))} threads — "
        f"HNSW single-writer invariant violated"
    )


def test_parallel_mine_skips_file_on_embedding_error(tmp_path, monkeypatch):
    """A producer that throws on one file must not abort the mine."""
    project = _make_project(tmp_path, n_files=5)
    palace = _make_palace(tmp_path)

    poison_name = "f2.py"

    class PickyEF:
        @staticmethod
        def name() -> str:
            return "picky-test-ef"

        def __call__(self, input):
            joined = "\n".join(input)
            if "POISON" in joined:
                raise RuntimeError("simulated embedding endpoint failure")
            return [[0.0, 0.0, 0.0] for _ in input]

    # Replace one file's content with the poison marker so its embed fails.
    (project / poison_name).write_text("POISON " * 50, encoding="utf-8")
    monkeypatch.setattr(embedding, "get_embedding_function", lambda: PickyEF())

    miner.mine(
        project_dir=str(project),
        palace_path=str(palace),
        wing_override="test_wing",
        workers=4,
    )

    client = chromadb.PersistentClient(path=str(palace))
    col = client.get_collection("mempalace_drawers")
    # The other 4 files filed; the poison file did not.
    sources = {meta["source_file"] for meta in col.get()["metadatas"]}
    poison_path = str(project / poison_name)
    assert poison_path not in sources, "poison file should be skipped, not partially filed"
    assert len(sources) == 4


def test_parallel_mine_dry_run_is_serial_and_files_nothing(tmp_path):
    """Dry-run keeps the serial path (no parallelism for print-only)."""
    project = _make_project(tmp_path, n_files=3)
    palace = _make_palace(tmp_path)

    miner.mine(
        project_dir=str(project),
        palace_path=str(palace),
        wing_override="test_wing",
        dry_run=True,
        workers=8,  # should be ignored in dry-run
    )

    # Palace shouldn't contain a chromadb at all — dry-run never opens one.
    assert not (palace / "chroma.sqlite3").exists()


def test_parallel_mine_workers_none_resolves_from_config(tmp_path, monkeypatch):
    """workers=None falls through to MempalaceConfig().workers."""
    project = _make_project(tmp_path, n_files=2)
    palace = _make_palace(tmp_path)

    # Force a specific config value via env.
    monkeypatch.setenv("MEMPALACE_WORKERS", "6")

    captured = {}

    class SpyPipeline(ParallelPipeline):
        def __init__(self, *args, **kwargs):
            captured["workers"] = kwargs.get("workers")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(miner, "ParallelPipeline", SpyPipeline)

    miner.mine(
        project_dir=str(project),
        palace_path=str(palace),
        wing_override="test_wing",
        workers=None,
    )

    assert captured["workers"] == 6
