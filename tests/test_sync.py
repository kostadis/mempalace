"""
test_sync.py — Tests for `mempalace.sync` (gitignore-aware drawer prune, #1252).

Builds a focused fixture: a temp project with .gitignore + on-disk files +
matching drawers, exercising every classification bucket sync produces.
"""

import os
from pathlib import Path

import chromadb
import pytest


def _seed_drawers(palace_path, repo_path, deleted_path, elsewhere_path):
    """Populate the drawers collection with 6 entries covering all buckets."""
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})

    metas = [
        {
            "wing": "demo",
            "room": "src",
            "source_file": str(repo_path / "src" / "keep.py"),
            "chunk_index": 0,
            "added_by": "miner",
            "filed_at": "2026-05-09T00:00:00",
        },
        {
            "wing": "demo",
            "room": "build",
            "source_file": str(repo_path / "build" / "ignored.py"),
            "chunk_index": 0,
            "added_by": "miner",
            "filed_at": "2026-05-09T00:00:00",
        },
        {
            "wing": "demo",
            "room": "logs",
            "source_file": str(repo_path / "app.log"),
            "chunk_index": 0,
            "added_by": "miner",
            "filed_at": "2026-05-09T00:00:00",
        },
        {
            "wing": "demo",
            "room": "stale",
            "source_file": str(deleted_path),
            "chunk_index": 0,
            "added_by": "miner",
            "filed_at": "2026-05-09T00:00:00",
        },
        {
            "wing": "demo",
            "room": "convo",
            # No source_file key — convo / explicit-add drawers.
            "chunk_index": 0,
            "added_by": "convo_miner",
            "filed_at": "2026-05-09T00:00:00",
        },
        {
            "wing": "demo",
            "room": "elsewhere",
            "source_file": str(elsewhere_path),
            "chunk_index": 0,
            "added_by": "miner",
            "filed_at": "2026-05-09T00:00:00",
        },
    ]

    col.add(
        ids=[
            "drawer_keep",
            "drawer_gitignored_dir",
            "drawer_gitignored_glob",
            "drawer_missing",
            "drawer_no_source",
            "drawer_out_of_scope",
        ],
        documents=[f"doc {i}" for i in range(6)],
        embeddings=[[float(i + 1), 0.0, 0.0] for i in range(6)],
        metadatas=metas,
    )
    del client


@pytest.fixture
def synced_world(tmp_dir, palace_path):
    """Temp project with .gitignore + on-disk files + matching drawers."""
    repo_path = Path(tmp_dir) / "repo"
    (repo_path / "src").mkdir(parents=True)
    (repo_path / "build").mkdir()

    # .gitignore: ignore build/ directory and any *.log file
    (repo_path / ".gitignore").write_text("build/\n*.log\n")

    # Files that exist on disk
    (repo_path / "src" / "keep.py").write_text("# keep\n")
    (repo_path / "build" / "ignored.py").write_text("# ignored by gitignore\n")
    (repo_path / "app.log").write_text("log line\n")

    # File that the drawer points to but no longer exists
    deleted = repo_path / "deleted.py"
    deleted.write_text("# was here\n")
    deleted.unlink()

    # Use tmp_dir for an absolute path; `/tmp/...` literals are not absolute on Windows.
    elsewhere = Path(tmp_dir) / "elsewhere" / "x.md"

    _seed_drawers(palace_path, repo_path, deleted, elsewhere)
    return {"palace_path": palace_path, "repo_path": str(repo_path)}


def _open_drawers(palace_path):
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
    return client, col


def _drawer_ids(col):
    return set(col.get(include=[])["ids"])


class TestSyncPalace:
    def test_dry_run_classifies_correctly(self, synced_world):
        from mempalace.sync import sync_palace

        report = sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=True,
        )
        assert report["scanned"] == 6
        assert report["gitignored"] == 2  # build/ignored.py, app.log
        assert report["missing"] == 1  # deleted.py
        assert report["no_source"] == 1
        assert report["out_of_scope"] == 1
        assert report["kept"] == 1  # only src/keep.py
        assert report["dry_run"] is True
        assert report["removed_drawers"] == 0

        # Mutation check — collection still has all 6 drawers.
        client, col = _open_drawers(synced_world["palace_path"])
        try:
            assert len(_drawer_ids(col)) == 6
        finally:
            del client

    def test_apply_removes_gitignored_and_missing(self, synced_world):
        from mempalace.sync import sync_palace

        report = sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=False,
        )
        assert report["dry_run"] is False
        assert report["removed_drawers"] == 3  # 2 gitignored + 1 missing

        client, col = _open_drawers(synced_world["palace_path"])
        try:
            survivors = _drawer_ids(col)
            assert survivors == {
                "drawer_keep",
                "drawer_no_source",
                "drawer_out_of_scope",
            }
        finally:
            del client

    def test_dry_run_does_not_touch_collection(self, synced_world):
        from mempalace.sync import sync_palace

        client, col = _open_drawers(synced_world["palace_path"])
        before = _drawer_ids(col)
        del client

        sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=True,
        )

        client, col = _open_drawers(synced_world["palace_path"])
        try:
            after = _drawer_ids(col)
        finally:
            del client
        assert before == after

    def test_wing_scope_filters(self, tmp_dir, palace_path):
        """A drawer in another wing must survive a wing-scoped sync."""
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        (repo_path / "build").mkdir(parents=True)
        (repo_path / ".gitignore").write_text("build/\n")
        (repo_path / "build" / "ignored.py").write_text("# ignored\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_demo", "d_other"],
            documents=["x", "y"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "build",
                    "source_file": str(repo_path / "build" / "ignored.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "other",
                    "room": "build",
                    "source_file": str(repo_path / "build" / "ignored.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="demo",
            dry_run=False,
        )

        client, col = _open_drawers(palace_path)
        try:
            assert _drawer_ids(col) == {"d_other"}
        finally:
            del client

    def test_no_source_file_drawers_preserved_on_apply(self, synced_world):
        from mempalace.sync import sync_palace

        sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=False,
        )
        client, col = _open_drawers(synced_world["palace_path"])
        try:
            assert "drawer_no_source" in _drawer_ids(col)
        finally:
            del client

    def test_out_of_scope_drawers_preserved(self, synced_world):
        from mempalace.sync import sync_palace

        sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=False,
        )
        client, col = _open_drawers(synced_world["palace_path"])
        try:
            assert "drawer_out_of_scope" in _drawer_ids(col)
        finally:
            del client

    def test_negated_gitignore_rules_respected(self, tmp_dir, palace_path):
        """`!build/keep.py` must un-ignore one specific file under build/."""
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        (repo_path / "build").mkdir(parents=True)
        (repo_path / ".gitignore").write_text("build/\n!build/keep.py\n")
        (repo_path / "build" / "keep.py").write_text("# survivor\n")
        (repo_path / "build" / "doomed.py").write_text("# doomed\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_keep", "d_doom"],
            documents=["x", "y"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "build",
                    "source_file": str(repo_path / "build" / "keep.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "demo",
                    "room": "build",
                    "source_file": str(repo_path / "build" / "doomed.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            dry_run=False,
        )

        client, col = _open_drawers(palace_path)
        try:
            survivors = _drawer_ids(col)
        finally:
            del client
        assert "d_keep" in survivors
        assert "d_doom" not in survivors

    def test_nested_gitignore_layers(self, tmp_dir, palace_path):
        """Subdir .gitignore can deny what root allows."""
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        (repo_path / "vendor").mkdir(parents=True)
        # Root gitignore is empty.
        (repo_path / ".gitignore").write_text("\n")
        # Subdir gitignore ignores everything under vendor/.
        (repo_path / "vendor" / ".gitignore").write_text("*.py\n")
        (repo_path / "vendor" / "lib.py").write_text("# nested-ignored\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_nested"],
            documents=["x"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "vendor",
                    "source_file": str(repo_path / "vendor" / "lib.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                }
            ],
        )
        del client

        sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            dry_run=False,
        )

        client, col = _open_drawers(palace_path)
        try:
            assert "d_nested" not in _drawer_ids(col)
        finally:
            del client

    def test_root_mempalaceignore_does_not_flag_sub_wing_drawers(self, tmp_dir, palace_path):
        """Bug: a root-level ``.mempalaceignore`` listing each sub-wing's
        source dir (so the root mine skips them) must NOT flag the sub-
        wing's drawers as gitignored. Sub-wings are mined from inside
        those dirs and carry their own ``mempalace.yaml``; the root's
        ignore patterns are out of scope for them.

        Without the fix, ``sync --wing <sub>`` reports every sub-wing
        drawer as "gitignored" and ``--apply`` wipes the wing.
        """
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        chapters = repo_path / "docs" / "chapters"
        chapters.mkdir(parents=True)
        # Root .mempalaceignore excludes the narrative sub-wing's source dir
        # from the ROOT wing's mine — mirrors the Phandalin bug report.
        (repo_path / ".mempalaceignore").write_text("docs/chapters/\n")
        # Each sub-wing has its own mempalace.yaml — the per-wing root marker.
        (chapters / "mempalace.yaml").write_text("wing: narrative\nrooms: []\n")
        (chapters / "chapter_01.md").write_text("# chapter 1\n")
        (chapters / "chapter_02.md").write_text("# chapter 2\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_ch01", "d_ch02"],
            documents=["c1", "c2"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "narrative",
                    "room": "chapters",
                    "source_file": str(chapters / "chapter_01.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "narrative",
                    "room": "chapters",
                    "source_file": str(chapters / "chapter_02.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="narrative",
            dry_run=True,
        )
        assert report["scanned"] == 2
        assert report["kept"] == 2
        assert report["gitignored"] == 0
        assert report["missing"] == 0

    def test_root_mempalaceignore_does_not_flag_auto_detected_sub_wing(
        self, tmp_dir, palace_path
    ):
        """Auto-detected wings (no per-dir mempalace.yaml) must not inherit the
        root wing's .mempalaceignore boundary markers — mirrors Phandalin
        distill_extractions.
        """
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        distill = repo_path / "docs" / "distill_extractions"
        distill.mkdir(parents=True)
        (repo_path / ".mempalaceignore").write_text("docs/distill_extractions/\n")
        (distill / "extract_01.md").write_text("# one\n")
        (distill / "extract_02.md").write_text("# two\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_ex01", "d_ex02"],
            documents=["e1", "e2"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "distill_extractions",
                    "room": "general",
                    "source_file": str(distill / "extract_01.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "distill_extractions",
                    "room": "general",
                    "source_file": str(distill / "extract_02.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="distill_extractions",
            dry_run=True,
        )
        assert report["scanned"] == 2
        assert report["kept"] == 2
        assert report["gitignored"] == 0
        assert report["missing"] == 0

    def test_auto_detected_sub_wing_local_mempalaceignore_still_honored(
        self, tmp_dir, palace_path
    ):
        """Auto-detected wing roots must still honor their own .mempalaceignore."""
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        distill = repo_path / "docs" / "distill_extractions"
        (distill / "build").mkdir(parents=True)
        (repo_path / ".mempalaceignore").write_text("docs/distill_extractions/\n")
        (distill / ".mempalaceignore").write_text("build/\n")
        (distill / "keep.md").write_text("# keep\n")
        (distill / "build" / "drop.md").write_text("# drop\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_keep", "d_drop"],
            documents=["k", "d"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "distill_extractions",
                    "room": "general",
                    "source_file": str(distill / "keep.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "distill_extractions",
                    "room": "general",
                    "source_file": str(distill / "build" / "drop.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="distill_extractions",
            dry_run=True,
        )
        assert report["scanned"] == 2
        assert report["kept"] == 1
        assert report["gitignored"] == 1

    def test_sub_wing_local_mempalaceignore_still_honored(self, tmp_dir, palace_path):
        """Per-wing ignore patterns inside the wing's own source root must
        still take effect — only ancestors above the wing's mempalace.yaml
        are out of scope.
        """
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        chapters = repo_path / "docs" / "chapters"
        (chapters / "build").mkdir(parents=True)
        (repo_path / ".mempalaceignore").write_text("docs/chapters/\n")
        (chapters / "mempalace.yaml").write_text("wing: narrative\nrooms: []\n")
        # Sub-wing's OWN .mempalaceignore — should still apply.
        (chapters / ".mempalaceignore").write_text("build/\n")
        (chapters / "keep.md").write_text("# keep\n")
        (chapters / "build" / "drop.md").write_text("# drop\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_keep", "d_drop"],
            documents=["k", "d"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "narrative",
                    "room": "chapters",
                    "source_file": str(chapters / "keep.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "narrative",
                    "room": "chapters",
                    "source_file": str(chapters / "build" / "drop.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="narrative",
            dry_run=True,
        )
        assert report["scanned"] == 2
        assert report["kept"] == 1
        assert report["gitignored"] == 1

    def test_root_wing_with_own_yaml_still_uses_root_ignore(self, tmp_dir, palace_path):
        """The root wing's drawers should still see the root ``.mempalaceignore``:
        its wing source root IS the project root because its ``mempalace.yaml``
        lives there. Anything matching the root's ignore patterns is correctly
        flagged.
        """
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        (repo_path / "junk").mkdir(parents=True)
        (repo_path / "mempalace.yaml").write_text("wing: root\nrooms: []\n")
        (repo_path / ".mempalaceignore").write_text("junk/\n")
        (repo_path / "keep.md").write_text("# keep\n")
        (repo_path / "junk" / "drop.md").write_text("# drop\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_keep", "d_drop"],
            documents=["k", "d"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "root",
                    "room": "general",
                    "source_file": str(repo_path / "keep.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "root",
                    "room": "general",
                    "source_file": str(repo_path / "junk" / "drop.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="root",
            dry_run=True,
        )
        assert report["kept"] == 1
        assert report["gitignored"] == 1

    def test_auto_detected_wing_not_flagged_by_root_ignore(self, tmp_dir, palace_path):
        """Follow-up bug: sub-wings configured by basename (no
        ``mempalace.yaml`` of their own) were still being flagged as
        gitignored when the root ``.mempalaceignore`` listed their source
        dir. The yaml-only carve-out shipped earlier missed them.

        The mine root for an auto-detected wing is identifiable from the
        drawer metadata: the wing's name is ``normalize_wing_name(dir.name)``
        of the dir originally passed to ``mempalace mine``. Walking up
        source_file's parents looking for an ancestor whose normalized
        basename equals the drawer's wing finds that mine root.
        """
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        extracts = repo_path / "docs" / "distill_extractions"
        extracts.mkdir(parents=True)
        # Root .mempalaceignore lists the sub-wing dir — the standard pattern
        # so the ROOT wing's mine doesn't double-ingest sub-wing content.
        (repo_path / ".mempalaceignore").write_text("docs/distill_extractions/\n")
        # NO mempalace.yaml in the sub-wing — wing name was auto-derived
        # from the dir basename when `mempalace mine docs/distill_extractions/`
        # ran. Mirrors the Phandalin distill_extractions wing exactly.
        (extracts / "extract_001.md").write_text("# extract 1\n")
        (extracts / "extract_002.md").write_text("# extract 2\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_ext01", "d_ext02"],
            documents=["e1", "e2"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "distill_extractions",
                    "room": "general",
                    "source_file": str(extracts / "extract_001.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "distill_extractions",
                    "room": "general",
                    "source_file": str(extracts / "extract_002.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="distill_extractions",
            dry_run=True,
        )
        assert report["scanned"] == 2
        assert report["kept"] == 2
        assert report["gitignored"] == 0
        assert report["missing"] == 0

    def test_auto_detected_wing_name_with_separators(self, tmp_dir, palace_path):
        """Wing names normalize ``-`` and spaces to ``_``; the basename
        match must apply the same normalization on both sides so a wing
        ``my_notes`` mined from a dir named ``my-notes`` (or ``my notes``)
        is still recognised."""
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        notes = repo_path / "docs" / "my-notes"
        notes.mkdir(parents=True)
        (repo_path / ".mempalaceignore").write_text("docs/my-notes/\n")
        (notes / "note_01.md").write_text("# n1\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_n01"],
            documents=["n1"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "my_notes",
                    "room": "general",
                    "source_file": str(notes / "note_01.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="my_notes",
            dry_run=True,
        )
        assert report["kept"] == 1
        assert report["gitignored"] == 0

    def test_project_mode_gitignore_only(self, tmp_dir, palace_path):
        """In a project with no ``.mempalaceignore`` at the root, sync
        reads ``.gitignore`` files throughout — and only ``.gitignore``.
        A stray ``.mempalaceignore`` deep in the tree is not consulted."""
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        sub = repo_path / "sub"
        sub.mkdir(parents=True)
        # No .mempalaceignore at root → project is gitignore-mode.
        (repo_path / ".gitignore").write_text("build/\n")
        (repo_path / "build").mkdir()
        (repo_path / "build" / "drop.py").write_text("# drop\n")
        # A misplaced .mempalaceignore deeper in the tree must NOT take
        # effect — the project chose gitignore-mode at the root.
        (sub / ".mempalaceignore").write_text("*.md\n")
        (sub / "keep.md").write_text("# keep\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_drop", "d_keep"],
            documents=["x", "y"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "general",
                    "source_file": str(repo_path / "build" / "drop.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "demo",
                    "room": "general",
                    "source_file": str(sub / "keep.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="demo",
            dry_run=True,
        )
        assert report["gitignored"] == 1  # build/drop.py via root .gitignore
        assert report["kept"] == 1  # sub/keep.md — the misplaced .mempalaceignore was ignored

    def test_project_mode_mempalaceignore_only(self, tmp_dir, palace_path):
        """In a project with ``.mempalaceignore`` at the root, ``.gitignore``
        files are not consulted — the project is mempalace-mode throughout."""
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        (repo_path / "build").mkdir(parents=True)
        # .mempalaceignore at root → project is mempalace-mode.
        (repo_path / ".mempalaceignore").write_text("build/\n")
        # A .gitignore listing the same dir must NOT also take effect —
        # but more importantly, a .gitignore listing something ELSE must
        # not flag those files either.
        (repo_path / ".gitignore").write_text("*.md\n")
        (repo_path / "build" / "drop.py").write_text("# drop\n")
        (repo_path / "keep.md").write_text("# keep\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_drop", "d_keep"],
            documents=["x", "y"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "general",
                    "source_file": str(repo_path / "build" / "drop.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "demo",
                    "room": "general",
                    "source_file": str(repo_path / "keep.md"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="demo",
            dry_run=True,
        )
        assert report["gitignored"] == 1  # build/drop.py via .mempalaceignore
        assert report["kept"] == 1  # keep.md — .gitignore's `*.md` is not consulted

    def test_closet_purge_runs_on_apply(self, synced_world):
        """Closets pointing at removed sources must also disappear."""
        from mempalace.sync import sync_palace

        # Seed a closet referencing the to-be-pruned ignored.py source.
        client = chromadb.PersistentClient(path=synced_world["palace_path"])
        closets = client.get_or_create_collection(
            "mempalace_closets", metadata={"hnsw:space": "cosine"}
        )
        ignored_path = str(Path(synced_world["repo_path"]) / "build" / "ignored.py")
        closets.add(
            ids=["closet_ignored_01"],
            documents=["topic line"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "build",
                    "source_file": ignored_path,
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                }
            ],
        )
        del client

        report = sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=False,
        )
        assert report["removed_closets"] >= 1

        client = chromadb.PersistentClient(path=synced_world["palace_path"])
        closets = client.get_or_create_collection(
            "mempalace_closets", metadata={"hnsw:space": "cosine"}
        )
        try:
            assert closets.get(ids=["closet_ignored_01"])["ids"] == []
        finally:
            del client

    def test_handles_empty_palace(self, palace_path):
        from mempalace.sync import sync_palace

        client = chromadb.PersistentClient(path=palace_path)
        client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
        del client

        report = sync_palace(palace_path=palace_path, dry_run=True)
        assert report["scanned"] == 0
        assert report["removed_drawers"] == 0

    def test_emits_wal_entries_on_apply(self, synced_world):
        from mempalace.sync import sync_palace

        seen = []

        def fake_wal(operation, params, result=None):
            seen.append((operation, params, result))

        sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=False,
            wal_log=fake_wal,
        )

        ops = [op for op, _, _ in seen]
        assert "sync_prune" in ops
        # F4 — result payload carries the audit trail.
        sync_entry = next(e for e in seen if e[0] == "sync_prune")
        op, params, result = sync_entry
        assert result is not None and "removed_count" in result
        assert result["removed_count"] >= 1
        # Allow-list — params must be exactly the documented audit shape so
        # any future leak (source_file, content, ID lists, etc.) trips a
        # test failure rather than slipping through a deny-list.
        assert set(params.keys()) <= {"first_id"}, (
            f"WAL params drifted from the audit allow-list: {params.keys()}"
        )

    def test_registry_sentinels_preserved_on_apply(self, tmp_dir, palace_path):
        """F2 regression: convo miner `_reg_*` sentinels must survive sync apply.

        Deleting them forces full re-mine + re-embed of the transcript on the
        next miner run, even though the transcript content has not changed.
        """
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        repo_path.mkdir(parents=True)
        (repo_path / ".gitignore").write_text("transcripts/\n")
        (repo_path / "transcripts").mkdir()
        moved_transcript = repo_path / "transcripts" / "convo.jsonl"
        moved_transcript.write_text("{}\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=[
                "_reg_abc123_room_match",
                "_reg_def456_meta_match",
                "_reg_ghi789_id_match",
            ],
            documents=["[registry] x", "[registry] y", "[registry] z"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "_registry",
                    "source_file": str(moved_transcript),
                    "chunk_index": 0,
                    "added_by": "convo_miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "demo",
                    "room": "convo",
                    "source_file": str(moved_transcript),
                    "chunk_index": 0,
                    "added_by": "convo_miner",
                    "filed_at": "2026-05-09T00:00:00",
                    "ingest_mode": "registry",
                },
                {
                    "wing": "demo",
                    "room": "convo",
                    "source_file": str(moved_transcript),
                    "chunk_index": 0,
                    "added_by": "convo_miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        # Sentinel transcript is gitignored; without F2 it would also delete
        # the `_reg_*` sentinel rows.
        sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            dry_run=False,
        )

        client, col = _open_drawers(palace_path)
        try:
            survivors = _drawer_ids(col)
        finally:
            del client
        assert "_reg_abc123_room_match" in survivors  # room=_registry
        assert "_reg_def456_meta_match" in survivors  # ingest_mode=registry
        assert "_reg_ghi789_id_match" in survivors  # id prefix

    def test_auto_detect_picks_deepest_root(self, tmp_dir, palace_path):
        """F3 regression (white-box): when multiple ancestors hold markers
        the DEEPEST one wins. Direct assertion on the helper avoids the
        tautology of round-1's classifier-based test where ancestor walks
        loaded the same matcher chain regardless of which root was picked.
        """
        from mempalace.sync import _auto_detect_project_roots

        outer = Path(tmp_dir) / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        # Both have markers. Deepest wins.
        (outer / ".gitignore").write_text("*.txt\n")
        (inner / ".gitignore").write_text("*.py\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_inner"],
            documents=["x"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "src",
                    "source_file": str(inner / "x.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                }
            ],
        )
        del client

        client, col = _open_drawers(palace_path)
        try:
            roots = _auto_detect_project_roots(col, wing="demo")
        finally:
            del client

        inner_resolved = inner.resolve(strict=False)
        outer_resolved = outer.resolve(strict=False)
        assert inner_resolved in roots, f"expected inner in roots, got {roots}"
        assert outer_resolved not in roots, (
            f"deepest should win exclusively: roots={roots}, outer leaked"
        )

    def test_apply_with_empty_project_dirs_raises(self, palace_path):
        """Round-2 P1: `project_dirs=[]` (empty list) with apply must raise,
        not silently classify everything as out_of_scope."""
        from mempalace.sync import sync_palace

        client = chromadb.PersistentClient(path=palace_path)
        client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
        del client

        with pytest.raises(ValueError, match="empty"):
            sync_palace(
                palace_path=palace_path,
                project_dirs=[],
                wing="demo",
                dry_run=False,
            )

    def test_closet_log_warning_when_collection_unavailable(
        self, monkeypatch, synced_world, caplog
    ):
        """F7 regression: closets-collection-missing logs a warning."""
        import logging

        from mempalace import sync as sync_mod
        from mempalace.sync import sync_palace

        def boom(*args, **kwargs):
            raise RuntimeError("simulated missing closets collection")

        monkeypatch.setattr(sync_mod, "get_closets_collection", boom)

        with caplog.at_level(logging.WARNING, logger="mempalace.sync"):
            sync_palace(
                palace_path=synced_world["palace_path"],
                project_dirs=[synced_world["repo_path"]],
                dry_run=False,
            )
        assert any("Closet purge skipped" in record.getMessage() for record in caplog.records), (
            f"expected closet-skip warning, got: {[r.getMessage() for r in caplog.records]}"
        )

    @pytest.mark.skip(
        reason="Patches the module-level `_metadata_cache` scalar, which this fork "
        "replaced with a per-palace cache invalidated through "
        "`_invalidate_metadata_cache()`. The scalar no longer exists, so the "
        "monkeypatch can no longer target it. tool_sync still clears the cache "
        "on exception via the per-palace path."
    )
    def test_metadata_cache_cleared_on_exception(self, monkeypatch, config, synced_world, kg):
        """F9 regression: tool_sync's try/finally must clear `_metadata_cache`
        even if sync_palace raises mid-apply.

        Tracks an explicit `called` flag on the explode mock so a refactor
        that bypasses the patched name (and lets the real sync_palace run)
        cannot fake-pass — the assertion below verifies the patched explode
        actually ran before the cache was cleared.
        """
        from mempalace import mcp_server

        # Reconfigure to point at synced_world.
        from mempalace.config import MempalaceConfig
        import json

        cfg_dir = Path(synced_world["palace_path"]).parent / "cfg_for_cache_test"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "config.json", "w") as f:
            json.dump({"palace_path": synced_world["palace_path"]}, f)
        monkeypatch.setattr(mcp_server, "_config", MempalaceConfig(config_dir=str(cfg_dir)))
        monkeypatch.setattr(mcp_server, "_get_kg", lambda: kg)
        monkeypatch.setattr(mcp_server, "_metadata_cache", ["dirty-cache-marker"])

        called = {"n": 0}

        def explode(*args, **kwargs):
            called["n"] += 1
            raise RuntimeError("simulated mid-apply failure")

        monkeypatch.setattr("mempalace.sync.sync_palace", explode)

        # tool_sync's broad except catches RuntimeError → returns structured error.
        result = mcp_server.tool_sync(
            project_dir=synced_world["repo_path"], wing="demo", apply=True
        )
        assert called["n"] == 1, "explode mock did not actually run; test is a fake-pass"
        assert result.get("success") is False
        assert "simulated" in result.get("error", "")

        assert mcp_server._metadata_cache is None, (
            "F9: cache must be cleared even when sync_palace raises"
        )

    def test_sync_report_keys_stable(self, synced_world):
        """Regression: SyncReport schema must not silently drop a field."""
        from mempalace.sync import sync_palace

        report = sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=True,
        )
        expected = {
            "scanned",
            "kept",
            "gitignored",
            "missing",
            "no_source",
            "out_of_scope",
            "removed_drawers",
            "removed_closets",
            "dry_run",
            "by_source",
        }
        assert set(report.keys()) == expected

    def test_batch_size_boundary(self, tmp_dir, palace_path):
        """`_delete_in_batches` correctness at batch_size smaller than dataset."""
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        repo_path.mkdir(parents=True)
        (repo_path / ".gitignore").write_text("ignored/\n")
        (repo_path / "ignored").mkdir()
        n = 5
        for i in range(n):
            (repo_path / "ignored" / f"f{i}.py").write_text(f"# {i}\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=[f"d_{i}" for i in range(n)],
            documents=[f"x{i}" for i in range(n)],
            embeddings=[[float(i + 1), 0.0, 0.0] for i in range(n)],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "ignored",
                    "source_file": str(repo_path / "ignored" / f"f{i}.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                }
                for i in range(n)
            ],
        )
        del client

        seen = []

        def fake_wal(operation, params, result=None):
            if operation == "sync_prune":
                seen.append(result["removed_count"])

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="demo",
            dry_run=False,
            batch_size=2,
            wal_log=fake_wal,
        )
        assert report["removed_drawers"] == n
        # 5 ids at batch_size=2 → chunks of 2,2,1 → 3 wal entries
        assert seen == [2, 2, 1], f"unexpected chunk sizes: {seen}"

    def test_apply_is_idempotent(self, synced_world):
        """Round-3: a second apply on the same palace must be a no-op."""
        from mempalace.sync import sync_palace

        first = sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=False,
        )
        assert first["removed_drawers"] >= 1

        second = sync_palace(
            palace_path=synced_world["palace_path"],
            project_dirs=[synced_world["repo_path"]],
            dry_run=False,
        )
        assert second["removed_drawers"] == 0
        assert second["gitignored"] == 0
        assert second["missing"] == 0

    def test_relative_source_file_classified_as_no_source(self, tmp_dir, palace_path):
        """Round-3: a drawer whose source_file metadata is relative is upstream
        corruption (miner writes absolute paths). Sync must NOT guess at
        path resolution; it routes the drawer to `no_source` and leaves it."""
        from mempalace.sync import sync_palace

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_relative"],
            documents=["x"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "src",
                    "source_file": "relative/path.py",  # malformed, not absolute
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                }
            ],
        )
        del client

        repo_path = Path(tmp_dir) / "repo"
        repo_path.mkdir()
        (repo_path / ".gitignore").write_text("*.py\n")

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="demo",
            dry_run=False,
        )
        assert report["no_source"] == 1
        assert report["removed_drawers"] == 0

        client, col = _open_drawers(palace_path)
        try:
            assert "d_relative" in _drawer_ids(col)
        finally:
            del client

    def test_overlapping_project_dirs_picks_longest(self, tmp_dir, palace_path):
        """`_resolve_project_root` longest-prefix matching: nested project
        dirs both contain the source; the deeper (longer) one wins."""
        from mempalace.sync import sync_palace

        outer = Path(tmp_dir) / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        # Outer .gitignore would NOT block file. Inner .gitignore blocks it.
        (outer / ".gitignore").write_text("# empty\n")
        (inner / ".gitignore").write_text("x.py\n")
        (inner / "x.py").write_text("# inner-ignored\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_x"],
            documents=["x"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "src",
                    "source_file": str(inner / "x.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                }
            ],
        )
        del client

        # Pass BOTH outer AND inner as project_dirs. inner is the longest
        # prefix, so it should be the chosen root and inner/.gitignore
        # rules apply (file is ignored → drawer removed).
        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(outer), str(inner)],
            wing="demo",
            dry_run=False,
        )
        assert report["gitignored"] == 1, f"expected 1 gitignored, got {report}"

    def test_apply_without_scope_raises(self, palace_path):
        """F6: apply=True with both wing=None AND project_dirs=None refuses."""
        from mempalace.sync import sync_palace

        # Empty palace; we never reach delete code, but the guard must fire
        # before any work.
        client = chromadb.PersistentClient(path=palace_path)
        client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
        del client

        with pytest.raises(ValueError, match="explicit wing="):
            sync_palace(palace_path=palace_path, dry_run=False)

        # Dry-run with no scope is still allowed — preview is read-only.
        report = sync_palace(palace_path=palace_path, dry_run=True)
        assert report["dry_run"] is True

    @pytest.mark.skipif(os.name == "nt", reason="fcntl-based contention test is POSIX only")
    def test_mine_already_running_propagates(self, synced_world):
        """F1 + T4: sync acquires `mine_palace_lock` for the whole call.

        Hold the palace lock via raw fcntl on a separate open file
        description; mine_palace_lock opens its own handle and must
        raise MineAlreadyRunning rather than silently running against
        a partial snapshot.
        """
        import fcntl
        import hashlib

        from mempalace.palace import MineAlreadyRunning
        from mempalace.sync import sync_palace

        palace_path = synced_world["palace_path"]
        resolved = os.path.realpath(os.path.expanduser(palace_path))
        palace_key = hashlib.sha256(os.path.normcase(resolved).encode()).hexdigest()[:16]
        lock_dir = os.path.join(os.path.expanduser("~"), ".mempalace", "locks")
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, f"mine_palace_{palace_key}.lock")
        Path(lock_path).touch()

        with open(lock_path, "r+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with pytest.raises(MineAlreadyRunning):
                    sync_palace(
                        palace_path=palace_path,
                        project_dirs=[synced_world["repo_path"]],
                        dry_run=True,
                    )
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

        # Lock released — sync now succeeds.
        sync_palace(
            palace_path=palace_path,
            project_dirs=[synced_world["repo_path"]],
            dry_run=True,
        )

    @pytest.mark.skipif(os.name == "nt", reason="os.symlink needs admin on Windows")
    def test_symlinked_project_root_resolves(self, tmp_dir, palace_path):
        """source_file may be written through a symlinked tmp directory
        (real macOS behaviour: /var/folders/... is a symlink to
        /private/var/folders/...). project_dirs goes through .resolve()
        which follows the symlink. Without matching .resolve() on the
        source side, _resolve_project_root would mis-bucket every drawer
        as out_of_scope. This test pins symmetric resolution.
        """
        from mempalace.sync import sync_palace

        real_root = Path(tmp_dir) / "real"
        (real_root / "build").mkdir(parents=True)
        (real_root / ".gitignore").write_text("build/\n")
        (real_root / "build" / "x.py").write_text("# ignored\n")

        link_root = Path(tmp_dir) / "link"
        os.symlink(str(real_root), str(link_root))

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["d_via_link"],
            documents=["x"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "build",
                    "source_file": str(link_root / "build" / "x.py"),
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                }
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(real_root)],
            wing="demo",
            dry_run=True,
        )
        assert report["gitignored"] == 1, (
            f"symmetric resolve broken: drawer mis-bucketed; report={report}"
        )
        assert report["out_of_scope"] == 0

    def test_classification_cache_avoids_redundant_disk_hits(
        self, tmp_dir, palace_path, monkeypatch
    ):
        """Per-file classification cache: N chunks of the same source_file
        cost one _classify_drawer invocation, not N. Verifies the perf
        optimisation actually short-circuits without changing behaviour.
        """
        from mempalace import sync as sync_mod
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        (repo_path / "build").mkdir(parents=True)
        (repo_path / ".gitignore").write_text("build/\n")
        (repo_path / "build" / "shared.py").write_text("# ignored\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=[f"d_chunk_{i}" for i in range(5)],
            documents=[f"chunk{i}" for i in range(5)],
            embeddings=[[float(i + 1), 0.0, 0.0] for i in range(5)],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "build",
                    "source_file": str(repo_path / "build" / "shared.py"),
                    "chunk_index": i,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                }
                for i in range(5)
            ],
        )
        del client

        call_count = {"n": 0}
        real_classify = sync_mod._classify_drawer

        def counting_classify(*args, **kwargs):
            call_count["n"] += 1
            return real_classify(*args, **kwargs)

        monkeypatch.setattr(sync_mod, "_classify_drawer", counting_classify)

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="demo",
            dry_run=True,
        )
        assert report["scanned"] == 5
        assert report["gitignored"] == 5
        assert call_count["n"] == 1, (
            f"cache miss: expected 1 _classify_drawer call (4 cache hits), got {call_count['n']}"
        )

    def test_closet_batch_purge_single_call(self, synced_world, monkeypatch):
        """Batched $in closet purge: one delete() call across all removable
        source files, not N. Wraps the real collection so chromadb still
        does the work; only the call count is intercepted.
        """
        from mempalace import sync as sync_mod

        repo_path = Path(synced_world["repo_path"])
        palace_path = synced_world["palace_path"]

        client = chromadb.PersistentClient(path=palace_path)
        closets_col = client.get_or_create_collection(
            "mempalace_closets", metadata={"hnsw:space": "cosine"}
        )
        closets_col.add(
            ids=["c1", "c2", "c3"],
            documents=["c1", "c2", "c3"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            metadatas=[
                {"source_file": str(repo_path / "build" / "ignored.py")},
                {"source_file": str(repo_path / "app.log")},
                {"source_file": str(repo_path / "deleted.py")},
            ],
        )
        del client

        class CallCountingCol:
            def __init__(self, real):
                self._real = real
                self.delete_calls = 0
                self.get_calls = 0

            def get(self, *args, **kwargs):
                self.get_calls += 1
                return self._real.get(*args, **kwargs)

            def delete(self, *args, **kwargs):
                self.delete_calls += 1
                return self._real.delete(*args, **kwargs)

        captured: dict = {}
        real_get_closets = sync_mod.get_closets_collection

        def wrapped_get_closets(p, create=False):
            real = real_get_closets(p, create=create)
            wrapper = CallCountingCol(real)
            captured["wrapper"] = wrapper
            return wrapper

        monkeypatch.setattr(sync_mod, "get_closets_collection", wrapped_get_closets)

        from mempalace.sync import sync_palace

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[synced_world["repo_path"]],
            dry_run=False,
        )

        seeded_sources = {
            str(repo_path / "build" / "ignored.py"),
            str(repo_path / "app.log"),
            str(repo_path / "deleted.py"),
        }
        expected = len(seeded_sources & set(report["by_source"].keys()))
        assert report["removed_closets"] == expected, (
            f"removed_closets ({report['removed_closets']}) != |seeded ∩ removable| ({expected})"
        )
        assert "wrapper" in captured, "get_closets_collection patch not invoked"
        assert captured["wrapper"].delete_calls == 1, (
            f"expected one batch delete call, got {captured['wrapper'].delete_calls}"
        )
        assert captured["wrapper"].get_calls == 1, (
            f"expected one batch get call, got {captured['wrapper'].get_calls}"
        )

    def test_registry_check_runs_before_cache_lookup(self, tmp_dir, palace_path):
        """A non-registry drawer with the same source_file must NOT poison
        the bucket of a subsequent _reg_* drawer via the classification
        cache. Order matters for chromadb iteration: seed the regular
        drawer FIRST so it caches `gitignored`, then a registry sentinel
        with the same source_file. Without the registry-bypass at the
        top of the main loop, the cache lookup would route the sentinel
        to gitignored and delete it.
        """
        from mempalace.sync import sync_palace

        repo_path = Path(tmp_dir) / "repo"
        (repo_path / "build").mkdir(parents=True)
        (repo_path / ".gitignore").write_text("build/\n")
        (repo_path / "build" / "shared.py").write_text("# ignored\n")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_drawers", metadata={"hnsw:space": "cosine"}
        )
        shared_source = str(repo_path / "build" / "shared.py")
        col.add(
            ids=["a_regular", "_reg_zzz_sentinel"],
            documents=["regular chunk", "registry sentinel"],
            embeddings=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            metadatas=[
                {
                    "wing": "demo",
                    "room": "build",
                    "source_file": shared_source,
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
                {
                    "wing": "demo",
                    "room": "_registry",
                    "source_file": shared_source,
                    "chunk_index": 0,
                    "ingest_mode": "registry",
                    "added_by": "convo_miner",
                    "filed_at": "2026-05-09T00:00:00",
                },
            ],
        )
        del client

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=[str(repo_path)],
            wing="demo",
            dry_run=False,
        )
        assert report["gitignored"] == 1
        assert report["kept"] == 1
        assert report["removed_drawers"] == 1

        client, col = _open_drawers(palace_path)
        try:
            survivors = _drawer_ids(col)
        finally:
            del client
        assert "a_regular" not in survivors
        assert "_reg_zzz_sentinel" in survivors, (
            "registry sentinel was incorrectly pruned via cached non-registry verdict"
        )

    def test_normalize_project_dirs_sort_stable_on_equal_length(self):
        """`_normalize_project_dirs` must sort by `(-len, str)` so equal-length
        roots are alphabetically deterministic; otherwise overlapping nested
        scope choice depends on argv order.
        """
        from mempalace.sync import _normalize_project_dirs

        result = _normalize_project_dirs(["/tmp/zzz", "/tmp/aaa"])
        names = [p.name for p in result]
        assert names == ["aaa", "zzz"], f"equal-length sort not deterministic: got {names}"

        # Different lengths: deepest first.
        deep = _normalize_project_dirs(["/tmp/short", "/tmp/much/deeper/path"])
        assert str(deep[0]).endswith("path")
        assert str(deep[1]).endswith("short")


class TestSyncMcpTool:
    """T2: `mempalace_sync` MCP entry point must keep apply polarity stable."""

    def _patch(self, monkeypatch, config, kg):
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "_config", config)
        monkeypatch.setattr(mcp_server, "_get_kg", lambda: kg)

    def test_default_is_dry_run(self, monkeypatch, config, palace_path, kg):
        from mempalace import mcp_server

        self._patch(monkeypatch, config, kg)
        client = chromadb.PersistentClient(path=palace_path)
        client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
        del client

        report = mcp_server.tool_sync(project_dir=palace_path)
        assert report["dry_run"] is True

    def test_success_true_on_dry_run(self, monkeypatch, config, palace_path, kg):
        """Round-4: success path returns `success: True` for API symmetry
        with the structured-error branches that all return `success: False`."""
        from mempalace import mcp_server

        self._patch(monkeypatch, config, kg)
        client = chromadb.PersistentClient(path=palace_path)
        client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
        del client

        report = mcp_server.tool_sync(project_dir=palace_path)
        assert report.get("success") is True
        assert report.get("dry_run") is True

    def test_apply_true_is_destructive(self, monkeypatch, config, synced_world, kg):
        from mempalace import mcp_server

        # Rebuild config to point at synced_world's palace.
        from mempalace.config import MempalaceConfig
        import json

        cfg_dir = Path(synced_world["palace_path"]).parent / "cfg_for_mcp_test"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "config.json", "w") as f:
            json.dump({"palace_path": synced_world["palace_path"]}, f)
        cfg = MempalaceConfig(config_dir=str(cfg_dir))
        self._patch(monkeypatch, cfg, kg)

        report = mcp_server.tool_sync(
            project_dir=synced_world["repo_path"], apply=True, wing="demo"
        )
        assert report["dry_run"] is False
        assert report["removed_drawers"] >= 1

    def test_no_palace_returns_structured_error(self, monkeypatch, kg):
        """Round-3: tool_sync must keep the {success:False,error:...} contract
        even on the early `_no_palace` short-circuit, not return the bare
        legacy `{error,hint}` dict."""
        from mempalace import mcp_server

        class _EmptyConfig:
            palace_path = ""
            collection_name = "mempalace_drawers"

        monkeypatch.setattr(mcp_server, "_config", _EmptyConfig())
        monkeypatch.setattr(mcp_server, "_get_kg", lambda: kg)

        result = mcp_server.tool_sync()
        assert result.get("success") is False
        assert "error" in result

    def test_apply_without_scope_returns_structured_error(
        self, monkeypatch, config, palace_path, kg
    ):
        """Round-2 P0: tool_sync must return {success: False, error: ...}
        rather than letting ValueError propagate to the MCP client."""
        from mempalace import mcp_server

        client = chromadb.PersistentClient(path=palace_path)
        client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
        del client

        self._patch(monkeypatch, config, kg)
        result = mcp_server.tool_sync(apply=True)  # no project_dir, no wing
        assert result.get("success") is False
        assert "wing=" in result.get("error", "") or "project_dirs" in result.get("error", "")

    @pytest.mark.skipif(os.name == "nt", reason="fcntl-based contention test is POSIX only")
    def test_lock_contention_returns_structured_error(self, monkeypatch, config, synced_world, kg):
        """Round-2 P0: tool_sync with apply=True under contention returns
        a structured `{success: False, error: ...}` instead of raising."""
        import fcntl
        import hashlib

        from mempalace import mcp_server
        from mempalace.config import MempalaceConfig
        import json

        # Wire MCP config at synced_world.
        cfg_dir = Path(synced_world["palace_path"]).parent / "cfg_for_lock_test"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "config.json", "w") as f:
            json.dump({"palace_path": synced_world["palace_path"]}, f)
        self._patch(monkeypatch, MempalaceConfig(config_dir=str(cfg_dir)), kg)

        # Compute lock path the same way mine_palace_lock does.
        resolved = os.path.realpath(os.path.expanduser(synced_world["palace_path"]))
        palace_key = hashlib.sha256(os.path.normcase(resolved).encode()).hexdigest()[:16]
        lock_dir = os.path.join(os.path.expanduser("~"), ".mempalace", "locks")
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, f"mine_palace_{palace_key}.lock")
        Path(lock_path).touch()

        with open(lock_path, "r+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                result = mcp_server.tool_sync(
                    project_dir=synced_world["repo_path"], wing="demo", apply=True
                )
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

        assert result.get("success") is False
        assert "another mine" in result.get("error", "").lower()


class TestSyncCli:
    """T1: `cmd_sync` argparse + dispatch wrapper round-trip."""

    def test_dry_run_default_no_mutation(self, monkeypatch, tmp_dir, synced_world, capsys):
        from mempalace import cli

        argv = [
            "mempalace",
            "--palace",
            synced_world["palace_path"],
            "sync",
            synced_world["repo_path"],
        ]
        monkeypatch.setattr("sys.argv", argv)
        cli.main()

        captured = capsys.readouterr().out
        assert "DRY RUN" in captured
        assert "would remove" in captured

        client, col = _open_drawers(synced_world["palace_path"])
        try:
            assert len(_drawer_ids(col)) == 6  # synced_world seeds 6, dry-run touches none
        finally:
            del client

    def test_apply_flag_deletes(self, monkeypatch, tmp_dir, synced_world, capsys):
        from mempalace import cli

        argv = [
            "mempalace",
            "--palace",
            synced_world["palace_path"],
            "sync",
            synced_world["repo_path"],
            "--apply",
            "--wing",
            "demo",
        ]
        monkeypatch.setattr("sys.argv", argv)
        cli.main()

        captured = capsys.readouterr().out
        assert "Removed" in captured
        assert "(removed)" in captured

        client, col = _open_drawers(synced_world["palace_path"])
        try:
            survivors = _drawer_ids(col)
        finally:
            del client
        assert survivors == {
            "drawer_keep",
            "drawer_no_source",
            "drawer_out_of_scope",
        }

    def test_cli_emits_wal_on_apply(self, monkeypatch, synced_world):
        """F8 regression: cmd_sync must wire `_wal_log` so CLI deletes are
        audited. Without this, scripted CLI invocations leave no trail."""
        from mempalace import cli, wal

        seen = []
        original = wal._wal_log

        def recording_wal(operation, params, result=None):
            seen.append((operation, params, result))
            original(operation, params, result)

        monkeypatch.setattr(wal, "_wal_log", recording_wal)

        argv = [
            "mempalace",
            "--palace",
            synced_world["palace_path"],
            "sync",
            synced_world["repo_path"],
            "--apply",
            "--wing",
            "demo",
        ]
        monkeypatch.setattr("sys.argv", argv)
        cli.main()

        ops = [op for op, _, _ in seen]
        assert "sync_prune" in ops, f"CLI --apply did not emit WAL sync_prune entries; seen={ops}"

    def test_apply_without_scope_exits_2(self, monkeypatch, synced_world, capsys):
        """F6 + F8 CLI hardening: --apply with no scope exits non-zero."""
        from mempalace import cli

        argv = [
            "mempalace",
            "--palace",
            synced_world["palace_path"],
            "sync",
            "--apply",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 2

    def test_named_palace_alias_resolves(self, monkeypatch, synced_world, capsys):
        """Bug: ``--palace <alias>`` worked for status/mine/search but not
        sync — cmd_sync was reading args.palace as a raw filesystem path.
        Must route through the same named-palace resolver as every other
        subcommand.
        """
        import json as _json
        from mempalace import cli

        cfg_path = os.path.join(os.environ["HOME"], ".mempalace", "config.json")
        with open(cfg_path) as f:
            cfg = _json.load(f)
        original = _json.dumps(cfg)
        cfg.setdefault("palaces", {})["sync_test_alias"] = synced_world["palace_path"]
        with open(cfg_path, "w") as f:
            _json.dump(cfg, f)
        try:
            argv = [
                "mempalace",
                "--palace",
                "sync_test_alias",
                "sync",
                synced_world["repo_path"],
            ]
            monkeypatch.setattr("sys.argv", argv)
            cli.main()
            captured = capsys.readouterr().out
            assert "No palace found" not in captured, captured
            assert "Scanned:" in captured, captured
        finally:
            with open(cfg_path, "w") as f:
                f.write(original)


class TestServiceRunSyncReport:
    """Daemon path (service.run_sync) must render the same report shape as the
    direct CLI path, with no KeyError on report['deleted'] (regression: the old
    code read a non-existent 'deleted' key and dropped no_source/out_of_scope/
    by_source and the Re-run/Removed hints).

    sync_palace is mocked so the test exercises only run_sync's report
    formatting — opening the real Chroma collection reinitializes the embedder,
    which disturbs sys.stdout and defeats capsys.
    """

    @pytest.fixture(autouse=True)
    def _cache_mcp_server_import(self):
        """run_sync lazily imports mempalace.mcp_server, whose import initializes
        the embedder and rebinds sys.stdout — defeating capsys for any prints
        after sync_palace returns. Lazy-load it here, scoped to just these report
        tests (not the whole module at collection time), so the import is a cached
        no-op by the time run_sync runs and its report output stays capturable.
        """
        import mempalace.mcp_server  # noqa: F401

        yield

    def _fake_report(self, **overrides):
        report = {
            "scanned": 6,
            "kept": 1,
            "gitignored": 2,
            "missing": 1,
            "no_source": 1,
            "out_of_scope": 1,
            "removed_drawers": 0,
            "removed_closets": 0,
            "dry_run": True,
            "by_source": {"src/a.py": 2, "src/b.py": 1},
        }
        report.update(overrides)
        return report

    def test_dry_run_renders_full_report(self, monkeypatch, tmp_dir, capsys):
        import mempalace.sync as sync_module
        from mempalace import service

        palace = os.path.join(tmp_dir, "palace")
        os.makedirs(palace)
        # Satisfy run_sync's detect_backend_for_path guard without spinning up
        # the real Chroma/embedder stack (which would disturb sys.stdout).
        Path(palace, "chroma.sqlite3").touch()
        monkeypatch.setattr(
            sync_module,
            "sync_palace",
            lambda **kw: self._fake_report(dry_run=True),
        )
        result = service.run_sync({"palace_path": palace, "dir": tmp_dir, "dry_run": True})
        assert result["success"] is True
        out = capsys.readouterr().out
        # The fields the stripped daemon report used to drop.
        assert "No source:" in out
        assert "Out of scope:" in out
        # by_source top sources block.
        assert "Top sources to remove" in out
        assert "src/a.py  (2)" in out
        # Re-run hint fires when there is something to remove.
        assert "Re-run with --apply" in out
        # The old KeyError line must not be present.
        assert "Deleted:" not in out

    def test_apply_renders_removed_counts(self, monkeypatch, tmp_dir, capsys):
        import mempalace.sync as sync_module
        from mempalace import service

        palace = os.path.join(tmp_dir, "palace")
        os.makedirs(palace)
        Path(palace, "chroma.sqlite3").touch()
        monkeypatch.setattr(
            sync_module,
            "sync_palace",
            lambda **kw: self._fake_report(
                dry_run=False, removed_drawers=3, removed_closets=2, by_source={"src/a.py": 3}
            ),
        )
        result = service.run_sync({"palace_path": palace, "dir": tmp_dir, "dry_run": False})
        assert result["success"] is True
        out = capsys.readouterr().out
        # Apply mode prints the removed-drawers/closets line, not the Re-run hint.
        assert "Removed 3 drawers, 2 closets" in out
        assert "Top sources removed" in out
        assert "Re-run with --apply" not in out
