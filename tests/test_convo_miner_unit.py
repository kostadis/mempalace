"""Unit tests for convo_miner pure functions (no chromadb needed)."""

import contextlib

from mempalace.convo_miner import (
    _ConvoBatch,
    _PreparedConvo,
    _write_prepared_convo,
    chunk_exchanges,
    detect_convo_room,
    scan_convos,
)


class TestChunkExchanges:
    def test_exchange_chunking(self):
        content = (
            "> What is memory?\n"
            "Memory is persistence of information over time.\n\n"
            "> Why does it matter?\n"
            "It enables continuity across sessions and conversations.\n\n"
            "> How do we build it?\n"
            "With structured storage and retrieval mechanisms.\n"
        )
        chunks = chunk_exchanges(content)
        assert len(chunks) >= 2
        assert all("content" in c and "chunk_index" in c for c in chunks)

    def test_paragraph_fallback(self):
        """Content without '>' lines falls back to paragraph chunking."""
        content = (
            "This is a long paragraph about memory systems. " * 10 + "\n\n"
            "This is another paragraph about storage. " * 10 + "\n\n"
            "And a third paragraph about retrieval. " * 10
        )
        chunks = chunk_exchanges(content)
        assert len(chunks) >= 2

    def test_paragraph_line_group_fallback(self):
        """Long content with no paragraph breaks chunks by line groups."""
        lines = [f"Line {i}: some content that is meaningful" for i in range(60)]
        content = "\n".join(lines)
        chunks = chunk_exchanges(content)
        assert len(chunks) >= 1

    def test_empty_content(self):
        chunks = chunk_exchanges("")
        assert chunks == []

    def test_short_content_skipped(self):
        chunks = chunk_exchanges("> hi\nbye")
        # Too short to produce chunks (below MIN_CHUNK_SIZE)
        assert isinstance(chunks, list)

    def test_long_ai_response_not_truncated(self):
        """AI responses longer than 8 lines must be stored in full (verbatim principle)."""
        lines = [f"Step {i}: important detail that must be stored" for i in range(1, 14)]
        content = "> How do I implement authentication?\n" + "\n".join(lines)
        chunks = chunk_exchanges(content)
        assert len(chunks) >= 1
        stored = chunks[0]["content"]
        # All 13 lines must be present — none silently dropped
        for i in range(1, 14):
            assert f"Step {i}:" in stored, f"Step {i} was truncated and not stored"


class TestDetectConvoRoom:
    def test_technical_room(self):
        content = "Let me debug this python function and fix the code error in the api"
        assert detect_convo_room(content) == "technical"

    def test_planning_room(self):
        content = "We need to plan the roadmap for the next sprint and set milestone deadlines"
        assert detect_convo_room(content) == "planning"

    def test_architecture_room(self):
        content = "The architecture uses a service layer with component interface and module design"
        assert detect_convo_room(content) == "architecture"

    def test_decisions_room(self):
        content = "We decided to switch and migrated to the new framework after we chose it"
        assert detect_convo_room(content) == "decisions"

    def test_general_fallback(self):
        content = "Hello, how are you doing today? The weather is nice."
        assert detect_convo_room(content) == "general"


class TestScanConvos:
    def test_scan_finds_txt_and_md(self, tmp_path):
        (tmp_path / "chat.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "notes.md").write_text("world", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"fake")
        files = scan_convos(str(tmp_path))
        extensions = {f.suffix for f in files}
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".png" not in extensions

    def test_scan_skips_git_dir(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.txt").write_text("git stuff", encoding="utf-8")
        (tmp_path / "chat.txt").write_text("hello", encoding="utf-8")
        files = scan_convos(str(tmp_path))
        assert len(files) == 1

    def test_scan_skips_meta_json(self, tmp_path):
        (tmp_path / "chat.meta.json").write_text("{}", encoding="utf-8")
        (tmp_path / "chat.json").write_text("{}", encoding="utf-8")
        files = scan_convos(str(tmp_path))
        names = [f.name for f in files]
        assert "chat.json" in names
        assert "chat.meta.json" not in names

    def test_scan_empty_dir(self, tmp_path):
        files = scan_convos(str(tmp_path))
        assert files == []


class TestWritePreparedConvo:
    """The post-refactor equivalent of the old _file_chunks_locked test.

    Convo mining is now split into _prepare_convo (file IO + chunk + build
    IDs/metadata) and _write_prepared_convo (single-writer chromadb upsert
    with pre-computed embeddings). This test pins the batching invariant
    end-to-end on the write phase.
    """

    def test_uses_bounded_upsert_batches(self, monkeypatch):
        import mempalace.convo_miner as convo_miner

        class FakeCol:
            def __init__(self):
                self.batch_sizes = []
                self.embedding_batch_sizes = []

            def delete(self, *args, **kwargs):
                pass

            def upsert(self, documents, ids, metadatas, embeddings=None):
                self.batch_sizes.append(len(documents))
                self.embedding_batch_sizes.append(
                    len(embeddings) if embeddings is not None else None
                )

        monkeypatch.setattr(
            convo_miner, "file_already_mined", lambda collection, source_file: False
        )
        monkeypatch.setattr(convo_miner, "mine_lock", lambda source_file: contextlib.nullcontext())

        # Build a prepared convo by hand with 5 chunks split across 3
        # batches of sizes [2, 2, 1]. Mirrors what _prepare_convo would
        # have produced with DRAWER_UPSERT_BATCH_SIZE=2.
        batches = [
            _ConvoBatch(
                documents=["chunk 0 " * 20, "chunk 1 " * 20],
                ids=["d0", "d1"],
                metadatas=[{"wing": "wing", "room": "general"}, {"wing": "wing", "room": "general"}],
                rooms=["general", "general"],
            ),
            _ConvoBatch(
                documents=["chunk 2 " * 20, "chunk 3 " * 20],
                ids=["d2", "d3"],
                metadatas=[{"wing": "wing", "room": "general"}, {"wing": "wing", "room": "general"}],
                rooms=["general", "general"],
            ),
            _ConvoBatch(
                documents=["chunk 4 " * 20],
                ids=["d4"],
                metadatas=[{"wing": "wing", "room": "general"}],
                rooms=["general"],
            ),
        ]
        prepared = _PreparedConvo(
            source_file="chat.txt",
            chunks=[{"content": f"chunk {i} " * 20, "chunk_index": i} for i in range(5)],
            room="general",
            extract_mode="exchange",
            batches=batches,
        )
        # Pre-computed embeddings — one fixed vector per chunk; chromadb
        # would skip its EF when these are passed.
        embeddings_batches = [
            [[0.0, 0.0, 0.0] for _ in batch.documents] for batch in batches
        ]

        col = FakeCol()
        drawers, room_counts, skipped = _write_prepared_convo(
            prepared, embeddings_batches, col, "wing"
        )

        assert drawers == 5
        assert dict(room_counts) == {}  # exchange mode → no per-chunk room delta
        assert skipped is False
        assert col.batch_sizes == [2, 2, 1]
        # New contract: pre-computed embeddings always flow through to chromadb.
        assert col.embedding_batch_sizes == [2, 2, 1]


class TestMineConvosWorkersPlumbing:
    """Pin the --workers plumbing without spinning up chromadb.

    Heavier integration (real palace, real drawer counts) lives in
    test_convo_miner.py. This is just "the flag reaches the pipeline".
    """

    def test_workers_argument_reaches_pipeline(self, tmp_path, monkeypatch):
        import mempalace.convo_miner as convo_miner
        from mempalace.parallel import ParallelPipeline

        # Make scan find one file with enough content to chunk.
        (tmp_path / "chat.txt").write_text(
            "> hi\n" + ("This is a long enough response. " * 20) + "\n", encoding="utf-8"
        )

        captured = {}

        class SpyPipeline(ParallelPipeline):
            def __init__(self, *args, **kwargs):
                captured["workers"] = kwargs.get("workers")
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(convo_miner, "ParallelPipeline", SpyPipeline)

        # Patch out the real collection writer so we don't need chromadb.
        class FakeCol:
            def get(self, *args, **kwargs):
                return {"ids": []}

            def delete(self, *args, **kwargs):
                pass

            def upsert(self, **kwargs):
                pass

        monkeypatch.setattr(convo_miner, "get_collection", lambda palace_path: FakeCol())
        monkeypatch.setattr(
            convo_miner, "file_already_mined", lambda collection, source_file: False
        )

        # Stub the embedding function — convo_miner imports it lazily
        # inside mine_convos, so we patch the source module.
        from mempalace import embedding

        class StubEF:
            @staticmethod
            def name():
                return "stub"

            def __call__(self, texts):
                return [[0.0, 0.0, 0.0] for _ in texts]

        monkeypatch.setattr(embedding, "get_embedding_function", lambda: StubEF())

        convo_miner.mine_convos(
            convo_dir=str(tmp_path),
            palace_path=str(tmp_path / "palace"),
            wing="testwing",
            workers=4,
        )

        assert captured["workers"] == 4
