import pytest
from mempalace.convo_miner import _chunk_by_exchange, _chunk_by_paragraph, CHUNK_SIZE

def test_exchange_chunking_respects_limit():
    # Simulate a very large exchange
    large_content = "User: " + ("A" * (CHUNK_SIZE * 3)) + "\nResponse: " + ("B" * (CHUNK_SIZE * 3))
    lines = large_content.split("\n")
    # Mocking the logic for a simple test
    chunks = _chunk_by_exchange(lines)
    for chunk in chunks:
        assert len(chunk["content"]) <= CHUNK_SIZE, f"Chunk too large: {len(chunk['content'])}"

def test_paragraph_chunking_respects_limit():
    large_content = "A" * (CHUNK_SIZE * 5)
    chunks = _chunk_by_paragraph(large_content)
    for chunk in chunks:
        assert len(chunk["content"]) <= CHUNK_SIZE, f"Chunk too large: {len(chunk['content'])}"


def test_paragraph_chunking_unique_chunk_index_across_paragraphs():
    # Two oversized paragraphs — each used to restart chunk_index at 0,
    # causing drawer-id collisions on upsert. Indices must be unique.
    para = "A" * (CHUNK_SIZE * 2)
    content = para + "\n\n" + ("B" * (CHUNK_SIZE * 2))
    chunks = _chunk_by_paragraph(content)
    indices = [c["chunk_index"] for c in chunks]
    assert len(indices) == len(set(indices)), f"Duplicate chunk_index values: {indices}"


def test_paragraph_chunking_unique_chunk_index_line_group_branch():
    # Single long line-blob (no paragraph breaks, many newlines) — the
    # line-group branch also used to restart idx per group.
    line = "A" * (CHUNK_SIZE * 2)
    content = "\n".join([line] * 30)
    chunks = _chunk_by_paragraph(content)
    indices = [c["chunk_index"] for c in chunks]
    assert len(indices) == len(set(indices)), f"Duplicate chunk_index values: {indices}"


if __name__ == "__main__":
    pytest.main([__file__])
