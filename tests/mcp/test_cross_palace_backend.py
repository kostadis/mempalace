"""Cross-palace ``palace=`` reads must open the target palace with its own backend.

A non-default palace used to be opened through the MCP server's chroma-only
client cache. Against a turbovec palace that reported 0 drawers, named the
default palace's backend, and created ``chroma.sqlite3`` inside the turbovec
palace -- after which every backend-detected open of that palace failed with
"multiple backend artifacts".
"""

import os

import pytest

pytest.importorskip("turbovecdb")

from _mcp_server_helpers import _patch_mcp_server  # noqa: E402

from mempalace.backends import PalaceRef, detect_backends_for_path, get_backend  # noqa: E402

DIM = 8


def _seed_turbovec_palace(path):
    os.makedirs(path, exist_ok=True)
    backend = type(get_backend("turbovec"))()
    try:
        col = backend.get_collection(
            PalaceRef(id=path, local_path=path),
            collection_name="mempalace_drawers",
            create=True,
        )
        col.upsert(
            documents=["the zeppelin design review kept the pipeline verbatim"],
            ids=["drawer_beta_1"],
            metadatas=[{"wing": "beta", "room": "general", "source_file": "/src/notes.md"}],
            embeddings=[[1.0] + [0.0] * (DIM - 1)],
        )
    finally:
        backend.close()


@pytest.fixture
def turbovec_palace(tmp_dir):
    path = os.path.join(tmp_dir, "other_turbovec_palace")
    _seed_turbovec_palace(path)
    return path


def test_status_reads_turbovec_palace_with_its_own_backend(
    monkeypatch, config, kg, turbovec_palace
):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace import mcp_server

    result = mcp_server.tool_status(palace=turbovec_palace)

    assert "error" not in result, result
    assert result["total_drawers"] == 1
    assert result["wings"] == {"beta": 1}
    assert result["backend"] == "turbovec"


def test_cross_palace_read_never_creates_chroma_store_in_turbovec_palace(
    monkeypatch, config, kg, turbovec_palace
):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace import mcp_server

    mcp_server.tool_status(palace=turbovec_palace)
    mcp_server.tool_status(palace=turbovec_palace)

    assert not os.path.exists(os.path.join(turbovec_palace, "chroma.sqlite3"))
    assert detect_backends_for_path(turbovec_palace) == ["turbovec"]


def test_cross_palace_read_refuses_palace_with_mixed_backend_artifacts(
    monkeypatch, config, kg, turbovec_palace
):
    """Two backends' artifacts in one palace: report it, never guess."""
    import chromadb

    chromadb.PersistentClient(path=turbovec_palace)  # leaves chroma.sqlite3 behind
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace import mcp_server

    result = mcp_server.tool_status(palace=turbovec_palace)

    assert "error" in result
    assert "multiple backend artifacts" in str(result)
