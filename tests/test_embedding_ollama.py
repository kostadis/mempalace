"""Tests for mempalace.embedding_ollama.

HTTP is mocked throughout — these tests do not require a running Ollama
or network access.
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from mempalace.embedding_ollama import (
    OllamaEmbeddingError,
    OllamaEmbeddingFunction,
)


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


# ── construction ────────────────────────────────────────────────────────


def test_init_strips_trailing_slash_from_endpoint():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x:11434/")
    assert ef.endpoint == "http://x:11434"


def test_init_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        OllamaEmbeddingFunction(model="", endpoint="http://x")


def test_init_rejects_empty_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="")


def test_name_is_stable_provider_tag():
    # Static method — chromadb persists this on the collection. Must NOT
    # vary per-instance or by model, otherwise rehydration breaks.
    assert OllamaEmbeddingFunction.name() == "ollama"


def test_get_config_round_trip():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x:11434", timeout=30)
    cfg = ef.get_config()
    assert cfg == {"model": "nomic-embed-text", "endpoint": "http://x:11434", "timeout": 30}
    rebuilt = OllamaEmbeddingFunction.build_from_config(cfg)
    assert rebuilt.model == ef.model
    assert rebuilt.endpoint == ef.endpoint
    assert rebuilt.timeout == ef.timeout


def test_validate_config_update_blocks_model_swap():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    with pytest.raises(ValueError, match="Cannot change Ollama embedding model"):
        ef.validate_config_update({"model": "nomic-embed-text"}, {"model": "mxbai-embed-large"})


# ── batch path (/api/embed) ─────────────────────────────────────────────


def test_call_uses_batch_endpoint():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x:11434")

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _mock_response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    with patch("mempalace.embedding_ollama.urlopen", side_effect=fake_urlopen):
        vectors = ef(["alpha", "beta"])

    assert captured["url"] == "http://x:11434/api/embed"
    assert captured["body"] == {"model": "nomic-embed-text", "input": ["alpha", "beta"]}
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    # Subsequent calls should not re-probe — _supports_batch is sticky True.
    assert ef._supports_batch is True


def test_embed_query_and_embed_documents_call_through_to_call():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    with patch(
        "mempalace.embedding_ollama.urlopen",
        return_value=_mock_response({"embeddings": [[0.1, 0.2]]}),
    ):
        q = ef.embed_query(["query text"])
        d = ef.embed_documents(["doc text"])
    assert q == [[0.1, 0.2]]
    assert d == [[0.1, 0.2]]


def test_call_accepts_single_string():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    with patch(
        "mempalace.embedding_ollama.urlopen",
        return_value=_mock_response({"embeddings": [[1.0]]}),
    ):
        vectors = ef("hello")
    assert vectors == [[1.0]]


def test_call_empty_input_returns_empty_list():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    # No HTTP call should happen.
    with patch("mempalace.embedding_ollama.urlopen") as mock_urlopen:
        assert ef([]) == []
    mock_urlopen.assert_not_called()


def test_batch_response_length_mismatch_raises():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    with patch(
        "mempalace.embedding_ollama.urlopen",
        return_value=_mock_response({"embeddings": [[0.1]]}),  # only 1 vector for 2 inputs
    ):
        with pytest.raises(OllamaEmbeddingError, match="returned 1 vectors for 2"):
            ef(["a", "b"])


# ── fallback path (/api/embeddings) ─────────────────────────────────────


def test_call_falls_back_to_single_endpoint_on_404():
    """Older Ollama returns 404 from /api/embed. We must drop to per-input
    /api/embeddings calls and never re-try the batch path for this instance."""
    from urllib.error import HTTPError

    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x:11434")

    call_log = []

    def fake_urlopen(req, timeout):
        call_log.append(req.full_url)
        if req.full_url.endswith("/api/embed"):
            raise HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))
        # legacy single-input endpoint
        return _mock_response({"embedding": [0.5, 0.6]})

    with patch("mempalace.embedding_ollama.urlopen", side_effect=fake_urlopen):
        vectors = ef(["alpha", "beta"])

    assert call_log == [
        "http://x:11434/api/embed",
        "http://x:11434/api/embeddings",
        "http://x:11434/api/embeddings",
    ]
    assert vectors == [[0.5, 0.6], [0.5, 0.6]]
    assert ef._supports_batch is False

    # A second invocation should skip /api/embed entirely.
    call_log.clear()
    with patch("mempalace.embedding_ollama.urlopen", side_effect=fake_urlopen):
        ef(["gamma"])
    assert call_log == ["http://x:11434/api/embeddings"]


def test_non_404_http_error_is_not_swallowed():
    """A 500 from /api/embed should surface, not silently fall back —
    the fallback would mask a real server problem."""
    from urllib.error import HTTPError

    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")

    err = HTTPError("http://x/api/embed", 500, "Server Error", {}, io.BytesIO(b"boom"))
    with patch("mempalace.embedding_ollama.urlopen", side_effect=err):
        with pytest.raises(OllamaEmbeddingError, match="HTTP 500"):
            ef(["a"])
    # _supports_batch stays None because we never confirmed either way.
    assert ef._supports_batch is None


def test_single_endpoint_empty_vector_raises():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    ef._supports_batch = False  # skip batch path
    with patch(
        "mempalace.embedding_ollama.urlopen",
        return_value=_mock_response({"embedding": []}),
    ):
        with pytest.raises(OllamaEmbeddingError, match="no vector"):
            ef(["hello"])


# ── network errors ──────────────────────────────────────────────────────


def test_url_error_wraps_as_ollama_error():
    from urllib.error import URLError

    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    with patch(
        "mempalace.embedding_ollama.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(OllamaEmbeddingError, match="Cannot reach"):
            ef(["a"])


# ── check_available probe ──────────────────────────────────────────────


def test_check_available_ok_when_model_present():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    payload = {"models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:14b"}]}
    with patch(
        "mempalace.embedding_ollama.urlopen",
        return_value=_mock_response(payload),
    ):
        ok, msg = ef.check_available()
    assert ok is True
    assert msg == "ok"


def test_check_available_reports_missing_model():
    ef = OllamaEmbeddingFunction(model="nomic-embed-text", endpoint="http://x")
    with patch(
        "mempalace.embedding_ollama.urlopen",
        return_value=_mock_response({"models": [{"name": "qwen2.5:14b"}]}),
    ):
        ok, msg = ef.check_available()
    assert ok is False
    assert "ollama pull nomic-embed-text" in msg
