"""HTTP keep-alive path tests for embedding clients.

The production embedding clients (``embedding_openai.py``,
``embedding_ollama.py``) default to ``urllib3.PoolManager`` so the
parallel miner can hold many concurrent connections without serializing
on TCP setup. The legacy ``urlopen`` path is preserved behind
``MEMPALACE_HTTP_KEEPALIVE=0``.

These tests exercise the pool path explicitly. They opt out of the
session-wide ``MEMPALACE_HTTP_KEEPALIVE=0`` set by ``conftest.py`` and
patch ``urllib3.PoolManager.request`` instead of ``urlopen``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mempalace import embedding_ollama, embedding_openai


@pytest.fixture
def keepalive_on(monkeypatch):
    """Force the pool path. Reset the module-cached pool too."""
    monkeypatch.setenv("MEMPALACE_HTTP_KEEPALIVE", "1")
    monkeypatch.setattr(embedding_openai, "_HTTP_POOL", None)
    monkeypatch.setattr(embedding_ollama, "_HTTP_POOL", None)


def _fake_response(status: int, payload: dict | bytes, reason: str = "OK"):
    resp = MagicMock()
    resp.status = status
    resp.reason = reason
    if isinstance(payload, bytes):
        resp.data = payload
    else:
        resp.data = json.dumps(payload).encode("utf-8")
    return resp


# ── OpenAI-compat pool path ──────────────────────────────────────────────


def test_openai_pool_reuses_single_manager_across_calls(keepalive_on):
    """The pool manager is cached at module scope and reused per call."""
    embedding_openai._HTTP_POOL = None
    pool_a = embedding_openai._get_http_pool()
    pool_b = embedding_openai._get_http_pool()
    assert pool_a is pool_b


def test_openai_pool_max_size_honors_workers_env(monkeypatch):
    """maxsize is sized for the configured worker count.

    A producer pool of N must not bottleneck on connection acquisition.
    """
    monkeypatch.setenv("MEMPALACE_WORKERS", "20")
    monkeypatch.setattr(embedding_openai, "_HTTP_POOL", None)
    pool = embedding_openai._get_http_pool()
    # urllib3 PoolManager doesn't expose maxsize on the manager itself; we
    # check the connection_pool_kw which holds the per-host args.
    assert pool.connection_pool_kw["maxsize"] >= 40  # workers*2


def test_openai_post_via_pool_translates_max_retry_error(keepalive_on):
    """urllib3.MaxRetryError → OpenAIEmbeddingError, preserving the URL."""
    import urllib3

    def boom(*args, **kwargs):
        raise urllib3.exceptions.MaxRetryError(None, "http://spark:8000/v1/embeddings", "no host")

    with patch.object(embedding_openai._get_http_pool(), "request", side_effect=boom):
        with pytest.raises(embedding_openai.OpenAIEmbeddingError) as exc_info:
            embedding_openai._post_json(
                "http://spark:8000/v1/embeddings",
                {"input": ["hi"], "model": "x"},
                headers={},
                timeout=30,
            )
        assert "Cannot reach" in str(exc_info.value)


def test_openai_post_via_pool_translates_4xx_to_embedding_error(keepalive_on):
    """A 4xx response body is included in the error message."""

    def fake_request(*args, **kwargs):
        return _fake_response(400, b"bad input", reason="Bad Request")

    with patch.object(embedding_openai._get_http_pool(), "request", side_effect=fake_request):
        with pytest.raises(embedding_openai.OpenAIEmbeddingError) as exc_info:
            embedding_openai._post_json(
                "http://spark:8000/v1/embeddings",
                {"input": ["hi"], "model": "x"},
                headers={},
                timeout=30,
            )
        assert "HTTP 400" in str(exc_info.value)
        assert "bad input" in str(exc_info.value)


def test_openai_post_via_pool_returns_decoded_json(keepalive_on):
    """Happy path: the pool response is decoded the same as urlopen."""

    def fake_request(*args, **kwargs):
        return _fake_response(
            200,
            {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}], "model": "x"},
        )

    with patch.object(embedding_openai._get_http_pool(), "request", side_effect=fake_request):
        result = embedding_openai._post_json(
            "http://spark:8000/v1/embeddings",
            {"input": ["hi"], "model": "x"},
            headers={"Authorization": "Bearer test"},
            timeout=30,
        )
        assert result["model"] == "x"
        assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_openai_legacy_urlopen_path_when_keepalive_disabled(monkeypatch):
    """MEMPALACE_HTTP_KEEPALIVE=0 routes through urlopen (existing-test shape)."""
    monkeypatch.setenv("MEMPALACE_HTTP_KEEPALIVE", "0")

    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps({"ok": True}).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False

    def fake_urlopen(req, timeout=None):
        # Confirm we're on the urlopen path (the request object differs).
        from urllib.request import Request

        assert isinstance(req, Request)
        return fake_resp

    with patch("mempalace.embedding_openai.urlopen", side_effect=fake_urlopen):
        result = embedding_openai._post_json(
            "http://spark:8000/v1/embeddings",
            {"input": ["hi"]},
            headers={},
            timeout=30,
        )
        assert result == {"ok": True}


# ── Ollama pool path ──────────────────────────────────────────────────────


def test_ollama_pool_reuses_single_manager_across_calls(keepalive_on):
    embedding_ollama._HTTP_POOL = None
    pool_a = embedding_ollama._get_http_pool()
    pool_b = embedding_ollama._get_http_pool()
    assert pool_a is pool_b


def test_ollama_post_via_pool_translates_max_retry_error(keepalive_on):
    import urllib3

    def boom(*args, **kwargs):
        raise urllib3.exceptions.MaxRetryError(None, "http://spark:11434/api/embed", "no host")

    with patch.object(embedding_ollama._get_http_pool(), "request", side_effect=boom):
        with pytest.raises(embedding_ollama.OllamaEmbeddingError) as exc_info:
            embedding_ollama._post_json(
                "http://spark:11434/api/embed", {"input": "hi", "model": "x"}, timeout=30
            )
        assert "Cannot reach" in str(exc_info.value)


def test_ollama_legacy_urlopen_path_when_keepalive_disabled(monkeypatch):
    monkeypatch.setenv("MEMPALACE_HTTP_KEEPALIVE", "0")

    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps({"embeddings": [[0.0, 0.0]]}).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False

    with patch(
        "mempalace.embedding_ollama.urlopen",
        return_value=fake_resp,
    ):
        result = embedding_ollama._post_json(
            "http://spark:11434/api/embed", {"input": "hi", "model": "x"}, timeout=30
        )
        assert "embeddings" in result
