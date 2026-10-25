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


# ── sub-batching (#57) ──────────────────────────────────────────────────


def _recording_urlopen(log):
    """Fake urlopen that echoes one vector per input, tagged by the input."""

    def fake(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        log.append(body["input"])
        return _mock_response({"embeddings": [[float(t)] for t in body["input"]]})

    return fake


def _no_batch_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_OLLAMA_BATCH_SIZE", raising=False)
    monkeypatch.setenv("MEMPALACE_CONFIG_DIR", str(tmp_path))


def test_splits_into_sub_batches_and_preserves_order(monkeypatch, tmp_path):
    _no_batch_env(monkeypatch, tmp_path)
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x", batch_size=64)
    log = []
    texts = [str(i) for i in range(150)]
    with patch("mempalace.embedding_ollama.urlopen", side_effect=_recording_urlopen(log)):
        vectors = ef(texts)
    assert [len(b) for b in log] == [64, 64, 22]
    assert vectors == [[float(i)] for i in range(150)]


def test_default_batch_size_is_64(monkeypatch, tmp_path):
    _no_batch_env(monkeypatch, tmp_path)
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x")
    assert ef.batch_size == 64
    log = []
    with patch("mempalace.embedding_ollama.urlopen", side_effect=_recording_urlopen(log)):
        ef([str(i) for i in range(130)])
    assert [len(b) for b in log] == [64, 64, 2]


def test_batch_size_larger_than_input_is_one_request(monkeypatch, tmp_path):
    _no_batch_env(monkeypatch, tmp_path)
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x", batch_size=1000)
    log = []
    with patch("mempalace.embedding_ollama.urlopen", side_effect=_recording_urlopen(log)):
        ef(["1", "2", "3"])
    assert [len(b) for b in log] == [3]


def test_env_override_sets_batch_size(monkeypatch, tmp_path):
    _no_batch_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMPALACE_OLLAMA_BATCH_SIZE", "10")
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x")
    assert ef.batch_size == 10
    log = []
    with patch("mempalace.embedding_ollama.urlopen", side_effect=_recording_urlopen(log)):
        ef([str(i) for i in range(25)])
    assert [len(b) for b in log] == [10, 10, 5]


def test_explicit_batch_size_beats_env(monkeypatch, tmp_path):
    _no_batch_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMPALACE_OLLAMA_BATCH_SIZE", "10")
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x", batch_size=5)
    assert ef.batch_size == 5


@pytest.mark.parametrize("bad", ["0", "-3", "abc", ""])
def test_invalid_env_falls_back_to_default(monkeypatch, tmp_path, bad):
    _no_batch_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMPALACE_OLLAMA_BATCH_SIZE", bad)
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x")
    assert ef.batch_size == 64


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_constructor_batch_size_rejected(bad):
    with pytest.raises(ValueError, match="batch_size"):
        OllamaEmbeddingFunction(model="m", endpoint="http://x", batch_size=bad)


def test_config_json_key_sets_batch_size(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_OLLAMA_BATCH_SIZE", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"ollama_batch_size": 7}))
    from mempalace.config import MempalaceConfig

    assert MempalaceConfig(config_dir=str(tmp_path)).ollama_batch_size == 7
    monkeypatch.setenv("MEMPALACE_OLLAMA_BATCH_SIZE", "3")
    assert MempalaceConfig(config_dir=str(tmp_path)).ollama_batch_size == 3
    monkeypatch.setenv("MEMPALACE_OLLAMA_BATCH_SIZE", "0")
    assert MempalaceConfig(config_dir=str(tmp_path)).ollama_batch_size == 64


def test_error_in_second_sub_batch_propagates(monkeypatch, tmp_path):
    _no_batch_env(monkeypatch, tmp_path)
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x", batch_size=2)
    calls = []

    def fake(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body["input"])
        if len(calls) == 2:
            return _mock_response({"embeddings": [[0.0]]})  # short: 1 for 2
        return _mock_response({"embeddings": [[0.0] for _ in body["input"]]})

    with patch("mempalace.embedding_ollama.urlopen", side_effect=fake):
        with pytest.raises(OllamaEmbeddingError, match="returned 1 vectors for 2"):
            ef(["a", "b", "c", "d", "e"])
    assert len(calls) == 2


def test_http_error_in_second_sub_batch_propagates(monkeypatch, tmp_path):
    from urllib.error import HTTPError

    _no_batch_env(monkeypatch, tmp_path)
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x", batch_size=2)
    calls = []

    def fake(req, timeout):
        calls.append(1)
        if len(calls) == 2:
            raise HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b"x"))
        return _mock_response({"embeddings": [[0.0], [0.0]]})

    with patch("mempalace.embedding_ollama.urlopen", side_effect=fake):
        with pytest.raises(OllamaEmbeddingError, match="HTTP 500"):
            ef(["a", "b", "c", "d"])


def test_404_on_first_sub_batch_falls_back_for_whole_call(monkeypatch, tmp_path):
    from urllib.error import HTTPError

    _no_batch_env(monkeypatch, tmp_path)
    ef = OllamaEmbeddingFunction(model="m", endpoint="http://x", batch_size=2)
    log = []

    def fake(req, timeout):
        log.append(req.full_url)
        if req.full_url.endswith("/api/embed"):
            raise HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))
        return _mock_response({"embedding": [0.5]})

    with patch("mempalace.embedding_ollama.urlopen", side_effect=fake):
        vectors = ef(["a", "b", "c"])
    assert log == ["http://x/api/embed"] + ["http://x/api/embeddings"] * 3
    assert vectors == [[0.5]] * 3
    assert ef._supports_batch is False
