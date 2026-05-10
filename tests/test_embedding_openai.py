"""Tests for mempalace.embedding_openai (OpenAI-compatible EF, used for vLLM
and other /v1/embeddings servers).

HTTP is mocked throughout — no live server required.
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from mempalace.embedding_openai import (
    OpenAICompatEmbeddingFunction,
    OpenAIEmbeddingError,
)


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _data_payload(vectors):
    """Build an OpenAI-spec /v1/embeddings response from a list of vectors."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


# ── construction ────────────────────────────────────────────────────────


def test_init_appends_v1_embeddings_when_endpoint_lacks_it():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x:8000/")
    assert ef._embeddings_url == "http://x:8000/v1/embeddings"


def test_init_keeps_v1_when_endpoint_already_has_it():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x:8000/v1")
    assert ef._embeddings_url == "http://x:8000/v1/embeddings"


def test_init_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        OpenAICompatEmbeddingFunction(model="", endpoint="http://x")


def test_init_rejects_empty_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        OpenAICompatEmbeddingFunction(model="m", endpoint="")


def test_init_picks_up_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    assert ef.api_key == "sk-from-env"


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x", api_key="sk-explicit")
    assert ef.api_key == "sk-explicit"


def test_name_is_distinct_from_ollama():
    # Important: chromadb persists this on the collection. A user who flips
    # provider from ollama → openai-compat must hit an EF identity error,
    # not silently get garbage.
    assert OpenAICompatEmbeddingFunction.name() == "openai-compat"


def test_get_config_excludes_api_key():
    ef = OpenAICompatEmbeddingFunction(
        model="nomic-ai/nomic-embed-text-v1.5",
        endpoint="http://spark:8000",
        api_key="sk-secret",
    )
    cfg = ef.get_config()
    assert "api_key" not in cfg
    assert cfg["model"] == "nomic-ai/nomic-embed-text-v1.5"


def test_validate_config_update_blocks_model_swap():
    ef = OpenAICompatEmbeddingFunction(model="m1", endpoint="http://x")
    with pytest.raises(ValueError, match="Cannot change embedding model"):
        ef.validate_config_update({"model": "m1"}, {"model": "m2"})


# ── happy path ──────────────────────────────────────────────────────────


def test_call_posts_to_v1_embeddings_with_correct_body():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x:8000")

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.header_items())
        return _mock_response(_data_payload([[0.1, 0.2], [0.3, 0.4]]))

    with patch("mempalace.embedding_openai.urlopen", side_effect=fake_urlopen):
        vectors = ef(["alpha", "beta"])

    assert captured["url"] == "http://x:8000/v1/embeddings"
    assert captured["body"] == {
        "model": "m",
        "input": ["alpha", "beta"],
        "encoding_format": "float",
    }
    # No Authorization header when no api_key.
    assert not any(k.lower() == "authorization" for k in captured["headers"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_call_sends_bearer_when_api_key_present():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x", api_key="sk-test")

    captured_headers = {}

    def fake_urlopen(req, timeout):
        captured_headers.update(dict(req.header_items()))
        return _mock_response(_data_payload([[0.1]]))

    with patch("mempalace.embedding_openai.urlopen", side_effect=fake_urlopen):
        ef(["x"])

    auth = next(v for k, v in captured_headers.items() if k.lower() == "authorization")
    assert auth == "Bearer sk-test"


def test_call_accepts_single_string():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    with patch(
        "mempalace.embedding_openai.urlopen",
        return_value=_mock_response(_data_payload([[1.0]])),
    ):
        v = ef("hello")
    assert v == [[1.0]]


def test_call_empty_input_returns_empty_list():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    with patch("mempalace.embedding_openai.urlopen") as mock_urlopen:
        assert ef([]) == []
    mock_urlopen.assert_not_called()


def test_embed_query_and_embed_documents_call_through():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    with patch(
        "mempalace.embedding_openai.urlopen",
        return_value=_mock_response(_data_payload([[0.1, 0.2]])),
    ):
        q = ef.embed_query(["q"])
        d = ef.embed_documents(["d"])
    assert q == [[0.1, 0.2]]
    assert d == [[0.1, 0.2]]


# ── defensive: server returns items out of order ────────────────────────


def test_response_items_are_sorted_by_index():
    """Per OpenAI spec, ``data`` entries can arrive in any order. We sort
    by ``index`` so the caller always gets vectors in submitted order.
    Trusting array order is the silent-corruption hazard."""
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    out_of_order = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.3, 0.3], "index": 2},
            {"object": "embedding", "embedding": [0.1, 0.1], "index": 0},
            {"object": "embedding", "embedding": [0.2, 0.2], "index": 1},
        ],
        "model": "m",
    }
    with patch(
        "mempalace.embedding_openai.urlopen",
        return_value=_mock_response(out_of_order),
    ):
        vectors = ef(["a", "b", "c"])
    assert vectors == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]


# ── error surfacing ─────────────────────────────────────────────────────


def test_response_count_mismatch_raises():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    with patch(
        "mempalace.embedding_openai.urlopen",
        return_value=_mock_response(_data_payload([[0.1]])),  # 1 vec for 2 inputs
    ):
        with pytest.raises(OpenAIEmbeddingError, match="returned 1 items for 2"):
            ef(["a", "b"])


def test_missing_index_field_raises():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    bad = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1]}],  # no index
    }
    with patch(
        "mempalace.embedding_openai.urlopen",
        return_value=_mock_response(bad),
    ):
        with pytest.raises(OpenAIEmbeddingError, match="missing 'index' field"):
            ef(["a"])


def test_http_error_wraps_as_openai_error():
    from urllib.error import HTTPError

    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    err = HTTPError("http://x/v1/embeddings", 500, "Server Error", {}, io.BytesIO(b"boom"))
    with patch("mempalace.embedding_openai.urlopen", side_effect=err):
        with pytest.raises(OpenAIEmbeddingError, match="HTTP 500"):
            ef(["a"])


def test_url_error_wraps_as_openai_error():
    from urllib.error import URLError

    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    with patch(
        "mempalace.embedding_openai.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(OpenAIEmbeddingError, match="Cannot reach"):
            ef(["a"])


# ── check_available probe ──────────────────────────────────────────────


def test_check_available_ok():
    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    with patch(
        "mempalace.embedding_openai.urlopen",
        return_value=_mock_response(_data_payload([[0.1]])),
    ):
        ok, msg = ef.check_available()
    assert ok is True


def test_check_available_reports_failure():
    from urllib.error import URLError

    ef = OpenAICompatEmbeddingFunction(model="m", endpoint="http://x")
    with patch(
        "mempalace.embedding_openai.urlopen",
        side_effect=URLError("nope"),
    ):
        ok, msg = ef.check_available()
    assert ok is False
    assert "nope" in msg
