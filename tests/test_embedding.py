import pytest

import mempalace.embedding as embedding


@pytest.fixture(autouse=True)
def isolate_embedding_state(monkeypatch):
    monkeypatch.setattr(embedding, "_EF_CACHE", {})
    monkeypatch.setattr(embedding, "_WARNED", set())


def test_auto_picks_cuda(monkeypatch):
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert embedding._resolve_providers("auto") == (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda",
    )


def test_auto_falls_to_cpu(monkeypatch):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("auto") == (["CPUExecutionProvider"], "cpu")


def test_cuda_missing_warns_with_gpu_extra(monkeypatch, caplog):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("cuda") == (["CPUExecutionProvider"], "cpu")
    assert "mempalace[gpu]" in caplog.text


def test_coreml_missing_warns_with_coreml_extra(monkeypatch, caplog):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("coreml") == (["CPUExecutionProvider"], "cpu")
    assert "mempalace[coreml]" in caplog.text


def test_dml_missing_warns_with_dml_extra(monkeypatch, caplog):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("dml") == (["CPUExecutionProvider"], "cpu")
    assert "mempalace[dml]" in caplog.text


def test_unknown_device_warns_once(monkeypatch, caplog):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("bogus") == (["CPUExecutionProvider"], "cpu")
    assert embedding._resolve_providers("bogus") == (["CPUExecutionProvider"], "cpu")
    assert caplog.text.count("Unknown embedding_device") == 1


def test_onnxruntime_import_error_falls_back_to_cpu(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert embedding._resolve_providers("cuda") == (["CPUExecutionProvider"], "cpu")


def test_get_embedding_function_caches_by_resolved_provider_tuple(monkeypatch):
    class DummyEF:
        def __init__(self, preferred_providers, intra_op_num_threads=0):
            self.preferred_providers = preferred_providers

    monkeypatch.setattr(embedding, "_build_ef_class", lambda: DummyEF)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )

    first = embedding.get_embedding_function("cpu", "minilm")
    second = embedding.get_embedding_function("auto", "minilm")

    assert first is second
    assert first.preferred_providers == ["CPUExecutionProvider"]


def test_intra_op_session_options_caps_threads():
    so = embedding._intra_op_session_options(3)
    assert so is not None
    assert so.intra_op_num_threads == 3


def test_intra_op_session_options_uncapped_returns_none():
    assert embedding._intra_op_session_options(0) is None
    assert embedding._intra_op_session_options(-1) is None


def test_get_embedding_function_threads_cap_passed_to_minilm_ef(monkeypatch):
    captured = {}

    class DummyEF:
        def __init__(self, preferred_providers, intra_op_num_threads=0):
            captured["threads"] = intra_op_num_threads

    monkeypatch.setattr(embedding, "_build_ef_class", lambda: DummyEF)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )
    monkeypatch.setattr(embedding, "_resolve_intra_op_threads", lambda: 2)

    embedding.get_embedding_function("cpu", "minilm")

    assert captured["threads"] == 2


def test_get_embedding_function_threads_cap_passed_to_embeddinggemma(monkeypatch):
    captured = {}

    class DummyGemma:
        def __init__(self, preferred_providers=None, intra_op_num_threads=0):
            captured["threads"] = intra_op_num_threads

    monkeypatch.setattr(embedding, "EmbeddinggemmaONNX", DummyGemma)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )
    monkeypatch.setattr(embedding, "_resolve_intra_op_threads", lambda: 4)

    embedding.get_embedding_function("cpu", "embeddinggemma")

    assert captured["threads"] == 4


def test_minilm_ef_model_override_applies_thread_cap(monkeypatch):
    """The ``_MempalaceONNX.model`` override must construct the ORT session
    with the configured ``intra_op_num_threads`` (#1068). We stub
    ``InferenceSession`` to capture the ``SessionOptions`` it receives, so the
    test never downloads or loads the real model."""
    import onnxruntime as ort

    captured = {}

    def fake_session(model_path, providers=None, sess_options=None):
        captured["sess_options"] = sess_options
        captured["providers"] = providers
        return object()

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    ef_cls = embedding._build_ef_class()
    ef = ef_cls(preferred_providers=["CPUExecutionProvider"], intra_op_num_threads=2)
    _ = ef.model  # triggers the cached_property build

    assert captured["sess_options"] is not None
    assert captured["sess_options"].intra_op_num_threads == 2
    assert "CoreMLExecutionProvider" not in captured["providers"]


def test_minilm_ef_model_override_falls_back_when_uncapped(monkeypatch):
    """With no cap (0), the override must defer to the parent build via
    ``super().model`` — not reach into ``cached_property`` internals (#1068
    review). Proves super() resolves the parent descriptor without error."""
    import onnxruntime as ort

    captured = {}

    def fake_session(model_path, providers=None, sess_options=None):
        captured["sess_options"] = sess_options
        return object()

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    ef_cls = embedding._build_ef_class()
    ef = ef_cls(preferred_providers=["CPUExecutionProvider"], intra_op_num_threads=0)
    session = ef.model  # cap <= 0 → super().model (upstream builder)

    assert session is not None
    # Upstream leaves intra_op at ORT's default (0 = unset), confirming we
    # deferred to it rather than applying our cap.
    assert captured["sess_options"].intra_op_num_threads == 0


def test_describe_device_uses_resolved_effective_device(monkeypatch):
    monkeypatch.setattr(
        embedding,
        "_resolve_providers",
        lambda device: (["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"),
    )

    assert embedding.describe_device("auto") == "cuda"


# ── provider routing ─────────────────────────────────────────────────────


class _StubConfig:
    """Stand-in for MempalaceConfig used in routing tests. Holds whatever
    attributes the test sets; embedding.py reads attributes by name."""

    def __init__(self, **kwargs):
        self.embedding_provider = kwargs.get("embedding_provider", "onnx")
        self.embedding_device = kwargs.get("embedding_device", "auto")
        self.embedding_model = kwargs.get("embedding_model", "")
        self.embedding_endpoint = kwargs.get("embedding_endpoint", "")


def _patch_config(monkeypatch, **kwargs):
    """Make ``from .config import MempalaceConfig`` inside embedding.py
    return our stub."""
    import mempalace.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "MempalaceConfig", lambda: _StubConfig(**kwargs))


def test_get_embedding_function_routes_to_ollama(monkeypatch):
    _patch_config(
        monkeypatch,
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        embedding_endpoint="http://192.168.1.147:11434",
    )

    captured = {}

    def fake_build(model, endpoint):
        captured["model"] = model
        captured["endpoint"] = endpoint
        return object()  # opaque sentinel — real EF not constructed

    monkeypatch.setattr(embedding, "_build_ollama_ef", fake_build)

    ef = embedding.get_embedding_function()

    assert captured == {"model": "nomic-embed-text", "endpoint": "http://192.168.1.147:11434"}
    # Cached: second call returns the same sentinel without re-invoking _build.
    monkeypatch.setattr(
        embedding,
        "_build_ollama_ef",
        lambda *a, **kw: pytest.fail("should not rebuild"),
    )
    assert embedding.get_embedding_function() is ef


def test_get_embedding_function_falls_back_to_onnx_for_unknown_provider(monkeypatch, caplog):
    _patch_config(monkeypatch, embedding_provider="bogus")

    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )

    class DummyEF:
        def __init__(self, preferred_providers, intra_op_num_threads=0):
            self.preferred_providers = preferred_providers
            self.intra_op_num_threads = intra_op_num_threads

    monkeypatch.setattr(embedding, "_build_ef_class", lambda: DummyEF)

    ef = embedding.get_embedding_function()
    assert isinstance(ef, DummyEF)
    assert "Unknown embedding_provider" in caplog.text


def test_get_embedding_function_routes_to_openai_compat(monkeypatch):
    _patch_config(
        monkeypatch,
        embedding_provider="openai-compat",
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_endpoint="http://192.168.1.147:8000",
    )

    captured = {}

    def fake_build(model, endpoint):
        captured["model"] = model
        captured["endpoint"] = endpoint
        return object()

    monkeypatch.setattr(embedding, "_build_openai_compat_ef", fake_build)

    ef = embedding.get_embedding_function()
    assert captured == {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "endpoint": "http://192.168.1.147:8000",
    }
    monkeypatch.setattr(
        embedding,
        "_build_openai_compat_ef",
        lambda *a, **kw: pytest.fail("should not rebuild"),
    )
    assert embedding.get_embedding_function() is ef


def test_describe_device_for_openai_compat_provider(monkeypatch):
    _patch_config(
        monkeypatch,
        embedding_provider="openai-compat",
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_endpoint="http://spark:8000",
    )
    assert (
        embedding.describe_device()
        == "openai-compat:nomic-ai/nomic-embed-text-v1.5 @ http://spark:8000"
    )


def test_describe_device_for_ollama_provider(monkeypatch):
    _patch_config(
        monkeypatch,
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        embedding_endpoint="http://spark:11434",
    )

    assert embedding.describe_device() == "ollama:nomic-embed-text @ http://spark:11434"


def test_describe_device_for_onnx_provider_uses_resolved_device(monkeypatch):
    _patch_config(monkeypatch, embedding_provider="onnx", embedding_device="cuda")
    monkeypatch.setattr(
        embedding,
        "_resolve_providers",
        lambda device: (["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"),
    )

    assert embedding.describe_device() == "cuda"
