"""Embedding function factory with hardware acceleration.

Returns a ChromaDB-compatible embedding function. Two providers are
supported, selected by ``MempalaceConfig.embedding_provider``:

* ``onnx`` (default) — local ``all-MiniLM-L6-v2`` via ONNX Runtime, 384-dim,
  matches ChromaDB's library default so palaces created with the default EF
  remain readable. Hardware device selected by
  :attr:`MempalaceConfig.embedding_device` (auto/cpu/cuda/coreml/dml).
* ``ollama`` — remote (or local) Ollama HTTP endpoint, model and dimension
  chosen by :attr:`MempalaceConfig.embedding_model`. Useful for offloading
  embedding work to a separate GPU box. See ``embedding_ollama.py``.

Switching providers (or models within the Ollama provider) requires wiping
and re-mining the palace: vectors at one dimension cannot be queried by an
EF that produces a different dimension.

Supported ``embedding_device`` values for the ONNX provider (env
``MEMPALACE_EMBEDDING_DEVICE`` or config key):

* ``auto`` — prefer CUDA ▸ CoreML ▸ DirectML, fall back to CPU
* ``cpu`` — force CPU (the historical default)
* ``cuda`` — NVIDIA GPU via ``onnxruntime-gpu`` (``pip install mempalace[gpu]``)
* ``coreml`` — Apple Neural Engine (macOS)
* ``dml`` — DirectML (Windows / AMD / Intel GPUs)

Requesting an unavailable accelerator emits a warning and falls back to CPU
rather than hard-failing — mining must still work on a laptop without CUDA.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}

_DEVICE_EXTRA = {
    "cuda": "mempalace[gpu]",
    "coreml": "mempalace[coreml]",
    "dml": "mempalace[dml]",
}

_AUTO_ORDER = [
    ("CUDAExecutionProvider", "cuda"),
    ("CoreMLExecutionProvider", "coreml"),
    ("DmlExecutionProvider", "dml"),
]

_EF_CACHE: dict = {}
_WARNED: set = set()


def _resolve_providers(device: str) -> tuple[list, str]:
    """Return ``(provider_list, effective_device)`` for ``device``.

    Falls back to CPU (with a one-shot warning) when the requested
    accelerator is not compiled into the installed ``onnxruntime``.
    """
    device = (device or "auto").strip().lower()

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return (["CPUExecutionProvider"], "cpu")

    if device == "auto":
        for provider, name in _AUTO_ORDER:
            if provider in available:
                return ([provider, "CPUExecutionProvider"], name)
        return (["CPUExecutionProvider"], "cpu")

    requested = _PROVIDER_MAP.get(device)
    if requested is None:
        if device not in _WARNED:
            logger.warning("Unknown embedding_device %r — falling back to cpu", device)
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    preferred = requested[0]
    if preferred == "CPUExecutionProvider":
        return (requested, "cpu")

    if preferred not in available:
        if device not in _WARNED:
            extra = _DEVICE_EXTRA.get(device, "the matching mempalace extra for your device")
            logger.warning(
                "embedding_device=%r requested but %s is not installed — "
                "falling back to CPU. Install %s.",
                device,
                preferred,
                extra,
            )
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    return (requested, device)


def _build_ef_class():
    """Subclass ``ONNXMiniLM_L6_V2`` with name ``"default"``.

    Why the rename: ChromaDB 1.5 persists the EF identity on the collection
    and rejects reads that pass a differently-named EF (``onnx_mini_lm_l6_v2``
    vs ``default``). The vectors and model are identical — only the
    ``name()`` tag differs — so spoofing the name lets one EF class serve
    palaces created with ``DefaultEmbeddingFunction`` *and* palaces we
    create ourselves, with the same GPU-capable ``preferred_providers``.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _MempalaceONNX(ONNXMiniLM_L6_V2):
        @staticmethod
        def name() -> str:
            return "default"

    return _MempalaceONNX


def _build_ollama_ef(model: str, endpoint: str):
    """Construct an :class:`OllamaEmbeddingFunction`. Lazy import keeps
    ``mempalace.embedding`` loadable on machines that do not have the
    Ollama module installed (it is in-tree, but the import still pulls
    chromadb which we want to keep optional at import time)."""
    from .embedding_ollama import OllamaEmbeddingFunction

    return OllamaEmbeddingFunction(model=model, endpoint=endpoint)


def _build_openai_compat_ef(model: str, endpoint: str):
    """Construct an :class:`OpenAICompatEmbeddingFunction`. Same lazy-import
    pattern as the Ollama builder."""
    from .embedding_openai import OpenAICompatEmbeddingFunction

    return OpenAICompatEmbeddingFunction(model=model, endpoint=endpoint)


def get_embedding_function(device: Optional[str] = None):
    """Return a cached embedding function for the configured provider.

    Provider is read from :attr:`MempalaceConfig.embedding_provider`:

    * ``onnx`` — local ONNX MiniLM EF on the requested ``device``
      (``device=None`` falls back to ``MempalaceConfig.embedding_device``).
    * ``ollama`` — :class:`OllamaEmbeddingFunction` (``/api/embed`` shape)
      with model/endpoint from config; the ``device`` parameter is ignored.
    * ``openai-compat`` — :class:`OpenAICompatEmbeddingFunction`
      (``/v1/embeddings`` shape — vLLM, LM Studio, OpenAI, etc.) with
      model/endpoint from config; ``device`` ignored.

    The returned function is cached per (provider, key) so we only pay
    model-load cost once per process.
    """
    from .config import MempalaceConfig

    config = MempalaceConfig()
    provider = config.embedding_provider

    if provider == "ollama":
        model = config.embedding_model
        endpoint = config.embedding_endpoint
        cache_key = ("ollama", endpoint, model)
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached
        ef = _build_ollama_ef(model=model, endpoint=endpoint)
        _EF_CACHE[cache_key] = ef
        logger.info(
            "Embedding function initialized (provider=ollama model=%s endpoint=%s)",
            model,
            endpoint,
        )
        return ef

    if provider == "openai-compat":
        model = config.embedding_model
        endpoint = config.embedding_endpoint
        cache_key = ("openai-compat", endpoint, model)
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached
        ef = _build_openai_compat_ef(model=model, endpoint=endpoint)
        _EF_CACHE[cache_key] = ef
        logger.info(
            "Embedding function initialized (provider=openai-compat model=%s endpoint=%s)",
            model,
            endpoint,
        )
        return ef

    if provider != "onnx":
        if provider not in _WARNED:
            logger.warning("Unknown embedding_provider %r — falling back to onnx", provider)
            _WARNED.add(provider)

    if device is None:
        device = config.embedding_device

    providers, effective = _resolve_providers(device)
    cache_key = ("onnx", tuple(providers))
    cached = _EF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ef_cls = _build_ef_class()
    ef = ef_cls(preferred_providers=providers)
    _EF_CACHE[cache_key] = ef
    logger.info("Embedding function initialized (device=%s providers=%s)", effective, providers)
    return ef


def describe_device(device: Optional[str] = None) -> str:
    """Return a short human-readable label for the active embedding setup.

    Used by the miner CLI header so users can see at a glance which
    provider/device is doing the work.

    Format:
      * onnx provider          → ``"cpu"`` / ``"cuda"`` / ``"coreml"`` / ``"dml"``
      * ollama provider        → ``"ollama:{model} @ {endpoint}"``
      * openai-compat provider → ``"openai-compat:{model} @ {endpoint}"``
    """
    from .config import MempalaceConfig

    config = MempalaceConfig()
    if config.embedding_provider == "ollama":
        return f"ollama:{config.embedding_model} @ {config.embedding_endpoint}"
    if config.embedding_provider == "openai-compat":
        return f"openai-compat:{config.embedding_model} @ {config.embedding_endpoint}"

    if device is None:
        device = config.embedding_device
    _, effective = _resolve_providers(device)
    return effective
