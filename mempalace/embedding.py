"""Embedding function factory with hardware acceleration.

Returns a ChromaDB-compatible embedding function. Providers are selected by
``MempalaceConfig.embedding_provider``; within the local ``onnx`` provider,
``MempalaceConfig.embedding_model`` picks the model — ``minilm``
(all-MiniLM-L6-v2, English-only, ChromaDB's default) or ``embeddinggemma``
(onnx-community/embeddinggemma-300m-ONNX q8, MRL→384-dim, multilingual;
cross-lingual cosine ~0.88 vs MiniLM's ~0.35, lazy-downloaded on first use):

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


# Embeddinggemma-300m ONNX (q8) — 100+ languages, MRL-truncated to 384 dims so
# it drops into existing ChromaDB collections without a schema change. Lazy:
# the model (~300 MB) downloads on first call and is cached by huggingface_hub.
_EMBEDDINGGEMMA_REPO = "onnx-community/embeddinggemma-300m-ONNX"
_EMBEDDINGGEMMA_ONNX = "model_quantized.onnx"
_EMBEDDINGGEMMA_PREFIX = "task: sentence similarity | query: "
_EMBEDDINGGEMMA_DIM = 384  # Matryoshka truncation — first 384 dims of the 768
_EMBEDDINGGEMMA_MAX_LEN = 2048


class EmbeddinggemmaONNX:
    """ChromaDB-compatible EF using embeddinggemma-300m ONNX (q8, MRL→384d).

    Cross-lingual cosine similarity on parallel-translated text averages 0.88
    across DE/FR/HI/IT/KO/RU vs 0.35 for ``all-MiniLM-L6-v2``. Output dim is
    truncated to 384 via Matryoshka Representation Learning so the model is a
    drop-in replacement for the MiniLM-shaped 384-dim collections ChromaDB
    creates by default — same vector width, no schema change.

    Switching an existing palace from minilm → embeddinggemma still requires
    re-embedding (different vector space) — collections persist the EF name
    and ChromaDB rejects mismatched reads. Run ``mempalace repair rebuild-index``.
    """

    @staticmethod
    def name() -> str:
        # ChromaDB persists this on the collection and refuses reads with a
        # mismatched EF — that's the signal that forces users to rebuild_index
        # when switching models. Keep it stable.
        return "embeddinggemma_300m"

    def __init__(self, preferred_providers=None):
        self._providers = (
            list(preferred_providers) if preferred_providers else ["CPUExecutionProvider"]
        )
        self._session = None
        self._tokenizer = None
        self._np = None
        self._output_idx = None

    def _lazy_load(self) -> None:
        if self._session is not None:
            return
        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "EmbeddinggemmaONNX requires huggingface_hub, tokenizers, and "
                "numpy — these ship with mempalace core, so this error usually "
                "means one was uninstalled or pinned to an incompatible version. "
                "Reinstall with: pip install --upgrade --force-reinstall mempalace"
            ) from e

        logger.info(
            "Downloading %s/%s (cached after first run)…",
            _EMBEDDINGGEMMA_REPO,
            _EMBEDDINGGEMMA_ONNX,
        )
        model_path = hf_hub_download(
            _EMBEDDINGGEMMA_REPO, subfolder="onnx", filename=_EMBEDDINGGEMMA_ONNX
        )
        tok_path = hf_hub_download(_EMBEDDINGGEMMA_REPO, filename="tokenizer.json")

        self._session = ort.InferenceSession(model_path, providers=self._providers)
        out_names = [o.name for o in self._session.get_outputs()]
        # Model card: sentence_embedding is the pooled output (last_hidden_state
        # is the per-token output we don't want).
        self._output_idx = (
            out_names.index("sentence_embedding") if "sentence_embedding" in out_names else 1
        )

        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.enable_padding()
        tokenizer.enable_truncation(max_length=_EMBEDDINGGEMMA_MAX_LEN)
        self._tokenizer = tokenizer
        self._np = np

    def __call__(self, input):  # noqa: A002 — ChromaDB EF protocol uses `input`
        self._lazy_load()
        np = self._np
        texts = [_EMBEDDINGGEMMA_PREFIX + t for t in input]
        encs = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        outputs = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        sent_emb = outputs[self._output_idx][:, :_EMBEDDINGGEMMA_DIM]
        # L2-normalize so cosine similarity == dot product (matches what the
        # MTEB methodology assumes; ChromaDB's distance is configured for it).
        norms = np.linalg.norm(sent_emb, axis=1, keepdims=True) + 1e-12
        return (sent_emb / norms).tolist()


def get_embedding_function(device: Optional[str] = None, model: Optional[str] = None):
    """Return a cached embedding function for the configured provider + model.

    Provider is read from :attr:`MempalaceConfig.embedding_provider`:

    * ``onnx`` — local ONNX EF on the requested ``device`` (``device=None``
      falls back to ``MempalaceConfig.embedding_device``). The model is chosen
      by ``model`` / :attr:`MempalaceConfig.embedding_model`: ``"minilm"``
      (all-MiniLM-L6-v2, English) or ``"embeddinggemma"`` (multilingual,
      MRL→384d). Unrecognized values fall back to minilm.
    * ``ollama`` — :class:`OllamaEmbeddingFunction` (``/api/embed`` shape) with
      model/endpoint from config; ``device``/``model`` args ignored.
    * ``openai-compat`` — :class:`OpenAICompatEmbeddingFunction`
      (``/v1/embeddings`` shape — vLLM, LM Studio, OpenAI, etc.) with
      model/endpoint from config; ``device``/``model`` args ignored.

    The returned function is cached so we only pay model-load cost once per
    process.
    """
    from .config import MempalaceConfig

    config = MempalaceConfig()
    provider = config.embedding_provider

    if provider == "ollama":
        remote_model = config.embedding_model
        endpoint = config.embedding_endpoint
        cache_key = ("ollama", endpoint, remote_model)
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached
        ef = _build_ollama_ef(model=remote_model, endpoint=endpoint)
        _EF_CACHE[cache_key] = ef
        logger.info(
            "Embedding function initialized (provider=ollama model=%s endpoint=%s)",
            remote_model,
            endpoint,
        )
        return ef

    if provider == "openai-compat":
        remote_model = config.embedding_model
        endpoint = config.embedding_endpoint
        cache_key = ("openai-compat", endpoint, remote_model)
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached
        ef = _build_openai_compat_ef(model=remote_model, endpoint=endpoint)
        _EF_CACHE[cache_key] = ef
        logger.info(
            "Embedding function initialized (provider=openai-compat model=%s endpoint=%s)",
            remote_model,
            endpoint,
        )
        return ef

    if provider != "onnx":
        if provider not in _WARNED:
            logger.warning("Unknown embedding_provider %r — falling back to onnx", provider)
            _WARNED.add(provider)

    if device is None:
        device = config.embedding_device
    if model is None:
        model = config.embedding_model
    # onnx model selection is case-insensitive; minilm is the back-compat default.
    model = (model or "minilm").strip().lower()

    providers, effective = _resolve_providers(device)
    cache_key = ("onnx", model, tuple(providers))
    cached = _EF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if model == "embeddinggemma":
        ef = EmbeddinggemmaONNX(preferred_providers=providers)
    else:
        # Default: minilm (or anything we don't recognize — back-compat win).
        ef_cls = _build_ef_class()
        ef = ef_cls(preferred_providers=providers)

    _EF_CACHE[cache_key] = ef
    logger.info(
        "Embedding function initialized (model=%s device=%s providers=%s)",
        model,
        effective,
        providers,
    )
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
