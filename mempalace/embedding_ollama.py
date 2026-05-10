"""Ollama-backed embedding function for ChromaDB.

Why this exists: the default ONNX path (``mempalace/embedding.py``) runs
``all-MiniLM-L6-v2`` on the CPU/GPU of the machine that runs MemPalace. For
users with a separate GPU box (DGX Spark, dedicated workstation) it is
useful to push embedding to a remote Ollama instance so the local box stays
free and a stronger model (e.g. ``nomic-embed-text`` at 768-dim) can be
used without installing CUDA wheels locally.

The class implements ChromaDB's ``EmbeddingFunction`` protocol: ``__call__``,
``name``, ``get_config``, and ``build_from_config``. ``name()`` returns
``"ollama:{model}"`` rather than spoofing ``"default"`` (the trick we use
for the ONNX path) — Ollama's vectors have a different dimension than
MiniLM, so a silent identity collision would produce a corrupt collection.
A noisy identity-mismatch error at read time is the safer failure mode.

Network calls use stdlib ``urllib`` only — no new dependency on requests.
The implementation prefers the modern ``/api/embed`` batch endpoint and
falls back to the legacy ``/api/embeddings`` single-input endpoint if the
remote returns 404 (older Ollama versions).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OllamaEmbeddingError(RuntimeError):
    """Raised when the Ollama embedding endpoint is unreachable, returns an
    error, or returns a malformed response. Surfaces as a hard failure so
    a misconfigured endpoint does not silently produce empty vectors."""


# ── HTTP transport ────────────────────────────────────────────────────────
#
# See ``mempalace.embedding_openai`` for the rationale on the dual transport
# (urllib3.PoolManager keep-alive default, urlopen behind a temporary env
# shim for test compatibility). Same shape, separate pool so callers can
# patch one without touching the other.

_HTTP_POOL = None  # urllib3.PoolManager — lazy-init, module-cached


def _get_http_pool():
    global _HTTP_POOL
    if _HTTP_POOL is None:
        import urllib3

        try:
            workers = max(1, int(os.environ.get("MEMPALACE_WORKERS", "8")))
        except ValueError:
            workers = 8
        _HTTP_POOL = urllib3.PoolManager(
            num_pools=4,
            maxsize=max(32, workers * 2),
            block=False,
        )
    return _HTTP_POOL


def _post_json_via_urlopen(url: str, body: dict, timeout: int) -> dict:
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise OllamaEmbeddingError(f"HTTP {e.code} from {url}: {detail or e.reason}") from e
    except (URLError, OSError) as e:
        raise OllamaEmbeddingError(f"Cannot reach {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise OllamaEmbeddingError(f"Malformed response from {url}: {e}") from e


def _post_json_via_pool(url: str, body: dict, timeout: int) -> dict:
    import urllib3

    pool = _get_http_pool()
    encoded = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        resp = pool.request(
            "POST",
            url,
            body=encoded,
            headers=headers,
            timeout=timeout,
            retries=False,
        )
    except urllib3.exceptions.MaxRetryError as e:
        raise OllamaEmbeddingError(f"Cannot reach {url}: {e}") from e
    except urllib3.exceptions.TimeoutError as e:
        raise OllamaEmbeddingError(f"Timeout reaching {url}: {e}") from e
    except urllib3.exceptions.HTTPError as e:
        raise OllamaEmbeddingError(f"Cannot reach {url}: {e}") from e
    except OSError as e:
        raise OllamaEmbeddingError(f"Cannot reach {url}: {e}") from e

    if resp.status >= 400:
        detail = ""
        try:
            detail = resp.data.decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise OllamaEmbeddingError(
            f"HTTP {resp.status} from {url}: {detail or resp.reason or 'no detail'}"
        )

    try:
        return json.loads(resp.data)
    except json.JSONDecodeError as e:
        raise OllamaEmbeddingError(f"Malformed response from {url}: {e}") from e


def _post_json(url: str, body: dict, timeout: int) -> dict:
    """Dispatch keep-alive pool (default) or urlopen (MEMPALACE_HTTP_KEEPALIVE=0)."""
    if os.environ.get("MEMPALACE_HTTP_KEEPALIVE", "1") == "0":
        return _post_json_via_urlopen(url, body, timeout)
    return _post_json_via_pool(url, body, timeout)


def _build_base_class():
    """Return the ChromaDB ``EmbeddingFunction`` base class lazily.

    Defined as a function (not an import at module load) so importing
    ``mempalace.embedding_ollama`` does not require chromadb to be present
    — useful for tests that monkey-patch the EF without touching the real
    ChromaDB install.
    """
    from chromadb.utils.embedding_functions import EmbeddingFunction

    return EmbeddingFunction


class OllamaEmbeddingFunction:
    """ChromaDB embedding function that delegates to an Ollama HTTP endpoint.

    Constructed with ``model`` and ``endpoint``. The class is intentionally
    *not* a subclass of ``EmbeddingFunction`` at definition time — chromadb's
    metaclass wraps ``__call__`` with a validator that calls
    ``validate_embeddings``, which we honor by returning a plain ``list`` of
    ``list[float]`` from ``__call__`` regardless of input type. We register
    as the protocol via ``isinstance``-style structural typing.
    """

    def __init__(
        self,
        model: str,
        endpoint: str,
        timeout: int = 120,
    ):
        if not model:
            raise ValueError("OllamaEmbeddingFunction requires a non-empty model name")
        if not endpoint:
            raise ValueError("OllamaEmbeddingFunction requires a non-empty endpoint URL")
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        # Set on first successful batch call. ``None`` means "not yet
        # discovered"; once flipped to False the class never tries
        # ``/api/embed`` again for this instance, avoiding a 404 round-trip
        # per batch on older Ollama servers.
        self._supports_batch: Optional[bool] = None

    @staticmethod
    def name() -> str:
        # Static name — chromadb persists this on the collection. Including
        # the model would let chromadb detect a model swap, but the
        # protocol requires ``name()`` to be a ``@staticmethod`` with no
        # access to instance state. We use a stable provider tag and rely
        # on dimension-mismatch errors at insert/query time to catch a
        # model swap. (See module docstring.)
        return "ollama"

    def get_config(self) -> dict:
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "timeout": self.timeout,
        }

    @staticmethod
    def build_from_config(config: dict) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(
            model=config["model"],
            endpoint=config["endpoint"],
            timeout=int(config.get("timeout", 120)),
        )

    @staticmethod
    def validate_config(config: dict) -> None:
        if not config.get("model"):
            raise ValueError("ollama embedding config missing 'model'")
        if not config.get("endpoint"):
            raise ValueError("ollama embedding config missing 'endpoint'")

    def validate_config_update(self, old_config: dict, new_config: dict) -> None:
        if old_config.get("model") != new_config.get("model"):
            raise ValueError(
                "Cannot change Ollama embedding model on an existing collection — "
                "vector dimensions would not match. Wipe and re-mine."
            )

    # ------------------------------------------------------------------
    # The work: take a list of strings, return a list of float lists.
    # ChromaDB's metaclass post-processes the return value via
    # ``validate_embeddings`` + ``normalize_embeddings``, so we keep the
    # output shape simple and let chromadb coerce.
    # ------------------------------------------------------------------

    def __call__(self, input):
        # chromadb passes ``Documents`` (list of strings). Defensive: also
        # accept a single string for parity with some test harnesses.
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        if not texts:
            return []

        if self._supports_batch is not False:
            try:
                vectors = self._embed_batch(texts)
                self._supports_batch = True
                return vectors
            except OllamaEmbeddingError as e:
                # 404 → older Ollama without /api/embed. Fall through to
                # the per-input loop and remember the result for next time.
                if "HTTP 404" in str(e):
                    logger.info(
                        "Ollama %s does not support /api/embed — falling back "
                        "to /api/embeddings (one request per input).",
                        self.endpoint,
                    )
                    self._supports_batch = False
                else:
                    raise

        return [self._embed_single(text) for text in texts]

    def embed_query(self, input):
        """Embed a query input. ChromaDB's search path calls this rather
        than ``__call__``; the protocol's base class provides a default
        passthrough only when you subclass ``EmbeddingFunction``, which
        we deliberately do not (the metaclass wraps ``__call__`` with
        validators we want to skip). Provide the passthrough explicitly."""
        return self.__call__(input)

    def embed_documents(self, input):
        """Symmetric counterpart to :meth:`embed_query` — some chromadb
        code paths call this when ingesting documents. Same passthrough."""
        return self.__call__(input)

    def _embed_batch(self, texts: list) -> list:
        body = {"model": self.model, "input": texts}
        data = _post_json(f"{self.endpoint}/api/embed", body, self.timeout)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaEmbeddingError(
                f"Ollama /api/embed returned {len(embeddings) if isinstance(embeddings, list) else '?'} "
                f"vectors for {len(texts)} inputs (model={self.model})"
            )
        return embeddings

    def _embed_single(self, text: str) -> list:
        body = {"model": self.model, "prompt": text}
        data = _post_json(f"{self.endpoint}/api/embeddings", body, self.timeout)
        vector = data.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise OllamaEmbeddingError(
                f"Ollama /api/embeddings returned no vector (model={self.model})"
            )
        return vector

    # ------------------------------------------------------------------
    # Probe used by the miner header / CLI sanity check.
    # ------------------------------------------------------------------

    def check_available(self) -> tuple[bool, str]:
        """Return ``(ok, message)``. Fast probe that ``model`` is loaded
        on the configured Ollama endpoint."""
        try:
            with urlopen(f"{self.endpoint}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
        except (URLError, HTTPError, OSError, json.JSONDecodeError) as e:
            return False, f"Cannot reach Ollama at {self.endpoint}: {e}"
        names = {m.get("name", "") for m in data.get("models", []) or []}
        wanted = {self.model, f"{self.model}:latest"}
        if not names & wanted:
            return (
                False,
                f"Model '{self.model}' not loaded in Ollama. Run: ollama pull {self.model}",
            )
        return True, "ok"


def register_with_chromadb() -> None:
    """Register :class:`OllamaEmbeddingFunction` in ChromaDB's known-EF
    registry so that ``build_from_config`` can rehydrate it from
    persisted collection metadata.

    Best-effort: chromadb's registry API has shifted between versions, so
    we tolerate ImportError/AttributeError silently. Without registration,
    palaces created with this EF still work in-process — they only fail
    when a different process tries to open them and asks chromadb to
    rehydrate the EF by name.
    """
    try:
        from chromadb.utils.embedding_functions import (
            register_embedding_function,
        )
    except ImportError:
        return
    try:
        register_embedding_function(OllamaEmbeddingFunction)
    except (TypeError, AttributeError, ValueError) as e:
        logger.debug("Could not register OllamaEmbeddingFunction: %s", e)


__all__ = [
    "OllamaEmbeddingFunction",
    "OllamaEmbeddingError",
    "register_with_chromadb",
]
