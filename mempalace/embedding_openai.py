"""OpenAI-compatible embedding function for ChromaDB.

Why this exists alongside ``embedding_ollama.py``: vLLM, LM Studio,
llama.cpp's server, Together, Fireworks, and OpenAI itself all expose
the same ``/v1/embeddings`` shape. One generic adapter covers them all.

We keep ``embedding_ollama.py`` for the ``/api/embed`` shape because the
Ollama variant is the most common local default and shaving the OpenAI
abstraction off it makes the error messages clearer when a user
mis-configures localhost:11434.

Spec we implement (per OpenAI Embeddings API):

  POST {endpoint}/v1/embeddings
  Headers: Content-Type: application/json
           Authorization: Bearer <key>   (optional — vLLM ignores by default)
  Body:    {"model": "...", "input": ["str", ...], "encoding_format": "float"}

  200 →    {"object": "list",
            "data": [{"object":"embedding","embedding":[float,...],"index":N}, ...],
            "model": "...",
            "usage": {...}}

The ``index`` field matters: per spec, the server may return entries
out of input order. We sort on receive so the caller always gets vectors
in the same order it submitted them. Quietly trusting array order is a
silent-corruption hazard.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OpenAIEmbeddingError(RuntimeError):
    """Raised when the OpenAI-compat endpoint is unreachable, returns an
    error, or returns a malformed response. Hard failure so a misconfigured
    endpoint does not silently produce empty vectors."""


# ── HTTP transport ────────────────────────────────────────────────────────
#
# The default keep-alive path uses urllib3.PoolManager so N producer threads
# (mempalace.parallel.ParallelPipeline) can saturate a remote endpoint
# without serializing on TCP setup. The legacy urlopen path is preserved
# behind ``MEMPALACE_HTTP_KEEPALIVE=0`` so existing tests that patch
# ``mempalace.embedding_openai.urlopen`` keep working unchanged.
#
# The follow-up parallelism work (docs/design/embrace-parallelism.md)
# ports those tests to the pool path and removes this env shim.

_HTTP_POOL = None  # urllib3.PoolManager — lazy-init, module-cached


def _get_http_pool():
    global _HTTP_POOL
    if _HTTP_POOL is None:
        import urllib3

        # maxsize must comfortably exceed worker count or producers stall
        # waiting for a free connection. Read MEMPALACE_WORKERS directly
        # (avoids importing MempalaceConfig which has its own lazy paths).
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


def _post_json_via_urlopen(url: str, body: dict, headers: dict, timeout: int) -> dict:
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
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
        raise OpenAIEmbeddingError(f"HTTP {e.code} from {url}: {detail or e.reason}") from e
    except (URLError, OSError) as e:
        raise OpenAIEmbeddingError(f"Cannot reach {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise OpenAIEmbeddingError(f"Malformed response from {url}: {e}") from e


def _post_json_via_pool(url: str, body: dict, headers: dict, timeout: int) -> dict:
    import urllib3

    pool = _get_http_pool()
    encoded = json.dumps(body).encode("utf-8")
    merged_headers = {"Content-Type": "application/json", **headers}
    try:
        resp = pool.request(
            "POST",
            url,
            body=encoded,
            headers=merged_headers,
            timeout=timeout,
            retries=False,
        )
    except urllib3.exceptions.MaxRetryError as e:
        raise OpenAIEmbeddingError(f"Cannot reach {url}: {e}") from e
    except urllib3.exceptions.TimeoutError as e:
        raise OpenAIEmbeddingError(f"Timeout reaching {url}: {e}") from e
    except urllib3.exceptions.HTTPError as e:
        raise OpenAIEmbeddingError(f"Cannot reach {url}: {e}") from e
    except OSError as e:
        raise OpenAIEmbeddingError(f"Cannot reach {url}: {e}") from e

    if resp.status >= 400:
        detail = ""
        try:
            detail = resp.data.decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise OpenAIEmbeddingError(
            f"HTTP {resp.status} from {url}: {detail or resp.reason or 'no detail'}"
        )

    try:
        return json.loads(resp.data)
    except json.JSONDecodeError as e:
        raise OpenAIEmbeddingError(f"Malformed response from {url}: {e}") from e


def _post_json(url: str, body: dict, headers: dict, timeout: int) -> dict:
    """Dispatch to the keep-alive pool by default, urlopen on opt-out.

    ``MEMPALACE_HTTP_KEEPALIVE=0`` selects the legacy urlopen path. Tests
    in this repo that pre-date the pool work patch
    ``mempalace.embedding_openai.urlopen`` directly; setting this env var
    in ``tests/conftest.py`` keeps those tests passing without rewriting
    every patch.
    """
    if os.environ.get("MEMPALACE_HTTP_KEEPALIVE", "1") == "0":
        return _post_json_via_urlopen(url, body, headers, timeout)
    return _post_json_via_pool(url, body, headers, timeout)


class OpenAICompatEmbeddingFunction:
    """ChromaDB EF backed by any OpenAI-spec /v1/embeddings endpoint.

    Constructed with ``model``, ``endpoint``, and an optional ``api_key``.
    The endpoint may end in ``/v1`` or not — the class normalises by
    appending ``/embeddings`` to ``{endpoint}/v1`` regardless.

    See module docstring for the wire format we speak.
    """

    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        if not model:
            raise ValueError("OpenAICompatEmbeddingFunction requires a non-empty model name")
        if not endpoint:
            raise ValueError("OpenAICompatEmbeddingFunction requires a non-empty endpoint URL")
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        # If the endpoint already ends in /v1 we keep it; otherwise append it.
        # vLLM defaults to /v1; LM Studio uses /v1; OpenAI uses /v1. Everyone
        # uses /v1.
        if not self.endpoint.endswith("/v1"):
            self._embeddings_url = f"{self.endpoint}/v1/embeddings"
        else:
            self._embeddings_url = f"{self.endpoint}/embeddings"
        # Allow the env var as a fallback so users do not have to put a key in
        # config.json. vLLM does not require a key; this matters only for
        # actual OpenAI / Together / Fireworks.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    @staticmethod
    def name() -> str:
        # Stable provider tag — chromadb persists this on the collection.
        # Distinct from ``"ollama"`` so a config swap from one to the other
        # surfaces as an EF identity error rather than silent vector
        # corruption (different model + different runtime → different
        # vectors even if dim happens to match).
        return "openai-compat"

    def get_config(self) -> dict:
        # Deliberately exclude api_key — it should not be persisted on the
        # collection. The reader rehydrates it from env.
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "timeout": self.timeout,
        }

    @staticmethod
    def build_from_config(config: dict) -> "OpenAICompatEmbeddingFunction":
        return OpenAICompatEmbeddingFunction(
            model=config["model"],
            endpoint=config["endpoint"],
            timeout=int(config.get("timeout", 120)),
        )

    @staticmethod
    def validate_config(config: dict) -> None:
        if not config.get("model"):
            raise ValueError("openai-compat embedding config missing 'model'")
        if not config.get("endpoint"):
            raise ValueError("openai-compat embedding config missing 'endpoint'")

    def validate_config_update(self, old_config: dict, new_config: dict) -> None:
        if old_config.get("model") != new_config.get("model"):
            raise ValueError(
                "Cannot change embedding model on an existing collection — "
                "vector dimensions would not match. Wipe and re-mine."
            )

    def __call__(self, input):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        if not texts:
            return []

        body = {"model": self.model, "input": texts, "encoding_format": "float"}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = _post_json(self._embeddings_url, body, headers, self.timeout)

        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise OpenAIEmbeddingError(
                f"OpenAI-compat /v1/embeddings returned "
                f"{len(items) if isinstance(items, list) else '?'} items "
                f"for {len(texts)} inputs (model={self.model})"
            )

        # Per OpenAI spec, ``data`` entries each have an ``index`` field and
        # MAY arrive out of order. Sort on receive — trusting array order is
        # how silent vector/document misalignment gets shipped to prod.
        try:
            ordered = sorted(items, key=lambda d: int(d["index"]))
        except (KeyError, TypeError, ValueError) as e:
            raise OpenAIEmbeddingError(
                f"OpenAI-compat response items missing 'index' field: {e}"
            ) from e

        try:
            return [item["embedding"] for item in ordered]
        except KeyError as e:
            raise OpenAIEmbeddingError(
                f"OpenAI-compat response items missing 'embedding' field: {e}"
            ) from e

    def embed_query(self, input):
        return self.__call__(input)

    def embed_documents(self, input):
        return self.__call__(input)

    def check_available(self) -> tuple[bool, str]:
        """Return ``(ok, message)``. Fast probe: do a 1-input embed.

        We use an actual embed (not ``/v1/models``) because vLLM and OpenAI
        differ on the models list shape and a successful embed is the only
        thing we actually care about anyway."""
        try:
            self(["ping"])
            return True, "ok"
        except OpenAIEmbeddingError as e:
            return False, str(e)


__all__ = [
    "OpenAICompatEmbeddingFunction",
    "OpenAIEmbeddingError",
]
