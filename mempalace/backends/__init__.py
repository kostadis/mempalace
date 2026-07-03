"""Storage backend implementations for MemPalace (RFC 001).

Public surface:

* :class:`BaseCollection` — per-collection read/write contract.
* :class:`BaseBackend` — per-palace factory contract.
* :class:`PalaceRef` — value object identifying a palace for a backend.
* :class:`QueryResult` / :class:`GetResult` — typed read returns.
* Error classes: :class:`PalaceNotFoundError`, :class:`BackendClosedError`,
  :class:`UnsupportedFilterError`, :class:`DimensionMismatchError`,
  :class:`EmbedderIdentityMismatchError`.
* Registry: :func:`get_backend`, :func:`register`, :func:`available_backends`,
  :func:`resolve_backend_for_palace`.
* In-tree Chroma default: :class:`ChromaBackend`, :class:`ChromaCollection`.
"""

from .base import (
    BackendClosedError,
    BackendError,
    BackendMismatchError,
    BaseBackend,
    BaseCollection,
    CollectionNotInitializedError,
    DimensionMismatchError,
    EmbedderIdentityMismatchError,
    GetResult,
    HealthStatus,
    LexicalHit,
    LexicalResult,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedCapabilityError,
    UnsupportedFilterError,
)
# Chroma is imported lazily below (PEP 562 __getattr__) so it stays optional.
# pgvector/qdrant/sqlite_exact import their heavy client deps lazily inside
# methods, so importing their classes here is cheap and safe on any deploy.
from .pgvector import PgVectorBackend, PgVectorCollection
from .qdrant import QdrantBackend, QdrantCollection
from .sqlite_exact import SQLiteExactBackend, SQLiteExactCollection
from .registry import (
    available_backends,
    detect_backend_for_path,
    detect_backends_for_path,
    get_backend,
    get_backend_class,
    register,
    reset_backends,
    resolve_backend_for_palace,
    unregister,
)


# Chroma is an OPTIONAL in-tree backend. Import it lazily (PEP 562) so that importing
# `mempalace.backends` on a turbovec-only deployment — where chromadb is absent or has
# broken transitive deps — does not fail at import time. `ChromaBackend`/`ChromaCollection`
# still resolve on first access if chromadb is importable.
def __getattr__(name: str):
    if name in ("ChromaBackend", "ChromaCollection"):
        from . import chroma

        return getattr(chroma, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BackendClosedError",
    "BackendError",
    "BackendMismatchError",
    "BaseBackend",
    "BaseCollection",
    "ChromaBackend",
    "ChromaCollection",
    "CollectionNotInitializedError",
    "DimensionMismatchError",
    "EmbedderIdentityMismatchError",
    "GetResult",
    "HealthStatus",
    "LexicalHit",
    "LexicalResult",
    "PalaceNotFoundError",
    "PalaceRef",
    "PgVectorBackend",
    "PgVectorCollection",
    "QdrantBackend",
    "QdrantCollection",
    "QueryResult",
    "SQLiteExactBackend",
    "SQLiteExactCollection",
    "UnsupportedCapabilityError",
    "UnsupportedFilterError",
    "available_backends",
    "detect_backend_for_path",
    "detect_backends_for_path",
    "get_backend",
    "get_backend_class",
    "register",
    "reset_backends",
    "resolve_backend_for_palace",
    "unregister",
]
