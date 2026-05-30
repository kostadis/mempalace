"""Contract + persistence tests for the experimental turbovec backend.

Skipped wholesale on platforms without a turbovec wheel. Vectors are supplied
explicitly so the unit tests are fully offline — the embedding-function path
(``query_texts``) is exercised with a fake EF; the real nomic/vllm-embed path
is covered by the on-Spark parity step, not here.
"""

import pytest

turbovec = pytest.importorskip("turbovec")

from mempalace.backends import (  # noqa: E402  (must follow importorskip)
    GetResult,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
)
from mempalace.backends.turbovec import (  # noqa: E402  (imports the turbovec dep)
    TurboVecBackend,
    TurboVecCollection,
)


# Near-one-hot 8-d vectors → unambiguous cosine ordering. turbovec requires
# dim to be a positive multiple of 8 (real embedders qualify: nomic 768,
# MiniLM 384), so the toy dimension is 8, not 4.
DIM = 8
VECS = {
    "a": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "c": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "d": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # closest neighbour of "a"
}


def _seed(collection):
    ids = ["a", "b", "c", "d"]
    collection.add(
        documents=[f"doc {i}" for i in ids],
        ids=ids,
        metadatas=[{"wing": "letters", "room": i} for i in ids],
        embeddings=[VECS[i] for i in ids],
    )


@pytest.fixture
def palace(tmp_path):
    return PalaceRef(id=str(tmp_path / "palace"), local_path=str(tmp_path / "palace"))


@pytest.fixture
def collection(palace):
    backend = TurboVecBackend()
    col = backend.get_collection(palace=palace, collection_name="drawers", create=True)
    _seed(col)
    yield col
    backend.close()


# ── contract: typed returns ────────────────────────────────────────────────


def test_query_returns_typed_result(collection):
    res = collection.query(query_embeddings=[VECS["a"]], n_results=2)
    assert isinstance(res, QueryResult)
    assert res.ids[0][0] == "a"            # exact match ranks first
    assert res.ids[0][1] == "d"            # nearest neighbour second
    assert res.embeddings is None           # not requested


def test_get_returns_typed_result(collection):
    res = collection.get(ids=["a"], include=["documents", "metadatas"])
    assert isinstance(res, GetResult)
    assert res.ids == ["a"]
    assert res.documents == ["doc a"]
    assert res.metadatas[0]["wing"] == "letters"


def test_distance_is_true_cosine(collection):
    res = collection.query(query_embeddings=[VECS["a"]], n_results=4)
    dists = dict(zip(res.ids[0], res.distances[0]))
    assert dists["a"] == pytest.approx(0.0, abs=1e-5)   # identical → 0
    assert dists["b"] == pytest.approx(1.0, abs=1e-5)   # orthogonal → 1
    assert 0.0 < dists["d"] < dists["b"]                # in between


def test_query_empty_collection_preserves_outer_dim(palace):
    backend = TurboVecBackend()
    col = backend.get_collection(palace=palace, collection_name="empty", create=True)
    res = col.query(query_embeddings=[VECS["a"], VECS["b"]], n_results=3)
    assert res.ids == [[], []]
    assert res.distances == [[], []]
    backend.close()


def test_embeddings_returned_only_when_requested(collection):
    res = collection.query(query_embeddings=[VECS["a"]], n_results=1, include=["embeddings"])
    assert res.embeddings is not None
    assert len(res.embeddings[0][0]) == DIM


# ── contract: input validation ─────────────────────────────────────────────


def test_query_rejects_missing_input(collection):
    with pytest.raises(ValueError):
        collection.query()


def test_query_rejects_both_inputs(collection):
    with pytest.raises(ValueError):
        collection.query(query_texts=["q"], query_embeddings=[VECS["a"]])


def test_query_rejects_empty_input_list(collection):
    with pytest.raises(ValueError):
        collection.query(query_embeddings=[])


# ── contract: filters ──────────────────────────────────────────────────────


def test_where_equality_filter(collection):
    res = collection.query(query_embeddings=[VECS["a"]], n_results=4, where={"room": "b"})
    assert res.ids[0] == ["b"]


def test_where_in_filter(collection):
    res = collection.query(
        query_embeddings=[VECS["a"]], n_results=4, where={"room": {"$in": ["b", "c"]}}
    )
    assert set(res.ids[0]) == {"b", "c"}


def test_where_and_filter(collection):
    res = collection.query(
        query_embeddings=[VECS["a"]],
        n_results=4,
        where={"$and": [{"wing": "letters"}, {"room": "d"}]},
    )
    assert res.ids[0] == ["d"]


def test_where_document_contains(collection):
    res = collection.query(
        query_embeddings=[VECS["a"]], n_results=4, where_document={"$contains": "doc c"}
    )
    assert res.ids[0] == ["c"]


def test_filter_with_no_matches_preserves_outer_dim(collection):
    res = collection.query(query_embeddings=[VECS["a"]], n_results=4, where={"room": "zzz"})
    assert res.ids == [[]]


@pytest.mark.parametrize("bad", [{"$regex": "x"}, {"$or": []}, {"room": {"$ne": "b"}}])
def test_unsupported_where_operator_raises(collection, bad):
    with pytest.raises(UnsupportedFilterError):
        collection.query(query_embeddings=[VECS["a"]], n_results=1, where=bad)


def test_unsupported_where_document_operator_raises(collection):
    with pytest.raises(UnsupportedFilterError):
        collection.query(query_embeddings=[VECS["a"]], n_results=1,
                         where_document={"$not_contains": "x"})


# ── contract: writes ───────────────────────────────────────────────────────


def test_count(collection):
    assert collection.count() == 4


def test_add_duplicate_id_raises(collection):
    with pytest.raises(ValueError):
        collection.add(documents=["dup"], ids=["a"], embeddings=[VECS["a"]])


def test_upsert_replaces(collection):
    moved = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # a direction no seed doc occupies
    collection.upsert(documents=["new a"], ids=["a"], metadatas=[{"room": "a2"}],
                      embeddings=[moved])
    assert collection.count() == 4                      # not a new row
    res = collection.get(ids=["a"])
    assert res.documents == ["new a"]
    # The vector moved → a query at the new direction ranks "a" first at distance 0.
    q = collection.query(query_embeddings=[moved], n_results=1)
    assert q.ids[0][0] == "a"
    assert q.distances[0][0] == pytest.approx(0.0, abs=1e-5)


def test_delete_by_id(collection):
    collection.delete(ids=["a"])
    assert collection.count() == 3
    res = collection.query(query_embeddings=[VECS["a"]], n_results=4)
    assert "a" not in res.ids[0]


def test_delete_by_where(collection):
    collection.delete(where={"room": "b"})
    assert collection.count() == 3
    assert collection.get(ids=["b"]).ids == []


# ── backend factory ────────────────────────────────────────────────────────


def test_get_collection_create_false_raises_on_missing_palace(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        TurboVecBackend().get_collection(
            palace=PalaceRef(id=str(missing), local_path=str(missing)),
            collection_name="drawers",
            create=False,
        )
    assert not missing.exists()


def test_get_collection_accepts_positional_string_path(tmp_path):
    backend = TurboVecBackend()
    col = backend.get_collection(str(tmp_path / "p"), collection_name="drawers", create=True)
    assert isinstance(col, TurboVecCollection)
    backend.close()


def test_detect(tmp_path):
    assert TurboVecBackend.detect(str(tmp_path)) is False
    (tmp_path / "turbovec").mkdir()
    assert TurboVecBackend.detect(str(tmp_path)) is True


# ── persistence: .tvim cache load + rebuild-from-SQLite ────────────────────


def test_persists_and_reloads_across_backend_instances(palace, tmp_path):
    b1 = TurboVecBackend()
    col = b1.get_collection(palace=palace, collection_name="drawers", create=True)
    _seed(col)
    b1.close()  # flushes .tvim

    import os
    tvim = os.path.join(palace.local_path, "turbovec", "drawers", "index.tvim")
    assert os.path.exists(tvim), "close() should flush the .tvim cache"

    b2 = TurboVecBackend()
    col2 = b2.get_collection(palace=palace, collection_name="drawers", create=False)
    assert col2.count() == 4
    assert col2.query(query_embeddings=[VECS["a"]], n_results=1).ids[0][0] == "a"
    b2.close()


def test_rebuilds_index_when_tvim_missing(palace, tmp_path):
    import os

    b1 = TurboVecBackend()
    b1.get_collection(palace=palace, collection_name="drawers", create=True)
    _seed(b1.get_collection(palace=palace, collection_name="drawers", create=True))
    b1.close()

    tvim = os.path.join(palace.local_path, "turbovec", "drawers", "index.tvim")
    os.remove(tvim)  # simulate a crash before the cache was flushed

    b2 = TurboVecBackend()
    col = b2.get_collection(palace=palace, collection_name="drawers", create=False)
    # SQLite is the source of truth → search still works, rebuilt from vectors.
    assert col.count() == 4
    assert col.query(query_embeddings=[VECS["a"]], n_results=2).ids[0] == ["a", "d"]
    b2.close()


# ── embedding-function path (query_texts) with a fake EF ───────────────────


def test_mempalace_backend_env_routes_palace_default():
    """MEMPALACE_BACKEND=turbovec makes palace._DEFAULT_BACKEND a TurboVecBackend.

    Run in a subprocess so the env-resolved module-level singleton doesn't leak
    into this test process (palace.py resolves the backend once at import).
    """
    import os
    import subprocess
    import sys

    def _backend_name(env_value):
        env = dict(os.environ)
        if env_value is None:
            env.pop("MEMPALACE_BACKEND", None)
        else:
            env["MEMPALACE_BACKEND"] = env_value
        out = subprocess.check_output(
            [sys.executable, "-c",
             "import mempalace.palace as p; print(type(p._DEFAULT_BACKEND).__name__)"],
            env=env, text=True,
        )
        return out.strip()

    assert _backend_name("turbovec") == "TurboVecBackend"
    assert _backend_name(None) == "ChromaBackend"  # default unchanged


def test_query_texts_uses_embedding_function(palace, monkeypatch):
    def fake_ef(texts):
        # Map any text containing 'a' → VECS['a'], else VECS['b'].
        return [VECS["a"] if "a" in t else VECS["b"] for t in texts]

    monkeypatch.setattr("mempalace.embedding.get_embedding_function", lambda *a, **k: fake_ef)

    backend = TurboVecBackend()
    col = backend.get_collection(palace=palace, collection_name="drawers", create=True)
    _seed(col)
    res = col.query(query_texts=["find a please"], n_results=1)
    assert res.ids[0][0] == "a"
    backend.close()
