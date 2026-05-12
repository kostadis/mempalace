"""Tests for closet_taxonomy.

The LLM call itself is not exercised here — that's a network test.
These cover the pure functions that turn classification labels into
closet rows, plus the vocabulary filter (which is load-bearing for
not propagating LLM hallucinations into the index).
"""

from mempalace import closet_taxonomy as ct


def test_filter_labels_keeps_only_vocab_entries():
    """Labels not in the fixed taxonomy must be dropped — hallucinated
    or paraphrased labels must NOT make it into closet rows."""
    vocab = ct.TAXONOMY["structural_role"]
    labels = ["wrapper", "client", "totally_made_up", "adapter", "ClassImagined"]
    result = ct._filter_labels(labels, vocab)
    assert result == ["wrapper", "client", "adapter"]


def test_filter_labels_caps_at_4():
    """Even when the LLM picks more, we cap to 4 per category to keep
    closet rows readable and the sentence templates predictable."""
    vocab = ct.TAXONOMY["structural_role"]
    labels = ["wrapper", "client", "adapter", "controller", "dispatcher", "factory"]
    result = ct._filter_labels(labels, vocab)
    assert result == ["wrapper", "client", "adapter", "controller"]


def test_filter_labels_handles_non_list_input():
    """LLM occasionally returns the wrong shape. Don't crash."""
    assert ct._filter_labels("wrapper", ct.TAXONOMY["structural_role"]) == []
    assert ct._filter_labels(None, ct.TAXONOMY["structural_role"]) == []
    assert ct._filter_labels({"a": 1}, ct.TAXONOMY["structural_role"]) == []


def test_filter_labels_drops_non_string_entries():
    """A list with mixed types — keep only the valid strings."""
    vocab = ct.TAXONOMY["structural_role"]
    labels = ["wrapper", 42, None, "client", {"x": "y"}]
    assert ct._filter_labels(labels, vocab) == ["wrapper", "client"]


def test_labels_to_sentences_basic_case():
    """A well-formed classification produces 4 natural sentences that
    each contain the abstract labels in human-readable phrasing."""
    classified = {
        "structural_role": ["wrapper", "client"],
        "function": ["fetch", "serialize"],
        "abstraction": ["external_service", "boundary"],
    }
    sentences = ct._labels_to_sentences("/foo/mempalace_client.py", classified)
    assert len(sentences) >= 3
    blob = " ".join(sentences).lower()
    # The whole point: labels must surface as readable phrases in the prose.
    assert "wrapper" in blob
    assert "external service" in blob  # humanized — underscore → space
    assert "boundary" in blob


def test_labels_to_sentences_humanizes_underscores():
    """`tiered_pipeline` must become `tiered pipeline` so an embedding
    of a query like "tiered retrieval pipeline" can match it."""
    classified = {
        "structural_role": ["orchestrator"],
        "function": ["tier", "route"],
        "abstraction": ["tiered_pipeline"],
    }
    sentences = ct._labels_to_sentences("/foo/rpg_retriever.py", classified)
    blob = " ".join(sentences).lower()
    assert "tiered pipeline" in blob
    assert "tiered_pipeline" not in blob  # the underscored form must not leak through


def test_labels_to_sentences_filters_hallucinated_labels():
    """The pipeline must not propagate labels outside the taxonomy
    even when they show up in the classification dict."""
    classified = {
        "structural_role": ["wrapper", "magical_thing"],
        "function": ["fetch"],
        "abstraction": ["external_service", "imaginary_layer"],
    }
    sentences = ct._labels_to_sentences("/foo/f.py", classified)
    blob = " ".join(sentences).lower()
    assert "magical" not in blob
    assert "imaginary" not in blob
    assert "wrapper" in blob
    assert "external service" in blob


def test_labels_to_sentences_empty_classification_returns_empty():
    """If the LLM picks nothing valid, return no sentences (caller will skip)."""
    assert ct._labels_to_sentences("/foo/x.py", {}) == []
    assert (
        ct._labels_to_sentences(
            "/foo/x.py",
            {"structural_role": ["nothing"], "function": [], "abstraction": []},
        )
        == []
    )


def test_labels_to_sentences_article_agreement():
    """First sentence picks an `a`/`an` article. Vowel-initial roles get `an`."""
    classified = {
        "structural_role": ["adapter"],  # vowel
        "function": ["dispatch"],
        "abstraction": ["boundary"],
    }
    sentences = ct._labels_to_sentences("/foo/x.py", classified)
    assert sentences[0].startswith("x.py is an adapter")

    classified["structural_role"] = ["wrapper"]  # consonant
    sentences = ct._labels_to_sentences("/foo/x.py", classified)
    assert sentences[0].startswith("x.py is a wrapper")


def test_source_hash_is_deterministic_and_path_aware():
    """Re-running the augmenter must produce the same IDs (so upsert
    replaces rows). Two files with the same basename in different
    directories must NOT collide."""
    h1 = ct._source_hash("/a/b/c.py")
    h2 = ct._source_hash("/a/b/c.py")
    h3 = ct._source_hash("/x/y/c.py")  # same basename, different path
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16  # short hash for readable IDs


def test_taxonomy_contains_specific_labels_the_specificity_rule_requires():
    """The specificity rule in the prompt references these labels by
    name. If we ever drop one, the prompt's example breaks. Lock them in."""
    must_have = {
        "structural_role": ["wrapper", "adapter", "client", "black_box"],
        "function": ["fetch", "serialize", "integrate", "dispatch"],
        "abstraction": [
            "external_service", "boundary", "black_box",
            "tiered_pipeline", "source_of_truth", "authoritative",
        ],
    }
    for category, labels in must_have.items():
        for label in labels:
            assert label in ct.TAXONOMY[category], (
                f"Specificity rule requires '{label}' in TAXONOMY['{category}'], "
                f"but it's missing — the prompt's worked example will reference "
                f"a label that can't be picked."
            )
