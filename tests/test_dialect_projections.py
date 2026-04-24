"""Unit tests for the rank-bucketed projection helpers in ``mempalace.dialect``.

These are pure, deterministic helpers used by ``recursive_indexer.py`` to
aggregate leaf-closet pointer lines into room- and wing-level indices
without invoking an LLM.
"""

from mempalace.dialect import (
    aggregate_entity_sets,
    frequency_top_n,
    parse_closet_line,
    project_closet_lines,
)


class TestParseClosetLine:
    def test_basic_line(self):
        assert parse_closet_line("jwt auth|Alice;Bob|→d1,d2") == (
            "jwt auth",
            ["Alice", "Bob"],
            ["d1", "d2"],
        )

    def test_empty_entities(self):
        assert parse_closet_line("some topic||→d1") == ("some topic", [], ["d1"])

    def test_quoted_topic_preserved(self):
        topic, entities, drawers = parse_closet_line(
            '"the stone giants"|Grundar;Xalvosh|→d_01,d_02'
        )
        assert topic == '"the stone giants"'
        assert entities == ["Grundar", "Xalvosh"]
        assert drawers == ["d_01", "d_02"]

    def test_ascii_arrow_fallback(self):
        assert parse_closet_line("topic|A|->d1") == ("topic", ["A"], ["d1"])

    def test_blank_line_returns_none(self):
        assert parse_closet_line("") is None
        assert parse_closet_line("   ") is None

    def test_missing_arrow_returns_none(self):
        assert parse_closet_line("topic|entities|no-arrow-here") is None

    def test_missing_topic_returns_none(self):
        assert parse_closet_line("|ents|→d1") is None

    def test_non_string_returns_none(self):
        assert parse_closet_line(None) is None
        assert parse_closet_line(42) is None

    def test_whitespace_stripped(self):
        assert parse_closet_line("  topic  |  A ; B  |→ d1 , d2 ") == (
            "topic",
            ["A", "B"],
            ["d1", "d2"],
        )

    def test_drawer_id_list_trimmed(self):
        topic, _ents, drawers = parse_closet_line("t|e|→d1,,,d2,")
        assert drawers == ["d1", "d2"]


class TestFrequencyTopN:
    def test_basic_counts(self):
        result = frequency_top_n(["a", "b", "a", "c", "b", "a"])
        assert result == [("a", 3), ("b", 2), ("c", 1)]

    def test_respects_n(self):
        result = frequency_top_n(["a", "b", "a", "c", "b", "a"], n=2)
        assert result == [("a", 3), ("b", 2)]

    def test_min_count_filter(self):
        result = frequency_top_n(["a", "b", "a", "c"], min_count=2)
        assert result == [("a", 2)]

    def test_first_seen_ties(self):
        # Three items each appear once — insertion order wins.
        assert frequency_top_n(["b", "c", "a"]) == [("b", 1), ("c", 1), ("a", 1)]

    def test_key_projection(self):
        items = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Alice"}]
        result = frequency_top_n(items, key=lambda d: d["name"])
        assert result == [("Alice", 2), ("Bob", 1)]

    def test_key_returning_empty_skips(self):
        # Falsy keys skip the item entirely — useful for filter+count.
        items = ["Alice", "", "Bob", None, "Alice"]
        result = frequency_top_n(items, key=lambda s: s)
        assert result == [("Alice", 2), ("Bob", 1)]

    def test_zero_n_returns_empty(self):
        assert frequency_top_n(["a"] * 10, n=0) == []

    def test_empty_input(self):
        assert frequency_top_n([]) == []

    def test_determinism(self):
        items = ["x", "y", "x", "z", "y", "x"]
        runs = [frequency_top_n(items, n=3) for _ in range(10)]
        assert all(r == runs[0] for r in runs)


class TestProjectClosetLines:
    def test_groups_duplicate_topics(self):
        lines = [
            "jwt auth|Alice|→d1",
            "jwt auth|Bob|→d2",
            "database|Carol|→d3",
        ]
        result = project_closet_lines(lines)
        # "jwt auth" wins on count; "database" follows.
        assert [line.split("|")[0] for line, _count in result] == ["jwt auth", "database"]
        assert result[0][1] == 2
        assert result[1][1] == 1

    def test_representative_is_first_seen(self):
        lines = [
            "topic|first|→d1",
            "topic|second|→d2",
        ]
        ((line, count),) = project_closet_lines(lines)
        assert line == "topic|first|→d1"
        assert count == 2

    def test_dedupe_on_line(self):
        lines = [
            "topic|A|→d1",
            "topic|B|→d2",  # same topic, different entities — counted separately
            "topic|A|→d1",  # exact dupe of first
        ]
        result = project_closet_lines(lines, dedupe_on="line")
        assert len(result) == 2
        assert result[0] == ("topic|A|→d1", 2)
        assert result[1] == ("topic|B|→d2", 1)

    def test_malformed_lines_skipped(self):
        lines = [
            "valid|A|→d1",
            "",
            "no arrow here",
            "valid|B|→d2",
        ]
        result = project_closet_lines(lines)
        # Both "valid" lines share the same topic → coalesced.
        assert len(result) == 1
        assert result[0][1] == 2

    def test_non_string_lines_skipped(self):
        lines = ["topic|A|→d1", None, 42, ["not", "a", "string"]]
        result = project_closet_lines(lines)
        assert len(result) == 1
        assert result[0][0] == "topic|A|→d1"

    def test_zero_n_returns_empty(self):
        assert project_closet_lines(["topic|A|→d1"], n=0) == []

    def test_invalid_dedupe_on_raises(self):
        import pytest

        with pytest.raises(ValueError):
            project_closet_lines(["topic|A|→d1"], dedupe_on="bogus")


class TestAggregateEntitySets:
    def test_string_inputs(self):
        result = aggregate_entity_sets(
            ["Alice;Bob", "Bob;Carol", "Alice"],
        )
        assert result[0] == ("Alice", 2)
        assert result[1] == ("Bob", 2)
        assert result[2] == ("Carol", 1)

    def test_list_inputs(self):
        result = aggregate_entity_sets([["Alice", "Bob"], ["Bob"]])
        assert dict(result) == {"Alice": 1, "Bob": 2}

    def test_mixed_string_and_list(self):
        result = aggregate_entity_sets(["Alice;Bob", ["Carol"], "Alice"])
        assert dict(result) == {"Alice": 2, "Bob": 1, "Carol": 1}

    def test_handles_none_and_empty(self):
        result = aggregate_entity_sets([None, "", "Alice", ";;", "Alice"])
        assert result == [("Alice", 2)]

    def test_respects_n(self):
        inputs = ["A;B;C;D;E;F"] * 3
        result = aggregate_entity_sets(inputs, n=2)
        assert len(result) == 2

    def test_case_preserved(self):
        result = aggregate_entity_sets(["alice;Alice;ALICE"])
        assert dict(result) == {"alice": 1, "Alice": 1, "ALICE": 1}

    def test_zero_n_returns_empty(self):
        assert aggregate_entity_sets(["Alice"], n=0) == []
