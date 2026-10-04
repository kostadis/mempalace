"""Unparsed native rollouts must not be archived as raw JSON or marked filed."""

import json
from unittest.mock import Mock

import pytest

from mempalace.convo_miner import _prepare_convo
from mempalace.normalize import (
    UnparsedCodexTranscriptError,
    normalize,
    normalize_conversations,
)


def _rollout(path, events):
    rows = [{"type": "session_meta", "payload": {"id": "synthetic-session"}}]
    rows.extend(events)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


@pytest.mark.parametrize("reader", [normalize, normalize_conversations])
@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"type": "event_msg", "payload": {"type": "future_conversation_format"}}],
        [{"type": "event_msg", "payload": {"type": "user_message", "message": "pending"}}],
        [
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "developer", "content": "DO NOT MINE"},
            }
        ],
    ],
)
def test_unparsed_rollout_cannot_fall_back_to_raw_json(tmp_path, reader, events):
    source = tmp_path / "rollout.jsonl"
    _rollout(source, events)
    original = source.read_bytes()
    with pytest.raises(UnparsedCodexTranscriptError, match="refusing raw JSON fallback"):
        reader(str(source))
    assert source.read_bytes() == original


@pytest.mark.parametrize("dry_run", [False, True])
def test_failed_source_is_not_registered_and_can_be_retried(tmp_path, monkeypatch, caplog, dry_run):
    source = tmp_path / "rollout.jsonl"
    _rollout(source, [{"type": "event_msg", "payload": {"type": "future_format"}}])
    register = Mock()
    monkeypatch.setattr("mempalace.convo_miner._register_file", register)
    # Local parallel convo miner: _prepare_convo (not upstream's
    # _normalize_convo_conversations) decides skip vs register. A ("register",
    # ...) payload would mark the source filed; None leaves it retryable.
    monkeypatch.setattr("mempalace.convo_miner.file_already_mined", lambda *a, **k: False)
    collection = Mock()

    def read():
        return _prepare_convo(
            source, "test", "agent", "exchange", dry_run, collection, min_chunk_size=30
        )

    assert read() is None
    register.assert_not_called()
    assert collection.mock_calls == []
    assert "source remains eligible for retry" in caplog.text

    _rollout(
        source,
        [
            {"type": "event_msg", "payload": {"type": "user_message", "message": "Where next?"}},
            {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "Meet at the north library desk."},
            },
        ],
    )
    prepared = read()
    assert prepared is not None and not isinstance(prepared, tuple)
    documents = [doc for batch in prepared.batches for doc in batch.documents]
    assert any("Meet at the north library desk." in doc for doc in documents)
    register.assert_not_called()


@pytest.mark.parametrize("reader", [normalize, normalize_conversations])
def test_unrelated_plain_text_and_json_fallback_are_unchanged(tmp_path, reader):
    source = tmp_path / "notes.json"
    text = '{"personal_note": "Remember the blue notebook"}'
    source.write_text(text, encoding="utf-8")
    expected = text if reader is normalize else [text]
    assert reader(str(source)) == expected
