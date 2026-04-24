"""Tests for the new room/wing index collections and dirty-flag plumbing.

These are the foundation for ``recursive_indexer.py``: the indexer reads
dirty flags, rebuilds the affected rooms/wings, and clears the flags. The
plumbing itself is simple filesystem-based state and must be
cross-process safe.
"""

import os
import threading

import pytest

from mempalace import palace


# ── Collection accessors ───────────────────────────────────────────────


def test_get_room_indices_collection_creates(palace_path):
    col = palace.get_room_indices_collection(palace_path)
    assert col is not None
    col.upsert(ids=["r1"], documents=["doc"], metadatas=[{"wing": "w", "room": "r"}])
    assert col.count() == 1


def test_get_wing_indices_collection_creates(palace_path):
    col = palace.get_wing_indices_collection(palace_path)
    assert col is not None
    col.upsert(ids=["w1"], documents=["doc"], metadatas=[{"wing": "w"}])
    assert col.count() == 1


def test_indices_collections_are_separate(palace_path):
    rooms = palace.get_room_indices_collection(palace_path)
    wings = palace.get_wing_indices_collection(palace_path)
    closets = palace.get_closets_collection(palace_path)
    # Writes to one do not leak into the others — prove they're distinct
    # Chroma collections by counting independently.
    rooms.upsert(ids=["r1"], documents=["room-doc"], metadatas=[{"wing": "w", "room": "r"}])
    wings.upsert(ids=["w1"], documents=["wing-doc"], metadatas=[{"wing": "w"}])
    assert rooms.count() == 1
    assert wings.count() == 1
    assert closets.count() == 0


# ── Dirty-flag plumbing ────────────────────────────────────────────────


def test_mark_room_dirty_also_marks_wing(palace_path):
    palace.mark_room_dirty(palace_path, "notes", "backend")
    rooms = list(palace.iter_dirty_rooms(palace_path))
    wings = list(palace.iter_dirty_wings(palace_path))
    assert len(rooms) == 1
    assert (rooms[0][0], rooms[0][1]) == ("notes", "backend")
    assert len(wings) == 1
    assert wings[0][0] == "notes"


def test_mark_room_dirty_is_idempotent(palace_path):
    for _ in range(5):
        palace.mark_room_dirty(palace_path, "notes", "backend")
    assert len(list(palace.iter_dirty_rooms(palace_path))) == 1
    assert len(list(palace.iter_dirty_wings(palace_path))) == 1


def test_clear_room_dirty_removes_flag(palace_path):
    palace.mark_room_dirty(palace_path, "notes", "backend")
    assert palace.clear_room_dirty(palace_path, "notes", "backend") is True
    assert list(palace.iter_dirty_rooms(palace_path)) == []
    # Second clear is a no-op returning False.
    assert palace.clear_room_dirty(palace_path, "notes", "backend") is False


def test_clear_room_does_not_clear_wing(palace_path):
    # Clearing a room does NOT auto-clear the wing — the wing might still
    # be dirty because other rooms in it are also pending.
    palace.mark_room_dirty(palace_path, "notes", "backend")
    palace.clear_room_dirty(palace_path, "notes", "backend")
    wings = list(palace.iter_dirty_wings(palace_path))
    assert [w for w, _ in wings] == ["notes"]


def test_clear_wing_dirty_removes_flag(palace_path):
    palace.mark_wing_dirty(palace_path, "notes")
    assert palace.clear_wing_dirty(palace_path, "notes") is True
    assert list(palace.iter_dirty_wings(palace_path)) == []
    assert palace.clear_wing_dirty(palace_path, "notes") is False


def test_multiple_rooms_in_same_wing(palace_path):
    palace.mark_room_dirty(palace_path, "notes", "backend")
    palace.mark_room_dirty(palace_path, "notes", "frontend")
    palace.mark_room_dirty(palace_path, "notes", "mobile")
    rooms = [(w, r) for w, r, _ in palace.iter_dirty_rooms(palace_path)]
    assert sorted(rooms) == [
        ("notes", "backend"),
        ("notes", "frontend"),
        ("notes", "mobile"),
    ]
    wings = [w for w, _ in palace.iter_dirty_wings(palace_path)]
    # Only one wing flag despite three room flags.
    assert wings == ["notes"]


def test_names_with_special_characters(palace_path):
    # Real wing/room names can have spaces, apostrophes, dots — must round-trip.
    palace.mark_room_dirty(palace_path, "Phandalin 2026", "Tolubb's Forge")
    palace.mark_room_dirty(palace_path, "Phandalin 2026", "v1.2 notes")
    rooms = sorted((w, r) for w, r, _ in palace.iter_dirty_rooms(palace_path))
    assert rooms == [
        ("Phandalin 2026", "Tolubb's Forge"),
        ("Phandalin 2026", "v1.2 notes"),
    ]


def test_iter_dirty_on_empty_palace(palace_path):
    # Fresh palace — no .dirty/ dir yet. Must not raise.
    assert list(palace.iter_dirty_rooms(palace_path)) == []
    assert list(palace.iter_dirty_wings(palace_path)) == []


def test_dirty_flags_survive_across_calls(palace_path):
    # Simulates process-restart durability: write, then list from scratch
    # by re-entering the API. State lives on disk.
    palace.mark_room_dirty(palace_path, "notes", "backend")
    palace.mark_room_dirty(palace_path, "notes", "frontend")
    # (no in-memory state to "reset" — the accessors re-read the dir each time)
    again = list(palace.iter_dirty_rooms(palace_path))
    assert len(again) == 2


def test_concurrent_marks_do_not_corrupt(palace_path):
    # Many threads mark overlapping rooms simultaneously. Every mark must
    # leave a well-formed JSON file (atomic tempfile+rename guarantees this).
    pairs = [("w1", f"r{i}") for i in range(20)] + [("w2", f"r{i}") for i in range(20)]

    def worker(pair):
        for _ in range(10):
            palace.mark_room_dirty(palace_path, pair[0], pair[1])

    threads = [threading.Thread(target=worker, args=(p,)) for p in pairs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rooms = list(palace.iter_dirty_rooms(palace_path))
    assert len(rooms) == 40
    # Well-formed: every entry has a valid (wing, room) pair.
    for wing, room, _ in rooms:
        assert isinstance(wing, str) and wing
        assert isinstance(room, str) and room


def test_dirty_dir_location(palace_path):
    palace.mark_room_dirty(palace_path, "w", "r")
    assert os.path.isdir(os.path.join(palace_path, ".dirty", "rooms"))
    assert os.path.isdir(os.path.join(palace_path, ".dirty", "wings"))
    # Dirty state is isolated to this palace — other palaces see nothing.


def test_dirty_flags_isolated_between_palaces(tmp_path):
    p1 = str(tmp_path / "palace1")
    p2 = str(tmp_path / "palace2")
    os.makedirs(p1)
    os.makedirs(p2)
    palace.mark_room_dirty(p1, "w", "r")
    assert list(palace.iter_dirty_rooms(p1)) != []
    assert list(palace.iter_dirty_rooms(p2)) == []


def test_malformed_dirty_file_is_skipped(palace_path):
    # A half-written JSON file should never crash the iterator — it should
    # be skipped and the valid files should still surface.
    palace.mark_room_dirty(palace_path, "w", "r")
    bogus = os.path.join(palace_path, ".dirty", "rooms", "deadbeefdeadbeef.json")
    with open(bogus, "w", encoding="utf-8") as f:
        f.write("{not json")
    rooms = list(palace.iter_dirty_rooms(palace_path))
    # Only the well-formed one survives.
    assert len(rooms) == 1
    assert rooms[0][:2] == ("w", "r")
