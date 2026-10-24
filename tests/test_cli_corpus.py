"""CLI coverage for the mneme-007 corpus verbs: `mine --files-from`, `sources`, `forget`."""

import io
import json
import os

import chromadb
import pytest

from mempalace import cli


def _run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["mempalace", *argv])
    cli.main()


def _seed(palace_path, rows):
    """rows: (id, wing, source_file, mtime|None). Returns nothing; closes the client."""
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
    metas = []
    for _i, wing, src, mtime in rows:
        meta = {"wing": wing, "room": "general", "source_file": src, "chunk_index": 0}
        if mtime is not None:
            meta["source_mtime"] = mtime
        metas.append(meta)
    col.add(ids=[r[0] for r in rows], documents=[f"doc {r[0]}" for r in rows], metadatas=metas)
    del client


def _ids(palace_path):
    client = chromadb.PersistentClient(path=palace_path)
    try:
        return set(client.get_collection("mempalace_drawers").get(include=[])["ids"])
    finally:
        del client


@pytest.fixture
def seeded(palace_path):
    _seed(
        palace_path,
        [
            ("a1", "w1", "/corp/a.md", 100.0),
            ("a2", "w1", "/corp/a.md", 200.0),
            ("b1", "w1", "/corp/b.md", 50.0),
            ("a3", "w2", "/corp/a.md", None),
            ("c1", "w2", "/corp/c.md", 7.0),
        ],
    )
    return palace_path


class TestFilesFrom:
    @pytest.fixture
    def captured(self, monkeypatch):
        calls = []
        monkeypatch.setattr("mempalace.miner.mine", lambda **kw: calls.append(kw))
        return calls

    def _tree(self, tmp_path):
        root = tmp_path / "corpus"
        (root / "sub").mkdir(parents=True)
        (root / "a.md").write_text("alpha")
        (root / "sub" / "b.txt").write_text("beta")
        (root / "pic.png").write_bytes(b"\x89PNG")
        (root / "real.md").write_text("real")
        (root / "link.md").symlink_to(root / "real.md")
        (root / "adir.md").mkdir()
        (root / ".gitignore").write_text("a.md\n")  # ignore files must not apply
        return root

    def test_happy_path_passes_resolved_paths(self, monkeypatch, tmp_path, palace_path, captured):
        root = self._tree(tmp_path)
        lst = tmp_path / "list.txt"
        lst.write_text(f"a.md\n\n  \n{root / 'sub' / 'b.txt'}\na.md\n")
        _run(
            monkeypatch,
            "--palace",
            palace_path,
            "mine",
            str(root),
            "--files-from",
            str(lst),
            "--wing",
            "w",
            "--dry-run",
        )
        assert len(captured) == 1
        kw = captured[0]
        assert kw["files"] == [root.resolve() / "a.md", root.resolve() / "sub" / "b.txt"]
        assert kw["wing_override"] == "w"
        assert kw["dry_run"] is True

    def test_stdin(self, monkeypatch, tmp_path, palace_path, captured):
        root = self._tree(tmp_path)
        monkeypatch.setattr("sys.stdin", io.StringIO("a.md\n"))
        _run(monkeypatch, "--palace", palace_path, "mine", str(root), "--files-from", "-")
        assert captured[0]["files"] == [root.resolve() / "a.md"]

    def test_skips_are_reported_not_dropped(
        self, monkeypatch, tmp_path, palace_path, captured, capsys
    ):
        root = self._tree(tmp_path)
        lst = tmp_path / "list.txt"
        lst.write_text("a.md\nmissing.md\nlink.md\npic.png\nadir.md\n../outside.md\n")
        _run(monkeypatch, "--palace", palace_path, "mine", str(root), "--files-from", str(lst))
        assert captured[0]["files"] == [root.resolve() / "a.md"]
        err = capsys.readouterr().err
        assert "SKIP: missing.md (missing)" in err
        assert "SKIP: link.md (symlink)" in err
        assert "SKIP: pic.png (unsupported extension .png)" in err
        assert "SKIP: adir.md (not a regular file)" in err
        assert "SKIP: ../outside.md (outside the mined directory)" in err

    @pytest.mark.parametrize(
        "flag", [["--include-ignored", "x"], ["--no-gitignore"], ["--no-mempalaceignore"]]
    )
    def test_mutually_exclusive(self, monkeypatch, tmp_path, palace_path, captured, capsys, flag):
        root = self._tree(tmp_path)
        lst = tmp_path / "list.txt"
        lst.write_text("a.md\n")
        with pytest.raises(SystemExit) as exc:
            _run(
                monkeypatch,
                "--palace",
                palace_path,
                "mine",
                str(root),
                "--files-from",
                str(lst),
                *flag,
            )
        assert exc.value.code == 2
        assert "--files-from cannot be combined" in capsys.readouterr().err
        assert captured == []

    def test_not_forwarded_to_hub(self, monkeypatch, tmp_path, palace_path, captured):
        root = self._tree(tmp_path)
        lst = tmp_path / "list.txt"
        lst.write_text("a.md\n")
        called = []
        monkeypatch.setattr(cli, "_forward_mine_to_hub", lambda *a, **k: called.append(1) or True)
        _run(monkeypatch, "--palace", palace_path, "mine", str(root), "--files-from", str(lst))
        assert called == [] and len(captured) == 1

    def test_unreadable_list_is_usage_error(self, monkeypatch, tmp_path, palace_path, captured):
        root = self._tree(tmp_path)
        with pytest.raises(SystemExit) as exc:
            _run(
                monkeypatch,
                "--palace",
                palace_path,
                "mine",
                str(root),
                "--files-from",
                str(tmp_path / "nope.txt"),
            )
        assert exc.value.code == 2


class TestSources:
    def test_json_rows_sorted(self, monkeypatch, seeded, capsys):
        _run(monkeypatch, "--palace", seeded, "sources", "--json")
        rows = json.loads(capsys.readouterr().out)
        assert rows == [
            {"wing": "w1", "source_file": "/corp/a.md", "source_mtime": 200.0, "drawers": 2},
            {"wing": "w1", "source_file": "/corp/b.md", "source_mtime": 50.0, "drawers": 1},
            {"wing": "w2", "source_file": "/corp/a.md", "source_mtime": None, "drawers": 1},
            {"wing": "w2", "source_file": "/corp/c.md", "source_mtime": 7.0, "drawers": 1},
        ]

    def test_wing_filter(self, monkeypatch, seeded, capsys):
        _run(monkeypatch, "--palace", seeded, "sources", "--wing", "w2", "--json")
        rows = json.loads(capsys.readouterr().out)
        assert {r["wing"] for r in rows} == {"w2"} and len(rows) == 2

    def test_read_only(self, monkeypatch, seeded, capsys):
        before = _ids(seeded)
        _run(monkeypatch, "--palace", seeded, "sources", "--json")
        assert _ids(seeded) == before

    def test_human_output(self, monkeypatch, seeded, capsys):
        _run(monkeypatch, "--palace", seeded, "sources")
        assert "/corp/b.md" in capsys.readouterr().out

    def test_unopenable_store_fails_without_json(self, monkeypatch, tmp_path, capsys):
        missing = str(tmp_path / "nope")
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--palace", missing, "sources", "--json")
        assert exc.value.code != 0
        out = capsys.readouterr()
        assert out.out == ""
        assert out.err

    def test_corrupt_store_fails_without_json(self, monkeypatch, palace_path, capsys):
        with open(os.path.join(palace_path, "chroma.sqlite3"), "wb") as fh:
            fh.write(b"this is not sqlite" * 100)
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--palace", palace_path, "sources", "--json")
        assert exc.value.code != 0
        assert capsys.readouterr().out == ""


class TestForget:
    def _list(self, tmp_path, *lines):
        f = tmp_path / "src.txt"
        f.write_text("\n".join(lines) + "\n")
        return str(f)

    def test_dry_run_deletes_nothing(self, monkeypatch, seeded, tmp_path, capsys):
        before = _ids(seeded)
        lst = self._list(tmp_path, "/corp/a.md", "/corp/none.md")
        _run(monkeypatch, "--palace", seeded, "forget", "--sources-from", lst)
        out = capsys.readouterr().out
        assert out.splitlines()[:3] == ["Sources: 1", "Drawers: 3", "  /corp/a.md  (3)"]
        assert _ids(seeded) == before

    def test_apply_deletes_only_listed(self, monkeypatch, seeded, tmp_path, capsys):
        lst = self._list(tmp_path, "/corp/a.md", "/corp/c.md")
        _run(monkeypatch, "--palace", seeded, "forget", "--sources-from", lst, "--apply")
        out = capsys.readouterr().out
        assert out.splitlines() == [
            "Sources: 2",
            "Drawers: 4",
            "  /corp/a.md  (3)",
            "  /corp/c.md  (1)",
        ]
        assert _ids(seeded) == {"b1"}

    def test_apply_respects_wing(self, monkeypatch, seeded, tmp_path, capsys):
        lst = self._list(tmp_path, "/corp/a.md")
        _run(
            monkeypatch,
            "--palace",
            seeded,
            "forget",
            "--sources-from",
            lst,
            "--wing",
            "w1",
            "--apply",
        )
        out = capsys.readouterr().out
        assert out.splitlines() == ["Sources: 1", "Drawers: 2", "  /corp/a.md  (2)"]
        assert _ids(seeded) == {"b1", "a3", "c1"}

    def test_stdin_and_relative_paths(self, monkeypatch, seeded, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin", io.StringIO("\nrel.md\n"))
        _seed(seeded, [("r1", "w1", str(tmp_path / "rel.md"), 1.0)])
        _run(monkeypatch, "--palace", seeded, "forget", "--sources-from", "-", "--apply")
        out = capsys.readouterr().out
        assert out.splitlines()[:2] == ["Sources: 1", "Drawers: 1"]
        assert "r1" not in _ids(seeded) and "b1" in _ids(seeded)

    def test_missing_palace_fails(self, monkeypatch, tmp_path, capsys):
        lst = self._list(tmp_path, "/corp/a.md")
        with pytest.raises(SystemExit) as exc:
            _run(
                monkeypatch,
                "--palace",
                str(tmp_path / "nope"),
                "forget",
                "--sources-from",
                lst,
                "--apply",
            )
        assert exc.value.code != 0
        assert "Sources:" not in capsys.readouterr().out
