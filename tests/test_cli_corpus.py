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


class _Proxy:
    """Delegating collection wrapper that lets a test override single methods."""

    def __init__(self, real, **overrides):
        self._real = real
        self._overrides = overrides

    def __getattr__(self, name):
        return self._overrides.get(name, getattr(self._real, name))


class TestSkipsJson:
    BODY = "The quick brown fox jumps over the lazy dog. " * 120  # many chunks

    def _root(self, tmp_path):
        root = tmp_path / "corpus"
        root.mkdir()
        return root

    def _mine(self, monkeypatch, palace_path, root, listing, out, *extra):
        lst = root.parent / "list.txt"
        lst.write_text(listing)
        _run(
            monkeypatch,
            "--palace",
            palace_path,
            "mine",
            str(root),
            "--files-from",
            str(lst),
            "--skips-json",
            str(out),
            "--wing",
            "w",
            "--workers",
            "1",
            *extra,
        )
        return json.loads(out.read_text())

    @staticmethod
    def _by_path(records):
        return {r["path"]: r for r in records}

    def test_listed_but_unminable_cmd_level_codes(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "real.md").write_text(self.BODY)
        (root / "link.md").symlink_to(root / "real.md")
        (root / "adir.md").mkdir()
        (root / "pic.png").write_bytes(b"\x89PNG")
        (root / "denied.md").write_text(self.BODY)
        orig_stat = type(root).stat

        def fake_stat(self, *a, **k):
            if self.name == "denied.md" and k.get("follow_symlinks", True):
                raise PermissionError(13, "Permission denied")
            return orig_stat(self, *a, **k)

        monkeypatch.setattr(type(root), "stat", fake_stat)
        out = tmp_path / "skips.json"
        recs = self._mine(
            monkeypatch,
            palace_path,
            root,
            "real.md\nmissing.md\nlink.md\nadir.md\npic.png\ndenied.md\n../outside.md\n",
            out,
        )
        got = {r["path"]: r["reason"] for r in recs}
        assert got == {
            "missing.md": "missing",
            "link.md": "symlink",
            "adir.md": "not_regular",
            "pic.png": "unsupported_ext",
            "denied.md": "stat_error",
            "../outside.md": "outside_dir",
        }
        assert all(set(r) == {"path", "reason", "detail"} for r in recs)
        assert "Permission denied" in self._by_path(recs)["denied.md"]["detail"]

    def test_too_large(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "big.md").write_text(self.BODY)
        monkeypatch.setattr("mempalace.miner.MAX_FILE_SIZE", 100)
        recs = self._mine(monkeypatch, palace_path, root, "big.md\n", tmp_path / "o.json")
        assert [(r["path"], r["reason"]) for r in recs] == [("big.md", "too_large")]

    def test_chunk_cap_and_stderr_unchanged(self, monkeypatch, tmp_path, palace_path, capsys):
        root = self._root(tmp_path)
        (root / "long.md").write_text(self.BODY)
        recs = self._mine(
            monkeypatch,
            palace_path,
            root,
            "long.md\n",
            tmp_path / "o.json",
            "--max-chunks-per-file",
            "1",
        )
        assert [(r["path"], r["reason"]) for r in recs] == [("long.md", "chunk_cap")]
        assert "! [skip] long.md" in capsys.readouterr().err

    def test_too_short(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "tiny.md").write_text("x")
        recs = self._mine(monkeypatch, palace_path, root, "tiny.md\n", tmp_path / "o.json")
        assert [(r["path"], r["reason"]) for r in recs] == [("tiny.md", "too_short")]

    def test_same_name_in_different_dirs_reports_exact_path(
        self, monkeypatch, tmp_path, palace_path
    ):
        root = self._root(tmp_path)
        (root / "a").mkdir()
        (root / "b").mkdir()
        (root / "a" / "notes.md").write_text(self.BODY)
        (root / "b" / "notes.md").write_text("x")
        recs = self._mine(
            monkeypatch, palace_path, root, "a/notes.md\nb/notes.md\n", tmp_path / "o.json"
        )
        assert [(r["path"], r["reason"]) for r in recs] == [("b/notes.md", "too_short")]

    def test_same_name_chunk_cap_path_not_basename(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "a").mkdir()
        (root / "b").mkdir()
        (root / "a" / "n.md").write_text("short but valid " * 5)
        (root / "b" / "n.md").write_text(self.BODY)
        recs = self._mine(
            monkeypatch,
            palace_path,
            root,
            "a/n.md\nb/n.md\n",
            tmp_path / "o.json",
            "--max-chunks-per-file",
            "2",
        )
        assert [(r["path"], r["reason"]) for r in recs] == [("b/n.md", "chunk_cap")]

    def test_excluded_and_limit(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "mempalace.yaml").write_text(
            "wing: w\nrooms:\n- name: general\n  description: x\nexclude_patterns:\n- skipme.md\n"
        )
        for n in ("skipme.md", "one.md", "two.md"):
            (root / n).write_text(self.BODY)
        recs = self._mine(
            monkeypatch,
            palace_path,
            root,
            "skipme.md\none.md\ntwo.md\n",
            tmp_path / "o.json",
            "--limit",
            "1",
        )
        assert {r["path"]: r["reason"] for r in recs} == {
            "skipme.md": "excluded",
            "two.md": "limit",
        }

    def _real_collection_proxy(self, monkeypatch, **overrides):
        import mempalace.miner as miner

        real_get = miner.get_collection
        monkeypatch.setattr(
            miner, "get_collection", lambda *a, **k: _Proxy(real_get(*a, **k), **overrides)
        )

    def test_purge_skipped(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "f.md").write_text(self.BODY)
        out = tmp_path / "o.json"
        assert self._mine(monkeypatch, palace_path, root, "f.md\n", out) == []
        (root / "f.md").write_text(self.BODY + " changed " * 50)
        os.utime(root / "f.md", (1, 1))  # force an mtime mismatch -> re-mine

        def boom(**kw):
            raise RuntimeError("delete refused")

        self._real_collection_proxy(monkeypatch, delete=boom)
        recs = self._mine(monkeypatch, palace_path, root, "f.md\n", out)
        assert [(r["path"], r["reason"]) for r in recs] == [("f.md", "purge_skipped")]
        assert "delete refused" in recs[0]["detail"]

    def test_write_failed(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "f.md").write_text(self.BODY)

        def boom(**kw):
            raise RuntimeError("upsert refused")

        self._real_collection_proxy(monkeypatch, upsert=boom)
        recs = self._mine(monkeypatch, palace_path, root, "f.md\n", tmp_path / "o.json")
        assert [(r["path"], r["reason"]) for r in recs] == [("f.md", "write_failed")]

    def test_producer_failure_is_failed(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "sub").mkdir()
        (root / "sub" / "f.md").write_text(self.BODY)

        def boom(*a, **k):
            raise RuntimeError("embed down")

        monkeypatch.setattr("mempalace.miner._embed_prepared", boom)
        recs = self._mine(monkeypatch, palace_path, root, "sub/f.md\n", tmp_path / "o.json")
        assert [(r["path"], r["reason"]) for r in recs] == [("sub/f.md", "failed")]
        assert "embed down" in recs[0]["detail"]

    def test_requires_files_from(self, monkeypatch, tmp_path, palace_path, capsys):
        root = self._root(tmp_path)
        out = tmp_path / "o.json"
        with pytest.raises(SystemExit) as exc:
            _run(
                monkeypatch,
                "--palace",
                palace_path,
                "mine",
                str(root),
                "--skips-json",
                str(out),
            )
        assert exc.value.code == 2
        assert "--skips-json requires --files-from" in capsys.readouterr().err
        assert not out.exists()

    def test_empty_array_when_nothing_skipped(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "ok.md").write_text(self.BODY)
        out = tmp_path / "o.json"
        assert self._mine(monkeypatch, palace_path, root, "ok.md\n", out) == []
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".skips-")]

    def test_atomic_write_leaves_old_file_and_no_temp_on_failure(
        self, monkeypatch, tmp_path, palace_path, capsys
    ):
        root = self._root(tmp_path)
        (root / "ok.md").write_text(self.BODY)
        out = tmp_path / "o.json"
        out.write_text('["previous"]')

        def boom(*a, **k):
            raise OSError("replace failed")

        monkeypatch.setattr(os, "replace", boom)
        lst = tmp_path / "list.txt"
        lst.write_text("ok.md\nmissing.md\n")
        _run(
            monkeypatch,
            "--palace",
            palace_path,
            "mine",
            str(root),
            "--files-from",
            str(lst),
            "--skips-json",
            str(out),
            "--dry-run",
        )
        assert out.read_text() == '["previous"]'
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".skips-")]
        assert "cannot write --skips-json" in capsys.readouterr().err

    def test_written_when_mine_fails_partway(self, monkeypatch, tmp_path, palace_path):
        root = self._root(tmp_path)
        (root / "ok.md").write_text(self.BODY)
        out = tmp_path / "o.json"

        def boom(*a, **k):
            raise RuntimeError("palace exploded")

        monkeypatch.setattr("mempalace.miner.get_collection", boom)
        lst = tmp_path / "list.txt"
        lst.write_text("ok.md\nmissing.md\n")
        with pytest.raises(RuntimeError, match="palace exploded"):
            _run(
                monkeypatch,
                "--palace",
                palace_path,
                "mine",
                str(root),
                "--files-from",
                str(lst),
                "--skips-json",
                str(out),
            )
        assert [(r["path"], r["reason"]) for r in json.loads(out.read_text())] == [
            ("missing.md", "missing")
        ]

    def test_help_lists_flag(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "mine", "--help")
        assert exc.value.code == 0
        assert "--skips-json" in capsys.readouterr().out


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
