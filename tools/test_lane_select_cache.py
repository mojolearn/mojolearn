"""lane_select.select_cached: the on-disk selection cache shared by release.py --plan,
reference_refresh --plan and the refresh run. Small git fixtures; the real selector is
replaced by a counting stub (the cache, not the selection, is under test)."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import lane_select as ls


class SelectCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = base / "repo"
        self.root.mkdir()
        self.tools = base / "tools"
        self.tools.mkdir()
        (self.tools / "lane_select.py").write_text("tool v1\n")
        self.cache = base / "cache"
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.test")
        self.git("config", "user.name", "Select Cache Test")
        (self.root / ".gitignore").write_text("gen/\n")
        (self.root / "kernel.mojo").write_text("v1\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "ref")
        self.ref = self.git("rev-parse", "HEAD")
        (self.root / "kernel.mojo").write_text("v2\n")
        self.git("commit", "-q", "-am", "head")
        self.calls = []
        self.during = None

        def select(paths, ref=None, sources=None, backend=None, stop_when=None):
            self.calls.append((tuple(paths), ref, backend, None if stop_when is None else sorted(stop_when)))
            if self.during:
                self.during()
            return dict(lanes=["lane_a"], unattributed=[], stopped_early=False, total=2,
                        reasons={"kernel.mojo": "1 lane(s)"})

        def lane_sources():
            return {"lane_a": ["kernel.mojo"], "lane_b": []}, {}

        self.select, self.lane_sources = select, lane_sources
        genuine = dict(ls._SELECT_CACHE_GENUINE, select=select, lane_sources=lane_sources)
        for target, value in ((ls, dict(ROOT=str(self.root), _SELECT_TOOL_DIR=str(self.tools),
                                         _SELECT_TOOL_DIGEST={}, _SELECT_FILE_DIGESTS={},
                                         select=select, lane_sources=lane_sources,
                                         _SELECT_CACHE_GENUINE=genuine)),):
            for name, v in value.items():
                p = patch.object(target, name, v)
                p.start()
                self.addCleanup(p.stop)
        env = patch.dict(os.environ, {ls.SELECT_CACHE_ENV: str(self.cache)})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(ls.RELEASE_BUILD_ONLY_ENV, None)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()

    def ask(self, ref=None, backend="cuda", stop_when=("lane_a",)):
        return ls.select_cached(["kernel.mojo"], ref=ref or self.ref, backend=backend,
                                stop_when=None if stop_when is None else set(stop_when))

    def entries(self):
        return sorted(p.name for p in self.cache.iterdir()) if self.cache.exists() else []

    # ------------------------------------------------------------------ hit / miss
    def test_second_call_is_a_hit_with_the_same_answer(self):
        first = self.ask()
        second = self.ask()
        self.assertEqual(first["cache"], "miss")
        self.assertEqual(second["cache"], "hit")
        self.assertEqual(len(self.calls), 1)
        for field in ls.SELECT_CACHE_FIELDS:
            self.assertEqual(first[field], second[field])
        self.assertEqual(len(self.entries()), 1)

    def test_a_new_commit_misses(self):
        self.ask()
        (self.root / "kernel.mojo").write_text("v3\n")
        self.git("commit", "-q", "-am", "next")
        self.assertEqual(self.ask()["cache"], "miss")
        self.assertEqual(len(self.calls), 2)

    def test_a_same_tree_different_commit_misses(self):
        self.ask()
        self.git("commit", "-q", "--allow-empty", "-m", "empty")
        self.assertEqual(self.ask()["cache"], "miss")
        self.assertEqual(len(self.calls), 2)

    def test_a_changed_tool_misses(self):
        self.ask()
        (self.tools / "lane_select.py").write_text("tool v2\n")
        ls._SELECT_TOOL_DIGEST.clear()          # a new process reads the tool again
        self.assertEqual(self.ask()["cache"], "miss")
        (self.tools / "helper.py").write_text("a new tool module\n")
        ls._SELECT_TOOL_DIGEST.clear()
        self.assertEqual(self.ask()["cache"], "miss")
        self.assertEqual(len(self.calls), 3)

    def test_ref_backend_stop_when_paths_and_environment_are_in_the_key(self):
        self.ask()
        self.assertEqual(self.ask(backend="hip")["cache"], "miss")
        self.assertEqual(self.ask(stop_when=("lane_a", "lane_b"))["cache"], "miss")
        self.assertEqual(self.ask(stop_when=None)["cache"], "miss")
        self.assertEqual(self.ask(ref=self.git("rev-parse", "HEAD"))["cache"], "miss")
        self.assertEqual(ls.select_cached(["kernel.mojo", "other.py"], ref=self.ref, backend="cuda",
                                          stop_when={"lane_a"})["cache"], "miss")
        with patch.dict(os.environ, {ls.RELEASE_BUILD_ONLY_ENV: "build_x"}):
            self.assertEqual(self.ask()["cache"], "miss")
        self.assertEqual(self.ask()["cache"], "hit")
        self.assertEqual(len(self.calls), 7)

    def test_an_ignored_generated_file_is_in_the_key(self):
        (self.root / "gen").mkdir()
        (self.root / "gen" / "table.mojo").write_text("generated A\n")
        self.assertEqual(self.ask()["cache"], "miss")
        self.assertEqual(self.ask()["cache"], "hit")
        (self.root / "gen" / "table.mojo").write_text("generated B\n")
        self.assertEqual(self.ask()["cache"], "miss")
        (self.root / "gen" / "table.mojo").unlink()
        self.assertEqual(self.ask()["cache"], "miss")
        # bytecode and pixi environments are never inputs
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "x.pyc").write_bytes(b"\0")
        (self.root / ".gitignore").write_text("gen/\n__pycache__/\n")
        self.git("commit", "-q", "-am", "ignore pycache")
        self.ask()
        (self.root / "__pycache__" / "x.pyc").write_bytes(b"\1")
        self.assertEqual(self.ask()["cache"], "hit")

    # ------------------------------------------------------------------ bypass
    def test_a_dirty_checkout_bypasses_and_stores_nothing(self):
        for name, dirty, clean in (
            ("tracked edit", lambda: (self.root / "kernel.mojo").write_text("edited\n"),
             lambda: self.git("checkout", "--", "kernel.mojo")),
            ("staged edit", lambda: ((self.root / "kernel.mojo").write_text("staged\n"),
                                     self.git("add", "kernel.mojo")),
             lambda: (self.git("reset", "-q", "HEAD", "kernel.mojo"), self.git("checkout", "--", "kernel.mojo"))),
            ("untracked file", lambda: (self.root / "new.py").write_text("x = 1\n"),
             lambda: (self.root / "new.py").unlink()),
        ):
            with self.subTest(name=name):
                dirty()
                count = len(self.calls)
                first, second = self.ask(), self.ask()
                self.assertTrue(first["cache"].startswith("bypass:"), first["cache"])
                self.assertTrue(second["cache"].startswith("bypass:"), second["cache"])
                self.assertEqual(len(self.calls), count + 2)
                self.assertEqual(self.entries(), [])
                clean()

    def test_an_unknown_ref_bypasses(self):
        out = self.ask(ref="0" * 40)
        self.assertTrue(out["cache"].startswith("bypass:"))
        self.assertEqual(self.entries(), [])

    def test_a_checkout_that_moves_during_the_selection_is_not_stored(self):
        self.during = lambda: (self.root / "kernel.mojo").write_text("moved\n")
        out = self.ask()
        self.assertTrue(out["cache"].startswith("bypass:"), out["cache"])
        self.assertEqual(self.entries(), [])

    def test_a_monkeypatched_selection_function_bypasses(self):
        with patch.object(ls, "_is_inert", lambda path: True):
            out = self.ask()
        self.assertTrue(out["cache"].startswith("bypass:"), out["cache"])
        self.assertEqual(self.entries(), [])

    def test_off_and_the_pytest_default_bypass(self):
        with patch.dict(os.environ, {ls.SELECT_CACHE_ENV: "off"}):
            self.assertTrue(self.ask()["cache"].startswith("bypass:"))
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "x"}):
            os.environ.pop(ls.SELECT_CACHE_ENV)
            self.assertIsNone(ls.select_cache_dir())
            self.assertTrue(self.ask()["cache"].startswith("bypass:"))
        self.assertEqual(self.entries(), [])

    def test_a_partial_or_foreign_entry_is_a_miss(self):
        self.ask()
        (entry,) = self.cache.glob("*.json")
        entry.write_text('{"key": ')
        self.assertEqual(self.ask()["cache"], "miss")
        body = json.loads(entry.read_text())
        body["key"]["ref"] = "f" * 40
        entry.write_text(json.dumps(body))
        self.assertEqual(self.ask()["cache"], "miss")
        self.assertEqual(self.ask()["cache"], "hit")
        self.assertEqual(len(self.calls), 3)

    def test_a_failed_lane_map_is_named_not_a_failed_diff(self):
        def broken():
            raise SystemExit("REFUSING: map")
        with patch.object(ls, "lane_sources", broken), \
                patch.object(ls, "_SELECT_CACHE_GENUINE", dict(ls._SELECT_CACHE_GENUINE, lane_sources=broken)):
            with self.assertRaises(ls.LaneMapFailed):
                self.ask()
        self.assertEqual(self.calls, [])

    # ------------------------------------------------------------------ concurrency
    def test_two_pipelines_racing_on_one_key_leave_one_complete_entry(self):
        gate = threading.Barrier(2)
        self.during = lambda: gate.wait(timeout=30)
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.ask())) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(sorted(r["cache"] for r in results), ["miss", "miss"])
        self.assertEqual([n for n in self.entries() if n.endswith(".json")], self.entries())
        self.assertEqual(len(self.entries()), 1)
        self.during = None
        self.assertEqual(self.ask()["cache"], "hit")

    def test_concurrent_writers_never_expose_a_partial_file(self):
        self.cache.mkdir()
        path = str(self.cache / "k.json")
        bad, stop = [], threading.Event()

        def write(n):
            for i in range(40):
                self.assertTrue(ls._select_cache_write(path, dict(writer=n, i=i, pad="x" * 200000)))

        def read():
            while not stop.is_set():
                try:
                    with open(path) as fh:
                        json.load(fh)
                except FileNotFoundError:
                    pass
                except ValueError as exc:
                    bad.append(str(exc))

        reader = threading.Thread(target=read)
        reader.start()
        writers = [threading.Thread(target=write, args=(n,)) for n in range(4)]
        for t in writers:
            t.start()
        for t in writers:
            t.join(timeout=120)
        stop.set()
        reader.join(timeout=30)
        self.assertEqual(bad, [])
        self.assertEqual(self.entries(), ["k.json"])


if __name__ == "__main__":
    unittest.main()
