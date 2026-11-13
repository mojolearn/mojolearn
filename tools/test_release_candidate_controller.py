# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""The measurement-build controller's fixes of 2026-11-13 (FOLLOWUPS 2026-11-12 22:50, 2026-11-13 09:35 and 10:45 ET):

1. concurrent controllers write the identities cache and state through per-process temp names (no rename race);
2. the leg collector polls with backoff and retries GitHub quota/"scraping"/5xx refusals, never a failed leg;
3. an artifact download is retried, and a succeeded run's failed download is DOWNLOAD PENDING (exit 75), not failed;
4. the binding-cache map in R2 outlives a crashed collector and is dropped only after a successful collection;
5. a relaunched leg whose recorded run succeeded is collected by --run-id, not dispatched again;
6. owed arm names (hyphens) are accepted as labels, and --owed-arm takes the define from the owed file;
7. a resume runs from a moved tooling ref that still holds the recorded tooling sha with the same route files.

Stdlib only; no network, no GitHub, no build. Local git repositories only for 7.

    python3 -m pytest -q tools/test_release_candidate_controller.py
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock as mock
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.argv = [sys.argv[0]]
import release  # noqa: E402
import release_candidate as RC  # noqa: E402
import release_github_build as G  # noqa: E402
import release_reuse  # noqa: E402

C = "c" * 40


def proc(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


class AtomicWrites(unittest.TestCase):
    """1: ten writers of one path never collide on a shared temp name."""

    def test_concurrent_writers_of_one_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "identities" / (C + ".json")
            errors = []

            def write(i):
                try:
                    for _ in range(20):
                        release_reuse.atomic_write_text(path, json.dumps(dict(writer=i)) + "\n")
                        release.write_json(path.with_name("state.json"), dict(writer=i))
                except Exception as exc:  # noqa: BLE001 - the test collects every failure
                    errors.append(exc)

            threads = [threading.Thread(target=write, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertIn("writer", json.loads(path.read_text()))
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), sorted([path.name, "state.json"]))

    def test_facts_cache_has_no_shared_temp_name(self):
        text = (TOOLS / "release_reuse.py").read_text()
        self.assertNotIn('cache.with_suffix(".tmp")', text)
        self.assertNotIn('path.with_name(path.name + ".tmp")', (TOOLS / "release.py").read_text())


class Polling(unittest.TestCase):
    """2: backoff, and quota refusals retried until the quota resets."""

    def test_poll_pause_backs_off_after_ten_minutes(self):
        self.assertEqual(G.poll_pause(60, None), 60)
        self.assertEqual(G.poll_pause(60, 599), 60)
        self.assertEqual(G.poll_pause(60, 601), 300)
        self.assertEqual(G.poll_pause(900, 3600), 900)

    def test_retryable_texts(self):
        for text in ("gh run view 1 exited 1: HTTP 403: API rate limit exceeded for user",
                     "HTTP 429: Too Many Requests", "HTTP 502: Bad Gateway",
                     "You have exceeded a secondary rate limit",
                     "... For more on scraping GitHub and how it may affect your rights ...",
                     "error connecting to api.github.com", "connection reset by peer"):
            self.assertTrue(G.gh_retryable(text), text)
        for text in ("HTTP 404: Not Found", "could not find any workflows named x"):
            self.assertFalse(G.gh_retryable(text), text)

    def test_gh_read_waits_for_the_quota_then_continues(self):
        calls = iter([RuntimeError("gh run view 1 exited 1: HTTP 403: API rate limit exceeded"),
                      RuntimeError("gh run view 1 exited 1: scraping"), '{"status": "completed"}'])

        def fake(*args, **kw):
            v = next(calls)
            if isinstance(v, Exception):
                raise v
            return v
        with mock.patch.object(G, "gh", side_effect=fake), \
                mock.patch.object(G, "rate_limit_wait", return_value=7) as wait, \
                mock.patch.object(G.time, "sleep") as sleep, mock.patch.object(G, "say"):
            self.assertEqual(G.gh_read("run", "view", "1"), '{"status": "completed"}')
        self.assertEqual(wait.call_count, 2)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [7, 7])

    def test_gh_read_raises_a_real_failure(self):
        with mock.patch.object(G, "gh", side_effect=RuntimeError("HTTP 404: Not Found")), \
                mock.patch.object(G.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError):
                G.gh_read("run", "view", "1")
        sleep.assert_not_called()

    def test_rate_limit_wait_reads_the_reset(self):
        reset = int(time.time()) + 120
        with mock.patch.object(G.subprocess, "run", return_value=proc(out=f"0 {reset}\n")):
            self.assertTrue(110 <= G.rate_limit_wait() <= 130)
        with mock.patch.object(G.subprocess, "run", return_value=proc(out=f"4000 {reset}\n")):
            self.assertEqual(G.rate_limit_wait(), 60)
        with mock.patch.object(G.subprocess, "run", return_value=proc(rc=1, out="")):
            self.assertEqual(G.rate_limit_wait(), 300)

    def test_wait_run_polls_slowly_once_the_run_is_old(self):
        old = "2020-01-01T00:00:00Z"
        views = iter([json.dumps(dict(status="in_progress", createdAt=old, jobs=[])),
                      json.dumps(dict(status="completed", conclusion="success", createdAt=old, jobs=[]))])
        with mock.patch.object(G, "gh_read", side_effect=lambda *a, **k: next(views)), \
                mock.patch.object(G.time, "sleep") as sleep, mock.patch.object(G, "say"):
            view = G.wait_run(1, 60, 60)
        self.assertEqual(view["conclusion"], "success")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [G.LONG_POLL_SECONDS])


class Downloads(unittest.TestCase):
    """3: retried downloads; a succeeded run never becomes a failed leg on a download error."""

    def test_six_failed_downloads_of_a_succeeded_run_are_pending(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(G.subprocess, "run", return_value=proc(1, err="connection reset by peer")) as run, \
                mock.patch.object(G.time, "sleep"), mock.patch.object(G, "say"):
            with self.assertRaises(SystemExit) as caught:
                G.download(1, "release-cuda-sm_89", td, [], "success")
        self.assertEqual(caught.exception.code, G.DOWNLOAD_PENDING_EXIT)
        self.assertEqual(run.call_count, G.DOWNLOAD_TRIES)

    def test_a_retry_that_succeeds_moves_the_artifact(self):
        attempts = []

        def fake(cmd, **kw):
            attempts.append(cmd)
            if len(attempts) < 3:
                return proc(1, err="unexpected EOF")
            dest = Path(cmd[cmd.index("-D") + 1])
            (dest / "leg").mkdir(parents=True, exist_ok=True)
            (dest / "leg" / "proof.json").write_text("{}")
            return proc(0)
        with tempfile.TemporaryDirectory() as td, mock.patch.object(G.subprocess, "run", side_effect=fake), \
                mock.patch.object(G.time, "sleep"), mock.patch.object(G, "say"):
            G.download(1, "release-cuda-sm_89", Path(td) / "out", [], "success")
            self.assertTrue((Path(td) / "out" / "proof.json").is_file())
        self.assertEqual(len(attempts), 3)

    def test_a_missing_artifact_of_a_failed_run_fails_at_once(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(G.subprocess, "run", return_value=proc(1, err="no valid artifacts found")) as run, \
                mock.patch.object(G.time, "sleep"):
            with self.assertRaises(SystemExit) as caught:
                G.download(1, "release-cuda-sm_89", td, [("assemble", "failure", "", "", 0, "u")], "failure")
        self.assertIsInstance(caught.exception.code, str)
        self.assertEqual(run.call_count, 1)


class R2MapLifetime(unittest.TestCase):
    """4: the map is deleted only after a successful collection."""

    def _attached(self, td, wait):
        out = Path(td) / "leg"
        (out / "GHA").mkdir(parents=True)
        (out / "GHA" / "dispatch.json").write_text(json.dumps(dict(tag="gha-T", r2_objects=["bincache/gha-maps/gha-T.tsv"])))
        run = json.dumps(dict(databaseId=5, displayTitle="release gha-T", headSha=C, url="u"))
        r2 = mock.MagicMock(return_value=(204, b""))
        with mock.patch.object(G, "read_creds", return_value={"R2_BUCKET": "b"}), \
                mock.patch.object(G, "gh_read", return_value=run), mock.patch.object(G, "wait_run", side_effect=wait), \
                mock.patch.object(G, "download"), mock.patch.object(G, "verify_tree", return_value=(True, "ok")), \
                mock.patch.object(G, "promote", return_value="none"), mock.patch.object(G, "say"), \
                mock.patch("bincache.r2_request", r2):
            try:
                rc = G.main(["run", "--commit", C, "--arch", "sm_89", "--out", str(out), "--run-id", "5"])
            except RuntimeError:
                rc = "crashed"
        return rc, r2

    def test_a_crashed_collector_leaves_the_map(self):
        with tempfile.TemporaryDirectory() as td:
            rc, r2 = self._attached(td, RuntimeError("collector died"))
        self.assertEqual(rc, "crashed")
        r2.assert_not_called()

    def test_a_successful_collection_drops_the_map(self):
        view = dict(status="completed", conclusion="success", jobs=[], url="u")
        with tempfile.TemporaryDirectory() as td:
            rc, r2 = self._attached(td, lambda *a, **k: view)
        self.assertEqual(rc, 0)
        self.assertEqual([c.args[1:3] for c in r2.call_args_list], [("DELETE", "bincache/gha-maps/gha-T.tsv")])

    def test_no_finally_delete_around_the_wait(self):
        text = (TOOLS / "release_github_build.py").read_text()
        body = text[text.index("def cmd_run(a):"):text.index("def next_steps(")]
        self.assertNotIn("finally:", body)


class AttachBySucceededRun(unittest.TestCase):
    """5: a relaunch collects the recorded run that succeeded."""

    def _gha(self, td, defines="A=1"):
        gha = Path(td) / "legs" / "cuda-sm_89" / "GHA"
        gha.mkdir(parents=True)
        (gha / "run-id.json").write_text(json.dumps(dict(databaseId=9, displayTitle="release gha-T")))
        (gha / "dispatch.json").write_text(json.dumps(dict(tag="gha-T", source_commit=C, defines=defines,
                                                           r2_objects=["k"])))
        return gha

    def test_succeeded_run(self):
        ok = json.dumps(dict(status="completed", conclusion="success", displayTitle="release gha-T"))
        bad = json.dumps(dict(status="completed", conclusion="failure", displayTitle="release gha-T"))
        with tempfile.TemporaryDirectory() as td:
            gha = self._gha(td)
            with mock.patch.object(G, "gh_read", return_value=ok):
                self.assertEqual(G.succeeded_run(gha, C, "A=1"), 9)
                self.assertIsNone(G.succeeded_run(gha, C, ""))            # another define set
                self.assertIsNone(G.succeeded_run(gha, "d" * 40, "A=1"))  # another commit
            with mock.patch.object(G, "gh_read", return_value=bad):
                self.assertIsNone(G.succeeded_run(gha, C, "A=1"))

    def test_launch_detached_attaches_instead_of_dispatching(self):
        with tempfile.TemporaryDirectory() as td:
            gha = self._gha(td)
            out = gha.parent
            leg = release.Leg("cuda-sm_89", "cuda", "sm_89",
                              ["python3", "tools/release_github_build.py", "run", "--commit", C, "--arch", "sm_89",
                               "--out", str(out), "--defines", "A=1"], {}, out / "cuda-sm_89" / "release-build",
                              Path(td) / "legs", out)
            said, spawned = [], []
            ctx = SimpleNamespace(commit=C, dry=False, args=SimpleNamespace(candidate_defines="A=1"), say=said.append,
                                  spawn=lambda cmd, env: spawned.append(cmd) or 4242)
            with mock.patch.object(release.release_github_build, "succeeded_run", return_value=9):
                self.assertEqual(release.launch_detached(ctx, [leg]), 1)
            self.assertEqual(leg.command[-2:], ["--run-id", "9"])
            self.assertIn("--run-id 9", spawned[0][-1])
            self.assertTrue((out / "GHA" / "dispatch.json").is_file())
            self.assertEqual(len(list(out.parent.glob("cuda-sm_89.failed-*"))), 1)

    def test_download_pending_is_a_relaunch_not_a_failure(self):
        self.assertEqual(G.DOWNLOAD_PENDING_EXIT, 75)
        text = (TOOLS / "release.py").read_text()
        self.assertIn("release_github_build.DOWNLOAD_PENDING_EXIT", text[text.index("def step_linux_wait"):])
        leg = release.Leg("cuda-sm_89", "cuda", "sm_89", ["python3", "tools/release_github_build.py", "run",
                                                          "--run-id", "12"], {}, "rb", "w", "o")
        self.assertEqual(release.github_attach_run(SimpleNamespace(dry=False), leg), "12")


class OwedArms(unittest.TestCase):
    """6: owed arm names are labels; --owed-arm takes the define."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.owed = Path(self.td.name) / "owed.json"
        self.owed.write_text(json.dumps(dict(b_arms={
            "small-mlp-generic": dict(define="MOJOLEARN_SMALL_MLP_GENERIC_OPS"),
            "em-check-1": dict(define="MOJOLEARN_EM_LOOP_CHECK_EVERY=1")})))

    def tearDown(self):
        self.td.cleanup()

    def test_label_with_hyphens(self):
        self.assertEqual(RC.resolve_owed("small-mlp-generic", "", "", str(self.owed)),
                         ("small.mlp.generic", "MOJOLEARN_SMALL_MLP_GENERIC_OPS", "small-mlp-generic"))
        self.assertEqual(RC.resolve_owed("small.mlp.generic", "", "MOJOLEARN_SMALL_MLP_GENERIC_OPS", str(self.owed)),
                         ("small.mlp.generic", "MOJOLEARN_SMALL_MLP_GENERIC_OPS", "small-mlp-generic"))
        self.assertTrue(RC.LABEL_RE.fullmatch(RC.resolve_owed("em-check-1", "", "", str(self.owed))[0]))

    def test_owed_arm_takes_the_define(self):
        self.assertEqual(RC.resolve_owed("", "em-check-1", "", str(self.owed)),
                         ("em.check.1", "MOJOLEARN_EM_LOOP_CHECK_EVERY=1", "em-check-1"))

    def test_refusals(self):
        with self.assertRaises(SystemExit):
            RC.resolve_owed("", "gbdtoff", "", str(self.owed))                           # not owed
        with self.assertRaises(SystemExit):
            RC.resolve_owed("", "em-check-1", "MOJOLEARN_EM_LOOP_CHECK_EVERY=8", str(self.owed))
        with self.assertRaises(SystemExit):
            RC.resolve_owed("small-mlp-generic", "em-check-1", "", str(self.owed))

    def test_plain_labels_are_unchanged(self):
        self.assertEqual(RC.resolve_owed("em1", "", "A=1", str(self.owed)), ("em1", "A=1", None))
        self.assertEqual(RC.resolve_owed("", "", "", str(self.owed)), ("", "", None))
        self.assertEqual(RC.resolve_owed("my-run", "", "A=1", str(self.owed)), ("my.run", "A=1", None))

    def test_the_parser_takes_owed_arm(self):
        a = RC.parse(["--commit", C, "--owed-arm", "em-check-1"])
        self.assertEqual(a.owed_arm, "em-check-1")


class RecordedTooling(unittest.TestCase):
    """7: a moved tooling ref is accepted while it holds the recorded sha and the same route files."""

    def git(self, cwd, *args):
        return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        base = Path(self.td.name)
        self.origin, self.tool, self.other = base / "origin.git", base / "tool", base / "other"
        env = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)], check=True)
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.tool)], check=True, capture_output=True)
        (self.tool / "tools").mkdir()
        (self.tool / "tools" / "release_github_build.py").write_text("v1\n")
        (self.tool / "README").write_text("a\n")
        self.git(self.tool, "add", "-A")
        self.git(self.tool, "commit", "-qm", "one")
        self.git(self.tool, "push", "-q", "origin", "HEAD:main")
        self.recorded = self.git(self.tool, "rev-parse", "HEAD")
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.other)], check=True, capture_output=True)
        self.root = mock.patch.object(G, "ROOT", self.tool)
        self.root.start()

    def tearDown(self):
        self.root.stop()
        self.env.stop()
        self.td.cleanup()

    def advance(self, path, text):
        (self.other / path).write_text(text)
        self.git(self.other, "commit", "-qam", "more")
        self.git(self.other, "push", "-q", "origin", "HEAD:main")
        return self.git(self.other, "rev-parse", "HEAD")

    def test_unmoved_ref(self):
        self.assertEqual(G.tooling_ref("main"), ("main", self.recorded))

    def test_moved_ref_with_the_same_route_files(self):
        tip = self.advance("README", "b\n")
        with self.assertRaises(SystemExit):
            G.tooling_ref("main")                    # no recorded sha: "push it", as before
        with mock.patch.object(G, "say"):
            self.assertEqual(G.tooling_ref("main", self.recorded), ("main", tip))

    def test_moved_ref_with_changed_route_files_is_refused(self):
        self.advance("tools/release_github_build.py", "v2\n")
        with self.assertRaises(SystemExit) as caught:
            G.tooling_ref("main", self.recorded)
        self.assertIn("route files", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
