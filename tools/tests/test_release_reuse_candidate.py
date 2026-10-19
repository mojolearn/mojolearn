# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/release_reuse.py's choice of the build reused bytes come from: the
newest PyPI record or a published candidate (a TestPyPI freeze under
<evidence>/release/<version>/<commit12>/). Fake records only: CPU, no
compiler, no network.

    python -m pytest -q tools/tests/test_release_reuse_candidate.py
"""
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tests"))
sys.path.insert(0, str(ROOT / "packaging" / "linux"))
import release_reuse as rr  # noqa: E402
from test_release_reuse import tiny_repo, sha  # noqa: E402

PLATFORMS = ("linux", "macos", "nvidia", "amd")
XCODE = dict(xcode="16")


def wheel_name(platform, version):
    if platform == "macos":
        return f"mojolearn-{version}-py3-none-macosx_11_0_arm64.whl"
    pkg = "mojolearn" if platform == "linux" else f"mojolearn_{platform}"
    return f"{pkg}-{version}-py3-none-manylinux_2_35_x86_64.whl"


def write_candidate(ev, version, commit, rows, runtime, channel="testpypi", published=PLATFORMS,
                    plan_commit=None, manifest_commit=None, done=True):
    """A freeze of `version` at `commit` as tools/release.py leaves it: state.json,
    reuse/plan.json and publish-<platform>/artifact/alpha-manifest.json."""
    d = pathlib.Path(ev) / "release" / version / commit[:12]
    (d / "reuse").mkdir(parents=True, exist_ok=True)
    (d / "reuse" / "plan.json").write_text(json.dumps(dict(
        schema=rr.PLAN_SCHEMA, commit=plan_commit or commit, rows=rows, runtime=dict(identity=runtime))))
    steps = {}
    for platform in published:
        whl = wheel_name(platform, version)
        art = d / f"publish-{platform}" / "artifact"
        art.mkdir(parents=True, exist_ok=True)
        (art / "alpha-manifest.json").write_text(json.dumps(dict(
            version=version, files={whl: sha(whl.encode())},
            qualification_reuse=dict(source_commit=manifest_commit or commit))))
        steps[rr.PUBLISH_STEP.get(platform, "publish-" + platform)] = dict(
            done=done, result=f"{channel} via alpha-api-{version}-{platform} (source {commit[:12]})")
    steps["record"] = dict(done=True, result="skipped (not published to PyPI)")
    (d.parent / "state.json").write_text(json.dumps(dict(commit=commit, version=version, steps=steps)))
    return d


class CandidateDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.ev = pathlib.Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def found(self):
        return rr.candidate_builds(self.ev, platforms=PLATFORMS)

    def test_a_testpypi_freeze_is_a_candidate_with_its_wheels(self):
        d = write_candidate(self.ev, "0.8.47", "9" * 40, [], {})
        (c,) = self.found()
        self.assertEqual((c["version"], c["source_commit"], c["channel"]), ("0.8.47", "9" * 40, "testpypi"))
        self.assertEqual(c["identities"], str(d / "reuse" / "plan.json"))
        self.assertEqual(c["build_dir"], str(d))
        for p in PLATFORMS:
            self.assertEqual(c[p], dict(wheel=wheel_name(p, "0.8.47"), sha256=sha(wheel_name(p, "0.8.47").encode())))

    def test_unpublished_or_mismatched_freezes_are_not_candidates(self):
        write_candidate(self.ev, "0.8.47", "1" * 40, [], {}, channel="skipped")
        write_candidate(self.ev, "0.8.48", "2" * 40, [], {}, done=False)
        write_candidate(self.ev, "0.8.49", "3" * 40, [], {}, plan_commit="4" * 40)
        write_candidate(self.ev, "0.8.50", "5" * 40, [], {}, manifest_commit="6" * 40)
        self.assertEqual(self.found(), [])

    def test_a_platform_not_published_is_none_and_candidates_sort_newest_first(self):
        write_candidate(self.ev, "0.8.9", "1" * 40, [], {}, published=("linux",))
        write_candidate(self.ev, "0.8.10", "2" * 40, [], {})
        got = self.found()
        self.assertEqual([c["version"] for c in got], ["0.8.10", "0.8.9"])
        self.assertIsNone(got[1]["macos"])
        self.assertIsNotNone(got[1]["linux"])

    def test_sources_keep_only_candidates_newer_than_pypi_and_not_this_commit(self):
        prev = dict(version="0.8.46", source_commit="a" * 40, channel="pypi")
        cands = [dict(version="0.8.47", source_commit="b" * 40), dict(version="0.8.46", source_commit="c" * 40),
                 dict(version="0.8.48", source_commit="d" * 40), dict(version="0.8.45", source_commit="e" * 40)]
        got = rr.reuse_sources(prev, cands, "d" * 40)
        self.assertEqual([s["version"] for s in got], ["0.8.47", "0.8.46"])
        self.assertIs(got[-1], prev)
        self.assertEqual(rr.reuse_sources(None, [], "d" * 40), [])


class CandidateSelectionTests(unittest.TestCase):
    """make_plan with a fake PyPI record and fake candidate freezes."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        (self.tmp / "repo").mkdir()
        self.repo, self.prev, self.cur, self.env = tiny_repo(self.tmp / "repo")
        self.cache = self.tmp / "cache"
        self.ev = self.tmp / "ev"
        for target in (rr.LINUX, rr.MACOS):
            for b in rr.bindings(target):
                p = self.repo / "bindings" / b.script
                if not p.exists():
                    p.write_text(f"mojo build k.mojo -o python/mojolearn/x/{b.name}.so\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True, env=self.env)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "all scripts"], check=True, env=self.env)
        self.cur = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
        self.facts = rr.facts_for_commit(self.cur, self.cache, self.repo)
        self.n_linux = len(rr.bindings(rr.LINUX))

    def tearDown(self):
        self._t.cleanup()

    def rows(self, changed=(), reasons=None):
        """Identity rows of this tree, with `changed` binding names moved."""
        out = []
        for target in (rr.LINUX, rr.MACOS):
            for b in rr.bindings(target):
                ident = rr.identity(b, self.facts, XCODE)
                if b.name in changed:
                    ident = dict(ident, closure=dict(scope="closure", digest="9" * 64))
                out.append(dict(key=b.key, identity=ident, reason=(reasons or {}).get(b.name, "changed: closure")))
        return out

    def pypi(self, changed):
        path = self.tmp / "binding-identities.json"
        path.write_text(json.dumps(dict(rows=self.rows(changed), runtime=dict(identity=rr.runtime_identity(self.facts)))))
        return dict(version="0.8.46", source_commit=self.prev, identities=str(path), channel="pypi",
                    linux=dict(wheel=wheel_name("linux", "0.8.46"), sha256="0" * 64),
                    macos=dict(wheel=wheel_name("macos", "0.8.46"), sha256="1" * 64))

    def candidate(self, changed=(), reasons=None, version="0.8.47"):
        write_candidate(self.ev, version, self.prev, self.rows(changed, reasons), rr.runtime_identity(self.facts))

    def plan(self, prev, candidates=None):
        return rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=XCODE, prev=prev,
                            evidence_root=self.ev if candidates is None else None, candidates=candidates)

    def builds(self, plan):
        return sorted({r["name"] for r in rr.plan_rows(plan, decision="BUILD")
                       if (r["vendor"], r["arch"]) != rr.PTX_SET})

    def test_no_candidate_keeps_the_pypi_plan(self):
        prev = self.pypi(["_mojolearn_gbdt"])
        plan = self.plan(prev)
        self.assertIs(plan["previous"], prev)
        self.assertEqual([(c["version"], c["channel"], c["chosen"]) for c in plan["considered"]],
                         [("0.8.46", "pypi", True)])
        self.assertEqual(self.builds(plan), ["_mojolearn_gbdt"])
        self.assertEqual({r["reuse_from"]["version"] for r in rr.plan_rows(plan, decision="REUSE")}, {"0.8.46"})
        self.assertTrue(all(r["reuse_from"] is None for r in rr.plan_rows(plan, decision="BUILD")))
        self.assertEqual(self.plan(prev, candidates=[])["rows"], plan["rows"])

    def test_a_candidate_that_built_the_change_wins_and_names_itself_per_row(self):
        # 0.8.48 after the 0.8.47 TestPyPI candidate: PyPI 0.8.46 differs in many
        # bindings, the candidate only in x_decomp
        many = ["_mojolearn_gbdt", "_mojolearn_svm", "_mojolearn", "_mojolearn_x_decomp"]
        self.candidate(["_mojolearn_x_decomp"])
        plan = self.plan(self.pypi(many))
        self.assertEqual((plan["previous"]["version"], plan["previous"]["channel"]), ("0.8.47", "testpypi"))
        self.assertEqual(self.builds(plan), ["_mojolearn_x_decomp"])
        origins = {(r["reuse_from"]["version"], r["reuse_from"]["channel"], r["reuse_from"]["source_commit"])
                   for r in rr.plan_rows(plan, decision="REUSE")}
        self.assertEqual(origins, {("0.8.47", "testpypi", self.prev)})
        self.assertEqual(plan["runtime"]["reuse_from"]["version"], "0.8.47")
        chosen = [c for c in plan["considered"] if c["chosen"]]
        self.assertEqual([c["version"] for c in plan["considered"]], ["0.8.47", "0.8.46"])
        self.assertEqual([c["version"] for c in chosen], ["0.8.47"])
        self.assertGreater(plan["considered"][0]["reuse"], plan["considered"][1]["reuse"])
        self.assertIn("0.8.47 testpypi", rr.render(plan))

    def test_identity_must_match_exactly_and_the_pypi_record_wins_when_it_reuses_more(self):
        self.candidate(["_mojolearn_gbdt", "_mojolearn_svm"])
        plan = self.plan(self.pypi(["_mojolearn_gbdt"]))
        self.assertEqual(plan["previous"]["version"], "0.8.46")
        self.assertEqual(self.builds(plan), ["_mojolearn_gbdt"])

    def test_a_tie_goes_to_the_newer_build(self):
        self.candidate(["_mojolearn_gbdt"])
        plan = self.plan(self.pypi(["_mojolearn_gbdt"]))
        self.assertEqual(plan["previous"]["version"], "0.8.47")

    def test_a_row_the_candidate_shipped_by_the_owner_override_is_never_reused_from_it(self):
        self.candidate(reasons={"_mojolearn_svm": "release owner: not in MOJOLEARN_RELEASE_BUILD_ONLY (changed: closure)"})
        plan = self.plan(self.pypi(["_mojolearn_gbdt", "_mojolearn_x_decomp"]))
        self.assertEqual(plan["previous"]["version"], "0.8.47")
        self.assertEqual(self.builds(plan), ["_mojolearn_svm"])
        self.assertTrue(all("release owner" in r["reason"] for r in rr.plan_rows(plan, decision="BUILD")
                            if r["name"] == "_mojolearn_svm" and (r["vendor"], r["arch"]) != rr.PTX_SET))

    def test_a_candidate_older_than_pypi_is_not_weighed(self):
        self.candidate(version="0.8.45")
        plan = self.plan(self.pypi(["_mojolearn_gbdt"]))
        self.assertEqual([c["version"] for c in plan["considered"]], ["0.8.46"])

    def test_an_unreadable_candidate_is_passed_over(self):
        d = write_candidate(self.ev, "0.8.47", self.prev, [], {})
        cand = rr.candidate_builds(self.ev, platforms=PLATFORMS)
        (d / "reuse" / "plan.json").write_text("{not json")
        plan = self.plan(self.pypi(["_mojolearn_gbdt"]), candidates=cand)
        self.assertEqual(plan["previous"]["version"], "0.8.46")
        bad = [c for c in plan["considered"] if c["version"] == "0.8.47"]
        self.assertTrue(bad and bad[0]["unusable"] and not bad[0]["chosen"])

    def test_a_candidate_without_a_macos_wheel_builds_every_macos_binding(self):
        write_candidate(self.ev, "0.8.47", self.prev, self.rows(), rr.runtime_identity(self.facts),
                        published=("linux", "nvidia", "amd"))
        plan = self.plan(None)      # the tree has no PyPI record: the candidate is the only source
        self.assertEqual(plan["previous"]["version"], "0.8.47")
        self.assertEqual({r["decision"] for r in rr.plan_rows(plan, rr.MACOS)}, {"BUILD"})
        self.assertEqual([r for r in rr.plan_rows(plan, rr.LINUX, "BUILD") if (r["vendor"], r["arch"]) != rr.PTX_SET], [])

    def test_the_record_names_each_reused_binary_s_build(self):
        self.candidate(["_mojolearn_x_decomp"])
        plan = self.plan(self.pypi(["_mojolearn_gbdt"]))
        doc = rr.record_identities(plan, None, None, self.tmp / "ids.json")
        self.assertEqual(doc["previous_channel"], plan["previous"]["channel"])
        for row in doc["rows"]:
            if row["decision"] == "REUSE":
                self.assertEqual(row["reuse_from"]["version"], plan["previous"]["version"])
            else:
                self.assertIsNone(row["reuse_from"])


class CandidateWheelTests(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.ev = pathlib.Path(self._t.name)
        self.name = wheel_name("linux", "0.8.47")
        self.data = b"candidate wheel bytes"
        self.prev = dict(version="0.8.47", source_commit="9" * 40, channel="testpypi",
                         build_dir=str(self.ev / "release" / "0.8.47" / ("9" * 12)),
                         linux=dict(wheel=self.name, sha256=sha(self.data)))

    def tearDown(self):
        self._t.cleanup()

    def test_the_candidate_s_own_build_directory_is_taken_when_its_bytes_verify(self):
        final = pathlib.Path(self.prev["build_dir"]) / "linux" / "final"
        final.mkdir(parents=True)
        (final / self.name).write_bytes(self.data)
        self.assertEqual(rr.published_wheel(self.prev, "linux", self.ev, download=False), final / self.name)
        (final / self.name).write_bytes(self.data + b"x")
        with self.assertRaises(SystemExit):
            rr.published_wheel(self.prev, "linux", self.ev, download=False)

    def fake_index(self, served):
        calls = []

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def urlopen(url, timeout=None):
            calls.append(url)
            if url.endswith("/json"):
                return Resp(json.dumps(dict(urls=[dict(filename=self.name, url="https://files/x.whl",
                                                       digests=dict(sha256=sha(self.data)))])).encode())
            return Resp(served)
        return urlopen, calls

    def test_otherwise_its_wheel_is_downloaded_from_testpypi_and_verified(self):
        urlopen, calls = self.fake_index(self.data)
        with mock.patch.object(rr.urllib.request, "urlopen", urlopen):
            got = rr.published_wheel(self.prev, "linux", self.ev)
        self.assertEqual(calls[0], "https://test.pypi.org/pypi/mojolearn/0.8.47/json")
        self.assertEqual(got.read_bytes(), self.data)

    def test_a_download_whose_bytes_differ_is_refused(self):
        urlopen, _ = self.fake_index(b"tampered")
        with mock.patch.object(rr.urllib.request, "urlopen", urlopen), self.assertRaises(SystemExit) as cm:
            rr.published_wheel(self.prev, "linux", self.ev)
        self.assertIn("does not match", str(cm.exception))

    def test_a_pypi_record_still_downloads_from_pypi(self):
        urlopen, calls = self.fake_index(self.data)
        prev = dict(self.prev, channel="pypi", build_dir=None)
        with mock.patch.object(rr.urllib.request, "urlopen", urlopen):
            rr.published_wheel(prev, "linux", self.ev)
        self.assertEqual(calls[0], "https://pypi.org/pypi/mojolearn/0.8.47/json")


if __name__ == "__main__":
    unittest.main()
