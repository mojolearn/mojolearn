"""The changed lanes in the release's own build steps (Andrew, 2026-10-13): the
NVIDIA sm_89 and AMD gfx942 build legs record and verify the changed lanes on
their boxes after the build, identity-admit admits those records before
linux-pack, and the per-lane summary is one grep-sized line per lane.

    python3 -m pytest -q tools/test_release_changed_lanes.py
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import changed_lanes_summary  # noqa: E402

FROZEN = "c" * 40


def args(**kw):
    base = dict(version="0.8.99", dry_run=False, publish=None, only="", redo="", build_backend="gpu-legs",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                cpu_column=False, status=False, refreeze=False, source_checkout="")
    base.update(kw)
    return argparse.Namespace(**base)


class Known:
    """A context that knows the changed lanes, as step_linux_builds leaves it."""

    def __init__(self, lanes):
        self.lanes = lanes

    def changed_lanes_known(self, backend):
        return self.lanes.get(backend)


class Graph(unittest.TestCase):
    def test_admit_sits_between_the_legs_and_the_pack(self):
        self.assertEqual(release.NEEDS["identity-admit"], ["linux-wait"])
        self.assertIn("identity-admit", release.NEEDS["linux-pack"])
        self.assertEqual(release.PIPELINE_OF["identity-admit"], "core-linux")
        self.assertEqual(release.RESOURCE["identity-admit"], "mac")
        self.assertIn("identity-admit", release.PIPELINES["core-linux"]["builds"])
        for mac in ("macos-build", "macos-smoke", "publish-macos"):
            self.assertNotIn("identity-admit", release.NEEDS[mac])


class LegEnv(unittest.TestCase):
    def test_only_the_sm89_and_gfx942_legs_get_the_lanes(self):
        ctx = Known({"cuda": ["ols", "knn"], "hip": ["ols"]})
        self.assertEqual(release.changed_lanes_env(ctx, "cuda-sm_89"),
                         {"MOJOLEARN_RELEASE_CHANGED_LANES": "ols,knn"})
        self.assertEqual(release.changed_lanes_env(ctx, "cuda-sm_90a"), {})
        self.assertEqual(release.changed_lanes_env(ctx, "cuda-sm_80"), {})
        self.assertEqual(release.changed_lanes_env(ctx, "hip-gfx942", amd_route="do"),
                         {"MOJOLEARN_RELEASE_CHANGED_LANES": "ols",
                          "MOJOLEARN_RELEASE_CHANGED_LABEL": "amd-mi325x-gfx942"})
        self.assertEqual(release.changed_lanes_env(ctx, "hip-gfx942", amd_route="hotaisle")
                         ["MOJOLEARN_RELEASE_CHANGED_LABEL"], "amd-mi300x-gfx942")
        # a CPU build box has no GPU to record on
        self.assertEqual(release.changed_lanes_env(ctx, "hip-gfx942", amd_route="cpu-box"), {})

    def test_nothing_known_or_nothing_changed_sends_nothing(self):
        self.assertEqual(release.changed_lanes_env(object(), "cuda-sm_89"), {})
        self.assertEqual(release.changed_lanes_env(Known({}), "cuda-sm_89"), {})
        self.assertEqual(release.changed_lanes_env(Known({"cuda": []}), "cuda-sm_89"), {})


class Admit(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        self._env = mock.patch.dict(os.environ, MOJOLEARN_EVIDENCE_ROOT=str(self.tmp / "ev"),
                                    MOJOLEARN_RELEASE_LEDGER="off")
        self._env.start()
        os.environ.pop(release.CHANGED_LANES_SWITCH, None)

    def tearDown(self):
        self._env.stop()
        self._t.cleanup()

    def release(self, lanes, legs=()):
        r = release.Release(args(state_dir=str(self.tmp / "state")))
        r.state["commit"] = FROZEN
        r.lines = []
        r.say = r.lines.append
        r.changed_lanes = lambda backend: lanes.get(backend, [])
        r.admitted_legs = lambda: {}
        self._legs = mock.patch.object(release, "linux_legs", return_value=list(legs))
        self._legs.start()
        self.addCleanup(self._legs.stop)
        return r

    def leg(self, name, vendor, arch):
        rb = self.tmp / "legs" / name / "release-build"
        return release.Leg(name, vendor, arch, ["true"], {}, rb, self.tmp / "legs", rb.parent)

    def test_off_skips(self):
        with mock.patch.dict(os.environ, {release.CHANGED_LANES_SWITCH: "off"}):
            self.assertTrue(self.release({}).step_identity_admit().startswith("SKIPPED"))

    def test_no_changed_lane_is_nothing_changed(self):
        r = self.release({"cuda": [], "hip": []})
        self.assertTrue(r.step_identity_admit().startswith("NOTHING-CHANGED"))
        # done at this freeze: a rerun reads its marker
        self.assertTrue(r.step_identity_admit().startswith("have NOTHING-CHANGED"))

    def test_a_vendor_whose_leg_did_not_build_is_owed_and_nothing_is_admitted(self):
        nv = self.leg("cuda-sm_89", "cuda", "sm_89")
        r = self.release({"cuda": ["ols"], "hip": ["ols"]}, legs=[nv])
        rec = nv.release_build / "changed-lanes" / "record"
        rec.mkdir(parents=True)
        (rec / "nvidia-l40s-sm_89.lanes-x.identical.json").write_text("{}")
        with mock.patch.object(r, "must") as must:
            out = r.step_identity_admit()
        self.assertTrue(out.startswith("OWED"), out)
        self.assertIn("hip-gfx942", out)
        must.assert_not_called()

    def test_a_leg_that_built_but_recorded_nothing_stops_the_pipeline(self):
        nv, amd = self.leg("cuda-sm_89", "cuda", "sm_89"), self.leg("hip-gfx942", "hip", "gfx942")
        for leg in (nv, amd):
            leg.release_build.mkdir(parents=True)
        r = self.release({"cuda": ["ols"], "hip": ["ols"]}, legs=[nv, amd])
        with self.assertRaises(release.StepFailed) as cm:
            r.step_identity_admit()
        self.assertIn("recorded none", str(cm.exception))


class Summary(unittest.TestCase):
    def test_one_line_per_lane_with_the_divergent_hashes(self):
        report = dict(
            lane_accounting=dict(lanes={
                "ols": dict(state="VERIFIED"),
                "knn": dict(state="DIVERGENT"),
                "pca": dict(state="OWED", reason="no committed record carries a reference for a part of it")}),
            cells=[dict(lane="knn", fixture="base", part="train", state="DIVERGENT",
                        value="aaaaaaaaaaaaaaaaaaaa", reference="bbbbbbbbbbbbbbbbbbbb"),
                   dict(lane="knn", fixture="base", part="infer", state="DIVERGENT", value="c", reference="d")])
        lines = changed_lanes_summary.summarize(report, "cuda", ["ols", "knn", "pca", "gone"])
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], "CHANGED-LANE column=cuda lane=ols state=VERIFIED")
        self.assertIn("state=DIVERGENT cell=base/train got=aaaaaaaaaaaa ref=bbbbbbbbbbbb +1 more", lines[1])
        self.assertIn("state=NO-REFERENCE", lines[2])
        self.assertIn("state=NOT-IN-REPORT", lines[3])
        self.assertEqual(lines[4], "CHANGED-LANES column=cuda lanes=4 VERIFIED=1 DIVERGENT=1 NO-REFERENCE=1 OTHER=1")

    def test_an_unreadable_report_is_never_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            bad = pathlib.Path(d) / "r.json"
            bad.write_text("not json")
            self.assertEqual(changed_lanes_summary.main([str(bad), "--column", "cpu", "--lanes", "ols"]), 2)


if __name__ == "__main__":
    unittest.main()
