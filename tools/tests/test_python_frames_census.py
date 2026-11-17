"""Tooling tests for tools/ops/python_frames_census.py (the census's own fixtures, the NEEDS_GPU status, the SUMMARY
lead, VIA_SUBCLASS through an alias) and the verify smoke's census skip rule (tools/ops/verify_smoke_dispatch.py).
Pure Python: nothing is installed, compiled or dispatched; no mojolearn import."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ops"))
import python_frames_census as census   # noqa: E402
import verify_smoke_dispatch as dispatch   # noqa: E402


def _stats(calls=0, frames=0, bindings=0):
    return dict(calls=calls, zero=calls if not frames else 0, frames=frames, frames_max=frames, bindings_max=bindings,
                bindings_total=bindings * calls, multi_binding=0, no_binding=0 if bindings else calls, exempt=0,
                imports=0, funcs={"m:f": frames} if frames else {}, bnames={})


class FixturePlanTest(unittest.TestCase):
    def test_committed_fixture_modules_load_with_the_standard_library_only(self):
        before = set(sys.modules)
        fixtures, needs_gpu, why_not = census.fixture_plan()
        self.assertGreater(len(fixtures), 50)
        self.assertFalse({m for m in set(sys.modules) - before if m.split(".")[0] in ("numpy", "mojolearn")})
        for fid, (fn, members, gpu) in fixtures.items():
            self.assertTrue(callable(fn), fid)
            self.assertTrue(members and all(m.startswith("mojolearn.") for m in members), fid)
        for table in (needs_gpu, why_not):
            self.assertTrue(all(isinstance(r, str) and r for r in table.values()))

    def test_plan_reads_gpu_marks_and_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "demo.py").write_text(
                "def a(ml, cx): pass\n"
                "FIXTURES = {'a': (a, ['mojolearn.A.f']), 'g': (a, ['mojolearn.A.g'], 'gpu')}\n"
                "NEEDS_GPU = {'mojolearn.A.g': 'two devices'}\nWHY_NOT = {'mojolearn.h': 'import time'}\n")
            fixtures, needs_gpu, why_not = census.fixture_plan(tmp)
        self.assertEqual(sorted(fixtures), ["demo.a", "demo.g"])
        self.assertFalse(fixtures["demo.a"][2])
        self.assertTrue(fixtures["demo.g"][2])
        self.assertEqual(needs_gpu, {"mojolearn.A.g": "two devices"})
        self.assertEqual(why_not, {"mojolearn.h": "import time"})


class ReportTest(unittest.TestCase):
    def test_statuses_lead_line_and_uncovered_list(self):
        members = [dict(id="mojolearn.A.f", family="a", kind="method", defined="mojolearn._a.A.f",
                        code="mojolearn._a.A.f"),
                   dict(id="mojolearn.A.g", family="a", kind="method", defined="mojolearn._a.A.g"),
                   dict(id="mojolearn.A.h", family="a", kind="method", defined="mojolearn._a.A.h"),
                   dict(id="mojolearn.B.save", family="b", kind="method", defined="mojolearn._b.B.save",
                        code="mojolearn._p.Archive.save"),
                   dict(id="mojolearn.p.Archive.save", family="p", kind="method",
                        defined="mojolearn._p.Archive.save", code="mojolearn._p.Archive.save"),
                   dict(id="mojolearn.C.py", family="c", kind="method", defined="mojolearn._c.C.py")]
        inv = dict(members=members, units=[dict(name="api:x")], needs_gpu={"mojolearn.A.g": "two devices"},
                   why_not={"mojolearn.A.h": "import time"})
        unit = dict(fixtures=[dict(fixture="f", ok=True)], members={
            "mojolearn.A.f": _stats(1, 0, 1), "mojolearn.B.save": _stats(2, 0, 1), "mojolearn.C.py": _stats(1, 3, 1)})
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "units").mkdir()
            (d / "inventory.json").write_text(json.dumps(inv))
            (d / "units" / "api_x.json").write_text(json.dumps(unit))
            lines = census.report(d)
            rows = {l.split("\t")[0]: l.split("\t")[3] for l in (d / "census.tsv").read_text().splitlines()[1:]}
            uncovered = (d / "uncovered.tsv").read_text().splitlines()[1:]
        self.assertTrue(lines[0].startswith("frames rc=0 measured=3 with_python=1 uncovered=1 needs_gpu=1 "),
                        lines[0])
        self.assertEqual(rows["mojolearn.A.g"], "NEEDS_GPU")
        self.assertEqual(rows["mojolearn.A.h"], "UNCOVERED")
        self.assertEqual(rows["mojolearn.p.Archive.save"], "VIA_SUBCLASS")   # ran as B.save (an alias)
        self.assertEqual(rows["mojolearn.C.py"], "PYTHON")
        self.assertIn("mojolearn.A.g\ta\tmethod\tNEEDS_GPU\ttwo devices", uncovered)
        self.assertIn("mojolearn.A.h\ta\tmethod\tUNCOVERED\timport time", uncovered)

    def test_function_key_follows_the_installed_function(self):
        class Archive:
            def save(self):
                pass

            @classmethod
            def load(cls):
                pass

        class B:
            save = Archive.save
            load = vars(Archive)["load"]
        self.assertEqual(census._function_key(vars(B)["save"]), census._function_key(vars(Archive)["save"]))
        self.assertEqual(census._function_key(vars(B)["load"]), census._function_key(vars(Archive)["load"]))
        self.assertEqual(census._function_key(len), "")


class InstanceHeldNativeMethodTest(unittest.TestCase):
    def test_a_native_method_held_by_a_public_instance_is_its_member(self):
        # a schedule's lr_at is its native block's bound method, stored in the schedule's own dictionary
        c = census.Census({"package": "/nonexistent/mojolearn", "classes": [
            {"id": "mojolearn.Sched", "module": "m", "qualname": "Sched", "attrs": {"lr_at": "method"}}],
            "functions": [], "members": []})

        class Block:
            pass

        class Sched:
            pass
        held = len.__call__   # any builtin bound method whose owner is no public class
        sched = Sched()
        sched.lr_at = held
        self.assertIsNone(c.native_member(held))
        c.instances["mojolearn.Sched"] = sched
        self.assertEqual(c.native_member(held), ["mojolearn.Sched.lr_at"])
        self.assertIsNone(c.native_member(Block().__init__))


class CensusSkipRuleTest(unittest.TestCase):
    def test_decision(self):
        inputs = {"python/mojolearn": "t1", "tools/ops/python_frames_census.py": "b1"}
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp) / "records.json"
            self.assertTrue(dispatch.census_decision("br", inputs, None, rec)[0])   # nothing recorded: runs
            dispatch.record_census("br", dict(commit="a" * 40, wheel="w.whl", inputs=inputs, token="t",
                                              made="now"), rec)
            run, line = dispatch.census_decision("br", dict(inputs), None, rec)
            self.assertFalse(run)
            self.assertIn("w.whl (aaaaaaaaaaaa", line)
            self.assertTrue(dispatch.census_decision("br", dict(inputs, **{"python/mojolearn": "t2"}), None, rec)[0])
            self.assertTrue(dispatch.census_decision("other", inputs, None, rec)[0])
            self.assertTrue(dispatch.census_decision("br", inputs, True, rec)[0])     # --census
            self.assertFalse(dispatch.census_decision("br", dict(inputs, x="y"), False, rec)[0])   # --no-census
            # an input missing at its revision never counts as unchanged
            missing = {"python/mojolearn": ""}
            dispatch.record_census("m", dict(commit="b" * 40, wheel="w", inputs=missing), rec)
            self.assertTrue(dispatch.census_decision("m", missing, None, rec)[0])

    def test_inputs_of_head(self):
        inputs = dispatch.census_inputs("HEAD", "HEAD")
        self.assertEqual(set(inputs), {"python/mojolearn", *dispatch.CENSUS_FILES})
        self.assertTrue(all(inputs.values()), inputs)


if __name__ == "__main__":
    unittest.main()
