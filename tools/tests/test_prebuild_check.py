"""Tooling tests for tools/prebuild_check.py: the source audit over a staged tree (part 1) and the probe wheel's file
mapping (part 3). Pure Python; nothing is compiled, dispatched or installed."""
import base64
import csv
import hashlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prebuild_check", ROOT / "tools" / "prebuild_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class SourceAuditTest(unittest.TestCase):
    """Part 1: this checkout's packaging/portable_math/wheel.py rules over a staged wheel layout."""

    def test_numpy_import_outside_the_oracles_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            write(stage / "mojolearn/__init__.py", "")
            write(stage / "mojolearn/_new_helper.py", "import numpy as np\n")
            errors = pc.math_errors(ROOT, stage)
            self.assertTrue(any("mojolearn/_new_helper.py" in e and "NumPy" in e for e in errors), errors)

    def test_clean_stage_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            write(stage / "mojolearn/__init__.py", "from ._x import f\n")
            write(stage / "mojolearn/_x.py", "def f():\n    return 1\n")
            self.assertEqual(pc.math_errors(ROOT, stage), [])

    def test_lastuse_audit_runs_the_commits_tool_and_skips_without_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pc.lastuse_audit(Path(tmp), "HEAD"), 0)   # a commit that predates the audit
            tree = Path(tmp) / "tree"
            write(tree / "tools/mojo_lastuse_audit.py", "import sys\nprint('ran', sys.argv[1:])\nsys.exit(3)\n")
            self.assertEqual(pc.lastuse_audit(tree, "abc123"), 3)       # its exit code fails the audits part

    def test_python_free_audit_runs_the_commits_script_and_skips_without_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pc.python_free_audit(Path(tmp), "HEAD"), 0)   # a commit that predates the script
            tree = Path(tmp) / "tree"
            write(tree / "tools/python_free_dirs.sh", 'echo "kernels/x.mojo imports Python ($1)"; exit 1\n')
            self.assertEqual(pc.python_free_audit(tree, "abc123"), 1)        # a refusal fails the audits part

    def test_python_free_audit_passes_on_head(self):
        self.assertEqual(pc.python_free_audit(ROOT, "HEAD"), 0)

    def test_stage_entries_lays_out_wheel_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = write(tmp / "tools/identity_break.py", "X = 1\n")
            stage = pc.stage_entries({"mojolearn/_identity_break.py": src}, tmp / "stage")
            self.assertEqual((stage / "mojolearn/_identity_break.py").read_text(), "X = 1\n")

    def test_wheel_entries_of_head_follow_the_packer(self):
        """The commit's own packer functions decide the mapping, tools/ copies included."""
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        with tempfile.TemporaryDirectory() as tmp:
            tree = pc.export_tree(head, Path(tmp) / "tree")
            # The child is this test's own interpreter, as prebuild_check runs its parts with sys.executable: the
            # commit's packer (packaging/linux/pack_wheel.py) needs Python 3.11+ (tomllib) and runs in the pkg env,
            # and a bare `python3` is the box's system 3.10 (Ubuntu 22.04), which the packer refuses.
            out = subprocess.run(
                [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import prebuild_check as pc; "
                 "e = pc.wheel_entries(sys.argv[2], sys.argv[3]); "
                 "print(e['mojolearn/_identity_break.py']); print(e['mojolearn/__init__.py']); "
                 "print(e['mojolearn_diagnostics.py']); print(any(m.startswith('mojolearn/tests/') for m in e))",
                 str(ROOT / "tools"), str(tree), head], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            lines = out.stdout.split()
            self.assertEqual(Path(lines[0]).resolve(), (tree / "tools/identity_break.py").resolve())
            self.assertEqual(Path(lines[1]), tree / "python/mojolearn/__init__.py")
            self.assertEqual(Path(lines[2]), tree / "python/mojolearn_diagnostics.py")
            self.assertEqual(lines[3], "False")   # tests never ship


class ProbeWheelTest(unittest.TestCase):
    """Part 3: which members of a packed core wheel the probe replaces, adds, drops and keeps."""

    NAMES = ["mojolearn/__init__.py", "mojolearn/_version.py", "mojolearn/_gone.py", "mojolearn/host/_x_host.so",
             "mojolearn/_identity_break.py", "mojolearn/verify_reference/table.json", "mojolearn_diagnostics.py",
             "mojolearn_core-1.dist-info/METADATA", "mojolearn_core-1.dist-info/RECORD"]

    def entries(self, tmp):
        tmp = Path(tmp)
        return {"mojolearn/__init__.py": write(tmp / "python/mojolearn/__init__.py", "NEW_INIT = 1\n"),
                "mojolearn/_version.py": write(tmp / "python/mojolearn/_version.py", '__version__ = "0.8.66"\n'),
                "mojolearn/_identity_break.py": write(tmp / "tools/identity_break.py", "NEW_BREAK = 1\n"),
                "mojolearn/_added.py": write(tmp / "python/mojolearn/_added.py", "ADDED = 1\n"),
                "mojolearn/verify_reference/table.json": write(tmp / "python/mojolearn/verify_reference/table.json",
                                                               '{"records": []}\n'),
                "mojolearn_diagnostics.py": write(tmp / "python/mojolearn_diagnostics.py", "D = 2\n")}

    def test_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            replace, add, drop = pc.probe_plan(self.NAMES, self.entries(tmp))
        self.assertEqual(replace, ["mojolearn/__init__.py", "mojolearn/_identity_break.py",
                                   "mojolearn/verify_reference/table.json", "mojolearn_diagnostics.py"])
        self.assertEqual(add, ["mojolearn/_added.py"])
        self.assertEqual(drop, ["mojolearn/_gone.py"])   # the generated _version.py, binaries and metadata stay

    def test_probe_wheel_bytes_and_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wheel = tmp / "mojolearn_core-1-py3-none-manylinux_2_35_x86_64.whl"
            with zipfile.ZipFile(wheel, "w") as z:
                for name in self.NAMES:
                    z.writestr(name, b"OLD " + name.encode())
            probe, _ = pc.build_probe_wheel(wheel, self.entries(tmp / "src"), tmp / pc.PROBE_DIR)
            self.assertEqual(probe.name, wheel.name)
            self.assertEqual(probe.parent.name, pc.PROBE_DIR)
            with zipfile.ZipFile(probe) as z:
                names = set(z.namelist())
                self.assertEqual(z.read("mojolearn/__init__.py"), b"NEW_INIT = 1\n")
                self.assertEqual(z.read("mojolearn/_identity_break.py"), b"NEW_BREAK = 1\n")
                self.assertEqual(z.read("mojolearn/_added.py"), b"ADDED = 1\n")
                self.assertEqual(z.read("mojolearn/_version.py"), b"OLD mojolearn/_version.py")   # kept as packed
                self.assertEqual(z.read("mojolearn/host/_x_host.so"), b"OLD mojolearn/host/_x_host.so")
                self.assertNotIn("mojolearn/_gone.py", names)
                rows = list(csv.reader(io.StringIO(z.read("mojolearn_core-1.dist-info/RECORD").decode())))
                record = {r[0]: r[1] for r in rows}
                self.assertEqual(set(record), names)
                for name in names - {"mojolearn_core-1.dist-info/RECORD"}:
                    digest = base64.urlsafe_b64encode(hashlib.sha256(z.read(name)).digest()).decode().rstrip("=")
                    self.assertEqual(record[name], "sha256=" + digest, name)



class HostParseRemovedTest(unittest.TestCase):
    """host-parse is gone (lane build-reuse-1, 2026-11-26): the build's legs compile every group it parsed, so the
    candidate no longer waits on it (1341 s of the acdf0ad303 build) and nothing dispatches native-checks.yml."""

    def test_the_default_parts_are_the_audits(self):
        self.assertEqual(pc.PARTS, ("audits", "probe"))
        self.assertFalse(hasattr(pc, "run_host_parse"))

    def test_asking_for_host_parse_is_refused_with_the_reason(self):
        with unittest.mock.patch.object(pc, "resolve_commit", return_value="c" * 40):
            with self.assertRaises(SystemExit) as cm:
                pc.main(["--commit", "c" * 40, "--only", "audits,host-parse", "--work", tempfile.mkdtemp()])
        self.assertIn("legs compile every host and device group", str(cm.exception))

    def test_a_pass_recorded_with_host_parse_is_reused_for_the_audits(self):
        work = Path(tempfile.mkdtemp())
        (work / "PASS.json").write_text('{"commit": "%s", "parts": ["audits", "host-parse"]}' % ("c" * 40))
        with unittest.mock.patch.object(pc, "resolve_commit", return_value="c" * 40), \
                unittest.mock.patch.object(pc.subprocess, "call", side_effect=AssertionError("ran a part")):
            self.assertEqual(pc.main(["--commit", "c" * 40, "--work", str(work), "--reuse-pass"]), 0)


if __name__ == "__main__":
    unittest.main()
