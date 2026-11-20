# SPDX-License-Identifier: Apache-2.0
"""tools/ops/job_preflight.sh and the jobs' own --preflight modes (lane fx-pod-job, 2026-11-18): a box job's inputs are
checked when it is submitted, not after hours in the queue. On 2026-11-18 tools/ops/call_profile_job.sh failed three
times in 90 s on conditions true at submit time: the venv held another wheel than the manifest (251f16771a), no wheel
manifest (bf3af46910), an install that could not run (eed2394df2). Offline: the blocks run in bash on temp dirs."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE / "ops" / "job_preflight.sh"
CP_JOB = HERE / "ops" / "call_profile_job.sh"
PC_JOB = HERE / "ops" / "paper_compare_job.sh"
CENTRAL = HERE / "nvidia_central.sh"


def block(path, name):
    text = path.read_text()
    return text[text.index("# >>> " + name):text.index("# <<< " + name)]


def run(script, env=None, args=(), timeout=60):
    return subprocess.run(["bash", "-c", script, "x", *args], env=dict(os.environ, **(env or {})),
                          capture_output=True, text=True, timeout=timeout)


class SubmitPreflight(unittest.TestCase):
    def preflight(self, td, script_text, *env, marker=False):
        s = Path(td) / "job.sh"
        s.write_text(("# job-preflight: --preflight\n" if marker else "") + script_text)
        return subprocess.run(["bash", str(PREFLIGHT), td, str(s), "", *env], capture_output=True, text=True,
                              timeout=60)

    def test_parses_and_env_paths(self):
        with tempfile.TemporaryDirectory() as td:
            here = Path(td) / "wheels"
            here.mkdir()
            p = self.preflight(td, "echo hi\n", "PC_WHEEL_DIR=%s" % here, "NC_FILTER=spectral_%")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("PREFLIGHT PASSED", p.stdout)
            p = self.preflight(td, "if then fi (\n")                                     # does not parse
            self.assertEqual(p.returncode, 4)
            self.assertIn("does not parse", p.stdout)
            p = self.preflight(td, "echo hi\n", "CP_MANIFEST=%s/none.json" % td)         # a missing --env path
            self.assertEqual(p.returncode, 4)
            self.assertIn("CP_MANIFEST=", p.stdout)
            self.assertIn("nothing queued", p.stdout)

    def test_runs_the_jobs_own_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            ok = '[ "${1:-}" = --preflight ] && { echo "PREFLIGHT ok mine $X"; exit 0; }\nexit 9\n'
            p = self.preflight(td, ok, "X=1", marker=True)
            self.assertEqual(p.returncode, 0, p.stdout)
            self.assertIn("PREFLIGHT ok mine 1", p.stdout)                              # the --env values reach it
            bad = '[ "${1:-}" = --preflight ] && { echo "PREFLIGHT FAIL no manifest"; exit 4; }\nexit 9\n'
            p = self.preflight(td, bad, marker=True)
            self.assertEqual(p.returncode, 4)
            self.assertIn("PREFLIGHT FAIL no manifest", p.stdout)
            p = self.preflight(td, bad)                                                  # no marker: not run
            self.assertEqual(p.returncode, 0)

    def test_submit_runs_it_before_anything_is_queued(self):
        text = CENTRAL.read_text()
        # the submit body is the function central_submit (lane fx-cpu-box-checks: tools/cpu_central.sh check runs the
        # same code), dispatched by `submit)`
        self.assertIn('\nsubmit) central_submit "$@" ;;\n', text)
        start = text.index("\ncentral_submit() {")
        sub = text[start:text.index("\n}\n", start)]
        self.assertIn('< "$ROOT/tools/ops/job_preflight.sh"', sub)
        self.assertLess(sub.index("job_preflight.sh"), sub.index("bp_prepare"))
        self.assertLess(sub.index("job_preflight.sh"), sub.index("$NVQ submit"))
        self.assertIn("--no-preflight) preflight=0", sub)
        for job in (CP_JOB, PC_JOB):
            self.assertIn("\n# job-preflight:", job.read_text())


class CallProfilePreflight(unittest.TestCase):
    """The three failure modes of 2026-11-18, each refused at submit time."""

    def setup(self, td, have=None, version="0.8.67.dev0+abcdefabcdef", source="a" * 40, wheels=1):
        res = Path(td) / "box-out" / "new"
        old = Path(td) / "box-out" / "old" / "job"
        old.mkdir(parents=True)
        man = old / "wheel-manifest-candidate.json"
        man.write_text('{"version": "%s", "source_commit": "%s"}' % (version, source))
        cache = Path(td) / "cache"
        if have:
            b = cache / "venv-candidate" / "bin"
            b.mkdir(parents=True)
            (b / "python").write_text("#!/bin/bash\necho %s\n" % have)
            (b / "python").chmod(0o755)
        wdir = Path(td) / "wheels"
        wdir.mkdir()
        for i in range(wheels):
            (wdir / ("w%d.whl" % i)).write_bytes(b"x")
        rows = Path(td) / "rows.tsv"
        rows.write_text("algos/a/taxi/rows=full\ttime\tx\n")
        env = {"PC_CACHE_ROOT": str(cache), "CP_WHEEL_DIR": str(wdir), "CP_ROWS_FILE": str(rows),
               "CP_PYTHON": sys.executable}
        return str(res), env

    def check(self, td, res, env):
        script = block(CP_JOB, "cp-preflight") + '\ncp_preflight "%s" "%s"\n' % (res, td)
        return run(script, env)

    def test_venv_holds_another_wheel_and_no_install(self):        # 251f16771a
        with tempfile.TemporaryDirectory() as td:
            res, env = self.setup(td, have="0.8.67.dev0+111111111111")
            p = self.check(td, res, dict(env, CP_PARTS="plan,run,report"))
            self.assertEqual(p.returncode, 4, p.stdout + p.stderr)
            self.assertIn("nothing would be timed", p.stdout)
            p = self.check(td, res, env)                                 # with the install part it reinstalls
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_no_manifest(self):                                        # bf3af46910
        with tempfile.TemporaryDirectory() as td:
            res, env = self.setup(td, have="0.8.67.dev0+abcdefabcdef")
            p = self.check(td, res, dict(env, CP_MANIFEST=os.path.join(td, "none.json")))
            self.assertEqual(p.returncode, 4)
            self.assertIn("no wheel manifest", p.stdout)
            p = self.check(td, res, env)                                 # the newest sibling results dir's
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("the venv holds the manifest's wheel", p.stdout)

    def test_install_cannot_run(self):                                 # eed2394df2
        with tempfile.TemporaryDirectory() as td:
            res, env = self.setup(td, wheels=0)
            p = self.check(td, res, env)
            self.assertEqual(p.returncode, 4)
            self.assertIn("the install cannot run", p.stdout)
        with tempfile.TemporaryDirectory() as td:
            res, env = self.setup(td, source="")
            p = self.check(td, res, env)
            self.assertEqual(p.returncode, 4)
        with tempfile.TemporaryDirectory() as td:
            res, env = self.setup(td)
            p = self.check(td, res, env)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("the install has 1 wheel(s)", p.stdout)


class PaperComparePreflight(unittest.TestCase):
    def check(self, env):
        script = block(PC_JOB, "pc-preflight") + '\npc_preflight nvidia\n'
        return run(script, env)

    def test_wheels_rows_and_b_sets(self):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "w"
            w.mkdir()
            (w / "mojolearn_core-1.0-x.whl").write_bytes(b"x")
            p = self.check({"PC_WHEEL_DIR": str(w), "PC_SOURCE_COMMIT": "a" * 40})
            self.assertEqual(p.returncode, 4)
            self.assertIn("holds 1 of the 2 wheels", p.stdout)
            (w / "mojolearn_nvidia-1.0-x.whl").write_bytes(b"x")
            p = self.check({"PC_WHEEL_DIR": str(w), "PC_SOURCE_COMMIT": "a" * 40})
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            p = self.check({"PC_WHEEL_DIR": str(w), "PC_SOURCE_COMMIT": "a" * 40, "PC_PARTS": "targeted",
                            "PC_ROWS": os.path.join(td, "none.tsv")})
            self.assertEqual(p.returncode, 4)
            arms = Path(td) / "b.txt"
            arms.write_text("x %s %s\ne env:MOJOLEARN_X=1 %s\nbad %s/none %s\n" % (w, "b" * 40, "b" * 40, td, "b" * 40))
            p = self.check({"PC_B_ARMS": str(arms)})
            self.assertEqual(p.returncode, 4)
            self.assertIn("B set bad", p.stdout)
            self.assertNotIn("B set x", p.stdout)
            self.assertNotIn("B set e", p.stdout)                       # an env arm has no wheels to stage


if __name__ == "__main__":
    unittest.main()
