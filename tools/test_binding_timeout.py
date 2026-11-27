"""CPU-only checks of the per-binding build bound
(packaging/linux/binding_timeout.sh, sourced by packaging/linux/build_sets.sh).
Runs small shell stand-ins for a build; no compiler, no GPU."""
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / 'packaging/linux/binding_timeout.sh'


def bash(script, env=None, timeout=60):
    return subprocess.run(['bash', '-c', '. "%s"\n%s' % (LIB, script)], capture_output=True, text=True,
                          env=dict(os.environ, **(env or {})), timeout=timeout)


def _running(pid):
    """True while `pid` exists and is not a zombie. In a container whose PID 1
    does not reap orphans a killed descendant stays a zombie, which `kill -0`
    still reports; a zombie runs nothing."""
    if subprocess.run(['kill', '-0', pid], capture_output=True).returncode != 0:
        return False
    try:
        state = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()[0]
    except OSError:
        state = subprocess.run(['ps', '-o', 'stat=', '-p', pid], capture_output=True, text=True).stdout.strip()
    return not state.startswith('Z')


class BindingTimeout(unittest.TestCase):
    def test_default_and_override_and_refusal(self):
        self.assertEqual(bash('binding_timeout_seconds', {'RELEASE_BINDING_TIMEOUT_SECONDS': ''}).stdout, '1200')
        self.assertEqual(bash('binding_timeout_seconds', {'RELEASE_BINDING_TIMEOUT_SECONDS': '30'}).stdout, '30')
        for bad in ('0', '-5', 'ten', '1.5'):
            r = bash('binding_timeout_seconds', {'RELEASE_BINDING_TIMEOUT_SECONDS': bad})
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn('positive integer', r.stderr)

    def test_a_finished_build_keeps_its_status_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'b.log'
            r = bash('with_binding_timeout 30 "%s" sh -c "echo compiled; exit 0"; echo rc=$?' % log)
            self.assertIn('rc=0', r.stdout)
            self.assertIn('compiled', log.read_text())
            r = bash('with_binding_timeout 30 "%s" sh -c "echo error: boom >&2; exit 3"; echo rc=$?' % log)
            self.assertIn('rc=3', r.stdout)
            self.assertIn('error: boom', log.read_text())
            self.assertNotIn('TIMEOUT', log.read_text())

    def test_a_function_with_prefix_assignments_sees_them(self):
        # build_sets.sh calls it as VAR=... with_binding_timeout T LOG run_binding ...
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'b.log'
            r = bash('f() { sh -c \'echo "mode=$MOJOLEARN_NUMERIC_MODE"\'; }\n'
                     'MOJOLEARN_NUMERIC_MODE=fast with_binding_timeout 30 "%s" f; echo rc=$?' % log)
            self.assertIn('rc=0', r.stdout)
            self.assertIn('mode=fast', log.read_text())

    def test_a_stuck_build_is_stopped_with_its_whole_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'b.log'
            pids = Path(tmp) / 'pids'
            # A stand-in for pixi -> bash -> mojo: a grandchild that ignores
            # nothing and would run for five minutes.
            stuck = ('sh -c \'sleep 300 & echo $! >> "%s"; sh -c "sleep 300 & echo \\$! >> %s; wait" & '
                     'echo $! >> "%s"; wait\'' % (pids, pids, pids))
            t0 = time.monotonic()
            r = bash('with_binding_timeout 2 "%s" %s; echo rc=$?' % (log, stuck))
            elapsed = time.monotonic() - t0
            self.assertIn('rc=124', r.stdout, r.stderr)
            self.assertLess(elapsed, 20)
            self.assertIn('TIMEOUT after 2s', log.read_text())
            time.sleep(0.5)
            alive = [p for p in pids.read_text().split() if _running(p)]
            self.assertEqual(alive, [], 'descendants survived the bound')

    def test_stall_default_override_and_refusal(self):
        self.assertEqual(bash('binding_stall_seconds', {'RELEASE_BINDING_STALL_SECONDS': ''}).stdout, '240')
        self.assertEqual(bash('binding_stall_seconds', {'RELEASE_BINDING_STALL_SECONDS': '0'}).stdout, '0')
        for bad in ('-5', 'ten', '1.5'):
            r = bash('binding_stall_seconds', {'RELEASE_BINDING_STALL_SECONDS': bad})
            self.assertEqual(r.returncode, 2, bad)

    @unittest.skipUnless(Path('/proc/self/stat').is_file(), 'the tree CPU probe reads Linux /proc')
    def test_a_stalled_build_is_stopped_by_name_before_its_bound(self):
        # tools/compile_watch.py, the rule tools/cross_compile_check.py uses: flat CPU and no output
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'b.log'
            t0 = time.monotonic()
            r = bash('with_binding_timeout 60 "%s" sh -c "sleep 300"; echo rc=$?' % log,
                     {'RELEASE_BINDING_STALL_SECONDS': '2', 'RELEASE_BINDING_STALL_POLL': '0.2'})
            self.assertIn('rc=125', r.stdout, r.stderr)
            self.assertLess(time.monotonic() - t0, 30)
            text = log.read_text()
            self.assertIn('STALLED: compiler CPU flat', text)
            self.assertNotIn('TIMEOUT', text)
            self.assertFalse(Path(str(log) + '.stalled').exists())

    @unittest.skipUnless(Path('/proc/self/stat').is_file(), 'the tree CPU probe reads Linux /proc')
    def test_a_busy_or_talking_build_is_not_stalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'b.log'
            spin = 'python3 -c "import time; t=time.time()\nwhile time.time()-t<6: pass"'
            r = bash('with_binding_timeout 60 "%s" %s; echo rc=$?' % (log, spin),
                     {'RELEASE_BINDING_STALL_SECONDS': '3', 'RELEASE_BINDING_STALL_POLL': '0.2'})
            self.assertIn('rc=0', r.stdout, r.stderr)
            talk = 'sh -c "for i in 1 2 3 4 5 6; do echo step; sleep 1; done"'
            r = bash('with_binding_timeout 60 "%s" %s; echo rc=$?' % (log, talk),
                     {'RELEASE_BINDING_STALL_SECONDS': '3', 'RELEASE_BINDING_STALL_POLL': '0.2'})
            self.assertIn('rc=0', r.stdout, r.stderr)
            self.assertNotIn('STALLED', log.read_text())

    def test_build_sets_uses_the_bound_for_every_binding(self):
        s = (ROOT / 'packaging/linux/build_sets.sh').read_text()
        self.assertIn('. "$REPO/packaging/linux/binding_timeout.sh"', s)
        self.assertEqual(s.count('with_binding_timeout "$BINDING_TIMEOUT" "$log"'), 2)
        self.assertIn('TIMED OUT after ${bound}s', s)  # bound: BINDING_TIMEOUT, or the merged bound
        self.assertIn('if [[ "$rc" = 125 ]]; then', s)  # STALLED is named as its own FINDING
        # The source line sits after the host checks, so the early refusals
        # (tools/test_linux_build_limits.py) still run first.
        self.assertLess(s.index('taskset -pc "$BUILD_CPUS" $$'), s.index('binding_timeout.sh"'))


if __name__ == '__main__':
    unittest.main()
