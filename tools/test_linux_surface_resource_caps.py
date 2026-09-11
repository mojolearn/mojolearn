"""Root-only Linux checks of the admission prefix; no build/model invocation.

Authored without execution. Run remotely from the root/main thread only.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('taskset'),
                     'Root-only remote Linux affinity checks')
class ResourceCaps(unittest.TestCase):
    def check_prefix(self, inherited, action='qualify', jobs=None):
        # DEVIATION 2501: the build action takes 2 x MOJOLEARN_BUILD_JOBS cores
        # (default 4 jobs); qualification stays on two with jobs forced to 1.
        expected_jobs = (jobs or 4) if action == 'build' else 1
        expected_cores = 2 * expected_jobs
        script = Path(__file__).with_name('linux_surface_qualification.sh')
        prefix = script.read_text().split('# RESOURCE_CAPS_END:', 1)[0]
        # The production prefix exits before any source inventory, build or GPU
        # work. Inspect an actual child after taskset, with hostile inherited caps.
        probe = '''
"$PY" - <<'PROBE'
import json, os
print(json.dumps({'cpus': sorted(os.sched_getaffinity(0)), 'env': dict(os.environ)}))
PROBE
'''
        env = dict(os.environ, MOJOLEARN_QUALIFY_PYTHON=sys.executable)
        keys = ('MOJOLEARN_BUILD_JOBS', 'MOJOLEARN_COMPILE_JOBS', 'MAX_JOBS',
                'CMAKE_BUILD_PARALLEL_LEVEL', 'MOJOLEARN_CPU_THREADS',
                'CARGO_BUILD_JOBS', 'RAYON_NUM_THREADS', 'OMP_NUM_THREADS',
                'OMP_THREAD_LIMIT', 'OMP_MAX_ACTIVE_LEVELS', 'BLIS_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS')
        env.update({key: '99' for key in keys})
        env.update(MAKEFLAGS='-j99', MFLAGS='-j99', GNUMAKEFLAGS='-j99')
        if jobs is not None:
            env['MOJOLEARN_BUILD_JOBS'] = str(jobs)
        result = subprocess.run(
            ['taskset', '-c', ','.join(map(str, inherited)), 'bash', '-c',
             prefix + probe, 'resource-cap-check', action],
            env=env, text=True, capture_output=True, timeout=15, check=True)
        record = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(record['cpus'], sorted(inherited)[:expected_cores])
        child = record['env']
        twos = {'MOJOLEARN_COMPILE_JOBS', 'MAX_JOBS', 'CMAKE_BUILD_PARALLEL_LEVEL',
                'MOJOLEARN_CPU_THREADS', 'CARGO_BUILD_JOBS', 'RAYON_NUM_THREADS'}
        for key in keys:
            if key == 'MOJOLEARN_BUILD_JOBS':
                self.assertEqual(child[key], str(expected_jobs), key)
                continue
            self.assertEqual(child[key], '2' if key in twos else '1', key)
        self.assertEqual(child['MAKEFLAGS'], '-j2')
        self.assertEqual(child['MFLAGS'], '-j2')
        self.assertEqual(child['GNUMAKEFLAGS'], '')

    def test_bounded_parent_and_hostile_thread_settings(self):
        # Even the test launcher is bounded to three cores before production
        # narrows it to two; never widen beyond the root's current allowance.
        self.check_prefix(sorted(os.sched_getaffinity(0))[:3])

    def test_single_core_parent_is_not_widened(self):
        self.check_prefix(sorted(os.sched_getaffinity(0))[:1])

    def test_build_action_takes_two_cores_per_job_never_beyond_parent(self):
        # DEVIATION 2501: hostile MOJOLEARN_BUILD_JOBS=99 in the environment is
        # NOT honored as-is by the prefix; the range check refuses it. A legal
        # value widens to 2 x jobs, bounded by the parent's allowance.
        parent = sorted(os.sched_getaffinity(0))
        self.check_prefix(parent[:16], action='build', jobs=4)
        self.check_prefix(parent[:3], action='build', jobs=4)
        self.check_prefix(parent[:16], action='build', jobs=1)


if __name__ == '__main__':
    unittest.main()
