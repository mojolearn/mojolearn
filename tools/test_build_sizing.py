"""CPU-only checks of the release build sizing (tools/build_sizing.py) and of
the guards and scripts that consume it. Launches nothing."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

import amd_serial_guard
import build_sizing as bs
import nvidia_serial_guard

GIB = 2 ** 30
ROOT = Path(__file__).resolve().parents[1]


class SizingRule(unittest.TestCase):
    """Linux build boxes (rented): every core, limited only by memory."""

    def size(self, **kw):
        return bs.size_build(platform='linux', **kw)

    def test_many_core_large_memory_box_uses_all_cores(self):
        # 128 cores, 1 TiB: 64 jobs x 2 compiler workers = 128 workers; the old
        # rule stopped at 16 jobs (32 cores) and left 96 cores idle.
        r = self.size(cores=128, available_bytes=1024 * GIB)
        self.assertEqual((r['jobs'], r['rss_cap_gib'], r['sizing'], r['bound']), (64, 194, 'auto', 'cores'))
        self.assertEqual(r['jobs'] * bs.WORKERS_PER_JOB, 128)
        r = self.size(cores=48, available_bytes=384 * GIB)
        self.assertEqual((r['jobs'], r['bound']), (24, 'cores'))

    def test_cores_bound_a_small_cpu_box(self):
        # An RTX 4090 pod shape: 16 vCPU, plenty of RAM -> 8 jobs, 16 workers.
        r = self.size(cores=16, available_bytes=120 * GIB)
        self.assertEqual((r['jobs'], r['rss_cap_gib'], r['bound']), (8, 26, 'cores'))
        self.assertEqual(r['jobs_by_cores'], 8)

    def test_memory_bound_box_names_memory(self):
        # 32 cores but 20 GiB: floor((20 - 4) / 3) = 5 jobs, cap 17.
        r = self.size(cores=32, available_bytes=20 * GIB)
        self.assertEqual((r['jobs'], r['rss_cap_gib'], r['jobs_by_memory'], r['bound']), (5, 17, 5, 'memory'))
        self.assertIn('bound=memory', bs.describe(r))

    def test_odd_core_count_leaves_no_core_idle(self):
        # 5 cores -> 3 jobs (6 workers): rounding down would idle one core.
        r = self.size(cores=5, available_bytes=64 * GIB)
        self.assertEqual((r['jobs'], r['bound']), (3, 'cores'))
        # The measured 13-core L40S pod: 7 jobs, not 6.
        self.assertEqual(self.size(cores=13, available_bytes=173 * GIB)['jobs'], 7)

    def test_sanity_ceiling_is_named(self):
        r = self.size(cores=1024, available_bytes=8192 * GIB)
        self.assertEqual((r['jobs'], r['bound']), (bs.MAX_JOBS, 'ceiling'))

    def test_jobs_fit_memory_at_the_per_job_figure(self):
        # Largest measured per-job peak (gfx942, four jobs): 6.65 GiB / 4; the
        # sized jobs at PER_JOB_GIB each plus the reserve never exceed memory,
        # and every core works unless memory or the ceiling binds.
        for cores in range(2, 300):
            for mem in (8, 16, 24, 48, 96, 400, 2048):
                r = self.size(cores=cores, available_bytes=mem * GIB)
                self.assertGreaterEqual(r['rss_cap_gib'], r['jobs'] * 6.65 / 4 + 1)
                self.assertLessEqual(r['jobs'] * bs.PER_JOB_GIB + bs.RESERVE_GIB, mem + 1e-9)
                if r['bound'] == 'cores':
                    self.assertGreaterEqual(r['jobs'] * bs.WORKERS_PER_JOB, cores)

    def test_one_core_is_refused(self):
        with self.assertRaisesRegex(bs.TooSmall, '1 usable core'):
            self.size(cores=1, available_bytes=64 * GIB)

    def test_too_little_memory_is_refused(self):
        with self.assertRaisesRegex(bs.TooSmall, 'too small'):
            self.size(cores=16, available_bytes=6 * GIB)

    def test_override_is_kept_even_on_a_small_box(self):
        r = self.size(cores=2, available_bytes=4 * GIB, override=4)
        self.assertEqual((r['jobs'], r['rss_cap_gib'], r['sizing'], r['bound']), (4, 14, 'override', 'override'))

    def test_override_out_of_range_is_a_usage_error(self):
        for bad in (0, bs.MAX_JOBS + 1):
            with self.assertRaises(ValueError):
                self.size(cores=16, available_bytes=64 * GIB, override=bad)
        self.assertEqual(self.size(cores=16, available_bytes=64 * GIB, override=17)['jobs'], 17)
        with self.assertRaises(ValueError):
            bs.parse_jobs('four')
        self.assertIsNone(bs.parse_jobs('auto'))
        self.assertEqual(bs.parse_jobs('6'), 6)

    def test_largest_cap_fits_both_guards(self):
        top = self.size(cores=4096, available_bytes=65536 * GIB)
        self.assertEqual(top['jobs'], bs.MAX_JOBS)
        self.assertLessEqual(top['rss_cap_gib'], nvidia_serial_guard.RSS_GIB_MAX)
        self.assertLessEqual(top['rss_cap_gib'], amd_serial_guard.RSS_GIB_MAX)


class LaptopRuleUnchanged(unittest.TestCase):
    """darwin keeps min(16, cores // 2, 0.6 x GiB / 3) exactly as before."""

    def size(self, **kw):
        return bs.size_build(platform='darwin', **kw)

    def test_old_numbers(self):
        self.assertEqual((self.size(cores=128, available_bytes=1024 * GIB)['jobs'],
                          self.size(cores=128, available_bytes=1024 * GIB)['rss_cap_gib']), (16, 50))
        self.assertEqual(self.size(cores=16, available_bytes=120 * GIB)['jobs'], 8)
        r = self.size(cores=32, available_bytes=20 * GIB)
        self.assertEqual((r['jobs'], r['rss_cap_gib'], r['jobs_by_memory'], r['bound']), (4, 14, 4, 'memory'))
        self.assertEqual(self.size(cores=5, available_bytes=64 * GIB)['jobs'], 2)
        self.assertEqual(self.size(cores=12, available_bytes=32 * GIB)['bound'], 'cores')

    def test_old_ceiling_and_refusals(self):
        r = self.size(cores=128, available_bytes=1024 * GIB)
        self.assertEqual(r['bound'], 'ceiling')
        with self.assertRaises(ValueError):
            self.size(cores=16, available_bytes=64 * GIB, override=17)
        with self.assertRaisesRegex(bs.TooSmall, 'too small'):
            self.size(cores=16, available_bytes=4 * GIB)
        with self.assertRaisesRegex(bs.TooSmall, '1 usable core'):
            self.size(cores=1, available_bytes=64 * GIB)

    def test_default_platform_is_this_process(self):
        with patch.object(bs.sys, 'platform', 'darwin'):
            self.assertEqual(bs.size_build(cores=128, available_bytes=1024 * GIB)['jobs'], 16)
        with patch.object(bs.sys, 'platform', 'linux'):
            self.assertEqual(bs.size_build(cores=128, available_bytes=1024 * GIB)['jobs'], 64)


class BoxReading(unittest.TestCase):
    def box(self, meminfo_kib, files):
        d = Path(tempfile.mkdtemp())
        (d / 'meminfo').write_text('MemTotal: 999999999 kB\nMemAvailable: %d kB\n' % meminfo_kib)
        cg = d / 'cg'
        cg.mkdir()
        for name, text in files.items():
            (cg / name).parent.mkdir(parents=True, exist_ok=True)
            (cg / name).write_text(text)
        return str(d / 'meminfo'), str(cg)

    def read(self, meminfo, cg, affinity):
        with patch.object(bs.os, 'sched_getaffinity', return_value=set(range(affinity)), create=True):
            return bs.box_resources(meminfo, cg)

    def test_no_cgroup_limits_uses_affinity_and_memavailable(self):
        meminfo, cg = self.box(64 * 2 ** 20, {'cpu.max': 'max 100000', 'memory.max': 'max'})
        self.assertEqual(self.read(meminfo, cg, 32), (32, 64 * GIB))

    def test_container_v2_limits_win_over_host_view(self):
        # RunPod-style: the container sees 128 host cores and 1 TiB MemAvailable.
        meminfo, cg = self.box(1024 * 2 ** 20, {'cpu.max': '1600000 100000',
                                                'memory.max': str(62 * GIB), 'memory.current': str(2 * GIB)})
        self.assertEqual(self.read(meminfo, cg, 128), (16, 60 * GIB))

    def test_cgroup_page_cache_is_reclaimable_headroom(self):
        # An 8-CPU/128 GiB build container whose usage is mostly inactive file
        # cache (source tree, Pixi env) still has its limit minus working set.
        meminfo, cg = self.box(1024 * 2 ** 20, {'cpu.max': '800000 100000',
                                                'memory.max': str(128 * GIB), 'memory.current': str(70 * GIB),
                                                'memory.stat': 'anon %d\ninactive_file %d\n' % (10 * GIB, 60 * GIB)})
        self.assertEqual(self.read(meminfo, cg, 128), (8, 118 * GIB))
        meminfo, cg = self.box(1024 * 2 ** 20, {'memory/memory.limit_in_bytes': str(40 * GIB),
                                                'memory/memory.usage_in_bytes': str(30 * GIB),
                                                'memory/memory.stat': 'total_inactive_file %d\n' % (20 * GIB)})
        self.assertEqual(self.read(meminfo, cg, 64)[1], 30 * GIB)

    def test_cgroup_v1_limits(self):
        meminfo, cg = self.box(1024 * 2 ** 20, {'cpu/cpu.cfs_quota_us': '800000', 'cpu/cpu.cfs_period_us': '100000',
                                                'memory/memory.limit_in_bytes': str(40 * GIB),
                                                'memory/memory.usage_in_bytes': str(0)})
        self.assertEqual(self.read(meminfo, cg, 64), (8, 40 * GIB))

    def test_cli_shell_and_refusal(self):
        meminfo, cg = self.box(120 * 2 ** 20, {})
        out, err = io.StringIO(), io.StringIO()
        with patch.object(bs.os, 'sched_getaffinity', return_value=set(range(16)), create=True), \
                patch.object(bs.sys, 'platform', 'linux'), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(bs.main(['--jobs', 'auto', '--meminfo', meminfo, '--cgroup-root', cg, '--shell']), 0)
        vals = dict(line.split('=', 1) for line in out.getvalue().split())
        self.assertEqual((vals['BUILD_JOBS'], vals['BUILD_RSS_GIB'], vals['BUILD_SIZING'], vals['BUILD_BOUND']),
                         ('8', '26', 'auto', 'cores'))
        self.assertIn('build sizing: jobs=8 bound=cores', err.getvalue())
        small, cg2 = self.box(3 * 2 ** 20, {})
        err = io.StringIO()
        with patch.object(bs.os, 'sched_getaffinity', return_value=set(range(16)), create=True), \
                patch.object(bs.sys, 'platform', 'linux'), contextlib.redirect_stderr(err):
            self.assertEqual(bs.main(['--meminfo', small, '--cgroup-root', cg2, '--jobs', 'auto', '--json']), 3)
        self.assertIn('REFUSED', err.getvalue())
        out = io.StringIO()
        with patch.object(bs.os, 'sched_getaffinity', return_value=set(range(16)), create=True), \
                patch.object(bs.sys, 'platform', 'linux'), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bs.main(['--meminfo', small, '--cgroup-root', cg2, '--jobs', '3', '--json']), 0)
        self.assertEqual(json.loads(out.getvalue())['jobs'], 3)


class GuardsTakeTheDerivedCap(unittest.TestCase):
    def test_nvidia_guard_accepts_the_sized_cap_and_reports_peak(self):
        cap = bs.size_build(cores=64, available_bytes=512 * GIB, platform='linux')['rss_cap_gib']
        proc = Mock(pid=4321, returncode=0)
        proc.poll.side_effect = [None, None, 0]
        rss = iter([(0, 400 * GIB), (20 * GIB, 380 * GIB), (7 * GIB, 390 * GIB)])
        out = io.StringIO()
        with tempfile.TemporaryFile() as lock, contextlib.ExitStack() as stack:
            g = nvidia_serial_guard
            stack.enter_context(patch.object(g.sys, 'platform', 'linux'))
            stack.enter_context(patch.object(g.Path, 'is_dir', return_value=True))
            stack.enter_context(patch('builtins.open', return_value=lock))
            stack.enter_context(patch.object(g.fcntl, 'flock'))
            stack.enter_context(patch.object(g, 'gpu_memory', return_value=(0, 24000)))
            stack.enter_context(patch.object(g, 'memory', side_effect=lambda group: next(rss)))
            stack.enter_context(patch.object(g.os, 'sched_getaffinity', return_value=set(range(64)), create=True))
            stack.enter_context(patch.object(g.signal, 'signal'))
            stack.enter_context(patch.object(g.time, 'sleep'))
            stack.enter_context(patch.object(g, 'stop_group'))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(patch.object(g.subprocess, 'Popen', return_value=proc))
            self.assertEqual(g.run(argparse.Namespace(command=['x'], seconds=10, rss_gib=cap, cores=32)), 0)
        record = json.loads(out.getvalue().strip().splitlines()[-1])
        self.assertIsNone(record['reason'])
        self.assertEqual((record['rss_cap_gib'], record['peak_rss_bytes']), (cap, 20 * GIB))
        self.assertEqual(record['min_host_available_bytes'], 380 * GIB)

    def test_nvidia_guard_refuses_above_its_maximum(self):
        g = nvidia_serial_guard
        with patch.object(g.sys, 'platform', 'linux'), patch.object(g.Path, 'is_dir', return_value=True):
            with self.assertRaisesRegex(ValueError, 'RSS cap of 1..512'):
                g.run(argparse.Namespace(command=['x'], seconds=10, rss_gib=513))

    def test_amd_guard_admits_the_sized_cap_past_the_old_twelve(self):
        g = amd_serial_guard
        cap = bs.size_build(cores=64, available_bytes=512 * GIB, platform='linux')['rss_cap_gib']
        self.assertGreater(cap, 12)
        with patch.object(g.sys, 'platform', 'linux'), patch.object(g.Path, 'is_dir', return_value=True), \
                patch.object(g.Path, 'exists', return_value=False):
            # Past the cap check, it stops at the missing /dev/kfd.
            with self.assertRaisesRegex(RuntimeError, 'kfd'):
                g.run(argparse.Namespace(command=['x'], seconds=10, rss_gib=cap))
            with self.assertRaisesRegex(ValueError, 'RSS cap of 1..512'):
                g.run(argparse.Namespace(command=['x'], seconds=10, rss_gib=513))


class ScriptsUseTheSizing(unittest.TestCase):
    def test_remote_build_sizes_and_hands_the_cap_to_the_build_guard(self):
        s = (ROOT / 'tools/release061_remote_build.sh').read_text()
        self.assertIn('tools/build_sizing.py" --jobs "${MOJOLEARN_BUILD_JOBS:-auto}" --shell', s)
        self.assertIsNone(re.search(r'^BUILD_RSS_GIB=12$', s, re.M))
        self.assertIn('[[ "$name" = full46-build ]] && guard_cores=$BUILD_CORES && guard_rss=$BUILD_RSS_GIB', s)
        for key in ('build_jobs=', 'build_rss_gib=', 'box_cores=', 'box_mem_available_gib=', 'build_sizing='):
            self.assertIn(key, s)
        self.assertIn('RELEASE_BUILD_SIZING_FILE', s)
        self.assertIn("record['build_resources']", (ROOT / 'tools/linux_surface_qualification.sh').read_text())

    def test_legs_default_to_auto(self):
        for name in ('gemm_remote_leg.sh', 'do_release061_leg.sh', 'hotaisle_release_leg.sh'):
            s = (ROOT / 'tools' / name).read_text()
            self.assertIn('MOJOLEARN_BUILD_JOBS:-auto', s, name)
            self.assertNotIn('MOJOLEARN_BUILD_JOBS:-4', s, name)
        u22 = (ROOT / 'tools/release_ubuntu22_build.sh').read_text()
        self.assertIn('tools/build_sizing.py --jobs "${MOJOLEARN_BUILD_JOBS:-auto}"', u22)
        self.assertIn('mem_gib >= BUILD_RSS_GIB + 4', u22)

    def test_no_new_keyed_build_variable(self):
        # Every MOJOLEARN_* variable reaches the binding cache key
        # (tools/bincache.py); the sizing must travel under another prefix.
        s = (ROOT / 'tools/release061_remote_build.sh').read_text()
        self.assertEqual(sorted(set(re.findall(r'export (MOJOLEARN_BUILD_[A-Z_]+)=', s))), ['MOJOLEARN_BUILD_JOBS', 'MOJOLEARN_BUILD_PIXI_ENV'])


if __name__ == '__main__':
    unittest.main()
