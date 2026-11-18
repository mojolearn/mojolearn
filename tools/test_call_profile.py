# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ops/call_profile_job.sh, tools/ops/call_profile.py and the harness profile window they use
(tools/bench_board_small_rows.py), on canned inputs. `python3 -m unittest tools.test_call_profile`."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.join(HERE, "ops", "call_profile_job.sh")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SR = _load("bench_board_small_rows_cp", os.path.join(HERE, "bench_board_small_rows.py"))
CP = _load("call_profile_t", os.path.join(HERE, "ops", "call_profile.py"))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


PERF_SCRIPT = """python3 4242  1000000
\t    7f00000010 cuStreamSynchronize+0x1 (/usr/lib/x86_64-linux-gnu/libcuda.so.1)
\t    7f00000020 run_layer (/root/v/lib/python3.12/site-packages/mojolearn/_mojolearn_nn.so)
\t    7f00000030 _PyEval_EvalFrameDefault (/usr/bin/python3.12)
\t    7f00000040 _start (/usr/bin/python3.12)

python3 4242  3000000
\t    7f00000050 malloc (/usr/lib/x86_64-linux-gnu/libc.so.6)
\t    7f00000020 run_layer (/root/v/lib/python3.12/site-packages/mojolearn/_mojolearn_nn.so)
\t    7f00000040 _start (/usr/bin/python3.12)

perf 99  5000000
\t    7f00000060 write (/usr/lib/x86_64-linux-gnu/libc.so.6)

"""

STRACE = """\
100 1700000000.000100 mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7f <0.000005>
100 1700000001.000000 newfstatat(AT_FDCWD, "/callprof-window-open", 0x7ffc, 0) = -1 ENOENT <0.000003>
101 1700000001.100000 futex(0x55, FUTEX_WAIT_PRIVATE, 0, NULL <unfinished ...>
100 1700000001.200000 ioctl(5, _IOC(0, 0, 0x21, 0x30), 0x7ffc) = 0 <0.000010>
100 1700000001.300000 ioctl(5, _IOC(0, 0, 0x21, 0x30), 0x7ffc) = 0 <0.000010>
101 1700000001.400000 <... futex resumed>) = 0 <0.300000>
100 1700000001.500000 madvise(0x7f, 4096, MADV_DONTNEED) = 0 <0.000002>
100 1700000002.000000 newfstatat(AT_FDCWD, "/callprof-window-close", 0x7ffc, 0) = -1 ENOENT <0.000003>
100 1700000002.100000 munmap(0x7f, 4096) = 0 <0.000004>
"""


def window_lines(calls=2, wall_ns=4000000, m0=None, m1=None):
    return (json.dumps({"event": "open", "calls": calls, "phase": "fit", "pid": 1, "perf": True, "malloc": m0}) + "\n" +
            json.dumps({"event": "close", "calls": calls, "phase": "fit", "pid": 1, "perf": True, "wall_ns": wall_ns,
                        "malloc": m1}) + "\n")


class Rows(unittest.TestCase):
    def test_default_rows(self):
        rows = CP.load_rows()
        names = [r.name for r in rows]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("lr-at-control", names)
        self.assertIn("global-avgpool-fit", names)
        self.assertIn("global-avgpool-infer", names)
        for r in rows:
            self.assertTrue(r.race_id.startswith("algos/"), r.race_id)
            self.assertIn(r.phase, ("fit", "infer"))       # the algos driver's phase names (forced_calls matches them)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            CP.parse_rows("algos/x/y/rows=full predict\n")
        with self.assertRaises(ValueError):
            CP.parse_rows("algos/x/y/rows=full fit name=a\nalgos/z/y/rows=full fit name=a\n")
        with self.assertRaises(ValueError):
            CP.parse_rows("algos/x/y/rows=full fit bogus\n")
        rows = CP.parse_rows("algos/x/y/rows=full fit\nalgos/x/y/rows=full infer\n")
        self.assertEqual([r.name for r in rows], ["x", "x-infer"])

    def test_estimate_under_thirty_minutes(self):
        e = CP.estimate(CP.load_rows())
        self.assertLess(e["total_min"], 30.0)
        self.assertEqual(len(e["rows"]), len(CP.load_rows()))


class HarnessWindow(unittest.TestCase):
    def setUp(self):
        self.keys = [SR.FORCE_CALLS_ENV, SR.FORCE_CALLS_PHASE_ENV, SR.PROFILE_WINDOW_ENV, SR.PROFILE_SECONDS_ENV,
                     SR.PROFILE_MAX_CALLS_ENV]
        self.old = {k: os.environ.pop(k) for k in self.keys if k in os.environ}
        del SR._RANGE[:]

    def tearDown(self):
        for k in self.keys:
            os.environ.pop(k, None)
        os.environ.update(self.old)
        del SR._RANGE[:]

    def test_profile_calls_fill_the_seconds(self):
        env = {SR.PROFILE_SECONDS_ENV: "10"}
        self.assertEqual(SR.profile_calls(20, 0.2, env), 50001)
        self.assertEqual(SR.profile_calls(20, 900.0, env), 20)             # never fewer than forced
        self.assertEqual(SR.profile_calls(20, 0.001, dict(env, **{SR.PROFILE_MAX_CALLS_ENV: "500"})), 500)
        self.assertEqual(SR.profile_calls(20, 0.2, {}), 20)                 # no seconds: the forced count
        self.assertEqual(SR.profile_calls(20, None, env), 20)

    def test_calls_for_in_a_profile_run(self):
        os.environ.update({SR.FORCE_CALLS_ENV: "20", SR.FORCE_CALLS_PHASE_ENV: "infer",
                           SR.PROFILE_SECONDS_ENV: "1"})
        self.assertEqual(SR.calls_for(5.0, True, "fit"), 20)               # not a profile run: the small-row rule
        os.environ[SR.PROFILE_WINDOW_ENV] = "/nonexistent/w.jsonl"
        self.assertEqual(SR.calls_for(5.0, True, "infer"), 201)
        self.assertEqual(SR.calls_for(5.0, True, "fit"), 1)                # the phase not profiled: one call
        self.assertEqual(SR.calls_for(5.0, False, "infer"), 1)

    def test_window_lines_and_perf_control(self):
        with tempfile.TemporaryDirectory() as d:
            ctl, ack = os.path.join(d, "ctl"), os.path.join(d, "ack")
            os.mkfifo(ctl)
            os.mkfifo(ack)
            got = []

            def fake_perf():      # perf: reads the control FIFO, acks every command
                rfd = os.open(ctl, os.O_RDONLY | os.O_NONBLOCK)
                afd = os.open(ack, os.O_RDWR)
                import select
                try:
                    while len(got) < 2:
                        r, _, _ = select.select([rfd], [], [], 5.0)
                        if not r:
                            break
                        data = os.read(rfd, 64).decode()
                        for w in data.split():
                            got.append(w)
                            os.write(afd, b"ack\n")
                finally:
                    os.close(rfd)
                    os.close(afd)
            t = threading.Thread(target=fake_perf)
            t.start()
            import time
            time.sleep(0.2)
            w = os.path.join(d, "window.jsonl")
            env = {SR.FORCE_CALLS_ENV: "20", SR.FORCE_CALLS_PHASE_ENV: "fit", SR.PROFILE_WINDOW_ENV: w,
                   SR.PERF_CTL_ENV: "%s,%s" % (ctl, ack)}
            self.assertFalse(SR.range_open(20, "infer", env))                # not the forced phase
            self.assertTrue(SR.range_open(20, "fit", env))
            self.assertFalse(SR.range_open(20, "fit", env))                  # one window at a time
            SR.range_close()
            SR.range_close()                                                 # nothing open: no-op
            t.join(10)
            self.assertEqual(got, ["enable", "disable"])
            win = CP.read_window(w)
            self.assertEqual(win["calls"], 19)
            self.assertTrue(win["perf_enabled"])
            self.assertIsNone(win["malloc_per_call"])                        # no counter loaded
            self.assertIsNotNone(win["wall_ms_per_call"])

    def test_window_without_perf_reader_does_not_block(self):
        with tempfile.TemporaryDirectory() as d:
            ctl, ack = os.path.join(d, "ctl"), os.path.join(d, "ack")
            os.mkfifo(ctl)
            os.mkfifo(ack)
            w = os.path.join(d, "w.jsonl")
            env = {SR.FORCE_CALLS_ENV: "20", SR.PROFILE_WINDOW_ENV: w, SR.PERF_CTL_ENV: "%s,%s" % (ctl, ack)}
            self.assertTrue(SR.range_open(5, "fit", env))
            SR.range_close()
            self.assertFalse(CP.read_window(w)["perf_enabled"])

    def test_board_is_untouched(self):
        self.assertFalse(SR.range_open(20, "fit", {}))
        self.assertEqual(SR.calls_for(5.0, True, "fit"), SR.SMALL_ROW_CALLS)
        self.assertEqual(SR.calls_for(50.0, True, "fit"), 1)


class Parsers(unittest.TestCase):
    def test_read_window_allocations(self):
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "w.jsonl")
            write(w, window_lines(4, 8000000, {"malloc": 10, "free": 10, "bytes": 0},
                                  {"malloc": 50, "free": 46, "bytes": 4000}))
            win = CP.read_window(w)
            self.assertEqual(win["calls"], 4)
            self.assertAlmostEqual(win["wall_ms_per_call"], 2.0)
            self.assertEqual(win["malloc_per_call"], {"malloc": 10.0, "free": 9.0, "bytes": 1000.0})
            self.assertEqual(CP.read_window(os.path.join(d, "none"))["calls"], 0)

    def test_perf_script_to_top_frames(self):
        samples = list(CP.perf_samples(PERF_SCRIPT.splitlines(True)))
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0][0], "python3")
        self.assertEqual(samples[0][1], 1000000)
        self.assertEqual(samples[0][2][0], "_start [python3.12]")                 # root first
        self.assertEqual(samples[0][2][-1], "cuStreamSynchronize [libcuda.so.1]")  # offset dropped, leaf last
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "perf.script"), PERF_SCRIPT)
            write(os.path.join(d, "window.jsonl"), window_lines(2))
            rc = CP.main(["perf", "--script", os.path.join(d, "perf.script"), "--window", os.path.join(d, "window.jsonl"),
                          "--out", os.path.join(d, "perf.json"), "--folded", os.path.join(d, "perf.folded")])
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "perf.json")) as fh:
                p = json.load(fh)
            self.assertEqual(p["samples"], 2)                                    # perf's own sample left out
            self.assertEqual(p["skipped_comm"], {"perf": 1})
            self.assertAlmostEqual(p["cpu_ms_per_call"], 2.0)                    # 4 ms of cpu-clock over 2 calls
            self.assertEqual(p["self"][0]["frame"], "malloc [libc.so.6]")
            self.assertEqual(p["self"][0]["us_per_call"], 1500.0)
            incl = {x["frame"]: x for x in p["inclusive"]}
            self.assertEqual(incl["run_layer [_mojolearn_nn.so]"]["pct"], 100.0)
            hc = CP.largest_host_costs(p)
            self.assertEqual([h["frame"] for h in hc], ["malloc [libc.so.6]", "cuStreamSynchronize [libcuda.so.1]"])
            with open(os.path.join(d, "perf.folded")) as fh:
                self.assertEqual(len(fh.read().splitlines()), 2)

    def test_strace_window(self):
        st = CP.strace_counts(STRACE.splitlines(True))
        self.assertTrue(st["found_open"] and st["found_close"])
        self.assertEqual(st["calls"], {"futex": 1, "ioctl": 2, "madvise": 1})     # mmap before, munmap after: out
        self.assertAlmostEqual(st["seconds"]["futex"], 0.3)
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "s.txt"), STRACE)
            write(os.path.join(d, "w.jsonl"), window_lines(2))
            self.assertEqual(CP.main(["strace", "--trace", os.path.join(d, "s.txt"), "--window",
                                      os.path.join(d, "w.jsonl"), "--out", os.path.join(d, "s.json")]), 0)
            with open(os.path.join(d, "s.json")) as fh:
                s = json.load(fh)
            self.assertEqual(s["total_per_call"], 2.0)
            self.assertEqual(s["per_call"][0], {"syscall": "ioctl", "per_call": 1.0, "us_per_call": 10.0})

    def test_pyspy_under_the_runner_method(self):
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "c"), "process 1;main (/r/tools/bench_board_algos.py:7600);fit (/r/tools/bench_board_algos.py:3093);"
                  "fit (mojolearn/nn.py:10);run_layer (_mojolearn_nn.so) 30\n"
                  "process 1;main (/r/tools/bench_board_algos.py:7600);infer (/r/tools/bench_board_algos.py:3098);x 5\n"
                  "process 1;load (/r/tools/bench_board_algos.py:100) 7\n")
            write(os.path.join(d, "w.jsonl"), window_lines(3))
            rc = CP.main(["pyspy", "--collapsed", os.path.join(d, "c"), "--window", os.path.join(d, "w.jsonl"),
                          "--phase", "fit", "--out", os.path.join(d, "p.json")])
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "p.json")) as fh:
                p = json.load(fh)
            self.assertEqual(p["samples"], 1)
            self.assertEqual(p["outside_samples"], 12)
            self.assertEqual(p["self"][0]["frame"], "run_layer (_mojolearn_nn.so)")
            self.assertAlmostEqual(p["self"][0]["us_per_call"], 40000.0)          # 30 samples / 250 Hz / 3 calls


class Report(unittest.TestCase):
    def _res(self, d, with_nsys=True, with_time=True):
        rows = "algos/a/synthetic/rows=full\tfit\tname=a\n"
        write(os.path.join(d, "rows.tsv"), rows)
        rd = os.path.join(d, "rows", "a")
        if with_time:
            write(os.path.join(rd, "time", "window.jsonl"), window_lines(10, 5000000))
        write(os.path.join(rd, "perf", "window.jsonl"), window_lines(100, 60000000))
        write(os.path.join(rd, "perf", "perf.json"), json.dumps(
            {"samples": 50, "cpu_ms_per_call": 0.4, "weight": "w",
             "self": [{"frame": "_start [python3]", "pct": 50.0, "us_per_call": 200.0},
                      {"frame": "cuMemAlloc_v2 [libcuda.so.1]", "pct": 30.0, "us_per_call": 120.0}],
             "inclusive": [{"frame": "_start [python3]", "pct": 100.0, "us_per_call": 400.0}]}))
        if with_nsys:
            write(os.path.join(rd, "nsys", "nsys.json"), json.dumps(
                {"captured": True, "api_ms": 0.2, "kernel_ms": 0.05, "sync_ms": 0.1, "window_ms": 0.7,
                 "counts": {"launch": 3.0, "sync": 1.0, "alloc": 2.0}, "api": {"cuMemAlloc_v2": {"calls": 2.0, "ms": 0.1}},
                 "mem": {"h2d": {"count": 1.0, "mb": 0.01, "ms": 0.01}}}))
        write(os.path.join(rd, "strace", "window.jsonl"), window_lines(4, 1, {"malloc": 0, "free": 0}, {"malloc": 40, "free": 40}))
        write(os.path.join(rd, "strace", "strace.json"), json.dumps(
            {"total_per_call": 12.0, "per_call": [{"syscall": "ioctl", "per_call": 8.0, "us_per_call": 1.0}]}))

    def test_report_names_the_host_costs(self):
        with tempfile.TemporaryDirectory() as d:
            self._res(d)
            rc = CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv"), "--wheel", "w1"])
            self.assertEqual(rc, CP.EXIT_OK)
            with open(os.path.join(d, "CALL-PROFILE-nvidia.json")) as fh:
                doc = json.load(fh)
            r = doc["rows"][0]
            self.assertAlmostEqual(r["wall_ms_per_call"], 0.5)                    # the time pass, not perf's 0.6
            self.assertAlmostEqual(r["split"]["host_outside_cuda_ms"], 0.3)
            self.assertEqual(r["malloc_per_call"]["malloc"], 10.0)
            self.assertEqual(r["largest_host_costs"][0]["frame"], "cuMemAlloc_v2 [libcuda.so.1]")
            with open(os.path.join(d, "CALL-PROFILE-nvidia.md")) as fh:
                md = fh.read()
            self.assertIn("LARGEST HOST COSTS (perf, self time per call): cuMemAlloc_v2 [libcuda.so.1] 120.0 us", md)
            self.assertIn("syscalls per call (12.0 in all): ioctl 8.00", md)

    def test_report_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            self._res(d, with_nsys=False)
            self.assertEqual(CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv")]),
                             CP.EXIT_NO_PROFILE)
        with tempfile.TemporaryDirectory() as d:
            self._res(d, with_time=False)                                          # perf's window stands in
            self.assertEqual(CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv")]), CP.EXIT_OK)
            shutil.rmtree(os.path.join(d, "rows", "a", "perf"))
            self.assertEqual(CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv")]),
                             CP.EXIT_NO_WALL)


class Job(unittest.TestCase):
    def test_syntax_and_refusal(self):
        self.assertEqual(subprocess.run(["bash", "-n", JOB]).returncode, 0)
        env = {k: v for k, v in os.environ.items() if k not in ("MOJOLEARN_BOX_PERSIST", "MOJOLEARN_ALLOW_UNPERSISTED")}
        r = subprocess.run(["bash", JOB], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("box_persist_run", r.stderr)

    @unittest.skipUnless(shutil.which("cc"), "no C compiler")
    def test_malloc_counter_compiles(self):
        r = subprocess.run(["cc", "-fsyntax-only", os.path.join(HERE, "ops", "call_profile_malloc.c")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
