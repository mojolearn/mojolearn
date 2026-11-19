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
import time
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


def window_lines(calls=2, wall_ns=4000000, m0=None, m1=None, pid=1):
    return (json.dumps({"event": "open", "calls": calls, "phase": "fit", "pid": pid, "perf": True, "shim": m0}) + "\n" +
            json.dumps({"event": "close", "calls": calls, "phase": "fit", "pid": pid, "perf": True, "wall_ns": wall_ns,
                        "shim": m1}) + "\n")


#: a stand-in for py-spy: says it samples, writes its raw file when interrupted (the harness's attach and stop)
FAKE_PYSPY = """#!%s
import signal, sys, time
a = sys.argv
out = a[a.index("--output") + 1]
pid = a[a.index("--pid") + 1]
def stop(*_):
    with open(out, "w") as fh:
        fh.write("thread (1): MainThread;run (x.py:1) 7\\n")
    print("py-spy> Wrote raw flamegraph data to " + out, flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
print("py-spy> Sampling process 250 times a second (pid %%s)" %% pid, flush=True)
while True:
    time.sleep(0.05)
"""

#: a callgrind out file: f in a.so calls g in libc twice (inclusive 30 each), g recurses once (g'2), f self 100
CALLGRIND = """# callgrind format
version: 1
creator: callgrind-3.18.1
pid: 77
cmd:  python3 x.py
part: 1

desc: Trigger: Client Request: callprof-window

positions: line
events: Ir
summary: 160

ob=(1) /v/site-packages/mojolearn/_mojolearn_nn.so
fl=(1) ???
fn=(1) run_layer
10 100
cob=(2) /usr/lib/x86_64-linux-gnu/libc.so.6
cfn=(2) malloc
calls=2 0
+1 30
fn=(3) helper
5 20
cob=(2)
cfn=(4) g
calls=1 0
* 40

ob=(2)
fn=(2)
0 30
fn=(4)
0 30
cfn=(5) g'2
calls=1 0
* 10
fn=(5)
0 10
"""


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
                     SR.PROFILE_MAX_CALLS_ENV, SR.PYSPY_ENV, SR.SHIM_ENV, SR.CALLGRIND_ENV]
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
            ready, done = threading.Event(), threading.Event()

            def fake_perf():      # perf: reads the control FIFO, acks every command
                rfd = os.open(ctl, os.O_RDONLY | os.O_NONBLOCK)
                afd = os.open(ack, os.O_RDWR)
                ready.set()
                import select
                deadline = time.monotonic() + 20.0
                try:
                    while len(got) < 2:
                        r, _, _ = select.select([rfd], [], [], 5.0)
                        if not r:
                            break
                        # Between two writers the FIFO is at end-of-file and select keeps reporting it readable; a
                        # read that lands after the next writer has opened but before it has written raises
                        # BlockingIOError (run 44580558011: the thread died there and 'disable' was never read).
                        try:
                            data = os.read(rfd, 64).decode()
                        except BlockingIOError:
                            data = ""
                        if not data:
                            if time.monotonic() > deadline:
                                break
                            time.sleep(0.005)
                            continue
                        for w in data.split():
                            got.append(w)
                            os.write(afd, b"ack\n")
                finally:
                    # Hold the ack FIFO open until the caller has read its last ack: closing the only writer end
                    # discards a buffered ack the reader has not opened yet (lost under load, 2026-11-17 runs).
                    done.wait(10)
                    os.close(rfd)
                    os.close(afd)
            t = threading.Thread(target=fake_perf)
            t.start()
            # Wait for the reader: a nonblocking writer gets ENXIO before perf has the FIFO open (a fixed 0.2 s sleep
            # lost that race on a loaded laptop, 2026-11-17 full run).
            self.assertTrue(ready.wait(10))
            w = os.path.join(d, "window.jsonl")
            env = {SR.FORCE_CALLS_ENV: "20", SR.FORCE_CALLS_PHASE_ENV: "fit", SR.PROFILE_WINDOW_ENV: w,
                   SR.PERF_CTL_ENV: "%s,%s" % (ctl, ack)}
            self.assertFalse(SR.range_open(20, "infer", env))                # not the forced phase
            self.assertTrue(SR.range_open(20, "fit", env))
            self.assertFalse(SR.range_open(20, "fit", env))                  # one window at a time
            SR.range_close()
            SR.range_close()                                                 # nothing open: no-op
            done.set()
            t.join(10)
            self.assertEqual(got, ["enable", "disable"])
            win = CP.read_window(w)
            self.assertEqual(win["calls"], 19)
            self.assertTrue(win["perf_enabled"])
            self.assertIsNone(win["shim_per_call"])                          # no shim loaded
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

    def test_pyspy_attached_for_the_window_only(self):
        with tempfile.TemporaryDirectory() as d:
            spy = os.path.join(d, "py-spy")
            write(spy, FAKE_PYSPY % sys.executable)
            os.chmod(spy, 0o755)
            w, raw = os.path.join(d, "w.jsonl"), os.path.join(d, "pyspy.raw")
            env = {SR.FORCE_CALLS_ENV: "20", SR.PROFILE_WINDOW_ENV: w, SR.PYSPY_ENV: "%s,%s,250" % (spy, raw),
                   SR.CALLGRIND_ENV: os.path.join(d, "absent.so")}
            self.assertTrue(SR.range_open(5, "fit", env))
            self.assertFalse(os.path.exists(raw))                           # sampling, not yet written
            SR.range_close()
            with open(raw) as fh:
                self.assertIn("MainThread", fh.read())                      # stopped by SIGINT at the close
            win = CP.read_window(w)
            self.assertTrue(win["pyspy"])
            self.assertIsNone(win["pyspy_error"])
            self.assertFalse(win["callgrind"])                              # no library: reported, not raised
            with open(raw + ".log") as fh:
                self.assertIn("pid %d" % os.getpid(), fh.read())             # attached to this process

    def test_pyspy_that_cannot_attach_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "w.jsonl")
            env = {SR.FORCE_CALLS_ENV: "20", SR.PROFILE_WINDOW_ENV: w,
                   SR.PYSPY_ENV: "%s,%s,250" % (shutil.which("false"), os.path.join(d, "r"))}
            self.assertTrue(SR.range_open(5, "fit", env))
            SR.range_close()
            win = CP.read_window(w)
            self.assertFalse(win["pyspy"])
            self.assertIn("py-spy exited", win["pyspy_error"])

    def test_board_is_untouched(self):
        self.assertFalse(SR.range_open(20, "fit", {}))
        self.assertEqual(SR.calls_for(5.0, True, "fit"), SR.SMALL_ROW_CALLS)
        self.assertEqual(SR.calls_for(50.0, True, "fit"), 1)


class Parsers(unittest.TestCase):
    def test_read_window_shim(self):
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "w.jsonl")
            write(w, window_lines(4, 8000000, {"malloc": [10, 1000, 0], "cuStreamSynchronize": [1, 50, 0]},
                                  {"malloc": [50, 9000, 4000], "cuStreamSynchronize": [5, 4050, 0]}))
            win = CP.read_window(w)
            self.assertEqual(win["calls"], 4)
            self.assertAlmostEqual(win["wall_ms_per_call"], 2.0)
            self.assertEqual(win["shim_per_call"], {"malloc": {"calls": 10.0, "us": 2.0, "bytes": 1000.0},
                                                    "cuStreamSynchronize": {"calls": 1.0, "us": 1.0, "bytes": 0.0}})
            self.assertEqual(CP.read_window(os.path.join(d, "none"))["calls"], 0)

    def test_categories(self):
        self.assertEqual(CP.category("/v/site-packages/mojolearn/_mojolearn_nn.so"), "bindings")
        self.assertEqual(CP.category("libcuda.so.595.112.07"), "cuda_driver")
        self.assertEqual(CP.category("libAsyncRTMojoBindings.so"), "mojo_runtime")
        self.assertEqual(CP.category("libKGENCompilerRTShared.so"), "mojo_runtime")
        self.assertEqual(CP.category("/usr/lib/libpython3.11.so.1.0"), "cpython")
        self.assertEqual(CP.category("tools/bench_board_algos.py"), "cpython")
        self.assertEqual(CP.category("<__array_function__ internals>"), "cpython")
        self.assertEqual(CP.category("/usr/bin/python3.11"), "cpython")
        self.assertEqual(CP.category("libc.so.6"), "libc")
        self.assertEqual(CP.category("numpy/core/_multiarray_umath.cpython-311-x86_64-linux-gnu.so"), "other")
        self.assertEqual(CP.category(""), "unresolved")

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

    def test_pyspy_main_thread_and_shares(self):
        raw = ("thread (1): MainThread;<module> (b.py:9);Runner.fit (/r/tools/bench_board_algos.py:3094);"
               "seq_tensor_upload (_mojolearn_native_sequence.so);cuMemcpyHtoDAsync_v2 (libcuda.so.595) 20\n"
               "thread (1): MainThread;<module> (b.py:9);Runner.fit (/r/tools/bench_board_algos.py:3094);"
               "seq_tensor_upload (_mojolearn_native_sequence.so) 30\n"
               "thread (1): MainThread;<module> (b.py:9);Runner.fit (/r/tools/bench_board_algos.py:3094) 40\n"
               "thread (1): MainThread;<module> (b.py:9);0x7f00 (libKGENCompilerRTShared.so) 10\n"
               "thread (2): AsyncRT;0x7f01 (libAsyncRTRuntimeGlobals.so) 99\n")
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "c"), raw)
            write(os.path.join(d, "w.jsonl"), window_lines(4))
            rc = CP.main(["pyspy", "--raw", os.path.join(d, "c"), "--window", os.path.join(d, "w.jsonl"),
                          "--out", os.path.join(d, "p.json")])
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "p.json")) as fh:
                p = json.load(fh)
            self.assertEqual(p["thread"], "thread (1): MainThread")
            self.assertEqual(p["sample_count"], 100)
            self.assertEqual(p["other_threads"], {"thread (2): AsyncRT": 99})
            self.assertEqual(p["self"][0]["frame"], "Runner.fit (/r/tools/bench_board_algos.py)")   # lines merged
            self.assertAlmostEqual(p["self"][0]["us_per_call"], 40000.0)          # 40 samples / 250 Hz / 4 calls
            self.assertEqual(p["shim_share"], 0.2)                                # the libcuda leaf: the shim's
            self.assertEqual(p["shares"], {"bindings": 0.3, "cpython": 0.4, "mojo_runtime": 0.1})
            self.assertEqual(p["bindings"], [{"frame": "seq_tensor_upload (_mojolearn_native_sequence.so)",
                                              "share": 0.3}])

    def test_callgrind_self_inclusive_and_recursion(self):
        g = CP.callgrind_parse(CALLGRIND.splitlines(True))
        self.assertEqual(g["pid"], 77)
        self.assertEqual(g["trigger"], "Client Request: callprof-window")
        so, lc = "/v/site-packages/mojolearn/_mojolearn_nn.so", "/usr/lib/x86_64-linux-gnu/libc.so.6"
        self.assertEqual(g["self"][("run_layer", so)], 100)
        self.assertEqual(g["inclusive"][("run_layer", so)], 130)
        self.assertEqual(g["self"][("helper", so)], 20)
        self.assertEqual(g["inclusive"][("helper", so)], 60)
        self.assertEqual(g["self"][("g", lc)], 40)                 # g and g'2 merged for self
        self.assertEqual(g["inclusive"][("g", lc)], 40)            # g'2's inclusive is inside g's: not added again
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "cg", "callgrind.out.77.1"), CALLGRIND)
            write(os.path.join(d, "cg", "callgrind.out.77"), "pid: 77\nevents: Ir\nsummary: 0\n")   # the exit dump
            write(os.path.join(d, "w.jsonl"), window_lines(2, pid=77))
            rc = CP.main(["callgrind", "--dir", os.path.join(d, "cg"), "--window", os.path.join(d, "w.jsonl"),
                          "--out", os.path.join(d, "c.json")])
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "c.json")) as fh:
                c = json.load(fh)
            self.assertEqual(c["ir_per_call"], 95.0)                   # 190 Ir of self over 2 calls
            self.assertEqual(c["self"][0], {"frame": "run_layer [_mojolearn_nn.so]", "category": "bindings",
                                            "ir_per_call": 50.0, "pct": 52.63})
            self.assertEqual(c["categories"], {"bindings": 60.0, "libc": 20.0})
            self.assertEqual(c["shim_ir_per_call"], 15.0)               # malloc: the shim's
            self.assertEqual(len(c["files"]), 2)


class Report(unittest.TestCase):
    def _res(self, d, with_nsys=True, with_time=True, with_sources=True):
        rows = "algos/a/synthetic/rows=full\tfit\tname=a\n"
        write(os.path.join(d, "rows.tsv"), rows)
        rd = os.path.join(d, "rows", "a")
        if with_time:
            write(os.path.join(rd, "time", "window.jsonl"), window_lines(10, 5000000))       # 500 us per call
        write(os.path.join(rd, "pyspy", "window.jsonl"), window_lines(100, 60000000))
        if with_sources:
            write(os.path.join(rd, "pyspy", "pyspy.json"), json.dumps(
                {"samples": 50, "cpu_ms_per_call": 0.4, "weight": "w", "shim_share": 0.2,
                 "shares": {"bindings": 0.4, "cpython": 0.2, "mojo_runtime": 0.1, "unresolved": 0.1},
                 "bindings": [{"frame": "run_layer (_mojolearn_nn.so)", "share": 0.4}],
                 "self": [{"frame": "<module> (b.py)", "pct": 50.0, "us_per_call": 200.0},
                          {"frame": "run_layer (_mojolearn_nn.so)", "pct": 30.0, "us_per_call": 120.0}],
                 "inclusive": [{"frame": "<module> (b.py)", "pct": 100.0, "us_per_call": 400.0}]}))
            # shim: 100 us of CUDA (a sync of 80, two launches of 10) and 20 us of libc per call
            write(os.path.join(rd, "shim", "window.jsonl"), window_lines(
                4, 2000000, {"cuStreamSynchronize": [0, 0, 0], "cuLaunchKernelEx": [0, 0, 0], "malloc": [0, 0, 0]},
                {"cuStreamSynchronize": [4, 320000, 0], "cuLaunchKernelEx": [8, 80000, 0], "malloc": [40, 80000, 4096]}))
        if with_nsys:
            write(os.path.join(rd, "nsys", "nsys.json"), json.dumps(
                {"captured": True, "api_ms": 0.2, "kernel_ms": 0.05, "sync_ms": 0.1, "window_ms": 0.7,
                 "counts": {"launch": 3.0, "sync": 1.0, "alloc": 2.0}, "api": {"cuMemAlloc_v2": {"calls": 2.0, "ms": 0.1}},
                 "mem": {"h2d": {"count": 1.0, "mb": 0.01, "ms": 0.01}}}))

    def test_report_splits_the_host_time(self):
        with tempfile.TemporaryDirectory() as d:
            self._res(d)
            rc = CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv"), "--wheel", "w1"])
            self.assertEqual(rc, CP.EXIT_OK)
            with open(os.path.join(d, "CALL-PROFILE-nvidia.json")) as fh:
                doc = json.load(fh)
            r = doc["rows"][0]
            self.assertAlmostEqual(r["wall_ms_per_call"], 0.5)                    # the time pass, not py-spy's 0.6
            self.assertAlmostEqual(r["split"]["host_outside_cuda_ms"], 0.3)
            h = r["host"]
            self.assertEqual((h["cuda_us"], h["libc_us"], h["rest_us"]), (100.0, 20.0, 380.0))   # of the shim's 500
            self.assertEqual(h["split_wall_us"], 500.0)
            self.assertEqual(h["cuda"]["cuStreamSynchronize"], {"calls": 1.0, "us": 80.0, "bytes": 0.0})
            self.assertEqual(h["source"], "py-spy")
            # the rest (380 us) by the sampled shares without the shim's 20 %: bindings 0.4 / 0.8 -> 190 us
            self.assertEqual(h["parts_us"], {"bindings": 190.0, "cpython": 95.0, "mojo_runtime": 47.5})
            self.assertEqual(h["unexplained_us"], 47.5)
            self.assertEqual(h["bindings"], [{"frame": "run_layer (_mojolearn_nn.so)", "us": 190.0}])
            self.assertEqual(r["function_sources"], ["pyspy", "shim"])
            self.assertEqual(r["largest_host_costs"][0]["frame"], "run_layer (_mojolearn_nn.so)")
            with open(os.path.join(d, "CALL-PROFILE-nvidia.md")) as fh:
                md = fh.read()
            self.assertIn("LARGEST HOST COSTS (pyspy, self per call): run_layer (_mojolearn_nn.so) 120.0 us", md)
            self.assertIn("CUDA driver per call (shim): cuStreamSynchronize 1.00 x 80.00 us", md)
            self.assertIn("| a | fit | 0.500 | 0.050 | 0.200 | 3.0 | 1.0 | 2.0 | 1.0 (0.01 MB) | - | 190.0 | 95.0 | 47.5 | 100.0 "
                          "| 20.0 | 0.0 | 47.5 | py-spy |", md)

    def test_report_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            self._res(d, with_nsys=False)
            self.assertEqual(CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv")]),
                             CP.EXIT_NO_PROFILE)
        with tempfile.TemporaryDirectory() as d:
            self._res(d, with_sources=False)                                      # nsys alone is not enough
            self.assertEqual(CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv")]),
                             CP.EXIT_NO_PROFILE)
        with tempfile.TemporaryDirectory() as d:
            self._res(d, with_time=False)                                          # the shim's window stands in
            self.assertEqual(CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv")]), CP.EXIT_OK)
            for p in ("shim", "pyspy"):
                shutil.rmtree(os.path.join(d, "rows", "a", p))
            self.assertEqual(CP.main(["report", "--res", d, "--rows-file", os.path.join(d, "rows.tsv")]),
                             CP.EXIT_NO_WALL)


class Job(unittest.TestCase):
    def test_syntax_and_refusal(self):
        self.assertEqual(subprocess.run(["bash", "-n", JOB]).returncode, 0)
        env = {k: v for k, v in os.environ.items() if k not in ("MOJOLEARN_BOX_PERSIST", "MOJOLEARN_ALLOW_UNPERSISTED")}
        r = subprocess.run(["bash", JOB], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("box_persist_run", r.stderr)

    @unittest.skipUnless(shutil.which("cc") and sys.platform.startswith("linux"), "glibc and a C compiler")
    def test_shims_compile(self):
        r = subprocess.run(["cc", "-fsyntax-only", "-Wall", "-Werror", os.path.join(HERE, "ops", "call_profile_shim.c")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        if os.path.isfile("/usr/include/valgrind/callgrind.h"):
            r = subprocess.run(["cc", "-fsyntax-only", os.path.join(HERE, "ops", "call_profile_cg.c")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])

    def test_default_passes_are_known(self):
        with open(JOB) as fh:
            body = fh.read()
        self.assertIn("CP_PASSES:-%s}" % ",".join(CP.DEFAULT_PASSES), body)
        for p in CP.PASSES:
            self.assertIn("        %s)" % p, body)                               # run_pass has a case per pass


if __name__ == "__main__":
    unittest.main()


def test_board_order_rows_parse():
    # lane fx-harness (2026-11-18): fit+infer profiles each fit and the first predict right after it
    rows = CP.parse_rows("algos/mlp-reg/istella/rows=full fit+infer\nalgos/mlp-reg/istella/rows=full infer\n")
    assert [r.phase for r in rows] == ["fit+infer", "infer"]
    assert rows[0].name == "mlp-reg" and rows[1].name == "mlp-reg-infer"
    assert CP._h2d({"count": 2.0, "mb": 88.0}) == "2.0 (88.00 MB)" and CP._h2d(None) == "-"
    assert any(r.phase == "fit+infer" for r in CP.load_rows(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ops", "call_profile_rows_unexplained.tsv")))


def test_board_order_forces_both_phases_to_one_count():
    env = {SR.FORCE_CALLS_ENV: "20", SR.FORCE_CALLS_PHASE_ENV: SR.BOARD_ORDER}
    assert SR.forced_calls(True, "fit", env) == 20 and SR.forced_calls(True, "infer", env) == 20
    assert SR.forced_calls(False, "fit", env) is None              # opponents are never forced
    assert SR.paired_calls(900.0, 100.0, True, env) == 20
    env[SR.PROFILE_SECONDS_ENV] = "30"                              # 30 s at 1 s per pair: 31 pairs
    assert SR.paired_calls(900.0, 100.0, True, env) == 31
    assert SR.paired_calls(900.0, 100.0, True, {SR.FORCE_CALLS_ENV: "20", SR.FORCE_CALLS_PHASE_ENV: "infer"}) is None
    assert SR.fit_warmup_skipped(env) is False                      # the board's warmup fit runs in board order
