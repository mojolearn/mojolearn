# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ops/attribution_job.sh, tools/ops/attribution_report.py and the harness knobs they set
(tools/bench_board_small_rows.py forced calls, NVTX range, fit warmup; tools/bench_board.py own arms), on canned
inputs. `python3 -m unittest tools.test_attribution_job`."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.join(HERE, "ops", "attribution_job.sh")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SR = _load("bench_board_small_rows_t", os.path.join(HERE, "bench_board_small_rows.py"))
AR = _load("attribution_report_t", os.path.join(HERE, "ops", "attribution_report.py"))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


API_CSV = """Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
40.0,1900000,190,10000,10000,9000,11000,100,cuLaunchKernel
30.0,3800000,38,100000,100000,90000,110000,100,cuStreamSynchronize
10.0,190000,19,10000,10000,10000,10000,0,cuMemAlloc_v2
5.0,95000,19,5000,5000,5000,5000,0,cuMemFree_v2
5.0,380000,38,10000,10000,10000,10000,0,cuMemcpyHtoDAsync_v2
4.0,19000,19,1000,1000,1000,1000,0,cuMemHostRegister_v2
1.0,19000,19,1000,1000,1000,1000,0,cuCtxGetDevice
"""
KERN_CSV = """Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
90.0,9500000,190,50000,50000,40000,60000,10,pinned_distance_register_tile_kernel_float32
10.0,950000,190,5000,5000,5000,5000,0,reduce_kernel
"""
SIZE_CSV = """Total (MB),Count,Avg (MB),Med (MB),Min (MB),Max (MB),StdDev (MB),Operation
38.0,38,1.0,1.0,1.0,1.0,0.0,[CUDA memcpy Host-to-Device]
1.9,19,0.1,0.1,0.1,0.1,0.0,[CUDA memcpy Device-to-Host]
"""
TIME_CSV = """Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation
90.0,1900000,38,50000,50000,50000,50000,0,[CUDA memcpy Host-to-Device]
10.0,190000,19,10000,10000,10000,10000,0,[CUDA memcpy Device-to-Host]
"""
NVTX_CSV = """Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Style,Range
100.0,19000000,1,19000000,19000000,19000000,19000000,0,PushPop,:attr-calls
"""
TRACE_CSV = """Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MBps),SrcMemKd,DstMemKd,Device,Ctx,Strm,Name
100,50000,1,64,1,1,256,1,1,64,0.0,0.0,,,,,0,1,7,pinned_distance_register_tile_kernel_float32
300,40000,2,64,1,1,256,1,1,64,0.0,0.0,,,,,0,1,7,pinned_distance_register_tile_kernel_float32
500,5000,3,8,1,1,128,1,1,32,0.0,0.0,,,,,0,1,7,reduce_kernel
"""
NCU_CSV = """==PROF== Connected to process 123
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","123","python","h","pinned_distance_register_tile_kernel","1","7","(256, 1, 1)","(64, 1, 1)","0","8.9","LaunchStats","Registers Per Thread","register/thread","64"
"0","123","python","h","pinned_distance_register_tile_kernel","1","7","(256, 1, 1)","(64, 1, 1)","0","8.9","Occupancy","Theoretical Occupancy","%","66.67"
"0","123","python","h","pinned_distance_register_tile_kernel","1","7","(256, 1, 1)","(64, 1, 1)","0","8.9","Occupancy","Achieved Occupancy","%","41.20"
"0","123","python","h","pinned_distance_register_tile_kernel","1","7","(256, 1, 1)","(64, 1, 1)","0","8.9","GPU Speed Of Light Throughput","Duration","usecond","52.10"
"""


class SmallRowsKnobs(unittest.TestCase):
    def test_forced_calls_by_phase(self):
        env = {SR.FORCE_CALLS_ENV: "20", SR.FORCE_CALLS_PHASE_ENV: "infer/test"}
        self.assertEqual(SR.forced_calls(True, "infer/test", env), 20)
        self.assertIsNone(SR.forced_calls(True, "infer/large", env))
        self.assertIsNone(SR.forced_calls(True, "fit", env))
        self.assertIsNone(SR.forced_calls(False, "infer/test", env))        # opponents are never repeated
        self.assertIsNone(SR.forced_calls(True, None, env))                 # a site without a phase: no match
        self.assertEqual(SR.forced_calls(True, "infer/Xq", {SR.FORCE_CALLS_ENV: "20",
                                                            SR.FORCE_CALLS_PHASE_ENV: "infer"}), 20)
        self.assertEqual(SR.forced_calls(True, None, {SR.FORCE_CALLS_ENV: "1"}), 1)   # the one-call arm
        self.assertIsNone(SR.forced_calls(True, "fit", {SR.FORCE_CALLS_ENV: "x"}))
        self.assertIsNone(SR.forced_calls(True, "fit", {SR.FORCE_CALLS_ENV: "0"}))
        self.assertIsNone(SR.forced_calls(True, "fit", {}))

    def test_calls_for_unchanged_on_a_board_and_forced_in_an_attribution_run(self):
        old = {k: os.environ.pop(k) for k in (SR.FORCE_CALLS_ENV, SR.FORCE_CALLS_PHASE_ENV) if k in os.environ}
        try:
            self.assertEqual(SR.calls_for(5.0, True, "fit"), 20)
            self.assertEqual(SR.calls_for(50.0, True, "fit"), 1)
            os.environ[SR.FORCE_CALLS_ENV] = "20"
            os.environ[SR.FORCE_CALLS_PHASE_ENV] = "fit"
            self.assertEqual(SR.calls_for(500.0, True, "fit"), 20)
            self.assertEqual(SR.calls_for(500.0, True, "infer/test"), 1)
            self.assertEqual(SR.calls_for(500.0, False, "fit"), 1)
        finally:
            for k in (SR.FORCE_CALLS_ENV, SR.FORCE_CALLS_PHASE_ENV):
                os.environ.pop(k, None)
            os.environ.update(old)

    def test_fit_warmup_skipped_only_for_a_forced_inference(self):
        self.assertTrue(SR.fit_warmup_skipped({SR.FORCE_CALLS_ENV: "20", SR.FORCE_CALLS_PHASE_ENV: "infer/test"}))
        self.assertFalse(SR.fit_warmup_skipped({SR.FORCE_CALLS_ENV: "20", SR.FORCE_CALLS_PHASE_ENV: "fit"}))
        self.assertFalse(SR.fit_warmup_skipped({SR.FORCE_CALLS_PHASE_ENV: "infer/test"}))
        self.assertFalse(SR.fit_warmup_skipped({}))

    def test_nvtx_range_around_forced_extra_calls_only(self):
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "nvtx.py"), "LOG = []\n"
                  "def push_range(message=None):\n    LOG.append(('push', message))\n"
                  "def pop_range():\n    LOG.append(('pop',))\n")
            env = {SR.NVTX_RANGE_ENV: "attr-calls", SR.NVTX_PATH_ENV: d, SR.FORCE_CALLS_ENV: "20",
                   SR.FORCE_CALLS_PHASE_ENV: "fit"}
            sys.modules.pop("nvtx", None)
            try:
                self.assertFalse(SR.range_open(20, "fit", {}))                      # a board: no-op
                self.assertFalse(SR.range_open(1, "fit", env))                      # no extra calls
                self.assertFalse(SR.range_open(20, "infer/Xq", env))                # an unforced small row
                self.assertTrue(SR.range_open(20, "fit", env))
                self.assertFalse(SR.range_open(20, "fit", env))                     # one range at a time
                SR.range_close()
                SR.range_close()                                                    # nothing open: no-op
                import nvtx
                self.assertEqual(nvtx.LOG, [("push", "attr-calls"), ("pop",)])
            finally:
                sys.modules.pop("nvtx", None)
                if d in sys.path:
                    sys.path.remove(d)


class OwnArms(unittest.TestCase):
    def test_own_arms_narrow_the_race_so_the_preflight_expects_only_them(self):
        """fb3-attr-tool (2026-11-17 L40S job): `ours-prebinned` alone was refused by the preflight ("missing
        parameter/version probe: ours") because only the driver command was narrowed."""
        try:
            bb = _load("bench_board_t2", os.path.join(HERE, "bench_board.py"))
        except Exception as exc:                  # noqa: BLE001
            self.skipTest("tools/bench_board.py does not import here: %s" % exc)
        tree = {"id": "trees/gbdt-depthwise/istella/rows=full", "family": "trees", "arms": ["ours", "ours-prebinned",
                "xgboost-gpu"], "our_arms": {"ours": "identical", "ours-prebinned": "identical"},
                "opponents": ["xgboost-gpu"]}
        neural = {"id": "neural/samba-train-step/bytes/shape=full", "family": "neural", "arms": ["ours"],
                  "our_arms": {"ours": "identical"}, "opponents": []}
        races = [tree]
        self.assertIs(bb.select_own_arms(races, {}), races)                  # unset (every board): untouched
        got = bb.select_own_arms([tree, neural], {bb.OWN_ARMS_ENV: "ours-prebinned"})
        self.assertEqual(got[0]["our_arms"], {"ours-prebinned": "identical"})
        self.assertEqual(got[0]["arms"], ["ours-prebinned", "xgboost-gpu"])
        self.assertIs(got[1], neural)                                         # only tree races take own arms
        got = bb.select_own_arms([tree], {bb.OWN_ARMS_ENV: "ours-prebinned,ours"})
        self.assertEqual(list(got[0]["our_arms"]), ["ours-prebinned", "ours"])
        with self.assertRaises(ValueError):
            bb.select_own_arms([tree], {bb.OWN_ARMS_ENV: "ours-gain0"})

    def test_tree_cmd_takes_the_own_arms_in_order_and_refuses_unplanned(self):
        try:
            bb = _load("bench_board_t", os.path.join(HERE, "bench_board.py"))
        except Exception as exc:                  # noqa: BLE001
            self.skipTest("tools/bench_board.py does not import here: %s" % exc)
        ctx = {"python": "py", "tree_driver": "drv", "vendor": "nvidia", "rounds": 1, "arm_budget_s": 1,
               "race_deadline_s": 1}
        race = {"our_arms": {"ours": "identical", "ours-prebinned": "identical"}, "lane": "gbdt-depthwise",
                "dataset": "istella", "rows": None, "opponents": [], "family": "trees",
                "arms": ["ours", "ours-prebinned"]}

        def own(env):
            r = bb.select_own_arms([race], {} if env is None else {bb.OWN_ARMS_ENV: env})[0]
            cmd, _ = bb.tree_cmd(ctx, r)
            return cmd[cmd.index("--own-arms") + 1]
        self.assertEqual(own(None), "ours,ours-prebinned")
        self.assertEqual(own("ours-prebinned"), "ours-prebinned")
        self.assertEqual(own("ours-prebinned,ours"), "ours-prebinned,ours")
        with self.assertRaises(ValueError):
            own("ours-gain0")


class NsysParse(unittest.TestCase):
    def capture(self, d, trace=True):
        p = os.path.join(d, "knn")
        write(p + "_cuda_api_sum.csv", API_CSV)
        write(p + "_cuda_gpu_kern_sum.csv", KERN_CSV)
        write(p + "_cuda_gpu_mem_size_sum.csv", SIZE_CSV)
        write(p + "_cuda_gpu_mem_time_sum.csv", TIME_CSV)
        write(p + "_nvtx_sum.csv", NVTX_CSV)
        if trace:
            write(p + "_cuda_gpu_trace.csv", TRACE_CSV)
        return p

    def test_api_kinds(self):
        self.assertEqual(AR.api_kind("cuLaunchKernel"), "launch")
        self.assertEqual(AR.api_kind("cuLaunchKernelEx"), "launch")
        self.assertEqual(AR.api_kind("cudaLaunchKernel_v7000"), "launch")
        self.assertEqual(AR.api_kind("cuStreamSynchronize_ptsz"), "sync")
        self.assertEqual(AR.api_kind("cuCtxSynchronize"), "sync")
        self.assertEqual(AR.api_kind("cuMemAllocAsync"), "alloc")
        self.assertEqual(AR.api_kind("cuMemAlloc_v2"), "alloc")
        self.assertEqual(AR.api_kind("cuMemFreeAsync"), "free")
        self.assertEqual(AR.api_kind("cuMemcpyDtoHAsync_v2"), "copy_api")
        self.assertEqual(AR.api_kind("cuMemHostRegister_v2"), "host_register")
        self.assertEqual(AR.api_kind("cuCtxGetDevice"), "other")

    def test_per_call_counts_from_one_capture(self):
        with tempfile.TemporaryDirectory() as d:
            got = AR.parse_nsys(self.capture(d), 19)
        self.assertTrue(got["captured"])
        c = got["counts"]
        self.assertAlmostEqual(c["launch"], 10.0)
        self.assertAlmostEqual(c["sync"], 2.0)
        self.assertAlmostEqual(c["alloc"], 1.0)
        self.assertAlmostEqual(c["free"], 1.0)
        self.assertAlmostEqual(c["host_register"], 1.0)
        self.assertAlmostEqual(c["other_api"], 1.0)
        self.assertAlmostEqual(got["kernel_ms"], 10450000 / 19 / 1e6)
        self.assertAlmostEqual(got["kernel_instances"], 20.0)
        self.assertAlmostEqual(got["mem"]["h2d"]["count"], 2.0)
        self.assertAlmostEqual(got["mem"]["h2d"]["mb"], 2.0)
        self.assertAlmostEqual(got["mem"]["d2h"]["count"], 1.0)
        self.assertAlmostEqual(got["mem"]["d2h"]["ms"], 0.01)
        self.assertAlmostEqual(got["window_ms"], 1.0)
        self.assertAlmostEqual(got["sync_ms"], 0.2)
        api = (1900000 + 3800000 + 190000 + 95000 + 380000 + 19000 + 19000) / 19 / 1e6
        self.assertAlmostEqual(got["api_ms"], api)
        self.assertAlmostEqual(got["host_ms"], 1.0 - api)
        self.assertEqual([k["name"] for k in got["kernels"]], ["pinned_distance_register_tile_kernel_float32"])
        self.assertAlmostEqual(got["kernels"][0]["instances_per_call"], 10.0)
        self.assertAlmostEqual(got["kernels"][0]["avg_us"], 50.0)
        shapes = got["kernel_launch_shapes"]["pinned_distance_register_tile_kernel_float32"]
        self.assertEqual(len(shapes), 1)
        self.assertEqual((shapes[0]["block"], shapes[0]["regs"], shapes[0]["launches"]), (256, 64, 2))
        self.assertAlmostEqual(shapes[0]["med_us"], 45.0)

    def test_no_capture_is_reported_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            got = AR.parse_nsys(os.path.join(d, "none"), 19)
        self.assertFalse(got["captured"])
        self.assertIsNone(got["counts"]["launch"])
        self.assertIsNone(got["kernel_ms"])
        self.assertIsNone(got["host_ms"])

    def test_notice_lines_before_the_header_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x")
            write(p + "_cuda_api_sum.csv", "NOTICE: Processing\n\n" + API_CSV)
            self.assertAlmostEqual(AR.parse_nsys(p, 19)["counts"]["launch"], 10.0)

    def test_cli_writes_json_and_exits_3_without_capture(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.capture(d, trace=False)
            out = os.path.join(d, "knn.json")
            r = subprocess.run([sys.executable, os.path.join(HERE, "ops", "attribution_report.py"), "nsys",
                                "--prefix", p, "--calls", "19", "--out", out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("NSYS captured=True launches=10.0", r.stdout)
            self.assertTrue(json.load(open(out))["captured"])
            r = subprocess.run([sys.executable, os.path.join(HERE, "ops", "attribution_report.py"), "nsys",
                                "--prefix", os.path.join(d, "none"), "--calls", "19", "--out", out],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 3)


class Occupancy(unittest.TestCase):
    def test_sm89_limits(self):
        # 256 threads, 64 registers: 2048 registers per warp -> 32 warps by registers -> 4 blocks of 8 warps
        o = AR.occupancy_sm89(256, 64, 0)
        self.assertEqual((o["blocks_per_sm"], o["limiter"]), (4, "registers"))
        self.assertAlmostEqual(o["occupancy"], 32 / 48)
        # 256 threads, 32 registers: 6 blocks by warps (48 / 8)
        o = AR.occupancy_sm89(256, 32, 0)
        self.assertEqual((o["blocks_per_sm"], o["limiter"]), (6, "warps"))
        self.assertAlmostEqual(o["occupancy"], 1.0)
        # 48 KB shared memory per block: 2 blocks (100 KB / 49 KB)
        o = AR.occupancy_sm89(128, 32, 48 * 1024)
        self.assertEqual((o["blocks_per_sm"], o["limiter"]), (2, "shared_memory"))
        # 32 threads: the 24-block limit
        self.assertEqual(AR.occupancy_sm89(32, 16, 0)["limiter"], "blocks")
        self.assertIsNone(AR.occupancy_sm89(0, 32, 0))


class NcuParse(unittest.TestCase):
    def test_metrics_and_permission(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ncu.csv")
            write(p, NCU_CSV)
            got = AR.parse_ncu(p)
            self.assertEqual(got["status"], "ok")
            k = got["kernels"]["pinned_distance_register_tile_kernel"]
            self.assertEqual(k["regs"], 64.0)
            self.assertAlmostEqual(k["achieved_occupancy_pct"], 41.2)
            self.assertAlmostEqual(k["duration_us"], 52.1)
            write(p, "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU "
                     "Performance Counters on the target device 0.\n")
            self.assertEqual(AR.parse_ncu(p)["status"], "no-permission")
            self.assertEqual(AR.parse_ncu(os.path.join(d, "absent.csv"))["status"], "missing")


FSPEED_LOG = """FSPEED-WARMUP lane=gbdt-depthwise arm=ours shape=full ms=9000.0
FSPEED-INFER-WARMUP lane=gbdt-depthwise arm=ours-prebinned batch=test rows=500000 ms=80.000
FSPEED-INFER lane=gbdt-depthwise arm=ours-prebinned batch=test rows=500000 round=1 ms=37.500 hash=abc calls=20 min_ms=36.000000 max_ms=45.000000
FSPEED-INFER lane=gbdt-depthwise arm=ours batch=test rows=500000 round=1 ms=36.000 hash=abd calls=20 min_ms=35.500000 max_ms=40.000000
FSPEED-INFER lane=gbdt-depthwise arm=ours-prebinned batch=large rows=1000000 round=1 ms=73.000 hash=abe
"""


class Timing(unittest.TestCase):
    def test_driver_lines_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "race.log")
            write(p, FSPEED_LOG)
            got = AR.parse_driver_lines(p, "infer/test")
            self.assertEqual(got["ours-prebinned"][0]["order"], 0)
            self.assertEqual(got["ours"][0]["order"], 1)
            self.assertEqual(got["ours-prebinned"][0]["calls"], 20)
            self.assertAlmostEqual(got["ours-prebinned"][0]["max_ms"], 45.0)
            large = AR.parse_driver_lines(p, "infer/large")
            self.assertEqual(large["ours-prebinned"][0]["calls"], 1)
            self.assertEqual(AR.parse_driver_lines(p, "fit"), {})

    def test_board_cell(self):
        board = {"races": {"r": {"cells": [{"arm": "ours", "median_ms": 5.0, "calls": 20, "min_ms": 4.0,
                                            "max_ms": 9.0, "status": "ok"}],
                                 "infer_cells": [{"arm": "ours", "batch": "Xq", "median_ms": 2.0, "calls": 1}]}}}
        self.assertEqual(AR.board_cell(board, "r", "fit", "ours")["max_ms"], 9.0)
        self.assertEqual(AR.board_cell(board, "r", "infer/Xq", "ours")["ms"], 2.0)
        self.assertIsNone(AR.board_cell(board, "r", "fit", "ours-gain0"))


class Report(unittest.TestCase):
    def fake_res(self, d):
        knn = AR.ROW_BY_NAME["knn"]
        dep = AR.ROW_BY_NAME["gbdt-depthwise"]
        for arm, ms, launches in (("candidate", 30.0, 12), ("prior-0.8.55", 25.0, 10)):
            write(os.path.join(d, "time-%s" % arm, "knn", "planned", "board.json"), json.dumps(
                {"races": {knn.race_id: {"log": "logs/x.log", "cells": [
                    {"arm": "ours", "median_ms": ms, "calls": 20, "min_ms": ms - 1, "max_ms": ms + 3}]}}}))
            for tag in ("ours-prebinned", "ours+ours-prebinned", "ours-prebinned+ours"):
                write(os.path.join(d, "time-%s" % arm, "gbdt-depthwise", tag, "logs", "trees.log"), FSPEED_LOG)
            prof = {"captured": True, "counts": {"launch": launches, "sync": 2, "alloc": 1, "free": 1,
                                                 "host_register": 0},
                    "kernel_ms": ms * 0.5, "kernel_instances": launches, "sync_ms": 1.0, "api_ms": 3.0,
                    "window_ms": ms, "host_ms": ms - 3.0, "mem": {"h2d": {"count": 2, "mb": 1.0, "ms": 0.1},
                                                                  "d2h": {"count": 1, "mb": 0.1, "ms": 0.01}},
                    "kernels": [{"name": "pinned_distance_register_tile_kernel", "instances_per_call": 10,
                                 "total_ms_per_call": 0.5, "avg_us": 50.0, "med_us": 50.0, "min_us": 40.0,
                                 "max_us": 60.0}],
                    "kernel_launch_shapes": {"pinned_distance_register_tile_kernel": [
                        {"block": 256, "grid": 64, "regs": 64, "smem_bytes": 0, "launches": 10, "med_us": 50.0,
                         "occupancy": AR.occupancy_sm89(256, 64, 0)}]}}
            write(os.path.join(d, "profile-%s" % arm, "knn.json"), json.dumps(prof))
            write(os.path.join(d, "ncu-%s" % arm, "knn.json"), json.dumps({"status": "no-permission",
                                                                           "kernels": {}}))
            write(os.path.join(d, "job", "version-%s.txt" % arm), "0.8.x venv /v manifest none")
        write(os.path.join(d, "job", "cores.log"), "10:00:00 busy=1.0 total=32 phase=time source=cgroup\n"
                                                   "10:01:00 busy=30.5 total=32 phase=stats source=cgroup\n")
        write(os.path.join(d, "job", "tools.txt"), "nsys 2025.1 at /usr/local/cuda/bin/nsys (on the pod)\n")
        return knn, dep

    def test_report_reads_both_wheels_and_writes_md_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake_res(d)
            r = subprocess.run([sys.executable, os.path.join(HERE, "ops", "attribution_report.py"), "report",
                                "--res", d, "--vendor", "nvidia", "--arms", "candidate,prior-0.8.55"],
                               capture_output=True, text=True)
            # seven rows have no run here: exit 7 (a row without timing), each row named by a SUMMARY line
            self.assertEqual(r.returncode, AR.EXIT_NO_TIMING, r.stderr)
            self.assertIn("ATTRIBUTION rows=9 with_timing=2 with_profile=1", r.stdout)
            self.assertIn("SUMMARY row=knn timing=ok profile=ok ncu=candidate no-permission,prior-0.8.55 "
                          "no-permission", r.stdout)
            self.assertIn("SUMMARY row=mamba3-forward timing=FAILED(candidate planned: not run", r.stdout)
            md = open(os.path.join(d, "ATTRIBUTION-nvidia.md")).read()
            doc = json.load(open(os.path.join(d, "ATTRIBUTION-nvidia.json")))
        self.assertIn("SERIAL", md)
        self.assertIn("busy cores: max 30.5 of 32", md)
        knn = doc["rows"]["knn"]
        self.assertEqual(knn["timing"]["candidate"]["ms"], 30.0)
        self.assertIn("candidate +5.00 ms (+20%) vs prior-0.8.55", knn["reading"])
        self.assertIn("largest change: host outside CUDA +5.00 ms", knn["reading"])
        self.assertIn("launches 10->12", knn["reading"])
        self.assertIn("| kernel launches | 12.0 | 10.0 |", md)
        self.assertIn("ncu: no-permission", md)
        self.assertIn("occ 0.67 (registers)", md)
        dep = doc["rows"]["gbdt-depthwise"]
        self.assertEqual(dep["timing"]["candidate"]["ms"], 37.5)        # the prebinned arm alone
        self.assertEqual(set(dep["order_test"]), {"ours+ours-prebinned", "ours-prebinned+ours"})
        self.assertIn("Arm-order test", md)
        self.assertIn("ours-prebinned* 37.50", md)
        self.assertEqual(doc["rows"]["mamba3-forward"]["reading"], "no timing")

    def test_one_wheel_reading(self):
        row = AR.ROW_BY_NAME["ols"]
        t = {"candidate": {"ms": 16.0}}
        p = {"candidate": {"captured": True, "kernel_ms": 2.0, "sync_ms": 1.0, "api_ms": 3.0, "host_ms": 11.0}}
        self.assertIn("most of it host outside CUDA (11.00 ms)", AR.reading(row, ["candidate"], t, p))
        self.assertIn("no profile", AR.reading(row, ["candidate"], t, {}))


class PlanAndEstimate(unittest.TestCase):
    def test_plan_has_the_arm_order_test_and_profiles_every_row(self):
        lines = [ln.split("\t") for ln in AR.plan_lines()]
        dep = [x for x in lines if x[1] == "gbdt-depthwise" and x[0] == "time"]
        self.assertEqual([x[4] for x in dep], ["ours-prebinned", "ours,ours-prebinned", "ours-prebinned,ours"])
        self.assertTrue(all(x[3] == "infer/test" for x in dep))
        self.assertEqual({x[1] for x in lines if x[0] == "profile"}, set(AR.ROW_BY_NAME))
        self.assertEqual({x[1] for x in lines if x[0] == "kernels"}, {"hdbscan", "knn", "kde"})
        prof = {x[1]: x[4] for x in lines if x[0] == "profile"}
        self.assertEqual(prof["gbdt-lossguide"], "ours-gain0")
        self.assertEqual(prof["knn"], "-")

    def test_estimate_from_stored_times(self):
        rec = {"recipes": [
            {"race_id": AR.ROW_BY_NAME["gbdt-lossguide"].race_id, "phase": "fit", "arm": "ours-gain0", "batch": None,
             "v0855": {"median_ms": 900000.0}},
            {"race_id": AR.ROW_BY_NAME["gbdt-lossguide"].race_id, "phase": "infer", "arm": "ours-gain0",
             "batch": "test", "v0855": {"median_ms": 40.0}},
            {"race_id": AR.ROW_BY_NAME["gbdt-lossguide"].race_id, "phase": "fit", "arm": "ours", "batch": None,
             "v0855": {"median_ms": 900000.0}}]}
        est = AR.estimate(rec, overhead_s=0.0, profile_slowdown=1.0, setup_s=0.0)
        lg = next(p for p in est["rows"] if p["row"] == "gbdt-lossguide")
        # one timing run + one profile run, each: one fit (no warmup fit), the inference warmup + 20 calls,
        # the large batch unforced (none stored here)
        self.assertAlmostEqual(lg["seconds"], 2 * (900.0 + 21 * 0.040), places=1)
        self.assertEqual(lg["runs"], 2)


class RowsFile(unittest.TestCase):
    def test_default_file_holds_the_nine_rows(self):
        self.assertEqual([r.name for r in AR.ROWS], ["mamba3-forward", "samba-train-step", "ols", "gbdt-depthwise",
                                                     "gbdt-lossguide", "iforest", "hdbscan", "knn", "kde"])
        dep = AR.ROW_BY_NAME["gbdt-depthwise"]
        self.assertEqual(dep.timing, [["ours-prebinned"], ["ours", "ours-prebinned"], ["ours-prebinned", "ours"]])
        self.assertEqual((dep.phase, dep.arm), ("infer/test", "ours-prebinned"))
        self.assertEqual(AR.ROW_BY_NAME["ols"].timing, [None])
        self.assertTrue(AR.ROW_BY_NAME["knn"].kernels)

    def test_any_board_row_and_refusals(self):
        rows = AR.parse_rows("# c\nclassical/pca/taxi/rows=full fit ours\n"
                             "trees/gbdt-depthwise/higgs/rows=full infer/test ours name=dep-higgs timing=ours|planned\n")
        self.assertEqual([r.name for r in rows], ["pca", "dep-higgs"])
        self.assertEqual(rows[1].timing, [["ours"], None])
        lines = [ln.split("\t") for ln in AR.plan_lines(rows)]
        self.assertIn(["profile", "dep-higgs", "trees/gbdt-depthwise/higgs/rows=full", "infer/test", "ours",
                       "profile"], lines)
        for bad in ("classical/pca fit ours", "classical/pca/taxi/rows=full predict ours",
                    "classical/pca/taxi/rows=full fit ours colour=red",
                    "classical/pca/taxi/rows=full fit ours\nclassical/pca/higgs/rows=full fit ours", "# none\n"):
            with self.assertRaises(ValueError):
                AR.parse_rows(bad)

    def test_estimate_states_pod_minutes_per_row(self):
        with tempfile.TemporaryDirectory() as d:
            rec = os.path.join(d, "r.json")
            rows = os.path.join(d, "rows.tsv")
            write(rec, json.dumps({"recipes": []}))
            write(rows, "classical/pca/taxi/rows=full fit ours\n")
            r = subprocess.run([sys.executable, os.path.join(HERE, "ops", "attribution_report.py"), "estimate",
                                "--recipes", rec, "--rows-file", rows, "--wheels", "2"],
                               capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        # two runs (time, profile) of overhead only, on two wheels: 2 * 2 * 90 s = 6 pod minutes
        self.assertIn("ESTIMATE row=pca runs_per_wheel=2 seconds_per_wheel=180 pod_min=6.0", r.stdout)


class FailuresAndExit(unittest.TestCase):
    def test_first_error_names_the_preflight_refusal_not_the_last_line(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "ours-prebinned")
            write(out + ".log", "bench_board: board %s/BOARD.md\n" % out)
            write(os.path.join(out, "board.json"), json.dumps({"failed_preflight_recipes": {
                "trees/x/y/rows=full": {"problems": ["missing parameter/version probe: ours"]}}, "races": {}}))
            self.assertEqual(AR.first_error(out, out + ".log"),
                             "preflight refused trees/x/y/rows=full: missing parameter/version probe: ours")
            os.remove(os.path.join(out, "board.json"))
            write(os.path.join(out, "raw", "drv.log"), "tcmalloc warning: Unable to mbind\nok\n"
                                                         "RuntimeError: boom\nlater error\n")
            self.assertEqual(AR.first_error(out, out + ".log"), "drv.log: RuntimeError: boom")

    def test_exit_code_and_summary(self):
        ok = {"runs": {"a": {"planned": {"ok": True}}}, "ncu": {}}
        doc = {"rows": {"r1": dict(ok), "r2": dict(ok)}}
        self.assertEqual(AR.summarize(doc, ["a"])["code"], AR.EXIT_OK)
        doc["rows"]["r2"] = dict(ok, profile_error={"a": "no nsys capture"})
        s = AR.summarize(doc, ["a"])
        self.assertEqual(s["code"], AR.EXIT_NO_PROFILE)
        self.assertIn("SUMMARY row=r2 timing=ok profile=FAILED(a: no nsys capture)", s["lines"])
        doc["rows"]["r1"] = {"runs": {"a": {"x": {"error": "no time for x: boom"}}}, "ncu": {"a": {"status": "no-kernels"}}}
        s = AR.summarize(doc, ["a"])
        self.assertEqual(s["code"], AR.EXIT_NO_TIMING)
        self.assertIn("SUMMARY row=r1 timing=FAILED(a x: no time for x: boom) profile=ok ncu=a no-kernels", s["lines"])
        self.assertIn("exit=7", s["lines"][-1])


class TrainLaneForcedCalls(unittest.TestCase):
    def test_training_lanes_take_the_forced_calls_in_an_attribution_run_only(self):
        """fb3-attr-tool (2026-11-17): samba-train-step scored one call (min..max = median) and opened no NVTX
        range (no nsys report) in the attribution job."""
        try:
            nb = _load("bench_board_neural_t", os.path.join(HERE, "bench_board_neural.py"))
        except Exception as exc:                  # noqa: BLE001
            self.skipTest("tools/bench_board_neural.py does not import here: %s" % exc)

        class Small:
            def __init__(self, n):
                self.n = n

            def forced_calls(self, ours, phase=None):
                return self.n
        self.assertEqual(nb.attribution_train_calls("samba-train-step", Small(20)), 20)
        self.assertIsNone(nb.attribution_train_calls("samba-train-step", Small(None)))     # a board: one call
        self.assertIsNone(nb.attribution_train_calls("ops-session-step", Small(20)))       # fixed batch: one call
        self.assertIsNone(nb.attribution_train_calls("mamba3-forward", Small(20)))         # not a training lane


class JobScript(unittest.TestCase):
    def test_syntax_and_refuses_unpersisted(self):
        self.assertEqual(subprocess.run(["bash", "-n", JOB]).returncode, 0)
        env = {k: v for k, v in os.environ.items() if k not in ("MOJOLEARN_BOX_PERSIST", "MOJOLEARN_ALLOW_UNPERSISTED")}
        r = subprocess.run(["bash", JOB], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("box_persist_run.sh", r.stderr)

    def test_reuses_the_shared_functions_and_states_the_parallel_plan(self):
        text = open(JOB).read()
        self.assertIn('source "$(dirname "$0")/box_job_lib.sh"', text)
        for fn in ("host_probe_phase", "prune_persist_staging", "cores_sampler", "disk_gate", "gpu_other"):
            self.assertIn(fn, text)
            self.assertNotIn("\n%s() {" % fn, text)               # reused, never copied
        self.assertIn("SERIAL BY NATURE", text)
        self.assertIn('xargs -P "$NCORES"', text)
        self.assertIn("--capture-range=nvtx", text)
        self.assertIn("MOJOLEARN_BENCH_FORCE_CALLS", text)

    def test_rows_file_error_line_and_exit_from_the_report(self):
        text = open(JOB).read()
        self.assertIn("AT_ROWS_FILE", text)
        self.assertIn('--rows-file "$RES/plan/rows.tsv"', text)
        self.assertIn("attribution_report.py error --out", text)
        self.assertIn("REPORT_RC=$rc", text)
        self.assertIn('--wheels "${#ARMS[@]}"', text)


if __name__ == "__main__":
    unittest.main()
