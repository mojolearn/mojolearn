#!/usr/bin/env python3
"""tools/test_paper_compare_same_box.py -- AGENTS.md "Times are compared only on the same box" (Andrew, 2026-11-14):
a time flag comes only from two boards of the same host id; a stored column (the paper record, a stored compare,
the best ever) is reference only; a compare of two host ids refuses; the host probe round-trips into the boards.

2026-11-14: the paper measurement of 80a7f8aca3 (L40S host 9204fff9f8f7) against the stored 0.8.55 column and the
ba08dac006 run (host 1e08d01c3d48) raised 56 REGRESSION flags; rows whose code had not changed were 1.2-2.2x apart.
No board driver, no box, no mojolearn import.

  python -m pytest -q tools/test_paper_compare_same_box.py
"""
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ops"))
import paper_compare as PC  # noqa: E402
import measure_owed as MO  # noqa: E402
import host_probe as HP  # noqa: E402
from test_paper_compare import GPU, new_board, paper_board  # noqa: E402

KM = "classical/kmeans/taxi/rows=full fit ours"
HOST_A = "9204fff9f8f7/GPU-aaaa"      # the 2026-11-14 measurement's L40S pod
HOST_B = "1e08d01c3d48/GPU-bbbb"      # the ba08dac006 run's L40S pod


def probe_doc(host):
    return {"schema": HP.SCHEMA, "host_id": host, "cpu_model": "AMD EPYC 9354", "numa_nodes": 2,
            "cgroup_cores": 16, "memcpy_gbps": 11.0, "h2d_pageable_gbps": 9.5, "h2d_pinned_gbps": 24.1,
            "pcie": {"gen": 4, "width": 16, "read_during_copy": True}, "gpu": {"name": GPU, "driver": "570.1"},
            "kernel": "6.8.0-147-generic", "boot_id": "boot-1"}


def on_host(board, host):
    """BOARD as bench_board --host-probe writes it: the box's probe and each race's host.host_id."""
    b = copy.deepcopy(board)
    if host:
        b.setdefault("box", {})["host_probe"] = probe_doc(host)
        for race in b["races"].values():
            race["host"] = {"hostname": host.split("/")[0], "host_id": host}
    return b


def ours_board(ms_by_race, host, version):
    """A prior wheel's on-box board: our cells only, the paper's races, MS_BY_RACE overriding the fit times."""
    races = {}
    for rid, r in paper_board()["races"].items():
        cells = [dict(copy.deepcopy(c), library_version=version) for c in r["cells"] if c["arm"].startswith("ours")]
        for c in cells:
            if c["arm"] == "ours" and rid in ms_by_race:
                c["median_ms"] = ms_by_race[rid]
        races[rid] = {"cells": cells,
                      "infer_cells": [copy.deepcopy(c) for c in r["infer_cells"] if c["arm"].startswith("ours")]}
    return on_host({"box": {"gpu": {"name": GPU}}, "races": races}, host)


TIME_FLAGS = {"REGRESSION", "FASTER", "SAME", "FIXED", "BROKEN", "STILL-FAILING", "REGRESSION-VS-PRIOR",
              "FASTER-VS-PRIOR", "RECOVERED", "STILL-REGRESSED", "NOW-WINS", "NOW-LOSES", "BEST-EVER-REGRESSION",
              "BEST-EVER-REGRESSION-UNCHECKED", "BEST-EVER-SUSPECT"}


class CrossHostNeverFlags(unittest.TestCase):
    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")

    def test_stored_paper_column_is_reference_only(self):
        # the 2026-11-14 case: the stored 0.8.55 kmeans fit is 100 ms, the new pod times 130 ms on unchanged code
        res = PC.compare(self.doc, [on_host(new_board(), HOST_A)], {})
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertEqual((km["ref_ms"], km["old_ms"]), (100.0, None))
        self.assertAlmostEqual(km["ratio_vs_ref"], 1.3)          # shown
        self.assertIsNone(km["ratio"])
        self.assertIn("NOT-COMPARED", km["flags"])
        for r in res["rows"]:
            self.assertFalse(TIME_FLAGS & set(r["flags"]), (r["key"], r["flags"]))
        self.assertTrue(res["verdict_basis"].startswith("no verdict"))
        self.assertTrue(PC.render_summary(res).startswith("NO VERDICT, every row is reference only: no on-box"))

    def test_stored_compare_and_best_ever_of_another_run_never_flag(self):
        ref = {r["key"]: dict(r, new_ms=50.0, flags=["REGRESSION"]) for r in
               PC.compare(self.doc, [on_host(new_board(), HOST_B)], {})["rows"]}
        best = {"schema": "mojolearn-board-best-ever/1", "vendor": "nvidia", "mined_at_head": "abc", "rows": {
            KM: {"key": KM, "best": {"ms": 50.0, "params": {"n_clusters": 9}, "hash": "h-ours", "device": GPU,
                                     "release": "0.8.25", "date": "2026-09-30"},       # mined before host ids
                 "best_by_device": {}, "latest": None, "flags": [], "timeline": []}}}
        best["rows"][KM]["best_by_device"] = {GPU: best["rows"][KM]["best"]}
        best["rows"][KM]["timeline"] = [best["rows"][KM]["best"]]
        cand = on_host(new_board(), HOST_A)
        for race in cand["races"].values():
            for c in race.get("cells", []) + race.get("infer_cells", []):
                c["mode"] = "identical"           # as a real board writes our cells (board_best_ever reads them)
        res = PC.compare(self.doc, [cand], {},
                         prior_boards=[("0.8.55", ours_board({}, HOST_A, "0.8.55"))],
                         reference=ref, reference_label="ba08dac006 stored (host %s)" % HOST_B, best_ever=best)
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertEqual(km["ref_prior_ms"], 50.0)
        self.assertAlmostEqual(km["ratio_vs_best_ever"], 130.0 / 50.0)
        self.assertNotIn("BEST-EVER-REGRESSION", km["flags"])
        self.assertNotIn("BEST-EVER-REGRESSION-UNCHECKED", km["flags"])
        self.assertNotIn("REGRESSION-VS-PRIOR", km["flags"])
        self.assertIn("REGRESSION", km["flags"])                  # from the on-box 0.8.55 board (100 -> 130)
        self.assertIn("reference ms", PC.render_md(res))

    def test_a_board_without_a_probe_gives_no_verdict(self):
        res = PC.compare(self.doc, [new_board()], {}, prior_boards=[("0.8.55", ours_board({}, HOST_A, "0.8.55"))])
        self.assertIn("carries no host probe", res["verdict_basis"])
        for r in res["rows"]:
            self.assertFalse(TIME_FLAGS & set(r["flags"]), (r["key"], r["flags"]))


class SameHostFlags(unittest.TestCase):
    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")

    def test_on_box_baseline_and_prior_flag(self):
        cand = on_host(new_board(), HOST_A)                       # kmeans fit 130 ms
        base = ours_board({}, HOST_A, "0.8.55")                   # 100 ms, as the paper
        prior = ours_board({"classical/kmeans/taxi/rows=full": 160.0, "classical/ols/taxi/rows=full": 70.0},
                           HOST_A, "0.8.66")
        res = PC.compare(self.doc, [cand], {}, prior_boards=[("0.8.55", base), ("0.8.66", prior)])
        rows = {r["key"]: r for r in res["rows"]}
        km = rows[KM]
        self.assertEqual(km["old_ms"], 100.0)
        self.assertIn("REGRESSION", km["flags"])
        self.assertAlmostEqual(km["ratio"], 1.3)
        self.assertEqual(km["prior_ms"], 160.0)
        self.assertIn("FASTER-VS-PRIOR", km["flags"])              # 130 / 160
        self.assertIn("STILL-REGRESSED", km["flags"])              # 0.8.66 was 1.6x the on-box 0.8.55
        ols = rows["classical/ols/taxi/rows=full fit ours"]        # 50 -> 0.8.66 70 -> 52
        self.assertIn("RECOVERED", ols["flags"])
        rf = rows["trees/rf/istella/rows=full fit ours"]           # 0.8.55 REFUSED on this box, the new one times it
        self.assertIn("FIXED", rf["flags"])
        self.assertEqual(res["host_id"], HOST_A)
        self.assertEqual(res["prior_label"], "0.8.66")
        self.assertTrue(res["verdict_basis"].startswith("same box " + HOST_A))
        summary = PC.render_summary(res)
        self.assertIn("probe candidate: host " + HOST_A, summary)  # both probes beside the verdict
        self.assertIn("probe 0.8.55: host " + HOST_A, summary)
        self.assertIn("pinned 24.1 GB/s", summary)
        md = PC.render_md(res)
        self.assertIn("0.8.66 ms | new/0.8.66", md)

    def test_median_against_one_call_is_flagged(self):
        """SCORING-MIXED (2026-11-16): the new cell is a median of 20 calls, the on-box 0.8.55 cell one call (its first
        scored call was at or above 10 ms); the ratio is kept and the row says it compares a median with one call."""
        cand = on_host(new_board(), HOST_A)
        for c in cand["races"]["classical/kmeans/taxi/rows=full"]["cells"]:
            if c["arm"] == "ours":
                c.update(calls=20, scoring="median-of-calls", min_ms=120.0, max_ms=140.0)
        res = PC.compare(self.doc, [cand], {}, prior_boards=[("0.8.55", ours_board({}, HOST_A, "0.8.55"))])
        rows = {r["key"]: r for r in res["rows"]}
        km = rows[KM]
        self.assertIn("SCORING-MIXED", km["flags"])
        self.assertEqual(km["flags"].count("SCORING-MIXED"), 1)     # the baseline and its prior column: one flag
        self.assertAlmostEqual(km["ratio"], 1.3)
        self.assertTrue(any(m.startswith("baseline: new median-of-calls") for m in km["scoring_mixed"]))
        for key, r in rows.items():
            if key != KM:
                self.assertNotIn("SCORING-MIXED", r["flags"], key)   # one call against one call
        self.assertIn("SCORING-MIXED", PC.render_summary(res))
        # lane fx-metrics-ranking (2026-11-19): a median of calls against one call makes no REGRESSION / FASTER
        self.assertNotIn("REGRESSION", km["flags"])
        self.assertIn("REGRESSION", km["scoring_mixed_replaced"])

    def test_scoring_mixed_replaces_the_verdict_and_shows_both_first_calls(self):
        """The 2026-11-19 pair job's row: the prior a median of 20 calls (first call 9.74 ms), the new one call
        (16.27 ms) on a byte-identical path. The ratio's verdict is replaced; both first scored calls are shown."""
        prior = {"calls": 20, "host_usage": {"per_call": [{"wall_ms": 9.74}, {"wall_ms": 9.92}]}}
        self.assertEqual(PC.first_call_ms(prior), 9.74)
        self.assertIsNone(PC.first_call_ms({"median_ms": 16.27}))
        row = {"new_scoring": "one-call", "new_first_ms": 16.27, "flags": ["REGRESSION"]}
        PC.flag_scoring_mixed(row, "baseline", "median-of-calls", replaces=PC.SCORING_MIXED_REPLACES,
                              other_first_ms=PC.first_call_ms(prior))
        self.assertEqual(row["flags"], ["SCORING-MIXED"])
        self.assertEqual(row["scoring_mixed_replaced"], ["REGRESSION"])
        self.assertIn("first scored calls: new 16.27 ms, baseline 9.74 ms (1.67)", row["scoring_mixed"][0])
        same = {"new_scoring": "one-call", "flags": ["REGRESSION"]}
        PC.flag_scoring_mixed(same, "baseline", "one-call", replaces=PC.SCORING_MIXED_REPLACES)
        self.assertEqual(same["flags"], ["REGRESSION"])                 # one call against one call: the verdict stays

    def test_a_row_the_prior_wheel_does_not_run_is_not_run_never_a_regression(self):
        base = ours_board({}, HOST_A, "0.8.55")
        del base["races"]["classical/kmeans/taxi/rows=full"]       # 0.8.55 failed its smoke for kmeans
        res = PC.compare(self.doc, [on_host(new_board(), HOST_A)], {}, prior_boards=[("0.8.55", base)])
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertEqual(km["flags"][:1], ["BASELINE-NOT-RUN"])
        self.assertNotIn("REGRESSION", km["flags"])


class HostMismatchRefuses(unittest.TestCase):
    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")

    def test_prior_board_of_another_host(self):
        with self.assertRaises(SystemExit) as cm:
            PC.compare(self.doc, [on_host(new_board(), HOST_A)], {},
                       prior_boards=[("0.8.55", ours_board({}, HOST_B, "0.8.55"))])
        self.assertIn("REFUSED", str(cm.exception))
        self.assertIn(HOST_B, str(cm.exception))

    def test_candidate_boards_of_two_hosts(self):
        b = on_host(new_board(), HOST_A)
        b["races"]["classical/ols/taxi/rows=full"]["host"]["host_id"] = HOST_B     # a resume on another pod
        with self.assertRaises(SystemExit):
            PC.compare(self.doc, [b], {}, prior_boards=[("0.8.55", ours_board({}, HOST_A, "0.8.55"))])

    def test_cli_exits_non_zero_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as td:
            rec, nb, pb = (os.path.join(td, n) for n in ("recipes.json", "board.json", "prior.json"))
            json.dump(self.doc, open(rec, "w"))
            json.dump(on_host(new_board(), HOST_A), open(nb, "w"))
            json.dump(ours_board({}, HOST_B, "0.8.55"), open(pb, "w"))
            p = subprocess.run([sys.executable, os.path.join(HERE, "paper_compare.py"), "compare", "--recipes", rec,
                                "--board", nb, "--prior-board", "0.8.55=" + pb, "--out-md", os.path.join(td, "C.md")],
                               capture_output=True, text=True, timeout=120)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("times are compared only on the same box", p.stderr)
            self.assertFalse(os.path.exists(os.path.join(td, "C.md")), "a refused compare writes nothing")

    def test_bad_prior_board_spec(self):
        with self.assertRaises(SystemExit):
            PC.parse_prior_board("board.json")
        self.assertEqual(PC.parse_prior_board("0.8.55=/x/board.json"), ("0.8.55", "/x/board.json"))


class SameSystemSoftware(unittest.TestCase):
    """fb-harness (2026-11-17): the d95f805cce full-board arms ran on one host id and GPU, but the pod restarted between
    them (candidate NVIDIA driver 595.91.07, kernel 6.8.0-147; 0.8.55 on 595.112.07, 6.8.0-149) and the compare gave
    27 REGRESSION and 30 FASTER verdicts. A verdict needs one GPU uuid, driver, kernel and boot id on both arms."""

    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")

    def compare(self, cand, base):
        res = PC.compare(self.doc, [cand], {}, prior_boards=[("0.8.55", base)])
        for r in res["rows"]:
            self.assertFalse(TIME_FLAGS & set(r["flags"]), (r["key"], r["flags"]))
        return res, PC.render_summary(res)

    def test_another_driver_and_kernel_give_no_verdict_and_say_why_first(self):
        base = ours_board({}, HOST_A, "0.8.55")
        base["box"]["host_probe"].update(gpu={"name": GPU, "driver": "570.2"}, kernel="6.8.0-149-generic",
                                         boot_id="boot-2")
        res, summary = self.compare(on_host(new_board(), HOST_A), base)
        first = summary.splitlines()[0]
        self.assertTrue(first.startswith("NO VERDICT, every row is reference only: the arms did not run under the "
                                         "same system software: "), first)
        for part in ("GPU driver 570.1 (candidate) vs 570.2 (the 0.8.55 board)",
                     "kernel 6.8.0-147-generic (candidate) vs 6.8.0-149-generic (the 0.8.55 board)",
                     "boot id boot-1 (candidate) vs boot-2 (the 0.8.55 board)"):
            self.assertIn(part, first)
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertEqual(km["flags"][:1], ["NOT-COMPARED"])
        self.assertIsNone(km["ratio"])

    def test_a_board_that_spans_a_restart_gives_no_verdict(self):
        cand = on_host(new_board(), HOST_A)
        cand["races"]["classical/ols/taxi/rows=full"]["host"].update(boot_id="boot-9", driver="570.9")
        _, summary = self.compare(cand, ours_board({}, HOST_A, "0.8.55"))
        self.assertIn("the candidate board spans 2 system states (a restart or an update between its races: "
                      "GPU driver 570.1 / 570.9; boot id boot-1 / boot-9)", summary.splitlines()[0])

    def test_a_field_not_recorded_is_no_proof(self):
        base = ours_board({}, HOST_A, "0.8.55")
        del base["box"]["host_probe"]["boot_id"]
        _, summary = self.compare(on_host(new_board(), HOST_A), base)
        self.assertIn("boot id not recorded on the 0.8.55 board", summary.splitlines()[0])

    def test_the_box_fingerprint_stands_in_for_an_older_probe(self):
        """The stored d95f805cce boards: the probe has the driver but no kernel; box.os.release has the kernel."""
        cand, base = on_host(new_board(), HOST_A), ours_board({}, HOST_A, "0.8.55")
        for b, kernel in ((cand, "6.8.0-147-generic"), (base, "6.8.0-149-generic")):
            del b["box"]["host_probe"]["kernel"]
            b["box"]["os"] = {"release": kernel}
        _, summary = self.compare(cand, base)
        self.assertIn("kernel 6.8.0-147-generic (candidate) vs 6.8.0-149-generic", summary.splitlines()[0])

    def test_progress_and_boot_ids_per_arm(self):
        cand = on_host(new_board(), HOST_A)
        cand["plan"] = sorted(cand["races"]) + ["classical/never/taxi/rows=full"]
        for rid, race in cand["races"].items():
            race["status"] = "done"
        cand["races"]["classical/ols/taxi/rows=full"]["host"]["boot_id"] = "boot-9"
        res = PC.compare(self.doc, [cand], {}, prior_boards=[("0.8.55", ours_board({}, HOST_A, "0.8.55"))])
        n = len(cand["races"])
        line = next(x for x in PC.render_summary(res).splitlines() if x.startswith("progress: "))
        self.assertIn("candidate %d/%d races done (boot ids boot-1, boot-9: resumed after a restart)" % (n, n + 1), line)
        self.assertIn("0.8.55 0/", line)

    def test_left_counts_races_not_done(self):
        self.assertEqual(PC.ids_left(["a", "b", "c"], {"races": {"a": {"status": "done"}, "b": {"status": "failed"}}}),
                         ["b", "c"])
        self.assertEqual(PC.ids_left(["a"], None), ["a"])

    def test_pass_ids_prints_each_failure_with_its_reason(self):
        with tempfile.TemporaryDirectory() as td:
            full, smoke, out = (os.path.join(td, n) for n in ("full.txt", "smoke.json", "pass.txt"))
            with open(full, "w") as fh:
                fh.write("algos/adaboost-clf/taxi/rows=full\nalgos/ard/taxi/rows=full\n")
            json.dump({"races": {"algos/adaboost-clf/taxi": {"id": "algos/adaboost-clf/taxi/rows=2000", "pass": False,
                                                             "failures": [{"arm": "ours", "reason": "refused n_bins"}]},
                                 "algos/ard/taxi": {"id": "algos/ard/taxi/rows=2000", "pass": True}}},
                      open(smoke, "w"))
            p = subprocess.run([sys.executable, os.path.join(HERE, "paper_compare.py"), "pass-ids", "--full-ids", full,
                                "--smoke-json", smoke, "--out", out], capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr)
            lines = p.stdout.splitlines()
            self.assertIn("PAPER-PASS-IDS-FAILED algos/adaboost-clf/taxi/rows=full: smoke failed: ours: refused n_bins",
                          lines)
            self.assertTrue(lines[-1].startswith("PAPER-PASS-IDS planned=2 smoke_passed=1 smoke_failed=1"))


class ProbeRoundTrip(unittest.TestCase):
    def test_probe_json_round_trip_into_a_board(self):
        doc = HP.probe("none", mb=4)          # no GPU here: host-side fields only, every failure recorded
        self.assertEqual(doc["schema"], HP.SCHEMA)
        self.assertEqual(doc["host_id"], HP.host_id(doc))
        self.assertTrue(doc["host_id"].endswith("/-"))
        self.assertIsInstance(doc["memcpy_gbps"], float)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "job", "host-probe.json")
            HP.write(doc, path)
            back = HP.load(path)
            self.assertEqual(back, json.loads(json.dumps(doc)))
            # bench_board --host-probe: box.host_probe and each race's host.host_id
            import bench_board as BB
            old = os.environ.pop("MOJOLEARN_HOST_ID", None)
            try:
                probe = BB.load_host_probe(path)
                self.assertEqual(probe, back)
                self.assertEqual(BB.race_host()["host_id"], doc["host_id"])
                # the system state per race (2026-11-17): the probe's GPU uuid and driver, this process's kernel and
                # boot id (the probe records the same two)
                rh = BB.race_host()
                self.assertEqual((rh["kernel"], rh["boot_id"]), (doc["kernel"], doc["boot_id"]))
                self.assertEqual(rh["gpu_uuid"], doc.get("gpu_uuid"))
            finally:
                os.environ.pop("MOJOLEARN_HOST_ID", None)
                if old is not None:
                    os.environ["MOJOLEARN_HOST_ID"] = old
            board = {"box": {"host_probe": probe}, "races": {"r": {"host": {"host_id": doc["host_id"]}}}}
            self.assertEqual(PC.board_host_ids(board), {doc["host_id"]})
            self.assertIn("host " + doc["host_id"], PC.probe_line(PC.board_probe(board)))
            with open(path, "w") as fh:
                json.dump({"schema": "other"}, fh)
            with self.assertRaises(ValueError):
                HP.load(path)
            with self.assertRaises(SystemExit):
                BB.load_host_probe(path)          # a probe without a host id refuses the board

    def test_host_id_prefers_the_pod(self):
        self.assertEqual(HP.host_id({"pod_id": "abc123", "hostname": "9204fff9f8f7", "gpu_uuid": "GPU-1"}),
                         "abc123/GPU-1")
        self.assertEqual(HP.host_id({"hostname": "droplet", "gpu_uuid": None}), "droplet/-")

    def test_gen_of_sysfs_speed(self):
        self.assertEqual(HP._gen_of_speed("32.0 GT/s PCIe"), 5)
        self.assertEqual(HP._gen_of_speed("16.0 GT/s PCIe"), 4)
        self.assertIsNone(HP._gen_of_speed(None))


class JobScript(unittest.TestCase):
    JOB = os.path.join(HERE, "ops", "paper_compare_job.sh")

    def text(self):
        with open(self.JOB) as fh:
            return fh.read()

    def test_prior_versions_default_and_order(self):
        t = self.text()
        block = t[t.index("# >>> prior-versions"):t.index("# <<< prior-versions")]

        def run(env):
            e = {k: v for k, v in os.environ.items() if not k.startswith("PC_")}
            e.update(env)
            return subprocess.run(["bash", "-c", "set -u\n" + block + '\necho "PV=$PRIOR_VERSIONS LAST=$LAST_PRIOR"'],
                                  env=e, capture_output=True, text=True, timeout=60)
        # Andrew, 2026-11-16: a measurement job times ONE prior, 0.8.55 (the paper baseline), by default
        p = run({})
        self.assertIn("PV=0.8.55 LAST=\n", p.stdout, p.stderr)
        self.assertIn("PAPER_VERSION=%s\n" % PC.PAPER_VERSION, block)  # the job's baseline is the compare's
        p = run({"PC_PRIOR_VERSIONS": "0.8.55,0.8.66"})
        self.assertIn("PV=0.8.66,0.8.55 LAST=\n", p.stdout)        # an explicit extra prior: paper rows only
        # an explicit last release (PC_PRIOR_VERSION other than the paper's): timed, last, the owed prior arm
        p = run({"PC_PRIOR_VERSION": "0.8.66"})
        self.assertIn("PV=0.8.55,0.8.66 LAST=0.8.66", p.stdout, p.stderr)
        p = run({"PC_PRIOR_VERSION": "0.8.66", "PC_PRIOR_VERSIONS": "0.8.66,0.8.55,0.8.60"})
        self.assertIn("PV=0.8.55,0.8.60,0.8.66", p.stdout)        # the last release always last
        p = run({"PC_PRIOR_VERSION": "0.8.66", "PC_PRIOR_VERSIONS": "0.8.55"})
        self.assertIn("PV=0.8.55,0.8.66", p.stdout)               # the last release is always timed
        p = run({"PC_PRIOR_VERSION": "0.8.66", "PC_PRIOR_VERSIONS": "0.8.55,bad"})
        self.assertEqual(p.returncode, 2)

    def test_job_probes_first_and_compares_on_box_boards_only(self):
        t = self.text()
        self.assertEqual(subprocess.run(["bash", "-n", self.JOB]).returncode, 0)
        # the probe phase is tools/ops/box_job_lib.sh host_probe_phase (shared with tools/ops/attribution_job.sh)
        with open(os.path.join(HERE, "ops", "box_job_lib.sh")) as fh:
            lib = fh.read()
        self.assertIn("tools/ops/host_probe.py --vendor", lib)
        self.assertIn('export MOJOLEARN_HOST_PROBE="$J/host-probe.json"', lib)
        self.assertIn("phase probe", lib[lib.index("host_probe_phase() {"):])
        self.assertLess(t.index("\nhost_probe_phase\n"), t.index("phase all-plan"))
        self.assertLess(t.index("\nhost_probe_phase\n"), t.index("phase plan"))
        self.assertIn("prior_board_args", t)
        self.assertIn("--reference-compare", t)
        self.assertNotIn("--prior-compare", t)                     # the stored compare never makes the column
        self.assertIn('[ "$arm" = prior ] && [ "$rows" = full ] && continue', t)    # no second prior timing
        self.assertIn("all-prior", t)
        self.assertIn('${LAST_PRIOR:+--prior-version "$LAST_PRIOR"}', t)   # the owed prior is the last release only
        self.assertIn('for arm in candidate ${LAST_PRIOR:+prior}', t)
        # each prior is smoked only on the races it is compared on (2026-11-16: 0.8.55 planned on 159, lacked 98)
        self.assertIn('PRIOR_SMOKE_IDS[$arm]="$RES/plan/race-ids-smoke.txt"', t)
        self.assertIn('"$RES/plan-owed/smoke-ids-prior.txt"', t)
        self.assertNotIn('sids="$SMOKE_IDS"', t)

    def test_prior_board_args_order(self):
        t = self.text()
        start = t.index("prior_board_args() {")
        fn = t[start:t.index("\n}\n", start) + 3]
        p = subprocess.run(["bash", "-c", "RES=/r; PRIOR_ARMS=(prior-0.8.55 prior-0.8.66)\n" + fn +
                            'prior_board_args all-'], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.stdout.split("\n")[:4], ["--prior-board", "0.8.55=/r/board-all-prior-0.8.55/board.json",
                                                   "--prior-board", "0.8.66=/r/board-all-prior-0.8.66/board.json"])


class StoredBaseline(unittest.TestCase):
    """Andrew, 2026-11-17 (AGENTS.md part 8 "0.8.55 is not timed again"): the 0.8.55 board of the 2026-11-17 L40S full
    board is the baseline; a job times our new wheel only. Another host or driver is a WARNING, not a refusal; the
    first line says the baseline is stored; a row whose upload probe moved more than 15% is BOX-STATE."""

    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")

    @staticmethod
    def with_upload(board, ms, nbytes=880e6):
        for r in board["races"].values():
            for c in r["cells"]:
                if c["arm"] == "ours":
                    c.setdefault("comparability", {}).setdefault("span", {}).update(
                        upload_ms_separate=ms, upload_bytes_separate=nbytes)
        return board

    def stored(self, ms_upload=40.0):
        b = self.with_upload(ours_board({}, HOST_B, "0.8.55"), ms_upload)
        b["box"]["host_probe"]["gpu"]["driver"] = "595.112.07"
        b["box"]["host_probe"]["taken"] = "2026-11-17T06:43:38Z"
        return b

    def test_another_host_and_driver_warn_and_still_flag(self):
        cand = self.with_upload(on_host(new_board(), HOST_A), 41.0)
        with self.assertRaises(SystemExit):          # the two-arm mode still refuses two hosts
            PC.compare(self.doc, [cand], {}, prior_boards=[("0.8.55", ours_board({}, HOST_B, "0.8.55"))])
        res = PC.compare(self.doc, [cand], {}, stored_baseline=self.stored())
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertEqual((km["old_ms"], round(km["ratio"], 2)), (100.0, 1.3))
        self.assertIn("REGRESSION", km["flags"])
        self.assertNotIn("BOX-STATE", km["flags"])           # 41 vs 40 ms: 2.5%
        text = PC.render_summary(res).splitlines()
        self.assertTrue(text[0].startswith("STORED BASELINE, not timed in this job: 0.8.55 stored (host %s, GPU %s, "
                                           "driver 595.112.07" % (HOST_B, GPU)), text[0])
        self.assertIn("date 2026-11-17", text[0])
        self.assertIn("this job: host %s" % HOST_A, text[0])
        self.assertTrue(text[1].startswith("WARNING: the stored 0.8.55 baseline ran on host %s" % HOST_B))
        self.assertTrue(text[2].startswith("WARNING: GPU driver 595.112.07 (stored 0.8.55) vs 570.1"))
        self.assertTrue(PC.render_md(res).startswith("**STORED BASELINE"))
        with self.assertRaises(SystemExit):          # one baseline only
            PC.compare(self.doc, [cand], {}, prior_boards=[("0.8.55", ours_board({}, HOST_A, "0.8.55"))],
                       stored_baseline=self.stored())

    def test_upload_probe_moved_is_box_state(self):
        cand = self.with_upload(on_host(new_board(), HOST_A), 60.0)     # 880 MB in 60 ms vs 40 ms: -33%
        res = PC.compare(self.doc, [cand], {}, stored_baseline=self.stored())
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertIn("BOX-STATE", km["flags"])
        self.assertFalse({"REGRESSION", "FASTER", "SAME"} & set(km["flags"]))
        self.assertAlmostEqual(km["ratio"], 1.3)                       # shown
        self.assertEqual(km["upload_probe"]["source"], "race")
        self.assertAlmostEqual(km["upload_probe"]["diff"], 40.0 / 60.0 - 1.0)
        summ = PC.render_summary(res)
        self.assertIn("BOX-STATE %d rows" % len(PC.box_state_rows(res)), summ)
        self.assertNotIn("worst regressions", summ)

    @staticmethod
    def ours_fit_cells(board):
        return [c for r in board["races"].values() for c in r["cells"] if c["arm"] == "ours"]

    def test_host_stall_replaces_regression(self):
        # lane fx-harness (2026-11-18): system time above half the wall is a host stall, shown, never a REGRESSION
        cand = self.with_upload(on_host(new_board(), HOST_A), 41.0)
        for c in self.ours_fit_cells(cand):
            c.update(host_stall=True, host_usage={"stime_fraction": 0.9, "stime_ms": 117.0, "wall_ms": 130.0,
                                                  "minflt": 640000, "majflt": 0, "calls": 1},
                     worker_numa={"node_now": 1, "cpu_nodes": [1], "placement": {"node": None, "why": "x"}})
        res = PC.compare(self.doc, [cand], {}, stored_baseline=self.stored())
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertIn("HOST-STALL", km["flags"])
        self.assertNotIn("REGRESSION", km["flags"])
        self.assertEqual(km["worker_numa"]["node_now"], 1)
        self.assertAlmostEqual(km["ratio"], 1.3)                       # shown
        summ = PC.render_summary(res)
        self.assertIn("HOST-STALL", summ)
        self.assertNotIn("worst regressions", summ)

    def test_unbound_workers_on_different_nodes_are_numa_mixed(self):
        # lane fx-cagra-taxi (2026-11-19): cagra taxi fit 4,480 ms on node 1 (the GPU's) and 11,382 ms on node 0, the
        # same code objects; two unbound workers on different nodes are shown, never a REGRESSION or FASTER
        def nodes(new_node, base_node, bound=None):
            cand = self.with_upload(on_host(new_board(), HOST_A), 41.0)
            base = self.stored()
            for c in self.ours_fit_cells(cand):
                c.update(worker_numa={"node_now": new_node, "placement": {"node": bound, "why": "x"}})
            for c in self.ours_fit_cells(base):
                c.update(worker_numa={"node_now": base_node, "placement": {"node": bound, "why": "x"}})
            res = PC.compare(self.doc, [cand], {}, stored_baseline=base)
            return {r["key"]: r for r in res["rows"]}[KM], res
        km, res = nodes(0, 1)
        self.assertIn("NUMA-MIXED", km["flags"])
        self.assertNotIn("REGRESSION", km["flags"])
        self.assertEqual(km["numa_mixed"], {"new": 0, "baseline": 1, "replaced": ["REGRESSION"]})
        self.assertAlmostEqual(km["ratio"], 1.3)                       # shown
        summ = PC.render_summary(res)
        self.assertIn("NUMA-MIXED", summ)
        self.assertNotIn("worst regressions", summ)
        km, _ = nodes(1, 1)                                            # the same node: a verdict as before
        self.assertIn("REGRESSION", km["flags"])
        self.assertNotIn("NUMA-MIXED", km["flags"])
        km, _ = nodes(0, 1, bound=1)                                   # bound by the conductor: never mixed
        self.assertIn("REGRESSION", km["flags"])

    def test_harness_clock_change_is_never_faster(self):
        # lane fx-harness (2026-11-18): the stored 0.8.55 cell was timed with the harness's copies in the clock
        cand = self.with_upload(on_host(new_board(), HOST_A), 41.0)
        for c in self.ours_fit_cells(cand):
            c.update(median_ms=50.0, times_ms=[50.0], harness_clock="2026-11-18 outputs reduced after the clock")
        res = PC.compare(self.doc, [cand], {}, stored_baseline=self.stored())
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertIn("HARNESS-CLOCK", km["flags"])
        self.assertFalse({"FASTER", "SAME"} & set(km["flags"]))
        self.assertEqual(km["harness_clock"]["replaced"], ["FASTER"])
        # the kmeans fit row is not among the gains (the fixture's infer row, timed under one clock on both sides,
        # is a real 5 -> 4 ms gain and stays listed)
        lines = PC.render_summary(res).splitlines()
        gains = []
        if "biggest gains:" in lines:
            for line in lines[lines.index("biggest gains:") + 1:]:
                if not line.startswith("  "):
                    break
                gains.append(line)
        self.assertFalse([g for g in gains if KM in g], gains)
        # a regression under the new clock stays a regression
        for c in self.ours_fit_cells(cand):
            c.update(median_ms=130.0, times_ms=[130.0])
        km = {r["key"]: r for r in PC.compare(self.doc, [cand], {}, stored_baseline=self.stored())["rows"]}[KM]
        self.assertIn("REGRESSION", km["flags"])
        self.assertIn("HARNESS-CLOCK", km["flags"])

    def test_all_arms_stamp_is_listed_once_for_an_opponent_rescore(self):
        # lane fx-harness-honest (2026-11-18, H1/H2): a settings change on all arms (gcn/graphsage cached=True) is
        # HARNESS-CLOCK against a cell timed before, never FASTER; our time stays shown; the race is listed once as
        # waiting for an opponent re-score, by name, with the count
        stamp = "2026-11-18 graph layers cached=True, settings key changed (all arms)"
        cand = self.with_upload(on_host(new_board(), HOST_A), 41.0)
        for c in self.ours_fit_cells(cand):
            c.update(median_ms=50.0, times_ms=[50.0], harness_clock=stamp)
        res = PC.compare(self.doc, [cand], {}, stored_baseline=self.stored())
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertIn("HARNESS-CLOCK", km["flags"])
        self.assertNotIn("FASTER", km["flags"])
        self.assertEqual(km["new_ms"], 50.0)
        owed = res["opponent_rescore_owed"]
        self.assertEqual(sorted({o["race_id"] for o in owed}), sorted(cand["races"]))
        self.assertTrue(all(o["harness_clock"] == stamp for o in owed))
        summ = PC.render_summary(res)
        self.assertEqual(summ.count("waiting for an opponent re-score (%d races" % len(owed)), 1)
        for rid in cand["races"]:
            self.assertIn("  %s (%s)" % (rid, stamp), summ)
        self.assertIn("## Waiting for an opponent re-score", PC.render_md(res))
        # an "(our arm)" stamp changed only our clock: nothing waits for an opponent
        for c in self.ours_fit_cells(cand):
            c["harness_clock"] = "2026-11-18 outputs reduced after the clock (our arm)"
        res = PC.compare(self.doc, [cand], {}, stored_baseline=self.stored())
        self.assertEqual(res["opponent_rescore_owed"], [])
        self.assertNotIn("waiting for an opponent re-score", PC.render_summary(res))

    def test_dataset_median_fallback_reads_large_uploads_only(self):
        races = {"a/x/taxi/rows=full": {"dataset": "taxi", "cells": [
                     {"arm": "ours", "comparability": {"span": {"upload_ms_separate": 1.0,
                                                                "upload_bytes_separate": 1e3}}}]},
                 "a/y/taxi/rows=full": {"dataset": "taxi", "cells": [
                     {"arm": "ours", "comparability": {"span": {"upload_ms_separate": 2.0,
                                                                "upload_bytes_separate": 40e6}}}]}}
        self.assertEqual(PC.dataset_probes(races), {"taxi": 20.0})


class OwedPriorIsTheOnBoxBoard(unittest.TestCase):
    def test_board_dir(self):
        self.assertEqual(MO.board_dir("/r", "prior", "full", "0.8.66"), "/r/board-prior-0.8.66")
        self.assertEqual(MO.board_dir("/r", "candidate", "full", "0.8.66"), "/r/board")
        self.assertEqual(MO.board_dir("/r", "b1", "full", "0.8.66"), "/r/board-b1-full")


class VerdictPrior(unittest.TestCase):
    """Lane fx-same-pod-prior (Andrew, 2026-11-19: a speed verdict compares two wheels timed on the SAME host in the SAME
    job; the stored 0.8.55 board is a screen only): --verdict-prior NAME makes one of our own earlier built wheels,
    timed beside the candidate in this job, the baseline; the stored board is a column without flags."""

    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")

    def test_prior_wheel_is_the_baseline_and_the_stored_board_a_screen(self):
        cand = on_host(new_board(), HOST_A)                                     # kmeans fit 130 ms
        prev = ours_board({"classical/kmeans/taxi/rows=full": 200.0}, HOST_A, "0.8.67.dev0+d95f805cceba")
        stored = StoredBaseline.with_upload(ours_board({}, HOST_B, "0.8.55"), 10.0)      # 100 ms, another host
        cand = StoredBaseline.with_upload(cand, 40.0)                          # an upload probe 4x off the stored
        res = PC.compare(self.doc, [cand], {}, prior_boards=[("prev", prev)], stored_baseline=stored,
                         verdict_prior="prev")
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertEqual(km["old_ms"], 200.0)                                  # the on-box pair
        self.assertAlmostEqual(km["ratio"], 0.65)
        self.assertIn("FASTER", km["flags"])
        self.assertNotIn("REGRESSION", km["flags"])                            # 1.3x the stored 0.8.55: no flag
        self.assertNotIn("BOX-STATE", km["flags"])                             # no stored verdict to replace
        self.assertNotIn("REGRESSION-VS-PRIOR", km["flags"])
        self.assertAlmostEqual(km["priors"]["0.8.55"]["ratio"], 1.3)           # the screen column, shown
        self.assertFalse(res["baseline_stored"])
        self.assertTrue(res["screen_stored"])
        self.assertEqual(res["baseline_version"], "prev")
        self.assertIsNone(res["prior_label"])
        self.assertEqual(res["baseline_label"], "prev (this box)")
        summary = PC.render_summary(res)
        self.assertTrue(summary.startswith("verdict: same box %s: our wheel prev timed in this job" % HOST_A), summary)
        self.assertIn("probe candidate: host " + HOST_A, summary)              # both probes beside the verdict
        self.assertIn("probe prev: host " + HOST_A, summary)
        self.assertIn("probe 0.8.55: host %s" % HOST_B, summary)
        self.assertIn("screen column, never a flag", summary)
        self.assertIn("0.8.55 (stored, screen)", PC.render_md(res).splitlines()[0])

    def test_prior_wheel_of_another_host_refuses_and_a_bad_name_refuses(self):
        cand = on_host(new_board(), HOST_A)
        with self.assertRaises(SystemExit):
            PC.compare(self.doc, [cand], {}, prior_boards=[("prev", ours_board({}, HOST_B, "x"))],
                       verdict_prior="prev")
        with self.assertRaises(SystemExit) as cm:
            PC.compare(self.doc, [cand], {}, prior_boards=[("prev", ours_board({}, HOST_A, "x"))],
                       verdict_prior="other")
        self.assertIn("names no --prior-board", str(cm.exception))

    def test_without_the_stored_board_and_on_the_cli(self):
        cand = on_host(new_board(), HOST_A)
        prev = ours_board({"classical/kmeans/taxi/rows=full": 100.0}, HOST_A, "x")
        res = PC.compare(self.doc, [cand], {}, prior_boards=[("prev", prev)], verdict_prior="prev")
        km = {r["key"]: r for r in res["rows"]}[KM]
        self.assertIn("REGRESSION", km["flags"])                               # 130 / 100
        self.assertFalse(res["screen_stored"])
        with tempfile.TemporaryDirectory() as d:
            rec, nb, pb = (os.path.join(d, n) for n in ("rec.json", "new.json", "prev.json"))
            json.dump(self.doc, open(rec, "w"))
            json.dump(cand, open(nb, "w"))
            json.dump(prev, open(pb, "w"))
            out = os.path.join(d, "c.json")
            with contextlib.redirect_stdout(io.StringIO()):
                PC.main(["compare", "--recipes", rec, "--board", nb, "--prior-board", "prev=" + pb,
                         "--verdict-prior", "prev", "--out-md", os.path.join(d, "c.md"), "--out-json", out,
                         "--summary", os.path.join(d, "s.txt")])
            self.assertEqual(json.load(open(out))["baseline_version"], "prev")


if __name__ == "__main__":
    unittest.main()
