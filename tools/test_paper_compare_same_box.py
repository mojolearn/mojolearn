#!/usr/bin/env python3
"""tools/test_paper_compare_same_box.py -- AGENTS.md "Times are compared only on the same box" (Andrew, 2026-11-14):
a time flag comes only from two boards of the same host id; a stored column (the paper record, a stored compare,
the best ever) is reference only; a compare of two host ids refuses; the host probe round-trips into the boards.

2026-11-14: the paper measurement of 80a7f8aca3 (L40S host 9204fff9f8f7) against the stored 0.8.55 column and the
ba08dac006 run (host 1e08d01c3d48) raised 56 REGRESSION flags; rows whose code had not changed were 1.2-2.2x apart.
No board driver, no box, no mojolearn import.

  python -m pytest -q tools/test_paper_compare_same_box.py
"""
import copy
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
            "pcie": {"gen": 4, "width": 16, "read_during_copy": True}, "gpu": {"name": GPU, "driver": "570.1"}}


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
        self.assertIn("no verdict", PC.render_summary(res))

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
            return subprocess.run(["bash", "-c", "set -u\n" + block + '\necho "PV=$PRIOR_VERSIONS"'], env=e,
                                  capture_output=True, text=True, timeout=60)
        p = run({"PRIOR": "0.8.66"})
        self.assertIn("PV=0.8.55,0.8.66", p.stdout, p.stderr)
        p = run({"PRIOR": "0.8.66", "PC_PRIOR_VERSIONS": "0.8.66,0.8.55,0.8.60"})
        self.assertIn("PV=0.8.55,0.8.60,0.8.66", p.stdout)        # the last release always last
        p = run({"PRIOR": "0.8.66", "PC_PRIOR_VERSIONS": "0.8.55"})
        self.assertIn("PV=0.8.55,0.8.66", p.stdout)               # the last release is always timed
        p = run({"PRIOR": "0.8.66", "PC_PRIOR_VERSIONS": "0.8.55,bad"})
        self.assertEqual(p.returncode, 2)

    def test_job_probes_first_and_compares_on_box_boards_only(self):
        t = self.text()
        self.assertEqual(subprocess.run(["bash", "-n", self.JOB]).returncode, 0)
        self.assertIn("tools/ops/host_probe.py --vendor", t)
        self.assertIn('export MOJOLEARN_HOST_PROBE="$J/host-probe.json"', t)
        self.assertLess(t.index("phase probe"), t.index("phase all-plan"))
        self.assertLess(t.index("phase probe"), t.index("phase plan"))
        self.assertIn("prior_board_args", t)
        self.assertIn("--reference-compare", t)
        self.assertNotIn("--prior-compare", t)                     # the stored compare never makes the column
        self.assertIn('[ "$arm" = prior ] && [ "$rows" = full ] && continue', t)    # no second prior timing
        self.assertIn("all-prior", t)
        self.assertIn("--prior-version \"$PRIOR\"", t)

    def test_prior_board_args_order(self):
        t = self.text()
        start = t.index("prior_board_args() {")
        fn = t[start:t.index("\n}\n", start) + 3]
        p = subprocess.run(["bash", "-c", "RES=/r; PRIOR_ARMS=(prior-0.8.55 prior-0.8.66)\n" + fn +
                            'prior_board_args all-'], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.stdout.split("\n")[:4], ["--prior-board", "0.8.55=/r/board-all-prior-0.8.55/board.json",
                                                   "--prior-board", "0.8.66=/r/board-all-prior-0.8.66/board.json"])


class OwedPriorIsTheOnBoxBoard(unittest.TestCase):
    def test_board_dir(self):
        self.assertEqual(MO.board_dir("/r", "prior", "full", "0.8.66"), "/r/board-prior-0.8.66")
        self.assertEqual(MO.board_dir("/r", "candidate", "full", "0.8.66"), "/r/board")
        self.assertEqual(MO.board_dir("/r", "b1", "full", "0.8.66"), "/r/board-b1-full")


if __name__ == "__main__":
    unittest.main()
