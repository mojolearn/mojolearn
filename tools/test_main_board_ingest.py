# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/main_board_ingest.py on a fixture of lq result lines: no box, no git history needed.

    python3 -m unittest discover -s tools -p test_main_board_ingest.py -v

The fixture holds a newer slower default race (it must replace the older one), a newer failed run (a flag,
never a cell), an error-only race (FAILED table), a DIFFER pair, an A/B line and a
grid line (skipped), a line whose sha is not in the repo (skipped with a note), an infrastructure
status (never replaces), a board-runner summary of a default main job and one of a grid job, the
other vendor's digest at the same commit (identity), and stored opponent boards (copied, one of
them with changed lane settings: withheld).
"""
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MB = _load("main_board_ingest", os.path.join(HERE, "main_board_ingest.py"))
BB = MB.load_bench_board()

A, B = "a" * 40, "b" * 40
SHAS = {"aaaaaaaaa": (A, 1_000), "bbbbbbbbb": (B, 2_000), A: (A, 1_000), B: (B, 2_000)}


class FakeGit:
    """sha prefix -> (full, commit time); 'ccccccccc' is not in the repo."""

    def __call__(self, sha):
        return SHAS.get(sha)

    def on_main(self, full):
        return full in (A, B)


NV_RESULTS = """\
n0001 nvidia main@aaaaaaaaa ALGOS lane=ridge-cv dataset=taxi arm=ours status=ok median_ms=100.0 quality={"r2": 0.9} digest=1111111111111111
n0002 nvidia main@bbbbbbbbb ALGOS lane=ridge-cv dataset=taxi arm=ours status=ok median_ms=150.0 quality={"r2": 0.9} digest=2222222222222222
n0003 nvidia freeze-20261007@bbbbbbbbb [ MOJOLEARN_GRID_TAG=ge1234567.P001 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_X=1] ALGOS lane=ridge-cv dataset=taxi arm=ours status=ok median_ms=50.0 quality={} digest=3333333333333333
n0004 nvidia main@bbbbbbbbb [ MOJOLEARN_BUILD_DEFINES=MOJOLEARN_X=1] ALGOS lane=ridge-cv dataset=taxi arm=ours status=ok median_ms=40.0 quality={} digest=4444444444444444
n0005 nvidia main@ccccccccc ALGOS lane=ridge-cv dataset=istella arm=ours status=ok median_ms=10.0 quality={} digest=5555555555555555
n0006 nvidia main@bbbbbbbbb ALGOS lane=ridge-cv dataset=taxi arm=ours status=not_ready median_ms=None quality={} digest=none
n0007 nvidia main@bbbbbbbbb [MOJOLEARN_GRID_TAG=flips1.trees.nv.bb] CMD flips1.trees.nv.bb rc=0 builds=[libMojolearnMath rc=0 ] digest=4cce1743fce22092 last: GRIDBB-DONE tag=flips1.trees.nv.bb races=1 ok=1 bench_board_rc=0 out=/root/lq/out/n0007/bb
n0008 nvidia main@bbbbbbbbb [MOJOLEARN_GRID_TAG=ge1234567.P002.bb MOJOLEARN_BUILD_DEFINES=MOJOLEARN_Y=1] CMD ge1234567.P002.bb rc=0 builds=[libMojolearnMath rc=0 ] digest=none last: GRIDBB-DONE tag=ge1234567.P002.bb races=1 ok=1
n0010 nvidia main@bbbbbbbbb ALGOS lane=gaussian-nb dataset=taxi arm=ours status=error median_ms=None quality={} digest=none
n0011 nvidia main@bbbbbbbbb ALGOS lane=ridge-cv dataset=istella arm=ours status=ok median_ms=30.0 quality={"r2": 0.8} digest=8888888888888888
n0009 nvidia main@bbbbbbbbb IDCHECK ridge-cv taxi rows=small nvidia=2222222222222222 host=2222222222222222 MATCH
"""
NV_GRID_LOGS = """\
/root/lq/out/n0007/flips1.trees.nv.bb.log:GRIDBB-RUN --out /root/lq/out/n0007/bb
/root/lq/out/n0007/flips1.trees.nv.bb.log:GRIDBB tag=flips1.trees.nv.bb vendor=nvidia head=bbbbbbbbb family=trees lane=gbdt-depthwise dataset=taxi status=ok median_ms=4061.7 hash=4cce1743fce22092 quality={"auc": 0.63}
/root/lq/out/n0007/flips1.trees.nv.bb.log:GRIDBB-DONE tag=flips1.trees.nv.bb races=1 ok=1 bench_board_rc=0 out=/root/lq/out/n0007/bb
/root/lq/out/n0008/ge1234567.P002.bb.log:GRIDBB tag=ge1234567.P002.bb vendor=nvidia head=bbbbbbbbb family=trees lane=gbdt-depthwise dataset=istella status=ok median_ms=1.0 hash=6666666666666666 quality={}
"""
AMD_RESULTS = """\
a0001 amd main@bbbbbbbbb ALGOS lane=ridge-cv dataset=taxi arm=ours status=ok median_ms=120.0 quality={"r2": 0.9} digest=2222222222222222
a0002 amd main@bbbbbbbbb ALGOS lane=ridge-cv dataset=istella arm=ours status=ok median_ms=25.0 quality={"r2": 0.8} digest=7777777777777777
"""


def _opp_cell(family, lane, dataset, arm, library, ms, span=None):
    return {"family": family, "lane": lane, "dataset": dataset, "rows": None, "rows_tag": "full",
            "arm": arm, "library": library, "mode": "opponent", "device": "gpu", "times_ms": [ms],
            "warmup_ms": ms, "median_ms": ms, "min_ms": ms, "max_ms": ms, "rounds": 1, "status": "ok",
            "quality": {"r2": 0.9}, "hash": None, "hash_stable": None, "verdict": "COMPARABLE",
            "comparability": {"span": span or {}}, "peak_host_mb": None, "peak_gpu_mb": None, "memory": {},
            "ratio_ours_identical_over": 99.0}


def _opp_board(lane_config, vendor):
    rid = "algos/ridge-cv/taxi/rows=full"
    races = {rid: {"id": rid, "family": "algos", "lane": "ridge-cv", "dataset": "taxi", "rows": None,
                   "status": "done", "finished": "2026-10-06T07:00:00Z", "lane_config": lane_config,
                   "cells": [_opp_cell("algos", "ridge-cv", "taxi", "cuml-gpu", "cuml", 200.0,
                                       {"upload_ms_untimed": 5.0})]}}
    tid = "trees/gbdt-depthwise/taxi/rows=full"
    races[tid] = {"id": tid, "family": "trees", "lane": "gbdt-depthwise", "dataset": "taxi", "rows": None,
                  "status": "done", "cells": [_opp_cell("trees", "gbdt-depthwise", "taxi", "xgboost-gpu",
                                                        "xgboost", 2000.0)]}
    return {"schema": BB.SCHEMA, "box": {"gpu": {"vendor": vendor, "name": "NVIDIA L40S" if vendor == "nvidia" else "AMD Instinct MI325X"},
                                         "host": {"hostname": "fixture-box"}}, "races": races}


class MainBoardIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="main-board-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.paths = {}
        for name, text in (("nv-results.txt", NV_RESULTS), ("nv-grid-logs.txt", NV_GRID_LOGS),
                           ("amd-results.txt", AMD_RESULTS)):
            self.paths[name] = os.path.join(self.tmp, name)
            with open(self.paths[name], "w") as fh:
                fh.write(text)
        cfg = json.loads(json.dumps(BB.ALGOS.lane_config("ridge-cv"), default=str))
        changed = json.loads(json.dumps(cfg))
        changed["params"] = {"changed": True}
        for name, board in (("opp-nv.json", _opp_board(cfg, "nvidia")), ("opp-amd.json", _opp_board(changed, "amd"))):
            self.paths[name] = os.path.join(self.tmp, name)
            with open(self.paths[name], "w") as fh:
                json.dump(board, fh)
        self.out = os.path.join(self.tmp, "boards")

    def argv(self, check=False, results=None):
        a = ["--results", "nv=" + (results or self.paths["nv-results.txt"]),
             "--results", "amd=" + self.paths["amd-results.txt"],
             "--grid-logs", "nv=" + self.paths["nv-grid-logs.txt"],
             "--opponents", "nvidia-l40s=opp-fixture-nv=" + self.paths["opp-nv.json"],
             "--opponents", "amd-mi325x=opp-fixture-amd=" + self.paths["opp-amd.json"],
             "--out-root", self.out]
        return a + (["--check"] if check else [])

    def run_tool(self, **kw):
        buf = io.StringIO()
        MB.run(self.argv(**kw), BB=BB, resolve=FakeGit(), out=buf)
        return buf.getvalue()

    def board(self, column):
        with open(os.path.join(self.out, "main-%s" % column, "board.json")) as fh:
            return json.load(fh)

    def test_regexes_match_the_grid_collector(self):
        try:
            LQ = _load("six_lane_grid_lq_for_test", os.path.join(HERE, "six_lane_grid_lq.py"))
        except Exception as exc:      # noqa: BLE001  (its own imports missing here)
            self.skipTest("six_lane_grid_lq not loadable: %s" % exc)
        for name in ("RESULT_RE", "ALGOS_RE", "DIGEST_TAIL_RE", "GRIDBB_RE"):
            self.assertEqual(getattr(MB, name).pattern, getattr(LQ, name).pattern, name)

    def test_split_bracket(self):
        env, tail = MB.split_bracket(" [ MOJOLEARN_GRID_TAG=x MOJOLEARN_BUILD_DEFINES=A=1,B=2 word] ALGOS lane=l")
        self.assertEqual(env["MOJOLEARN_BUILD_DEFINES"], "A=1,B=2")
        self.assertEqual(env["_words"], ["word"])
        self.assertTrue(tail.strip().startswith("ALGOS"))
        self.assertEqual(MB.split_bracket(" ALGOS lane=l"), (None, " ALGOS lane=l"))

    def test_check_writes_nothing_and_counts(self):
        text = self.run_tool(check=True)
        self.assertFalse(os.path.exists(self.out))
        nv = next(l for l in text.splitlines() if l.startswith("MAINBOARD column=nvidia-l40s"))
        self.assertIn("races=3 add=3 change=0 drop=0 failed=2", nv)
        self.assertIn("DIFFER algos/ridge-cv/istella/rows=full main@bbbbbbbbb nv/n0011 8888888888888888 vs "
                      "amd-mi325x amd/a0002 7777777777777777", text)
        self.assertNotIn("gaussian-nb", text.split("MAINBOARD-SKIPS")[0])
        self.assertIn("label=main@bbbbbbbbb", nv)
        self.assertIn("ADD algos/ridge-cv/taxi/rows=full", text)
        self.assertIn("sha not in this repo", text)

    def test_board_newest_wins_skips_and_identity(self):
        self.run_tool()
        nv = self.board("nvidia-l40s")
        # a failed run is never a race: gaussian-nb has only an error run
        self.assertEqual(sorted(nv["races"]), ["algos/ridge-cv/istella/rows=full", "algos/ridge-cv/taxi/rows=full",
                                               "trees/gbdt-depthwise/taxi/rows=full"])
        rr = nv["races"]["algos/ridge-cv/taxi/rows=full"]
        ours = [c for c in rr["cells"] if c["library"] == "mojolearn"]
        self.assertEqual(len(ours), 1)
        c = ours[0]
        # the newer slower default race replaced the older one; A/B and grid arms never entered
        self.assertEqual(c["median_ms"], 150.0)
        self.assertEqual(c["main_board"]["sha"], B)
        self.assertEqual(c["main_board"]["job"], "n0002")
        self.assertEqual(c["mode"], "identical")
        self.assertEqual(c["hash"], "2222222222222222")
        self.assertEqual(c["main_board"]["identity"]["status"], "MATCH")
        self.assertIn("identity vs amd-mi325x: MATCH", c["source"])
        # the newer not_ready run did not replace the ok cell; it flags it
        self.assertEqual(c["main_board"]["newer_failed"], ["newer run bbbbbbbbb/n0006 failed: not_ready"])
        self.assertIn("newer run bbbbbbbbb/n0006 failed: not_ready", c["source"])
        self.assertEqual(nv["main_board"]["identity"], {"DIFFER": 1, "MATCH": 1, "n/a": 1})
        self.assertEqual(nv["main_board"]["failed_runs"], 2)
        # opponent copied from the stored board, ratio recomputed from the two stored medians
        opp = next(x for x in rr["cells"] if x["arm"] == "cuml-gpu")
        self.assertEqual(opp["copied_from"]["board"], "opp-fixture-nv")
        self.assertAlmostEqual(opp["ratio_ours_identical_over"], 150.0 / 200.0)
        self.assertTrue(opp["source"].startswith("copied from opp-fixture-nv"))
        # the default main bb summary is on the board; the grid bb job is not
        trees = nv["races"]["trees/gbdt-depthwise/taxi/rows=full"]
        t = next(x for x in trees["cells"] if x["library"] == "mojolearn")
        self.assertEqual((t["median_ms"], t["main_board"]["job"]), (4061.7, "n0007"))
        self.assertEqual(t["main_board"]["identity"]["status"], "n/a")
        self.assertEqual(nv["box"]["mojolearn"]["version"], "main@bbbbbbbbb")
        # the ledger keeps the replaced number and the ignored infrastructure status
        with open(os.path.join(self.out, "main-nvidia-l40s", "LEDGER.json")) as fh:
            led = json.load(fh)["entries"]
        older = [e for e in led if e["replaced"]["job"] == "n0001"]
        self.assertEqual(len(older), 1)
        self.assertEqual((older[0]["reason"], older[0]["replaced"]["median_ms"],
                          older[0]["replaced_by"]["median_ms"]), ("older commit", 100.0, 150.0))
        self.assertTrue(any(e["replaced"]["job"] == "n0006" and e["reason"].startswith("infrastructure")
                            for e in led))
        with open(os.path.join(self.out, "main-nvidia-l40s", "BOARD.md")) as fh:
            md = fh.read()
        self.assertIn("MAIN BOARD nvidia-l40s, version label main@bbbbbbbbb", md)
        self.assertIn("Unreleased: not reproducible by pip install", md)
        self.assertIsNone(re.search(r"\b(faster|slower)\b", md, re.I))
        # Identity section above the box, DIFFER cells by name with both digests; FAILED table with the reason
        self.assertLess(md.index("## Identity"), md.index("## Box"))
        self.assertIn("**DIFFER: 1 cells", md)
        self.assertIn("| algos/ridge-cv/istella/rows=full | main@bbbbbbbbb | NVIDIA L40S (nv, RunPod), nv/n0011 8888888888888888 | "
                      "AMD Instinct MI325X (amd, DO), amd/a0002 7777777777777777 |", md)
        self.assertIn("| gaussian-nb | taxi | main@bbbbbbbbb | NVIDIA L40S (nv, RunPod) | mojolearn (source build, main@bbbbbbbbb) | nv/n0010 | error | none |", md)
        self.assertIn("| ridge-cv | taxi | main@bbbbbbbbb | NVIDIA L40S (nv, RunPod) | mojolearn (source build, main@bbbbbbbbb) | nv/n0006 | not_ready | main@bbbbbbbbb nv/n0002 stays |", md)
        self.assertTrue(any(e["replaced"]["job"] == "n0010" and e["failed"] and "FAILED table" in e["reason"]
                            for e in led))
        # the other vendor: settings changed since its opponents were scored -> withheld
        amd = self.board("amd-mi325x")
        ar = amd["races"]["algos/ridge-cv/taxi/rows=full"]
        self.assertEqual([x["arm"] for x in ar["cells"]], ["ours"])
        self.assertIn("withheld", ar["opponent_source"]["withheld"])
        self.assertTrue(any("opponents withheld" in m for m in ar["lane_config"]["mismatches"]))
        self.assertEqual(ar["cells"][0]["main_board"]["identity"]["status"], "MATCH")

    def test_rerun_is_stable_and_keeps_cells_when_inputs_rotate(self):
        self.run_tool()
        empty = os.path.join(self.tmp, "empty-results.txt")
        open(empty, "w").close()
        text = self.run_tool(check=True, results=empty)
        nv = next(l for l in text.splitlines() if l.startswith("MAINBOARD column=nvidia-l40s"))
        self.assertIn("add=0 change=0 drop=0", nv)
        self.run_tool()
        with open(os.path.join(self.out, "main-nvidia-l40s", "LEDGER.json")) as fh:
            led = json.load(fh)["entries"]
        keys = [(e["race"], e["replaced"]["job"], e["replaced"]["kind"]) for e in led]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
