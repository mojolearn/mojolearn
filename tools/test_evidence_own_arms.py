"""CPU-only: what a measurement keeps (AGENTS.md "What a measurement keeps", Andrew, 2026-11-20).

One decision, bench_board_evidence.scores_opponent(ctx): a job with only our own arms leaves no *.tar.gz and no
*.npz in its results directory once a row completes (its quality and hashes stay in the row); a job that scores an
opponent keeps the per-race archive as before. Stubs only: no driver, no wheel, no GPU, no R2.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = load("bench_board_evidence")
S = load("bench_board_state")
B = load("bench_board")
COMMIT = "0123456789abcdef0123456789abcdef01234567"


class PublicSave:
    __module__ = "mojolearn.testing"

    def save(self, path):
        np.savez(path, weights=np.arange(3, dtype=np.float32))


def tiny_source(repo, drivers):
    """_source_members without reading the repository: one file, a fixed commit."""
    members = {"tools/x.py": b"code"}
    return members, {}, COMMIT, dict(commit=COMMIT, files={"tools/x.py": E.digest(b"code")}, scope="fixture")


class Decision(unittest.TestCase):
    def test_only_an_opponent_scoring_job_keeps_full_evidence(self):
        self.assertFalse(E.scores_opponent({}))
        self.assertFalse(E.scores_opponent(dict(with_opponents=False, retime=False, opponents_only=False,
                                                retime_cpu_opponents=False, capture_measurements=True)))
        for flag in ("with_opponents", "retime", "retime_cpu_opponents", "opponents_only"):
            self.assertTrue(E.scores_opponent({flag: True}), flag)

    def test_workers_get_the_retain_switch_only_from_the_decision(self):
        base = dict(data_root="/tmp/data", vendor="apple", measurement_receipts="/tmp/r")
        with patch.dict(os.environ, {E.RETAIN_BYTES: "1"}):          # never inherited from the shell
            self.assertNotIn(E.RETAIN_BYTES, B.child_env(dict(base, measurement_retain_bytes=False)))
        self.assertEqual(B.child_env(dict(base, measurement_retain_bytes=True))[E.RETAIN_BYTES], "1")
        self.assertNotIn(E.RETAIN_BYTES, E.runtime_environment({E.RETAIN_BYTES: "1"}))

    def test_own_arms_receipt_keeps_hashes_and_writes_no_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            workers = Path(d) / "workers"
            workers.mkdir()
            outputs = {"pred": np.arange(4, dtype=np.float32)}
            with patch.dict(os.environ, {E.RECEIPTS: str(workers)}):
                os.environ.pop(E.RETAIN_BYTES, None)
                receipt = S.scored_receipt(outputs, model=PublicSave())
            self.assertEqual(receipt["output_sha256"], S.canonical_hash(outputs))
            self.assertEqual(receipt["model"]["status"], "ok")
            self.assertTrue(receipt["model"]["sha256"])
            self.assertNotIn("retained_output", receipt)
            self.assertEqual(list(Path(d).rglob("*.npz")), [])


class Job(unittest.TestCase):
    """run_race end to end with a stub driver that writes what the real drivers write."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "board"
        self.out.mkdir()
        self.seen = {}

    def driver(self, ctx, race):
        self.seen["retain"] = ctx.get("measurement_retain_bytes")
        work = self.out / "work"
        work.mkdir(exist_ok=True)
        np.savez(work / "ridge-fixture-ours.npz", y=np.ones(8))                    # scored output / quality ref
        neural = self.out / "raw" / "neural" / "shape-full"
        neural.mkdir(parents=True, exist_ok=True)
        np.savez(neural / "ridge-fixture-ours.outputs.npz", y=np.ones(8))         # a kept neural output
        log = self.out / "raw" / "algos" / "ridge.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("ALGOS lane=ridge status=ok median_ms=1.0\n")
        with patch.dict(os.environ, B.child_env(dict(ctx, data_root="/tmp/data", vendor="apple")), clear=True):
            receipt = S.scored_receipt({"y": np.ones(8)}, model=PublicSave())
        return dict(id=race["id"], family=race["family"], lane=race["lane"], dataset=race["dataset"],
                    rows=None, arms=race["arms"], our_arms=race["our_arms"], status="done", rc=0,
                    log=str(log.relative_to(self.out)), cells=[], receipt=receipt)

    def run_one(self, **flags):
        race = dict(id="algos/ridge/fixture", family="algos", lane="ridge", dataset="fixture", rows=None,
                    arms=["ours"], opponents=[], our_arms={"ours": "identical"})
        ctx = dict(capture_measurements=True, out=str(self.out), box={}, evidence_r2=False, **flags)
        original = B._load_tool
        with patch.dict(os.environ, {"MOJOLEARN_SOURCE_ARCHIVE_CACHE": str(Path(self.tmp.name) / "cache")}), \
             patch.object(E, "_source_members", side_effect=tiny_source), \
             patch.object(B, "_load_tool", side_effect=lambda n: E if n == "bench_board_evidence" else original(n)), \
             patch.object(B, "race_data_files", return_value=[]), patch.object(B, "race_data_sha", return_value="in"), \
             patch.object(B, "race_settings", return_value={}), patch.object(B, "stored_opponents", return_value={}), \
             patch.object(B, "unsupported_our_arms", return_value={}), \
             patch.object(B, "preflight_check", return_value={"verdict": "MATCHED"}), \
             patch.object(B, "_run_race", side_effect=self.driver), patch.object(B, "store_opponents", return_value=0):
            return B.run_race(ctx, race)

    def bulky(self):
        return sorted(str(p.relative_to(self.out)) for p in self.out.rglob("*")
                      if p.is_file() and (p.name.endswith(".tar.gz") or p.suffix == ".npz"))

    def test_own_arms_job_leaves_no_archive_or_array_after_a_row(self):
        rec = self.run_one()
        self.assertIs(self.seen["retain"], False)
        self.assertEqual(self.bulky(), [])
        self.assertEqual(rec["evidence_scope"], "own-arms")
        self.assertEqual(rec["evidence_receipt"]["status"], "not-archived")
        self.assertEqual(rec["arrays_discarded"]["files"], 2)
        self.assertTrue(rec["receipt"]["output_sha256"])                       # the hash stays in the row
        self.assertTrue((self.out / rec["log"]).is_file())                      # the log stays
        state = json.loads((self.out / rec["evidence_directory"] / "upload.json").read_text())
        self.assertEqual(state["run"]["status"], "not-archived")
        self.assertEqual(state["source"]["status"], "commit")
        self.assertEqual(state["source"]["commit"], COMMIT)
        board = self.out / "board.json"
        E.write(board, {"races": {rec["id"]: rec}})
        with patch.object(E, "upload", side_effect=AssertionError("nothing to retain")):
            E.sync(board)                                                       # an own-arms race uploads nothing

    def test_failed_own_arms_row_keeps_small_arrays_for_diagnosis(self):
        original = self.driver
        self.driver = lambda ctx, race: dict(original(ctx, race), status="failed", rc=1)
        rec = self.run_one()
        self.assertEqual(rec["arrays_kept_for_diagnosis"]["files"], 2)
        self.assertNotIn("arrays_discarded", rec)
        self.assertFalse([p for p in self.bulky() if p.endswith(".tar.gz")])

    def test_opponent_job_keeps_the_race_archive(self):
        rec = self.run_one(with_opponents=True)
        self.assertIs(self.seen["retain"], True)
        self.assertEqual(rec["evidence_scope"], "opponent")
        bulky = self.bulky()
        self.assertTrue(any(p.endswith("run.tar.gz") for p in bulky), bulky)
        self.assertTrue(any(p.endswith("source.tar.gz") for p in bulky), bulky)
        archive = next(self.out.rglob("run.tar.gz"))
        with tarfile.open(archive) as t:
            names = t.getnames()
        self.assertIn("run/work/ridge-fixture-ours.npz", names)
        self.assertNotIn("arrays_discarded", rec)


if __name__ == "__main__":
    unittest.main()
