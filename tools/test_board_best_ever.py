#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""Tests of tools/board_best_ever.py: best-ever extraction from two small synthetic board blobs, the flags, and the
git walk over every committed version (including a blob that was later deleted)."""
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_best_ever as BE  # noqa: E402

RF = "trees/rf/istella/rows=full"
DB = "classical/dbscan/taxi/rows=full"
KM = "classical/kmeans/taxi/rows=full"


def ours(ms, version, hash_="h1", params=None, hw="NVIDIA L40S", batch=None, status="ok", mode="identical"):
    c = {"arm": "ours", "mode": mode, "status": status, "median_ms": ms, "hash": hash_, "version": version,
         "hardware_key": hw, "batch": batch}
    if params is not None:
        c["params"] = {"library": "mojolearn", "params": params}
    return c


def board(updated, cells_by_race, gpu="NVIDIA L40S", vendor="nvidia"):
    races = {}
    for rid, (fit, infer) in cells_by_race.items():
        fam, lane, ds = rid.split("/")[:3]
        races[rid] = {"id": rid, "family": fam, "lane": lane, "dataset": ds, "our_arms": {"ours": "identical"},
                      "cells": fit + [{"arm": "sklearn-cpu", "mode": "opponent", "status": "ok", "median_ms": 1.0}],
                      "infer_cells": infer}
    return {"updated": updated, "box": {"gpu": {"name": gpu, "vendor": vendor}}, "races": races}


def old_board():
    """0.8.25, 2026-09-30: RF fit 5.0 s, dbscan at eps 3.0, kmeans fit and infer."""
    return board("2026-09-30T12:00:00Z", {
        RF: ([ours(5000.0, "mojolearn 0.8.25", params={"n_estimators": 100, "max_depth": 16})], []),
        DB: ([ours(13000.0, "mojolearn 0.8.25", params={"eps": 3.0, "min_samples": 2})], []),
        KM: ([ours(100.0, "mojolearn 0.8.25", hash_="k1", params={"n_clusters": 8})],
             [ours(5.0, "mojolearn 0.8.25", hash_="k1", batch="Xq")]),
    })


def new_board():
    """0.8.66, 2026-11-08: RF fit 30 s with the same params and hash; dbscan slower with another eps; kmeans slower
    with another output hash;
    a fast-mode cell, an apple cell and a refused cell that must all be ignored."""
    b = board("2026-11-08T12:00:00Z", {
        RF: ([ours(30000.0, "mojolearn 0.8.66", params={"n_estimators": 100, "max_depth": 16}),
              dict(ours(1.0, "mojolearn 0.8.66"), mode="fast", arm="ours-fast")], []),
        DB: ([ours(900000.0, "mojolearn 0.8.66", params={"eps": 0.5, "min_samples": 2})], []),
        KM: ([ours(120.0, "mojolearn 0.8.66 (source build, main@abcdef1234)", hash_="k2", params={"n_clusters": 8}),
              dict(ours(None, "mojolearn 0.8.66"), status="REFUSED(timeout)")],
             [ours(6.0, "mojolearn 0.8.66", hash_="k1", batch="Xq")]),
    })
    return b


def recs_of(doc, commit, date):
    out = BE.cell_records(doc)
    for r in out:
        r.update(commit=commit, commit_date=date, path="bench/results/bench_board/main-nvidia-l40s/board.json")
        r["date"] = str(r.pop("board_updated") or "")[:10] or date
    return out


class Extraction(unittest.TestCase):
    def setUp(self):
        self.records = recs_of(old_board(), "aaaa", "2026-09-30") + recs_of(new_board(), "bbbb", "2026-11-08")
        self.rows = BE.summarize(self.records, "nvidia")

    def test_only_timed_identical_ours_cells(self):
        self.assertEqual(len(self.records), 8)        # 4 per board: rf, dbscan, kmeans fit, kmeans infer
        self.assertTrue(all(r["arm"] == "ours" for r in self.records))
        self.assertEqual(sorted(self.rows), sorted([RF + " fit ours", DB + " fit ours", KM + " fit ours",
                                                    KM + " infer ours Xq"]))
        self.assertEqual(BE.summarize(self.records, "amd"), {})

    def test_best_latest_and_timeline(self):
        rf = self.rows[RF + " fit ours"]
        self.assertEqual(rf["best"]["ms"], 5000.0)
        self.assertEqual(rf["best"]["release"], "0.8.25")
        self.assertEqual(rf["best"]["date"], "2026-09-30")
        self.assertEqual(rf["best"]["commit"], "aaaa")
        self.assertEqual(rf["best"]["device"], "NVIDIA L40S")
        self.assertEqual(rf["latest"]["ms"], 30000.0)
        self.assertAlmostEqual(rf["latest_over_best"], 6.0)
        self.assertEqual([t["ms"] for t in rf["timeline"]], [5000.0, 30000.0])
        self.assertEqual(rf["flags"], ["LATEST>BEST"])           # same params, same hash: a real loss

    def test_params_differ_and_hash_differ(self):
        db = self.rows[DB + " fit ours"]
        self.assertEqual(db["params_differ"], ["eps"])
        self.assertEqual(db["flags"], ["PARAMS-DIFFER"])         # the audit's dbscan "loss" was a params change
        km = self.rows[KM + " fit ours"]
        self.assertTrue(km["hash_differs"])
        self.assertEqual(km["params_differ"], [])
        self.assertEqual(km["flags"], ["PARAMS-DIFFER"])         # 1.2x, but another output hash
        self.assertEqual(km["best"]["ms"], 100.0)
        self.assertEqual(km["latest"]["release"], "0.8.66 (main@abcdef1234)")
        inf = self.rows[KM + " infer ours Xq"]
        self.assertEqual(inf["best"]["ms"], 5.0)
        self.assertIn("LATEST>BEST", inf["flags"])               # 6 / 5 = 1.2, same hash, no params recorded
        self.assertIn("PARAMS-UNKNOWN", inf["flags"])

    def test_same_cell_carried_by_later_commits_is_one_entry(self):
        again = recs_of(old_board(), "cccc", "2026-10-02")
        rows = BE.summarize(self.records + again, "nvidia")
        rf = rows[RF + " fit ours"]
        self.assertEqual(len(rf["timeline"]), 2)
        self.assertEqual(rf["best"]["commit"], "aaaa")

    def test_vendor_and_release(self):
        self.assertEqual(BE.vendor_of("AMD Instinct MI300X"), "amd")
        self.assertEqual(BE.vendor_of(None, "NVIDIA L40S"), "nvidia")
        self.assertIsNone(BE.vendor_of("Apple M2 Pro"))
        self.assertEqual(BE.vendor_of("", "", "bench/results/board-archive/amd-mi325x/x/board.json.gz"), "amd")
        self.assertEqual(BE.release_of("mojolearn 0.8.42 (source build, main@8899093fc)"), "0.8.42 (main@8899093fc)")
        self.assertEqual(BE.release_of("main@80ace6b7a"), "main@80ace6b7a")

    def test_compare_and_recipe_blobs(self):
        cmp_doc = {"vendor": "nvidia", "new_gpus": ["NVIDIA L40S"], "new_versions": ["0.8.66"], "new_label": "0.8.66",
                   "rows": [{"key": RF + " fit ours", "race_id": RF, "phase": "fit", "arm": "ours", "new_ms": 31000.0},
                            {"key": DB + " fit ours", "race_id": DB, "phase": "fit", "arm": "ours", "new_ms": None}]}
        recs = BE.blob_records("bench/results/paper_compare/compare-nvidia.json", json.dumps(cmp_doc).encode())
        self.assertEqual([(r["key"], r["ms"], r["hash"]) for r in recs], [(RF + " fit ours", 31000.0, None)])
        rec_doc = {"vendor": "amd", "paper_version": "0.8.55", "source": {"gpu": "AMD Instinct MI300X"},
                   "recipes": [{"key": KM + " fit ours", "race_id": KM, "phase": "fit", "arm": "ours",
                                "mode": "identical", "params": {"n_clusters": 8},
                                "v0855": {"status": "ok", "median_ms": 80.0, "hash": "k1",
                                          "library_version": "0.8.55"}}]}
        recs = BE.blob_records("bench/results/paper_compare/recipes-amd.json", json.dumps(rec_doc).encode())
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["vendor"], recs[0]["release"], recs[0]["device"]),
                         ("amd", "0.8.55", "AMD Instinct MI300X"))
        gz = BE.blob_records("bench/results/board-archive/nvidia-l40s/x/board.json.gz",
                             gzip.compress(json.dumps(old_board()).encode()))
        self.assertEqual(len(gz), 4)


class GitWalk(unittest.TestCase):
    """mine() over a throwaway repo: two versions of one board, the first later deleted, and an Apple board."""

    def git(self, *a):
        subprocess.run(["git", "-C", self.repo, *a], check=True, capture_output=True,
                       env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                                GIT_COMMITTER_EMAIL="t@t"))

    def put(self, rel, doc, msg):
        p = os.path.join(self.repo, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(doc, fh)
        self.git("add", rel)
        self.git("commit", "-q", "-m", msg)

    def test_mine_every_version(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo = d
            self.git("init", "-q")
            a = "bench/results/bench_board/run-a/board.json"
            self.put(a, old_board(), "a1")
            self.put(a, new_board(), "a2")
            self.git("rm", "-q", a)
            self.git("commit", "-q", "-m", "trim")
            self.put("bench/results/bench_board/m3/board.json",
                     board("2026-10-01", {RF: ([ours(1.0, "mojolearn 0.8.30", hw="Apple M3 Ultra")], [])},
                           gpu="Apple M3 Ultra", vendor="apple"), "apple")
            records, stats = BE.mine(d, jobs=2, rev_args=("--all",))
            self.assertEqual(stats["paths"], 2)
            self.assertEqual(stats["errors"], [])
            self.assertEqual(stats["blobs"], 4)               # a1, a2, the trim (no blob), the apple board
            self.assertEqual(stats["blobs_with_cells"], 2)    # a1, a2 (the apple board has no NV/AMD cell)
            out = os.path.join(d, "out")
            paths = BE.write(records, stats, out, d)
            doc = BE.load(paths["nvidia"][0])
            self.assertEqual(doc["rows"][RF + " fit ours"]["best"]["ms"], 5000.0)   # from the deleted blob
            self.assertEqual(BE.load(paths["amd"][0])["rows"], {})


if __name__ == "__main__":
    unittest.main()
