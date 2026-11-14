"""The per-race evidence archive stays small (2026-11-14, L40S pod nvc1, 80 GB disk).

The measurement of ba08dac006 filled the pod: every race's run.tar.gz held the bytes of each loaded native module
(about 150 MB per race, 27 GB over 179 races) and the scored receipt twice (scored/<id>/outputs.npz and
run/work/<lane>-<ds>-ours.npz), beside the loose scored copy on disk. Now a loaded module is retained once per
results dir (evidence/loaded/<sha256>.tar.gz) and the race records its sha256, path and size in loaded.json; a
byte-identical member is a tar hard link to the scored/ copy; the loose scored copy goes once it is archived.

Run: ~/mojolearn-evidence/venv-tools/bin/python -m pytest -q -p no:cacheprovider tools/test_bench_board_evidence_size.py
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = load("bench_board_evidence")
SO_BYTES = os.urandom(1024 * 1024)          # a 1 MB "native module"
RECEIPT = b"PK\x03\x04 scored receipt bytes " + os.urandom(4096)


class EvidenceSize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "board"
        self.out.mkdir()
        site = Path(self.tmp.name) / "venv" / "site-packages" / "mojolearn"
        site.mkdir(parents=True)
        self.so = site / "_mojolearn_native.cpython-312-x86_64-linux-gnu.so"
        self.so.write_bytes(SO_BYTES)
        self.so_sha = hashlib.sha256(SO_BYTES).hexdigest()
        self.ctx = dict(out=str(self.out), measurement_inventory={}, measurement_source={"status": "local"},
                        evidence_r2=False, artifact_identity={"wheel": "mojolearn_nvidia-0.8.67-candidate.whl"})

    def race(self, name):
        """One sealed race: a worker that loaded the module, one scored receipt and its run/work twin."""
        d = self.out / "evidence" / "runs" / name
        (d / "workers").mkdir(parents=True)
        E.write(d / "recipe.json", {"race": name})
        time.sleep(0.01)
        E.write(d / "workers" / "1.json", dict(status="recorded", arms={"ours": "mojolearn"},
                                               loaded_mojolearn_files={str(self.so): self.so_sha}))
        scored = d / "scored" / "1-abc"
        scored.mkdir(parents=True)
        (scored / "outputs.npz").write_bytes(RECEIPT)
        work = self.out / "work"
        work.mkdir(exist_ok=True)
        (work / ("kmeans-taxi-%s-ours.npz" % name)).write_bytes(RECEIPT)
        rec = dict(family="classical2", lane="kmeans", dataset="taxi", arms=["ours"], cells=[])
        E.seal(self.ctx, d, dict(source={}, dataset={"identity": "abc"}, hardware={}), rec)
        return d, rec

    def members(self, d):
        with tarfile.open(d / "run.tar.gz") as tf:
            out = {}
            for m in tf.getmembers():
                out[m.name] = (m, tf.extractfile(m).read() if (m.isfile() or m.islnk()) else None)
            return out

    def test_race_archive_holds_no_module_bytes_and_one_receipt(self):
        d, _ = self.race("race-1")
        members = self.members(d)
        # no native module inside the race archive: neither by name nor by bytes
        self.assertFalse([n for n in members if n.startswith("loaded/") or n.endswith(".so")], sorted(members))
        regular = {n: b for n, (m, b) in members.items() if m.isfile()}
        self.assertFalse([n for n, b in regular.items() if b == SO_BYTES])
        self.assertLess(max(len(b) for b in regular.values()), len(SO_BYTES))
        # the receipt is stored once (scored/), the run/work twin is a hard link that still reads back its bytes
        stored = [n for n, b in regular.items() if b == RECEIPT]
        self.assertEqual(stored, ["scored/1-abc/outputs.npz"])
        twin = "run/work/kmeans-taxi-race-1-ours.npz"
        self.assertTrue(members[twin][0].islnk())
        self.assertEqual(members[twin][0].linkname, "scored/1-abc/outputs.npz")
        self.assertEqual(members[twin][1], RECEIPT)
        # the module is recorded by sha256, path and size, with the wheel identity and the shared archive's ref
        loaded = json.loads(regular["loaded.json"])
        self.assertEqual(loaded["artifact_identity"], self.ctx["artifact_identity"])
        (mod,) = loaded["modules"]
        self.assertEqual((mod["sha256"], mod["path"], mod["bytes"]), (self.so_sha, str(self.so), len(SO_BYTES)))
        self.assertEqual(mod["archive"]["module_sha256"], self.so_sha)
        self.assertNotIn("local_path", mod["archive"])
        # the loose scored copy beside the archive is gone (the archive holds it)
        self.assertFalse((d / "scored" / "1-abc" / "outputs.npz").exists())
        self.assertLess((d / "run.tar.gz").stat().st_size, len(SO_BYTES) // 4)

    def test_module_retained_once_per_results_dir(self):
        self.race("race-1")
        d2, _ = self.race("race-2")
        shared = sorted((self.out / "evidence" / "loaded").iterdir())
        self.assertEqual([p.name for p in shared], [self.so_sha + ".tar.gz"], "one copy for both races")
        with tarfile.open(shared[0]) as tf:
            self.assertEqual(tf.extractfile(self.so_sha + "/" + self.so.name).read(), SO_BYTES)
        upload = json.loads((d2 / "upload.json").read_text())
        self.assertEqual(upload["loaded:" + self.so_sha]["sha256"], E.file_hash(shared[0]),
                         "sync retries the shared archive's transfer like the race's own")

    def test_changed_module_is_a_retention_failure(self):
        d = self.out / "evidence" / "runs" / "race-x"
        (d / "workers").mkdir(parents=True)
        E.write(d / "recipe.json", {"race": "x"})
        E.write(d / "workers" / "1.json", dict(status="recorded", arms={"ours": "mojolearn"},
                                               loaded_mojolearn_files={str(self.so): "0" * 64}))
        rec = dict(family="classical2", lane="kmeans", dataset="taxi", arms=["ours"],
                   cells=[dict(arm="ours", status="ok", mode="identical")])
        E.seal(self.ctx, d, dict(source={}, dataset={"identity": "abc"}, hardware={}), rec)
        self.assertIn("loaded bytes: " + str(self.so), rec["cells"][0]["measurement"]["missing"])
        self.assertFalse((self.out / "evidence" / "loaded").exists() and
                         list((self.out / "evidence" / "loaded").iterdir()))

    def test_archive_dedupe_is_deterministic_and_complete(self):
        p1, p2 = Path(self.tmp.name) / "a.tar.gz", Path(self.tmp.name) / "b.tar.gz"
        members = {"run/work/x.npz": b"same", "scored/1/outputs.npz": b"same", "log": b"other", "e1": b"", "e2": b""}
        r1, r2 = E.archive(p1, members), E.archive(p2, dict(reversed(list(members.items()))))
        self.assertEqual(r1["sha256"], r2["sha256"])
        with tarfile.open(p1) as tf:
            for name, payload in members.items():
                self.assertEqual(tf.extractfile(name).read(), payload)
            self.assertTrue(tf.getmember("run/work/x.npz").islnk())
            self.assertTrue(tf.getmember("e2").isfile(), "empty members stay regular files")
        x = Path(self.tmp.name) / "x"
        with tarfile.open(p1) as tf:
            tf.extractall(x, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(x)
        self.assertEqual((x / "run/work/x.npz").read_bytes(), b"same")


if __name__ == "__main__":
    unittest.main()
