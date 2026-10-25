"""Completed merged legs retain their physical binaries and indexes on reuse."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_reuse as rr


class MergedLegReuse(unittest.TestCase):
    def test_all_completed_merged_legs_supply_host_and_gpu_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            old = {}
            host = dict(target=rr.LINUX, tier="host", name="_host",
                        set_rel="host/_host.so", archive_path="mojolearn/host/_host.so",
                        decision="BUILD", identity_digest="host-identity")
            rows.append(host)
            for vendor, arch in rr.LINUX_SETS:
                name = f"{vendor}-{arch}"
                source = root / name
                rows.append(dict(target=rr.LINUX, tier="identical", vendor=vendor, arch=arch,
                                 name="_gpu", set_rel="identical/_gpu.so",
                                 archive_path=f"mojolearn/{vendor}/{arch}/identical/_gpu.so",
                                 decision="BUILD", identity_digest=name))
                for tier, member in (("host", "_host"), ("identical", "_gpu")):
                    folder = source / tier
                    folder.mkdir(parents=True)
                    (folder / "merged.so").write_bytes((tier + name).encode())
                    (folder / "_mojolearn_merged.json").write_text(json.dumps(
                        {"format": 1, "files": {"merged.so": [member]}}))
                (source / ".libs").mkdir()
                (source / ".libs/runtime.so").write_bytes(b"runtime")
                old[name] = dict(dir=source, origin=dict(source_commit="a" * 40,
                                                       set_identity_digest=name))
            plan = dict(previous=None, rows=rows, runtime=dict(decision="BUILD"))
            assembled = rr.assemble_linux(plan, None, {}, root / "assembled",
                                          say=lambda _: None, leg_reuse=old)
            for vendor, arch in rr.LINUX_SETS:
                folder = assembled[f"{vendor}/{arch}"]
                record = json.loads((folder / "reuse.json").read_text())
                for tier in ("host", "identical"):
                    for name in ("merged.so", "_mojolearn_merged.json"):
                        rel = f"{tier}/{name}"
                        self.assertEqual(record["files"][rel]["sha256"], rr.sha256(folder / rel))
                        self.assertEqual(record["files"][rel]["source"], "leg")
                self.assertFalse((folder / "host/_host.so").exists())

    def test_index_cannot_stand_in_for_a_missing_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "host").mkdir()
            (root / "host/_mojolearn_merged.json").write_text(json.dumps(
                {"format": 1, "files": {"merged.so": ["_host"]}}))
            with self.assertRaisesRegex(SystemExit, "missing or unsafe merged binary"):
                rr.binding_files(root, dict(name="_host", set_rel="host/_host.so"))


if __name__ == "__main__":
    unittest.main()
