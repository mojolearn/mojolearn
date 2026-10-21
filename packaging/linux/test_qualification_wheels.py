"""File-only qualification packaging tests; no compiler or native code runs."""
import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("qualification_wheels", HERE / "qualification_wheels.py")
qw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qw)


def installed_dist(site, name, payload):
    dist = site / (name.replace("-", "_") + "-0.8.55.dist-info")
    dist.mkdir(parents=True)
    metadata = b"Metadata-Version: 2.4\nName: " + name.encode() + b"\nVersion: 0.8.55\n\n"
    payload = dict(payload, **{dist.relative_to(site).as_posix() + "/METADATA": metadata})
    records = []
    for name, raw in payload.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        records.append((name, "sha256=" + digest, str(len(raw))))
    buffer = io.StringIO()
    csv.writer(buffer).writerows(records + [(dist.relative_to(site).as_posix() + "/RECORD", "", "")])
    (dist / "RECORD").write_text(buffer.getvalue())


class QualificationWheels(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.site = self.root / "old-env/site-packages"
        self.site.mkdir(parents=True)
        self.core = {"mojolearn/.libs/libretained.so": b"retained runtime",
                     "mojolearn/host/_mojolearn_tokenizer_host.so": b"old tokenizer",
                     "mojolearn/identity_columns/COMMIT": b"1111111111111111111111111111111111111111\n",
                     "mojolearn/obsolete.py": b"this module must not ship"}
        installed_dist(self.site, "mojolearn", self.core)
        installed_dist(self.site, "mojolearn-amd", {
            "mojolearn/hip_native/gfx942/identical/_mojolearn.so": b"retained GPU"})
        self.commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()

    def test_records_and_tampering(self):
        entries, provenance, distributions = qw.retained_payload(self.site)
        self.assertEqual(len(distributions), 2)
        self.assertNotIn("mojolearn/obsolete.py", entries)
        member = "mojolearn/hip_native/gfx942/identical/_mojolearn.so"
        self.assertEqual(provenance[member]["origin"], "installed-record")
        entries[member].write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "RECORD mismatch"):
            qw.retained_payload(self.site)

    def test_overlay_requires_evidence_and_rejects_non_elf(self):
        artifact = self.root / "new.so"
        artifact.write_bytes(b"not ELF")
        row = dict(member="mojolearn/host/_mojolearn_tokenizer_host.so", path=str(artifact),
                   source_commit=self.commit, build_evidence=[])
        with self.assertRaisesRegex(ValueError, "not an ELF"):
            qw.apply_native_overlay({}, {}, [row])
        artifact.write_bytes(b"\x7fELFtest")
        with self.assertRaisesRegex(ValueError, "needs retained build evidence"):
            qw.apply_native_overlay({}, {}, [row])

    def test_real_split_writer_current_metadata_and_exact_assets(self):
        # Capture every input byte to prove packaging does not mutate the old
        # installation. Only the local wheels contain new Python/native bytes.
        before = {p.relative_to(self.site).as_posix(): p.read_bytes() for p in self.site.rglob("*") if p.is_file()}
        artifact, receipt = self.root / "new.so", self.root / "build.log"
        artifact.write_bytes(b"\x7fELFfixture-new-tokenizer")
        receipt.write_text("file fixture, not a real compilation receipt\n")
        row = dict(member="mojolearn/host/_mojolearn_tokenizer_host.so", path=str(artifact),
                   source_commit=self.commit, build_evidence=[str(receipt)])
        out = self.root / "wheels"
        report = qw.assemble(ROOT, self.commit, self.site, out, [row])
        self.assertFalse(report["publishable"])
        self.assertEqual(len(report["wheels"]), 2)
        self.assertEqual(report["replaced_assets"][row["member"]]["sha256"],
                         hashlib.sha256(b"old tokenizer").hexdigest())
        for item in report["wheels"]:
            self.assertTrue(item["path"].endswith("-py3-none-linux_x86_64.whl"))
            with zipfile.ZipFile(item["path"]) as wheel:
                metadata_name = next(n for n in wheel.namelist() if n.endswith("/METADATA"))
                metadata = BytesParser().parsebytes(wheel.read(metadata_name))
                if metadata["Name"] == "mojolearn-core":
                    self.assertEqual(wheel.read(row["member"]), artifact.read_bytes())
                    self.assertEqual(wheel.read("mojolearn/identity_columns/COMMIT"), (self.commit + "\n").encode())
                    self.assertEqual(wheel.read("mojolearn/gpu_plugins.py"),
                                     (ROOT / "python/mojolearn/gpu_plugins.py").read_bytes())
                    self.assertNotIn("mojolearn/obsolete.py", wheel.namelist())
                    self.assertFalse(any("hip_native" in n for n in wheel.namelist()))
                else:
                    self.assertEqual(metadata["Name"], "mojolearn-amd")
                    self.assertIn("mojolearn-core==0.8.55", metadata.get_all("Requires-Dist"))
                    self.assertEqual(wheel.read("mojolearn/hip_native/gfx942/identical/_mojolearn.so"), b"retained GPU")
                # Independent wheel RECORD check, including generated metadata.
                record_name = metadata_name.removesuffix("METADATA") + "RECORD"
                for name, encoded, size in csv.reader(wheel.read(record_name).decode().splitlines()):
                    if name != record_name:
                        raw = wheel.read(name)
                        self.assertEqual(str(len(raw)), size)
                        self.assertEqual("sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("="), encoded)
        self.assertEqual(json.loads((out / "split-audit.json").read_text())["problems"], [])
        self.assertEqual(before, {p.relative_to(self.site).as_posix(): p.read_bytes() for p in self.site.rglob("*") if p.is_file()})
        with self.assertRaisesRegex(ValueError, "already exists"):
            qw.assemble(ROOT, self.commit, self.site, out, [row])

    def test_source_freeze_must_match(self):
        with self.assertRaisesRegex(ValueError, "not the requested freeze"):
            qw.assemble(ROOT, "a" * 40, self.site, self.root / "out")

    def test_old_ptx_source_is_preserved_and_current_source_is_not_forged(self):
        pw = qw.load_writer(ROOT)
        plugins = pw.gpu_plugins
        old_source = "1" * 40
        native_member = "mojolearn/cuda_native/sm_89/identical/_mojolearn.so"
        installed_dist(self.site, "mojolearn-nvidia", {native_member: b"retained sm89"})
        raw_ptx = b"retained ptx"
        manifest = dict(schema=plugins.PTX_MANIFEST_SCHEMA, code_format=plugins.PTX_CODE_FORMAT,
            vendor="cuda", target="sm_80", min_compute_capability=[8, 0], errors=[],
            source_commit=old_source, source_dirty=False, files=[dict(file="identical/_mojolearn.so",
                numeric_mode="identical", sha256=hashlib.sha256(raw_ptx).hexdigest(),
                ptx_modules=[dict(target="sm_80", sha256="a" * 64)])])
        manifest_bytes = json.dumps(manifest).encode()
        installed_dist(self.site, "mojolearn-nvidia-ptx", {
            "mojolearn/cuda_ptx/sm_80/identical/_mojolearn.so": raw_ptx,
            "mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json": manifest_bytes})
        report = qw.assemble(ROOT, self.commit, self.site, self.root / "wheels")
        self.assertFalse(report["ptx"]["current_source_admissible"])
        self.assertEqual(report["ptx"]["source_commit"], old_source)
        for item in report["wheels"]:
            with zipfile.ZipFile(item["path"]) as wheel:
                if item["profile"] == "nvidia-ptx":
                    self.assertEqual(wheel.read("mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json"), manifest_bytes)
                elif item["profile"] is None:
                    self.assertEqual(wheel.read("mojolearn/identity_columns/COMMIT"), (self.commit + "\n").encode())


if __name__ == "__main__":
    unittest.main()
