"""tools/bincache.py: key computation, manifest verification, corruption
rejection and the build-through-cache flow, with file:// URLs standing in for
R2. Stdlib only; no network, no Mojo, no GPU. Run from tools/:

    python3 -m unittest -v test_bincache
"""
import datetime
import io
import json
import os
import shutil

import sys
import tarfile
import tempfile
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bincache as bc  # noqa: E402

OS = dict(id="ubuntu", version_id="22.04", glibc="glibc 2.35", machine="x86_64",
          ld="GNU ld 2.38", cc="cc 11.4")
PLAT = bc.toolchain(Path("/nonexistent"))["platform"]

SCRIPT = """#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"
if false; then pixi run mojo build -j 2 --emit shared-lib -I . -I bindings bindings/_mojolearn_fake.mojo -o x; fi
[ -z "${BINCACHE_TEST_MARKER:-}" ] || echo ran >> "$BINCACHE_TEST_MARKER"
mkdir -p python/mojolearn/identical
printf 'fake-binding %s\\n' "${MOJOLEARN_NUMERIC_MODE:-identical}" > python/mojolearn/identical/.tmp.so
mv python/mojolearn/identical/.tmp.so python/mojolearn/identical/_mojolearn_fake.so
echo built python/mojolearn/identical/_mojolearn_fake.so
"""


def make_repo(root):
    root = Path(root)
    (root / "bindings").mkdir(parents=True)
    (root / "core").mkdir()
    (root / "bindings" / "build_fake.sh").write_text(SCRIPT)
    (root / "bindings" / "_mojolearn_fake.mojo").write_text(
        '"""A docstring that says\nfrom the kernel import nothing\n"""\n'
        "from std.os import getenv\nfrom core import (\n    a,\n    g,  # a module\n)\n")
    (root / "core" / "a.mojo").write_text("from layout import TileTensor\nfn f(): pass\n")
    (root / "core" / "g.mojo").write_text("fn g(): pass\n")
    (root / "core" / "unrelated.mojo").write_text("fn u(): pass\n")
    (root / "pixi.toml").write_text("[workspace]\n")
    (root / "pixi.lock").write_text(
        "      - conda: https://conda.modular.com/max/%s/mojo-compiler-1.0.0-release.conda\n" % PLAT)
    return root


def fields(repo, environ=None, image="runpod:img", arch="sm_89", os_info=None, script="bindings/build_fake.sh"):
    return bc.key_fields(repo, Path(repo) / script, [], environ or {}, image,
                         dev_arch=arch, os_info=os_info or dict(OS))


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.repo = make_repo(Path(self.td.name) / "repo").resolve()
        self.base = bc.key_of(fields(self.repo))

    def test_closure_follows_imports_and_parenthesized_names(self):
        src, rels = bc.source_digest(self.repo, self.repo / "bindings/build_fake.sh", [])
        self.assertEqual(src["scope"], "closure")
        self.assertIn("core/a.mojo", rels)
        self.assertIn("core/g.mojo", rels)          # from core import (a, g): both are modules
        self.assertNotIn("core/unrelated.mojo", rels)

    def test_each_input_moves_the_key(self):
        cases = {
            "toolchain": lambda: (self.repo / "pixi.lock").write_text(
                "      - conda: https://conda.modular.com/max/%s/mojo-compiler-1.0.1-release.conda\n" % PLAT),
            "imported source": lambda: (self.repo / "core/a.mojo").write_text("from layout import T\nfn f(): return\n"),
            "build script": lambda: (self.repo / "bindings/build_fake.sh").write_text(SCRIPT + "# edit\n"),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                saved = {p: p.read_bytes() for p in self.repo.rglob("*") if p.is_file()}
                mutate()
                self.assertNotEqual(bc.key_of(fields(self.repo)), self.base, name)
                for p, b in saved.items():
                    p.write_bytes(b)
        self.assertEqual(bc.key_of(fields(self.repo)), self.base)

    def test_environment_image_arch_and_os_move_the_key(self):
        moved = {
            "image": fields(self.repo, image="do:188571990"),
            "mode": fields(self.repo, {"MOJOLEARN_NUMERIC_MODE": "fast"}),
            "mode deterministic": fields(self.repo, {"MOJOLEARN_NUMERIC_MODE": "deterministic"}),
            "define": fields(self.repo, {"MOJOLEARN_BUILD_EXTRA_DEFINES": "-D MOJOLEARN_2560_MEMSET_FILL=1"}),
            "trial": fields(self.repo, {"MOJOLEARN_GEMM_ARM_TRIAL": "kpack_hg"}),
            "gpu archs": fields(self.repo, {"MOJOLEARN_GPU_ARCHS": "sm_90a"}),
            "column": fields(self.repo, {"MOJOLEARN_TARGET_COLUMN": "cpu"}),
            "jobs": fields(self.repo, {"MOJOLEARN_COMPILE_JOBS": "8"}),
            "device arch": fields(self.repo, arch="gfx942"),
            "os release": fields(self.repo, os_info=dict(OS, version_id="24.04")),
            "glibc": fields(self.repo, os_info=dict(OS, glibc="glibc 2.39")),
            # aarch64 only: bindings/build_*.sh leaves --target-cpu unset
            # there, so two cores that share machine=aarch64 build different
            # instructions. See bc.host_cpu.
            "host cpu": fields(self.repo, os_info=dict(OS, machine="aarch64",
                                                       cpu="CPU part=0xd4f")),
            "host cpu, other core": fields(self.repo, os_info=dict(OS, machine="aarch64",
                                                                   cpu="CPU part=0xd0c")),
        }
        keys = {n: bc.key_of(f) for n, f in moved.items()}
        for name, k in keys.items():
            self.assertNotEqual(k, self.base, name)
        self.assertEqual(len(set(keys.values())), len(keys))

    def test_the_host_cpu_is_keyed_exactly_where_the_build_does_not_pin_it(self):
        # Every bindings/build_*.sh pins `--target-cpu x86-64-v3` on Linux
        # x86-64 and pins NOTHING on any other Linux machine. So the host CPU
        # must be absent from the key on x86 (where keying a pod's EPYC model
        # would cost every hit and buy nothing) and present everywhere else
        # (where it is the instruction set).
        for m in ("x86_64", "amd64"):
            self.assertEqual(bc.host_cpu(m), "", m)
        # Never silently empty on an unpinned machine: with no /proc/cpuinfo
        # to read it says so, and "unknown" is still a value that partitions
        # the cache rather than a field that quietly vanishes.
        self.assertTrue(bc.host_cpu("aarch64"))

    def test_identical_is_the_unset_mode(self):
        self.assertEqual(bc.key_of(fields(self.repo, {"MOJOLEARN_NUMERIC_MODE": "identical"}))
                         == self.base, False)     # the variable itself is keyed
        self.assertEqual(fields(self.repo)["numeric_mode"], "identical")

    def test_non_build_variables_and_unrelated_files_do_not_move_it(self):
        env = {"MOJOLEARN_COMMIT": "abc", "MOJOLEARN_IDENTITY_VENDOR_LABEL": "x", "MOJOLEARN_BINCACHE": "1",
               "MOJOLEARN_SKIP_BUILD_GATE": "1", "HOME": "/elsewhere"}
        self.assertEqual(bc.key_of(fields(self.repo, env)), self.base)
        (self.repo / "core/unrelated.mojo").write_text("fn u(): return 1\n")
        (self.repo / "README.md").write_text("docs\n")
        self.assertEqual(bc.key_of(fields(self.repo)), self.base)

    def test_pixi_tasks_do_not_move_the_key_but_dependencies_do(self):
        toml = self.repo / "pixi.toml"
        toml.write_text('[workspace]\nname = "m"\n[dependencies]\nmojo = ">=1.0.0,<2"\n[tasks]\nprobe = "mojo run x"\n')
        k = bc.key_of(fields(self.repo))
        toml.write_text('[workspace]\nname = "m"\n# a comment\n[dependencies]\nmojo = ">=1.0.0,<2"\n[tasks]\n'
                        'probe = "mojo run x"\ncheck-new = """\n[dependencies]\nnot = 1\n"""\n'
                        '[feature.test.tasks]\nt = "pytest"\n')
        self.assertEqual(bc.key_of(fields(self.repo)), k)
        toml.write_text('[workspace]\nname = "m"\n[dependencies]\nmojo = ">=1.0.1,<2"\n[tasks]\nprobe = "mojo run x"\n')
        self.assertNotEqual(bc.key_of(fields(self.repo)), k)

    def test_unresolvable_import_widens_to_the_whole_tree(self):
        (self.repo / "core/a.mojo").write_text("from nowhere.at_all import thing\n")
        src, rels = bc.source_digest(self.repo, self.repo / "bindings/build_fake.sh", [])
        self.assertEqual(src["scope"], "tree")
        self.assertIn("core/unrelated.mojo", rels)
        k = bc.key_of(fields(self.repo))
        (self.repo / "core/unrelated.mojo").write_text("fn u(): return 2\n")
        self.assertNotEqual(bc.key_of(fields(self.repo)), k)

    def test_sabotage_is_refused(self):
        for env in ({"MOJOLEARN_BUILD_EXTRA_DEFINES": "-D MOJOLEARN_HOST_SABOTAGE=1"},
                    {"MOJOLEARN_MAMBA2_SABOTAGE_X": "1"},
                    {"MOJOLEARN_BYTE_LM_FAULT_INJECT": "3"}):
            with self.subTest(env):
                self.assertTrue(bc.refusal(fields(self.repo, env), env).startswith("sabotage:"))
        self.assertEqual(bc.refusal(fields(self.repo), {}), "")

    def test_host_shim_resolves_through_the_family_builder(self):
        (self.repo / "bindings/build_fake_host.sh").write_text(
            '#!/bin/sh\nexec sh "$(dirname -- "$0")/build_host_family.sh" fake "$@"\n')
        (self.repo / "bindings/build_host_family.sh").write_text(
            'source_file="bindings/_mojolearn_${family}_host.mojo"\n'
            'pixi run mojo build -j 2 --emit shared-lib "$@" -I . -I bindings \\\n    "$source_file" -o x\n')
        (self.repo / "bindings/_mojolearn_fake_host.mojo").write_text("from core.g import g\n")
        src, rels = bc.source_digest(self.repo, self.repo / "bindings/build_fake_host.sh", [])
        self.assertEqual(src["scope"], "closure")
        self.assertIn("bindings/_mojolearn_fake_host.mojo", rels)
        self.assertIn("core/g.mojo", rels)
        self.assertIn("bindings/build_host_family.sh", rels)


class PresignTests(unittest.TestCase):
    def test_aws_documented_vector(self):
        # docs.aws.amazon.com AmazonS3 sigv4-query-string-auth example
        creds = dict(R2_ACCOUNT_ID="x", R2_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE",
                     R2_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", R2_BUCKET="examplebucket")
        url = bc.presign("GET", "test.txt", 86400, creds,
                         now=datetime.datetime(2013, 5, 24, tzinfo=datetime.timezone.utc),
                         host="examplebucket.s3.amazonaws.com", region="us-east-1", path_style=False)
        self.assertTrue(url.endswith("X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.repo = make_repo(Path(self.td.name) / "repo").resolve()
        so = self.repo / "python/mojolearn/identical/_mojolearn_fake.so"
        so.parent.mkdir(parents=True)
        so.write_bytes(os.urandom(2048))
        self.fields = fields(self.repo)
        self.key = bc.key_of(self.fields)
        self.arc = Path(self.td.name) / "a.tar.gz"
        bc.pack(self.arc, self.key, self.fields, self.repo, ["python/mojolearn/identical/_mojolearn_fake.so"], {})

    def rewrite(self, edit_manifest=None, edit_blob=None, extra=None):
        with tarfile.open(self.arc, "r:gz") as tf:
            members = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}
        if edit_manifest:
            man = json.loads(members["manifest.json"])
            edit_manifest(man)
            members["manifest.json"] = json.dumps(man).encode()
        if edit_blob:
            name = "files/python/mojolearn/identical/_mojolearn_fake.so"
            members[name] = edit_blob(members[name])
        members.update(extra or {})
        with tarfile.open(self.arc, "w:gz") as tf:
            for name, data in members.items():
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))

    def assertRejected(self, fragment, key=None, flds=None):
        with self.assertRaises(bc.Reject) as cm:
            bc.verify_archive(self.arc, key or self.key, flds or self.fields)
        self.assertIn(fragment, str(cm.exception))

    def test_good_archive_verifies(self):
        manifest, blobs = bc.verify_archive(self.arc, self.key, self.fields)
        self.assertEqual(list(blobs), ["python/mojolearn/identical/_mojolearn_fake.so"])

    def test_one_flipped_byte_is_rejected(self):
        self.rewrite(edit_blob=lambda b: bytes([b[0] ^ 1]) + b[1:])
        self.assertRejected("sha256 mismatch")

    def test_truncated_archive_is_rejected(self):
        data = self.arc.read_bytes()
        self.arc.write_bytes(data[: len(data) // 2])
        self.assertRejected("unreadable")

    def test_another_key_or_fields_are_rejected(self):
        other = fields(self.repo, {"MOJOLEARN_NUMERIC_MODE": "fast"})
        self.assertRejected("key mismatch", key=bc.key_of(other), flds=other)

    def test_manifest_edited_to_match_a_new_blob_is_still_bound_to_the_key(self):
        def edit(man):
            man["fields"]["image"] = "forged"
        self.rewrite(edit_manifest=edit)
        self.assertRejected("key mismatch")

    def test_extra_member_and_traversal_are_rejected(self):
        self.rewrite(extra={"files/../../evil.so": b"x"})
        self.assertRejected("members differ")

        def trav(man):
            man["files"][0]["path"] = "../evil.so"
        bc.pack(self.arc, self.key, self.fields, self.repo, ["python/mojolearn/identical/_mojolearn_fake.so"], {})
        self.rewrite(edit_manifest=trav, extra={"files/../evil.so": b"x"})
        self.assertRejected("")


class BuildFlowTests(unittest.TestCase):
    """Fresh on one tree, cached on a second, corruption on a third; R2 is a
    directory reached through file:// URLs."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.base = Path(self.td.name).resolve()
        self.store = self.base / "r2"
        self.store.mkdir()

    def tree(self, name):
        # The repository path is keyed (the rpath into .pixi is baked into a
        # binding), and every box unpacks to /root/mojolearn, so each "box"
        # here is a fresh tree at ONE path; the previous box's tree moves aside.
        repo = self.base / "mojolearn"
        if repo.exists():
            repo.rename(self.base / ("gone-%d" % len(list(self.base.glob("gone-*")))))
        self.trees = getattr(self, "trees", {})
        self.trees[name] = make_repo(repo)
        return repo

    def write_map(self, name, keys=(), leg="leg1", skeys=()):
        d = self.base / ("map-" + name)
        d.mkdir()
        lines = ["#partition\tsm_89/runpod-img", "#image\trunpod:img", "#leg\t" + leg]
        for k in keys:
            lines.append("get\t%s\tfile://%s/%s.tar.gz" % (k, self.store, k))
        for k in skeys:
            lines.append("sget\t%s\tfile://%s/%s.tar.gz" % (k, self.store, k))
        for i in range(3):
            lines.append("put\t%03d\tfile://%s/inbox/%s/%03d.tar.gz" % (i, self.store, leg, i))
        (d / "urls.tsv").write_text("\n".join(lines) + "\n")
        return d / "urls.tsv"

    def build(self, repo, map_path, out, extra_env=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MOJOLEARN_")}
        env.update(MOJOLEARN_BINCACHE_MAP=str(map_path), MOJOLEARN_BINCACHE_OUT=str(out),
                   BINCACHE_TEST_MARKER=str(repo / "ran.txt"))
        env.update(extra_env or {})
        # deterministic box facts so three trees in one process share a key
        orig = (bc.device_arch, bc.os_fields)
        bc.device_arch, bc.os_fields = (lambda: "sm_89"), (lambda: dict(OS))
        try:
            rc = self._call(env, repo)
        finally:
            bc.device_arch, bc.os_fields = orig
        return rc

    def _call(self, env, repo):
        return bc.cmd_build([str(repo / "bindings/build_fake.sh")], environ=env)

    def provenance(self, out):
        return [l.split("\t") for l in (Path(out) / "provenance.tsv").read_text().splitlines()]

    def promote_locally(self, out):
        rows = (Path(out) / "uploads.tsv").read_text().splitlines()
        for row in rows:
            leg, slot, dest, key = bc.check_upload_row(row, Path(out) / "keys")
            shutil.move(str(self.store / "inbox" / leg / (slot + ".tar.gz")), str(self.store / (key + ".tar.gz")))
        return rows

    def test_fresh_then_cached_then_corrupt(self):
        a = self.tree("a")
        out_a = self.base / "out-a"
        self.assertEqual(self.build(a, self.write_map("a"), out_a), 0)
        prov = self.provenance(out_a)
        self.assertEqual(prov[0][2], "miss+built-uploaded")
        key = prov[0][3]
        self.assertTrue((a / "ran.txt").exists())
        rows = self.promote_locally(out_a)
        self.assertEqual(len(rows), 1)
        so_a = (a / "python/mojolearn/identical/_mojolearn_fake.so").read_bytes()

        b = self.tree("b")
        out_b = self.base / "out-b"
        self.assertEqual(self.build(b, self.write_map("b", [key]), out_b), 0)
        prov = self.provenance(out_b)
        self.assertEqual(prov[0][2], "hit")
        self.assertEqual(prov[0][3], key)
        self.assertFalse((b / "ran.txt").exists(), "a hit must not run the build script")
        self.assertEqual((b / "python/mojolearn/identical/_mojolearn_fake.so").read_bytes(), so_a)
        self.assertIn(bc.sha256_bytes(so_a), prov[0][5])
        self.assertFalse((out_b / "uploads.tsv").exists())

        # corrupt the stored object's binding, keep its manifest
        arc = self.store / (key + ".tar.gz")
        with tarfile.open(arc, "r:gz") as tf:
            members = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}
        name = [n for n in members if n.startswith("files/")][0]
        members[name] = b"X" + members[name][1:]
        with tarfile.open(arc, "w:gz") as tf:
            for n, data in members.items():
                ti = tarfile.TarInfo(n)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        c = self.tree("c")
        out_c = self.base / "out-c"
        self.assertEqual(self.build(c, self.write_map("c", [key]), out_c), 0)
        prov = self.provenance(out_c)
        self.assertTrue(prov[0][2].startswith("rejected:sha256 mismatch"), prov[0][2])
        self.assertTrue(prov[0][2].endswith("+built-uploaded"), prov[0][2])
        self.assertTrue((c / "ran.txt").exists())
        self.assertEqual((c / "python/mojolearn/identical/_mojolearn_fake.so").read_bytes(), so_a)

    def test_changed_mode_misses(self):
        a = self.tree("a")
        out_a = self.base / "out-a"
        self.build(a, self.write_map("a"), out_a)
        self.promote_locally(out_a)
        key = self.provenance(out_a)[0][3]
        b = self.tree("b")
        out_b = self.base / "out-b"
        self.build(b, self.write_map("b", [key]), out_b, {"MOJOLEARN_NUMERIC_MODE": "fast"})
        prov = self.provenance(out_b)
        self.assertEqual(prov[0][2], "miss+built-uploaded")
        self.assertNotEqual(prov[0][3], key)

    def test_sabotage_never_touches_the_cache(self):
        a = self.tree("a")
        out_a = self.base / "out-a"
        self.build(a, self.write_map("a"), out_a)
        self.promote_locally(out_a)
        key = self.provenance(out_a)[0][3]
        b = self.tree("b")
        out_b = self.base / "out-b"
        env = {"MOJOLEARN_BUILD_EXTRA_DEFINES": "-D MOJOLEARN_HOST_SABOTAGE=1"}
        self.assertEqual(self.build(b, self.write_map("b", [key]), out_b, env), 0)
        prov = self.provenance(out_b)
        self.assertTrue(prov[0][2].startswith("refused:sabotage:"), prov[0][2])
        self.assertTrue((b / "ran.txt").exists())
        self.assertFalse((out_b / "uploads.tsv").exists())
        self.assertFalse(list((out_b / "keys").glob("*.json")))

    def test_negative_control_has_its_own_namespace_and_never_serves_production(self):
        # tools/runpod_cpu_leg.sh, 2026-09-15: MOJOLEARN_BINCACHE_NEGATIVE=1
        # caches a sabotage build, under variant=sabotage and its own prefix.
        sab = {"MOJOLEARN_BUILD_EXTRA_DEFINES": "-D MOJOLEARN_HOST_SABOTAGE=1",
               "MOJOLEARN_BINCACHE_NEGATIVE": "1"}
        a = self.tree("a")
        out_a = self.base / "out-a"
        self.build(a, self.write_map("a"), out_a)
        self.promote_locally(out_a)
        prod_key = self.provenance(out_a)[0][3]
        so_prod = (a / "python/mojolearn/identical/_mojolearn_fake.so").read_bytes()

        b = self.tree("b")
        out_b = self.base / "out-b"
        self.assertEqual(self.build(b, self.write_map("b", [prod_key]), out_b, sab), 0)
        prov = self.provenance(out_b)
        self.assertEqual(prov[0][2], "negative-miss+built-uploaded")
        sab_key = prov[0][3]
        self.assertNotEqual(sab_key, prod_key)
        self.assertTrue((b / "ran.txt").exists(), "a production entry must never serve a sabotage build")
        row = (out_b / "uploads.tsv").read_text().splitlines()[0]
        self.assertTrue(row.endswith("sabotage=1"), row)
        leg, slot, dest, key = bc.check_upload_row(row, out_b / "keys")
        self.assertTrue(dest.startswith(bc.SABOTAGE_PREFIX + "/"), dest)
        with self.assertRaises(ValueError):
            bc.check_upload_row(row.replace("sabotage=1", "sabotage=0"), out_b / "keys")
        with self.assertRaises(ValueError):
            bc.check_upload_row(row.replace(bc.SABOTAGE_PREFIX, bc.OBJECT_PREFIX), out_b / "keys")
        self.promote_locally(out_b)
        so_sab = (b / "python/mojolearn/identical/_mojolearn_fake.so").read_bytes()

        # the negative control is served from its namespace only
        c = self.tree("c")
        out_c = self.base / "out-c"
        self.assertEqual(self.build(c, self.write_map("c", keys=[sab_key]), out_c, sab), 0)
        self.assertEqual(self.provenance(out_c)[0][2], "negative-miss+built-uploaded")
        d = self.tree("d")
        out_d = self.base / "out-d"
        self.assertEqual(self.build(d, self.write_map("d", skeys=[sab_key]), out_d, sab), 0)
        self.assertEqual(self.provenance(out_d)[0][2], "negative-hit")
        self.assertFalse((d / "ran.txt").exists())
        self.assertEqual((d / "python/mojolearn/identical/_mojolearn_fake.so").read_bytes(), so_sab)

        # a production build never reads an sget row, and a sabotage archive
        # planted under the production key is rejected on its fields
        e = self.tree("e")
        out_e = self.base / "out-e"
        self.assertEqual(self.build(e, self.write_map("e", skeys=[prod_key]), out_e), 0)
        self.assertEqual(self.provenance(out_e)[0][2], "miss+built-uploaded")
        shutil.copyfile(self.store / (sab_key + ".tar.gz"), self.store / (prod_key + ".tar.gz"))
        f = self.tree("f")
        out_f = self.base / "out-f"
        self.assertEqual(self.build(f, self.write_map("f", keys=[prod_key]), out_f), 0)
        self.assertTrue(self.provenance(out_f)[0][2].startswith("rejected:key mismatch"), self.provenance(out_f)[0][2])
        self.assertEqual((f / "python/mojolearn/identical/_mojolearn_fake.so").read_bytes(), so_prod)

    def test_off_is_a_pass_through(self):
        a = self.tree("a")
        out_a = self.base / "out-a"
        self.assertEqual(self.build(a, self.write_map("a"), out_a, {"MOJOLEARN_BINCACHE": "0"}), 0)
        self.assertTrue((a / "ran.txt").exists())
        self.assertFalse((out_a / "provenance.tsv").exists())
        b = self.tree("b")
        self.assertEqual(self.build(b, self.base / "no-such-map.tsv", self.base / "out-b"), 0)
        self.assertTrue((b / "ran.txt").exists())
        self.assertFalse((self.base / "out-b").exists())

    def test_failed_build_keeps_its_exit_code_and_uploads_nothing(self):
        a = self.tree("a")
        (a / "bindings/build_fake.sh").write_text(SCRIPT.replace("mkdir -p python", "exit 7\nmkdir -p python"))
        out_a = self.base / "out-a"
        self.assertEqual(self.build(a, self.write_map("a"), out_a), 7)
        self.assertTrue(self.provenance(out_a)[0][2].endswith("+build-failed"))
        self.assertFalse((out_a / "uploads.tsv").exists())

    def test_promote_refuses_rows_the_fields_do_not_support(self):
        a = self.tree("a")
        out_a = self.base / "out-a"
        self.build(a, self.write_map("a"), out_a)
        row = (out_a / "uploads.tsv").read_text().splitlines()[0]
        leg, slot, dest, key = bc.check_upload_row(row, out_a / "keys")
        kj = out_a / "keys" / (key + ".json")
        f = json.loads(kj.read_text())
        f["build_env"]["MOJOLEARN_BUILD_EXTRA_DEFINES"] = "-D MOJOLEARN_HOST_SABOTAGE=1"
        kj.write_text(json.dumps(f))
        with self.assertRaises(ValueError):
            bc.check_upload_row(row, out_a / "keys")
        with self.assertRaises(ValueError):
            bc.check_upload_row(row.replace("sabotage=0", "sabotage=1"), out_a / "keys")
        with self.assertRaises(ValueError):
            bc.check_upload_row(row.replace(dest, "bincache/v1/../x/%s.tar.gz" % key), out_a / "keys")


class ReleaseDeclaredTests(BuildFlowTests):
    """The Linux release build (packaging/linux/build_sets.sh through the CPU
    build box, 2026-09-22): declared outputs on the R2 path, the box-local hot
    directory shared by the three sets, and the per-hit placement record that
    names the commit an archive was compiled from."""

    OUT = "python/mojolearn/identical/_mojolearn_fake.so"

    def rbuild(self, repo, map_path, out, hot=None, extra=None):
        env = dict(MOJOLEARN_BINCACHE_OUTPUTS=self.OUT, MOJOLEARN_BINCACHE_SHELL="bash",
                   MOJOLEARN_COMMIT="c" * 40)
        if hot is not None:
            env["MOJOLEARN_BINCACHE_HOT_DIR"] = str(hot)
        env.update(extra or {})
        return self.build(repo, map_path, out, env)

    def test_declared_build_is_keyed_apart_and_archives_only_its_output(self):
        a = self.tree("a")
        # a neighbour's binary written during the build must not be archived
        (a / "bindings/build_fake.sh").write_text(SCRIPT + "printf n > python/mojolearn/identical/_mojolearn_other.so\n")
        out_a = self.base / "out-a"
        self.assertEqual(self.rbuild(a, self.write_map("a"), out_a), 0)
        row = self.provenance(out_a)[0]
        self.assertEqual(row[2], "miss+built-uploaded")
        self.assertNotIn("_mojolearn_other.so", row[5])
        fields_a = json.loads((out_a / "keys" / (row[3] + ".json")).read_text())
        self.assertEqual(fields_a["declared"]["outputs"], [self.OUT])
        self.assertIn("bindings/build_fake.sh", fields_a["declared"]["inputs"]["files"])
        b = self.tree("b")
        out_b = self.base / "out-b"
        self.build(b, self.write_map("b"), out_b)
        self.assertNotEqual(self.provenance(out_b)[0][3], row[3], "declared and undeclared keys must differ")

    def test_hot_directory_serves_the_next_set_and_records_the_commit(self):
        hot = self.base / "hot"
        a = self.tree("a")
        out = self.base / "out"
        self.assertEqual(self.rbuild(a, self.write_map("a"), out, hot=hot), 0)
        key = self.provenance(out)[0][3]
        self.assertTrue((hot / (key + ".tar.gz")).is_file())
        so = (a / self.OUT).read_bytes()
        (a / self.OUT).unlink()          # build_sets.sh moves each set out of the tree
        (a / "ran.txt").unlink()
        self.assertEqual(self.rbuild(a, self.write_map("a2", leg="leg2"), out, hot=hot), 0)
        row = self.provenance(out)[1]
        self.assertEqual(row[2], "hot-hit")
        self.assertFalse((a / "ran.txt").exists(), "a hot hit must not run the build script")
        self.assertEqual((a / self.OUT).read_bytes(), so)
        rec = [json.loads(l) for l in (out / "placements.jsonl").read_text().splitlines()]
        self.assertEqual(rec[0]["archive_source_commit"], "c" * 40)
        self.assertEqual(rec[0]["files"][0]["sha256"], bc.sha256_bytes(so))

    def test_a_declared_output_the_build_did_not_write_is_not_cached(self):
        a = self.tree("a")
        out = self.base / "out"
        env = dict(MOJOLEARN_BINCACHE_OUTPUTS="python/mojolearn/identical/_mojolearn_nothing.so")
        self.assertEqual(self.build(a, self.write_map("a"), out, env), 0)
        self.assertIn("declared-output-not-written", self.provenance(out)[0][2])
        self.assertFalse((out / "uploads.tsv").exists())

    def test_release_scheduling_variables_do_not_move_the_key(self):
        a = self.tree("a")
        base = fields(a, {"MOJOLEARN_NUMERIC_MODE": "identical"})
        moved = fields(a, {"MOJOLEARN_NUMERIC_MODE": "identical", "MOJOLEARN_BUILD_JOBS": "16",
                           "MOJOLEARN_RELEASE_BUILD_SECONDS": "2371", "MOJOLEARN_RELEASE_NO_DEVICE": "1",
                           "MOJOLEARN_QUALIFY_PYTHON": "/x/python", "MOJOLEARN_EXPECT_CORE_HOST_SHA256": "skip"})
        self.assertEqual(bc.key_of(base), bc.key_of(moved))
        self.assertNotEqual(bc.key_of(base), bc.key_of(fields(a, {"MOJOLEARN_COMPILE_JOBS": "4"})))


class LocalCacheTests(unittest.TestCase):
    """The macOS release build's local directory cache: declared outputs, a
    verified archive, never a replaced destination, fail closed."""

    OUT = "python/mojolearn/identical/_mojolearn_fake.so"

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.base = Path(self.td.name).resolve()
        self.cache = self.base / "cache"
        self.n = 0

    def tree(self, script=SCRIPT):
        repo = self.base / "mojolearn"
        if repo.exists():
            self.n += 1
            repo.rename(self.base / ("gone-%d" % self.n))
        make_repo(repo)
        (repo / "bindings" / "build_fake.sh").write_text(script)
        return repo

    def build(self, repo, outputs=OUT, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MOJOLEARN_")}
        env.update(MOJOLEARN_BINCACHE_DIR=str(self.cache), MOJOLEARN_BINCACHE_OUTPUTS=outputs,
                   BINCACHE_TEST_MARKER=str(repo / "ran.txt"))
        env.update(extra or {})
        orig = (bc.device_arch, bc.os_fields, bc.darwin_toolchain)
        bc.device_arch, bc.os_fields = (lambda: "none"), (lambda: dict(OS))
        bc.darwin_toolchain = lambda: dict(xcode="Xcode 26.0", metal="metal 32023")
        try:
            return _run_with_buffer(bc.cmd_build)([str(repo / "bindings/build_fake.sh")], environ=env)
        finally:
            bc.device_arch, bc.os_fields, bc.darwin_toolchain = orig

    def rows(self):
        return [l.split("\t") for l in (self.cache / "provenance" / "provenance.tsv").read_text().splitlines()]

    def test_fresh_then_hit_then_corrupt(self):
        a = self.tree()
        self.assertEqual(self.build(a), 0)
        self.assertEqual(self.rows()[-1][2], "miss+built-cached")
        key = self.rows()[-1][3]
        built = (a / self.OUT).read_bytes()
        b = self.tree()
        self.assertEqual(self.build(b), 0)
        self.assertEqual(self.rows()[-1][2], "hit")
        self.assertFalse((b / "ran.txt").exists(), "a hit must not run the build script")
        self.assertEqual((b / self.OUT).read_bytes(), built)
        arc = self.cache / (key + ".tar.gz")
        with tarfile.open(arc, "r:gz") as tf:
            members = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}
        members["files/" + self.OUT] = b"X" + members["files/" + self.OUT][1:]
        with tarfile.open(arc, "w:gz") as tf:
            for n, data in members.items():
                ti = tarfile.TarInfo(n)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        c = self.tree()
        self.assertEqual(self.build(c), 0)
        self.assertTrue(self.rows()[-1][2].startswith("rejected:sha256 mismatch"), self.rows()[-1][2])
        self.assertTrue((c / "ran.txt").exists(), "a corrupt archive must fall back to a real build")
        self.assertEqual((c / self.OUT).read_bytes(), built)

    def test_only_the_declared_output_is_archived(self):
        # The release script runs builds side by side in ONE tree, so a .so
        # another build writes at the same time must never enter this archive.
        script = SCRIPT.replace("echo built", "printf other > python/mojolearn/identical/_mojolearn_other.so\necho built")
        a = self.tree(script)
        self.assertEqual(self.build(a), 0)
        key = self.rows()[-1][3]
        with tarfile.open(self.cache / (key + ".tar.gz"), "r:gz") as tf:
            names = sorted(m.name for m in tf.getmembers())
        self.assertEqual(names, ["files/" + self.OUT, "manifest.json"])

    def test_no_declared_output_means_no_cache(self):
        a = self.tree()
        self.assertEqual(self.build(a, outputs=""), 0)
        self.assertEqual(self.rows()[-1][2], "refused:no-declared-outputs")
        self.assertFalse(list(self.cache.glob("*.tar.gz")))
        self.assertEqual(self.build(self.tree(), outputs="../escape.so"), 0)
        self.assertEqual(self.rows()[-1][2], "refused:no-declared-outputs")

    def test_a_declared_output_the_build_did_not_write_is_not_cached(self):
        a = self.tree()
        self.assertEqual(self.build(a, outputs="python/mojolearn/identical/_mojolearn_nothing.so"), 0)
        self.assertIn("declared-output-not-written", self.rows()[-1][2])
        self.assertFalse(list(self.cache.glob("*.tar.gz")))

    def test_a_helper_the_script_runs_moves_the_key_and_a_comment_does_not(self):
        helper = SCRIPT.replace("mkdir -p python", "sh tools/helper.sh\nmkdir -p python")
        a = self.tree(helper)
        (a / "tools").mkdir()
        (a / "tools" / "helper.sh").write_text("true\n")
        self.build(a)
        first = self.rows()[-1][3]
        b = self.tree(helper)
        (b / "tools").mkdir()
        (b / "tools" / "helper.sh").write_text("true # changed\n")
        self.build(b)
        self.assertEqual(self.rows()[-1][2], "miss+built-cached")
        self.assertNotEqual(self.rows()[-1][3], first)
        # a path named only in a comment is not an input
        commented = SCRIPT.replace("set -eu", "set -eu\n# see tools/unrelated.sh")
        c = self.tree(commented)
        self.assertEqual(bc.local_inputs(c, c / "bindings/build_fake.sh")["files"], ["bindings/build_fake.sh"])

    def test_sabotage_and_an_existing_destination_never_use_the_cache(self):
        a = self.tree()
        self.build(a)
        b = self.tree()
        self.assertEqual(self.build(b, extra={"MOJOLEARN_BUILD_EXTRA_DEFINES": "-D X_SABOTAGE=1"}), 0)
        self.assertTrue(self.rows()[-1][2].startswith("refused:sabotage:"))
        self.assertTrue((b / "ran.txt").exists())
        c = self.tree()
        (c / self.OUT).parent.mkdir(parents=True, exist_ok=True)
        (c / self.OUT).write_bytes(b"old")
        self.assertEqual(self.build(c), 0)
        self.assertTrue(self.rows()[-1][2].startswith("bypass-destination-exists"), self.rows()[-1][2])
        self.assertTrue((c / "ran.txt").exists())

    def test_a_failed_build_keeps_its_exit_code_and_caches_nothing(self):
        a = self.tree(SCRIPT.replace("mkdir -p python", "exit 7\nmkdir -p python"))
        self.assertEqual(self.build(a), 7)
        self.assertTrue(self.rows()[-1][2].endswith("+build-failed"))
        self.assertFalse(list(self.cache.glob("*.tar.gz")))


def _run_with_buffer(fn):
    """run_tee writes to sys.stdout.buffer, which unittest's capture may lack."""
    def wrapper(*a, **k):
        if not hasattr(sys.stdout, "buffer"):
            raw = io.BytesIO()
            old = sys.stdout
            sys.stdout = io.TextIOWrapper(raw)
            try:
                return fn(*a, **k)
            finally:
                sys.stdout = old
        return fn(*a, **k)
    return wrapper


BuildFlowTests._call = _run_with_buffer(BuildFlowTests._call)


if __name__ == "__main__":
    unittest.main()
