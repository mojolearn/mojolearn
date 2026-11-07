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
: "${MOJOLEARN_BUILD_EXTRA_DEFINES:-}${MOJOLEARN_GEMM_ARM_TRIAL:-}${MOJOLEARN_COMPILE_JOBS:-2}"
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

    def test_imported_symbol_does_not_alias_module_filename_case(self):
        package = self.repo / "core"
        (package / "__init__.mojo").write_text("from core.tape import Tape\n")
        (package / "tape.mojo").write_text("struct Tape: pass\n")
        entry = self.repo / "bindings/_mojolearn_fake.mojo"
        entry.write_text("from core import Tape\n")
        _, rels = bc.source_digest(self.repo, self.repo / "bindings/build_fake.sh", [])
        self.assertIn("core/tape.mojo", rels)
        self.assertNotIn("core/Tape.mojo", rels)
        # This must also reject a mismatched directory on case-insensitive Macs.
        self.assertFalse(bc.exact_source_file(self.repo / "Core/tape.mojo", self.repo))

    def test_each_input_moves_the_key(self):
        cases = {
            "toolchain": lambda: (self.repo / "pixi.lock").write_text(
                "      - conda: https://conda.modular.com/max/%s/mojo-compiler-1.0.1-release.conda\n" % PLAT),
            "imported source": lambda: (self.repo / "core/a.mojo").write_text("from layout import T\nfn f(): return\n"),
            "build script": lambda: (self.repo / "bindings/build_fake.sh").write_text(SCRIPT + "true\n"),
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

    def test_unresolvable_import_widens_the_v1_key_and_fails_the_closure_key(self):
        (self.repo / "core/a.mojo").write_text("from nowhere.at_all import thing\n")
        # v1 (kept verifiable): the whole tree.
        src, rels = bc.source_digest(self.repo, self.repo / "bindings/build_fake.sh", [])
        self.assertEqual(src["scope"], "tree")
        self.assertIn("core/unrelated.mojo", rels)
        k = bc.key_of(bc.key_fields(self.repo, self.repo / "bindings/build_fake.sh", [], {}, "i",
                                    dev_arch="x", os_info=dict(OS), rule="v1"))
        (self.repo / "core/unrelated.mojo").write_text("fn u(): return 2\n")
        self.assertNotEqual(bc.key_of(bc.key_fields(self.repo, self.repo / "bindings/build_fake.sh", [], {}, "i",
                                                    dev_arch="x", os_info=dict(OS), rule="v1")), k)
        # closure-v2: no fallback, a loud error naming the import and its file.
        with self.assertRaises(bc.UnresolvedImport) as caught:
            fields(self.repo)
        self.assertEqual(caught.exception.module, "nowhere.at_all")
        self.assertEqual(caught.exception.importer, "core/a.mojo")
        self.assertIn("nowhere.at_all", str(caught.exception))

    def test_unread_and_per_run_variables_do_not_move_the_key(self):
        for env in ({"MOJOLEARN_SLOT_TOKEN": "681867bcbd7740c6"},
                    {"MOJOLEARN_MERGED_KEY": "a4675865"},
                    {"MOJOLEARN_RUNPOD_KEY_FILE": "/Users/x/.key"},
                    {"MOJOLEARN_CUDA_CODE_FORMAT": "ptx", "MOJOLEARN_BUILD_TIERS": "identical"},
                    {"MOJOLEARN_SOMETHING_NO_SCRIPT_READS": "1"}):
            with self.subTest(env):
                self.assertEqual(bc.key_of(fields(self.repo, env)), self.base)
        # The toolchain's own variables stay keyed, read or not.
        self.assertNotEqual(bc.key_of(fields(self.repo, {"MODULAR_HOME": "/x"})), self.base)
        self.assertNotEqual(bc.key_of(fields(self.repo, {"MOJOLEARN_BUILD_PIXI_ENV": "cuda"})), self.base)

    def test_a_dynamic_variable_name_is_read_by_pattern(self):
        (self.repo / "bindings/build_fake.sh").write_text(
            SCRIPT + 'd=$(eval "printf %s \\"\\${MOJOLEARN_${FAMILY}_HOST_OUTDIR:-}\\"")\n')
        base = bc.key_of(fields(self.repo))
        self.assertNotEqual(bc.key_of(fields(self.repo, {"MOJOLEARN_FAKE_HOST_OUTDIR": "o"})), base)
        self.assertEqual(bc.key_of(fields(self.repo, {"MOJOLEARN_FAKE_OUTDIR": "o"})), base)

    def test_comment_edits_of_shell_inputs_do_not_move_the_key(self):
        (self.repo / "bindings/build_fake.sh").write_text(SCRIPT + "# a note\n\n")
        self.assertEqual(bc.key_of(fields(self.repo)), self.base)

    def test_a_sourced_helper_beside_the_script_is_an_input(self):
        (self.repo / "bindings/build_defines.sh").write_text("X=1\n")
        (self.repo / "bindings/build_fake.sh").write_text(
            '. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/build_defines.sh"\n' + SCRIPT)
        f = fields(self.repo)
        self.assertIn("bindings/build_defines.sh", f["inputs"]["files"])
        k = bc.key_of(f)
        (self.repo / "bindings/build_defines.sh").write_text("X=2\n")
        self.assertNotEqual(bc.key_of(fields(self.repo)), k)

    def test_unread_sabotage_variable_still_makes_a_negative_control(self):
        env = {"MOJOLEARN_UNREAD_SABOTAGE_X": "1"}
        f = fields(self.repo, env)
        self.assertEqual(f.get("variant"), "sabotage")
        self.assertTrue(bc.refusal(f, env).startswith("sabotage:"))

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
            'entry_file="bindings/entry/entry__mojolearn_${family}_host.mojo"\n'
            'pixi run mojo build -j 2 --emit shared-lib "$@" -I . -I bindings \\\n    "$entry_file" -o x\n')
        (self.repo / "bindings/entry").mkdir()
        (self.repo / "bindings/entry/entry__mojolearn_fake_host.mojo").write_text(
            "from _mojolearn_fake_host import build_fake\n")
        (self.repo / "bindings/_mojolearn_fake_host.mojo").write_text("from core.g import g\n")
        src, rels = bc.source_digest(self.repo, self.repo / "bindings/build_fake_host.sh", [])
        self.assertEqual(src["scope"], "closure")
        self.assertIn("bindings/_mojolearn_fake_host.mojo", rels)
        self.assertIn("core/g.mojo", rels)
        self.assertIn("bindings/build_host_family.sh", rels)


    def test_release_provisioning_switches_are_not_build_inputs(self):
        # MOJOLEARN_ORCHESTRATOR=1 is required by release.py gate-boxes and
        # boxes-down; it chooses who may rent, never what the compiler emits.
        env = dict(MOJOLEARN_ORCHESTRATOR="1", MOJOLEARN_AMD_PROVIDER="do",
                   MOJOLEARN_RELEASE_NV_BOX="nv", MOJOLEARN_NUMERIC_MODE="identical")
        self.assertEqual(bc.build_env(env), {"MOJOLEARN_NUMERIC_MODE": "identical"})
        self.assertNotIn("MOJOLEARN_ORCHESTRATOR",
                         bc.keyed_env(env, ({"MOJOLEARN_ORCHESTRATOR"}, [])))


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
        self.assertIn("bindings/build_fake.sh", fields_a["inputs"]["files"])
        self.assertEqual(fields_a["key_rule"], bc.KEY_RULE)
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
        self.assertEqual(self.rbuild(a, self.write_map("a2", leg="leg2"), out, hot=hot,
                                    extra={"MOJOLEARN_BINCACHE_REQUIRE_HIT": "1"}), 0)
        row = self.provenance(out)[1]
        self.assertEqual(row[2], "hot-hit")
        self.assertFalse((a / "ran.txt").exists(), "a hot hit must not run the build script")
        self.assertEqual((a / self.OUT).read_bytes(), so)
        rec = [json.loads(l) for l in (out / "placements.jsonl").read_text().splitlines()]
        self.assertEqual(rec[0]["archive_source_commit"], "c" * 40)
        self.assertEqual(rec[0]["files"][0]["sha256"], bc.sha256_bytes(so))

    def test_retained_assembly_never_compiles_on_miss_or_invalid_map(self):
        a = self.tree("a")
        out = self.base / "out"
        for path in (self.write_map("a"), self.base / "absent-map"):
            with self.subTest(path=path):
                self.assertEqual(self.rbuild(a, path, out,
                    extra={"MOJOLEARN_BINCACHE_REQUIRE_HIT": "1"}), 3)
                self.assertFalse((a / "ran.txt").exists())

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



class RealBuildScriptsAreReadable(unittest.TestCase):
    """Every bindings/build*.sh compile line must stay literally readable (2026-10-07: a trailing
    comment containing the words 'mojo build' on the define-snippet source line made script_plan
    refuse every script, which turned the release cross-compile check into a no-op)."""

    def test_every_binding_script_has_a_plan(self):
        import glob
        repo = Path(__file__).resolve().parents[1]
        refused = []
        for s in sorted(glob.glob(str(repo / 'bindings' / 'build*.sh'))):
            name = Path(s).name
            if name in ('build_host_family.sh', 'build_defines.sh'):
                continue
            if bc.script_plan(repo, s, []) is None:
                refused.append(name)
        self.assertEqual(refused, [])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# closure-v2 for bindings/merged_build.sh (2026-11-04)
# ---------------------------------------------------------------------------

MERGED_SCRIPT = """#!/bin/sh
. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/build_defines.sh"
PY=${MOJOLEARN_MERGED_PYTHON:-python3}
plan=$("$PY" tools/merged_bindings.py plan --dir x)
# `mojo build` of a generated root: script_plan cannot read it literally
"$@" build -j "${MOJOLEARN_COMPILE_JOBS:-2}" ${MOJOLEARN_MOJO_BUILD_FLAGS:-} -I . -I bindings "$root" -o x
mkdir -p python/mojolearn/identical
printf 'merged %s\\n' "$*" > python/mojolearn/identical/.tmp.so
mv python/mojolearn/identical/.tmp.so python/mojolearn/identical/_mojolearn_native_a.so
"""


def make_merged_repo(root):
    root = Path(root)
    make_repo(root)
    (root / "tools").mkdir(exist_ok=True)
    shutil.copy(Path(__file__).resolve().parent / "merged_bindings.py", root / "tools" / "merged_bindings.py")
    (root / "bindings" / "merged_build.sh").write_text(MERGED_SCRIPT)
    (root / "bindings" / "build_defines.sh").write_text("MOJOLEARN_BUILD_DEFINE_FLAGS=${MOJOLEARN_BUILD_DEFINES:-}\n")
    (root / "kernels").mkdir()
    for name, imports in (("_mojolearn_a1", "from kernels import ka\nfrom core import g\n"),
                          ("_mojolearn_a2", "from kernels import ka\n"),
                          ("_mojolearn_b1", "from kernels import kb\nfrom core import g\n"),
                          ("_mojolearn_t", "from tokenizer.impl.unicode_table_generated import TABLE\n"
                                           "from tokenizer.impl import unicode_class\n")):
        (root / "bindings" / (name + ".mojo")).write_text(
            imports + "from std.python import PythonObject\n\n\ndef build_%s() -> PythonObject:\n    pass\n" % name)
    (root / "kernels" / "ka.mojo").write_text("fn ka(): pass\n")
    (root / "kernels" / "kb.mojo").write_text("fn kb(): pass\n")
    (root / "tokenizer" / "impl").mkdir(parents=True)
    (root / "tokenizer" / "tools").mkdir()
    (root / "tokenizer" / "__init__.mojo").write_text("")
    (root / "tokenizer" / "impl" / "__init__.mojo").write_text("")
    (root / "tokenizer" / "impl" / "unicode_class.mojo").write_text('comptime UNICODE_VERSION_PINNED = "15"\n')
    (root / "tokenizer" / "tools" / "gen_unicode_categories.py").write_text("print('gen')\n")
    (root / "bindings" / "merged_groups.json").write_text(json.dumps({
        "index_file": "_mojolearn_merged.json",
        "dirs": {"identical": {"path": "python/mojolearn/identical", "optional": {},
                               "groups": {"_mojolearn_native_a": ["_mojolearn_a1", "_mojolearn_a2"],
                                          "_mojolearn_native_b": ["_mojolearn_b1"],
                                          "_mojolearn_native_t": ["_mojolearn_t"]}},
                 "host": {"path": "python/mojolearn/host", "optional": {}, "groups": {}}},
        "darwin_metal1": [], "darwin_metal1_suffix": "_m1"}))
    return root


class MergedClosureTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.repo = make_merged_repo(Path(self.td.name) / "repo").resolve()

    def key(self, group, env=None, plat="linux"):
        env = dict({"MOJOLEARN_GPU_ARCHS": "sm_89"}, **(env or {}))
        return bc.key_fields(self.repo, self.repo / "bindings/merged_build.sh", ["gpu", "--group", group], env,
                             "img", dev_arch="sm_89", os_info=dict(OS), plat=plat)

    def test_a_merged_group_keys_its_own_members_closure(self):
        f = self.key("_mojolearn_native_a")
        self.assertEqual(f["source"]["scope"], "closure")
        self.assertEqual(f["merged"][0]["members"], ["_mojolearn_a1", "_mojolearn_a2"])
        self.assertNotIn("device_arch", f, "a merged GPU build names its target; the box GPU is not an input")
        _, closure = bc.key_fields_closure(self.repo, self.repo / "bindings/merged_build.sh",
                                           ["gpu", "--group", "_mojolearn_native_a"],
                                           {"MOJOLEARN_GPU_ARCHS": "sm_89"}, "img", os_info=dict(OS), plat="linux")
        self.assertIn("kernels/ka.mojo", closure)
        self.assertNotIn("kernels/kb.mojo", closure)
        self.assertNotIn("core/unrelated.mojo", closure)

    def test_an_edit_in_one_group_leaves_the_other_groups_keys(self):
        a, b = bc.key_of(self.key("_mojolearn_native_a")), bc.key_of(self.key("_mojolearn_native_b"))
        (self.repo / "kernels" / "kb.mojo").write_text("fn kb(): return\n")
        self.assertEqual(bc.key_of(self.key("_mojolearn_native_a")), a)
        self.assertNotEqual(bc.key_of(self.key("_mojolearn_native_b")), b)
        # a shared file moves both
        (self.repo / "core" / "g.mojo").write_text("fn g(): return\n")
        self.assertNotEqual(bc.key_of(self.key("_mojolearn_native_a")), a)

    def test_membership_and_root_text_are_keyed_per_group(self):
        a, b = bc.key_of(self.key("_mojolearn_native_a")), bc.key_of(self.key("_mojolearn_native_b"))
        m = json.loads((self.repo / "bindings/merged_groups.json").read_text())
        m["dirs"]["identical"]["groups"]["_mojolearn_native_b"] = ["_mojolearn_b1", "_mojolearn_t"]
        del m["dirs"]["identical"]["groups"]["_mojolearn_native_t"]
        (self.repo / "bindings/merged_groups.json").write_text(json.dumps(m))
        self.assertEqual(bc.key_of(self.key("_mojolearn_native_a")), a, "another group's edit")
        self.assertNotEqual(bc.key_of(self.key("_mojolearn_native_b")), b)
        # A generator edit that does not change the root text moves nothing.
        mb = self.repo / "tools" / "merged_bindings.py"
        mb.write_text(mb.read_text() + "\n# a new subcommand would go here\n")
        self.assertEqual(bc.key_of(self.key("_mojolearn_native_a")), a)

    def test_the_slot_token_and_merged_key_do_not_split_a_group(self):
        base = bc.key_of(self.key("_mojolearn_native_a"))
        moved = self.key("_mojolearn_native_a", {"MOJOLEARN_SLOT_TOKEN": "18ecb160", "MOJOLEARN_MERGED_KEY": "a4",
                                                 "MOJOLEARN_MERGED_PYTHON": "/x/python3",
                                                 "MOJOLEARN_CUDA_CODE_FORMAT": "ptx"})
        self.assertEqual(bc.key_of(moved), base)
        self.assertNotEqual(bc.key_of(self.key("_mojolearn_native_a", {"MOJOLEARN_COMPILE_JOBS": "8"})), base)
        self.assertNotEqual(bc.key_of(self.key("_mojolearn_native_a", {"MOJOLEARN_BUILD_DEFINES": "X=1"})), base)

    def test_a_generated_source_is_keyed_by_its_generator_not_its_bytes(self):
        before = bc.key_of(self.key("_mojolearn_native_t"))
        (self.repo / "tokenizer/impl/unicode_table_generated.mojo").write_text("comptime TABLE = 1\n")
        self.assertEqual(bc.key_of(self.key("_mojolearn_native_t")), before, "the first generator run")
        (self.repo / "tokenizer/tools/gen_unicode_categories.py").write_text("print('gen2')\n")
        self.assertNotEqual(bc.key_of(self.key("_mojolearn_native_t")), before)
        a = bc.key_of(self.key("_mojolearn_native_a"))
        (self.repo / "tokenizer/tools/gen_unicode_categories.py").write_text("print('gen3')\n")
        self.assertEqual(bc.key_of(self.key("_mojolearn_native_a")), a, "only groups that import the table")

    def test_the_generated_table_matches_generated_sources(self):
        import generated_sources as gs
        self.assertEqual(set(bc.GENERATED_SOURCES), set(gs.GENERATED))
        for rel, (directory, pin) in bc.GENERATED_SOURCES.items():
            self.assertEqual(directory, gs.GENERATOR_DIRS[rel])
            self.assertIn(pin, gs.GENERATED[rel])

    def test_an_unknown_group_or_argument_is_unkeyable(self):
        with self.assertRaises(bc.UnkeyableBuild):
            self.key("_mojolearn_native_nope")
        with self.assertRaises(bc.UnkeyableBuild):
            bc.key_fields(self.repo, self.repo / "bindings/merged_build.sh", ["gpu", "--bogus"], {}, "i",
                          dev_arch="x", os_info=dict(OS), plat="linux")

    def test_the_real_tree_keys_every_merged_group_without_a_fallback(self):
        repo = Path(__file__).resolve().parents[1]
        import merged_bindings as mb
        for d in ("identical", "host"):
            for plat in ("linux", "darwin"):
                for stem, _, _ in mb.plan(d, plat, True):
                    env = {"MOJOLEARN_PACKAGE_BYTE_LM": "1"}
                    if d == "identical" and plat == "linux":
                        env["MOJOLEARN_GPU_ARCHS"] = "sm_89"
                    with self.subTest((d, plat, stem)):
                        f = bc.key_fields(repo, repo / "bindings/merged_build.sh",
                                          ["host" if d == "host" else "gpu", "--group", stem], env, "i",
                                          dev_arch="none", os_info=dict(OS), plat=plat)
                        self.assertEqual(f["source"]["scope"], "closure")
                        self.assertLess(f["source"]["files"], len(bc.tree_sources(repo)))


class ExplainSeedHotTests(unittest.TestCase):
    """explain (the plan line a build prints before compiling), seed
    directories of a release-scoped local cache, and the hot-only box route."""

    OUT = "python/mojolearn/identical/_mojolearn_fake.so"

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.base = Path(self.td.name).resolve()
        self.repo = make_repo(self.base / "mojolearn")
        self.orig = (bc.device_arch, bc.os_fields, bc.darwin_toolchain)
        bc.device_arch, bc.os_fields = (lambda: "none"), (lambda: dict(OS))
        bc.darwin_toolchain = lambda: {}
        self.addCleanup(self.restore)

    def restore(self):
        bc.device_arch, bc.os_fields, bc.darwin_toolchain = self.orig

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MOJOLEARN_")}
        env.update(MOJOLEARN_BINCACHE_OUTPUTS=self.OUT, MOJOLEARN_BINCACHE_MAP=str(self.base / "no-map.tsv"),
                   BINCACHE_TEST_MARKER=str(self.repo / "ran.txt"))
        env.update(extra)
        return env

    def explain(self, env):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = bc.cmd_explain([str(self.repo / "bindings/build_fake.sh")], environ=env)
        finally:
            sys.stdout = old
        return rc, buf.getvalue().strip()

    def test_explain_then_seed_a_release_scoped_cache(self):
        old_release = self.base / "release-old"
        env = self.env(MOJOLEARN_BINCACHE_DIR=str(old_release))
        rc, line = self.explain(env)
        self.assertEqual(rc, 0)
        self.assertTrue(line.startswith("compile: no earlier build"), line)
        self.assertEqual(_run_with_buffer(bc.cmd_build)([str(self.repo / "bindings/build_fake.sh")], environ=env), 0)
        self.assertTrue(self.explain(env)[1].startswith("hit (%s)" % old_release))
        (self.repo / self.OUT).unlink()
        (self.repo / "ran.txt").unlink()
        new_release = self.base / "release-new"
        env = self.env(MOJOLEARN_BINCACHE_DIR=str(new_release), MOJOLEARN_BINCACHE_SEED_DIRS=str(old_release))
        self.assertTrue(self.explain(env)[1].startswith("hit (%s)" % old_release))
        self.assertEqual(_run_with_buffer(bc.cmd_build)([str(self.repo / "bindings/build_fake.sh")], environ=env), 0)
        self.assertFalse((self.repo / "ran.txt").exists(), "a seeded hit compiles nothing")
        self.assertEqual(len(list(new_release.glob("*.tar.gz"))), 1, "the seed is copied into the release cache")
        rows = [l.split("\t") for l in (new_release / "provenance" / "provenance.tsv").read_text().splitlines()]
        self.assertEqual(rows[-1][2], "hit")
        # an edit: explain names the changed closure file
        (self.repo / self.OUT).unlink()
        (self.repo / "core/a.mojo").write_text("from layout import TileTensor\nfn f(): return\n")
        rc, line = self.explain(env)
        self.assertTrue(line.startswith("compile: closure changed: core/a.mojo"), line)

    def test_hot_only_route_caches_on_the_box_without_a_map(self):
        hot = self.base / "hot"
        env = self.env(MOJOLEARN_BINCACHE_HOT_DIR=str(hot), MOJOLEARN_BINCACHE_OUT=str(self.base / "out"),
                       MOJOLEARN_BINCACHE_SHELL="sh", MOJOLEARN_BINCACHE_IMAGE="box:nvidia")
        self.assertEqual(_run_with_buffer(bc.cmd_build)([str(self.repo / "bindings/build_fake.sh")], environ=env), 0)
        self.assertEqual(len(list(hot.glob("*.tar.gz"))), 1)
        self.assertTrue(self.explain(env)[1].startswith("hit (%s)" % hot))
        (self.repo / self.OUT).unlink()
        (self.repo / "ran.txt").unlink()
        self.assertEqual(_run_with_buffer(bc.cmd_build)([str(self.repo / "bindings/build_fake.sh")], environ=env), 0)
        self.assertFalse((self.repo / "ran.txt").exists())
        rows = [l.split("\t") for l in (self.base / "out" / "provenance.tsv").read_text().splitlines()]
        self.assertEqual([r[2] for r in rows], ["miss+built-not-uploaded:no-slot", "hot-hit"])

    def test_explain_reports_an_unresolved_import_and_exits_3(self):
        (self.repo / "core/a.mojo").write_text("from nowhere.at_all import thing\n")
        rc, line = self.explain(self.env(MOJOLEARN_BINCACHE_DIR=str(self.base / "c")))
        self.assertEqual(rc, 3)
        self.assertIn("nowhere.at_all", line)


# ---------------------------------------------------------------------------
# one key, one byte content (2026-11-05, finding 3 of the 0.8.66 release state)
# ---------------------------------------------------------------------------

#: The fake build's bytes follow FAKE_BYTES, which no key reads: two builds of
#: one key can then write different bytes, as the hip and cuda legs did.
VARYING_SCRIPT = SCRIPT.replace(
    "printf 'fake-binding %s\\n' \"${MOJOLEARN_NUMERIC_MODE:-identical}\"",
    "printf 'fake-binding %s%s\\n' \"${MOJOLEARN_NUMERIC_MODE:-identical}\" \"${FAKE_BYTES:-}\"")


class DirStore:
    """bincache.R2Store over a directory (the promote transport in tests)."""

    def __init__(self, root):
        self.root = Path(root)
        self.copies = []

    def fetch(self, name, dest):
        p = self.root / name
        if not p.is_file():
            return False
        shutil.copyfile(p, dest)
        return True

    def copy(self, src, dest):
        (self.root / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / src, self.root / dest)
        self.copies.append(dest)

    def delete(self, name):
        (self.root / name).unlink()


class OneKeyOneContentTests(unittest.TestCase):
    OUT = "python/mojolearn/host/_mojolearn_native_series_host.so"

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.base = Path(self.td.name).resolve()
        self.repo = make_repo(self.base / "repo")
        self.store = self.base / "r2"
        self.fields = dict(schema=bc.SCHEMA, script="bindings/merged_build.sh",
                           args=["host", "--group", "_mojolearn_native_series_host"], build_env={},
                           merged=[dict(dir="host", stem="_mojolearn_native_series_host")])
        self.key = bc.key_of(self.fields)

    def archive(self, name, payload, builder="a"):
        so = self.repo / self.OUT
        so.parent.mkdir(parents=True, exist_ok=True)
        so.write_bytes(payload)
        arc = self.base / name
        bc.pack(arc, self.key, self.fields, self.repo, [self.OUT], dict(builder=dict(leg=builder)))
        return arc

    def upload(self, leg, slot, arc):
        """An inbox upload plus the box's record of it (uploads.tsv, keys/)."""
        d = self.base / ("out-" + leg)
        (d / "keys").mkdir(parents=True, exist_ok=True)
        (d / "keys" / (self.key + ".json")).write_text(json.dumps(self.fields))
        dest = "%s/none/img/%s.tar.gz" % (bc.OBJECT_PREFIX, self.key)
        inbox = self.store / bc.INBOX_PREFIX / leg / (slot + ".tar.gz")
        inbox.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(arc, inbox)
        with open(d / "uploads.tsv", "a") as fh:
            fh.write("%s\t%s\t%s\t%s\tsabotage=0\n" % (leg, slot, dest, self.key))
        return d, dest, inbox

    def promote(self, d, store):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = bc.cmd_promote([str(d)], store=store)
        finally:
            sys.stdout = old
        return rc, buf.getvalue()

    def test_content_conflict_compares_files_not_archive_bytes(self):
        a = self.archive("a.tar.gz", b"cuda bytes", builder="sm_89")
        b = self.archive("b.tar.gz", b"cuda bytes", builder="gfx942")   # other manifest extras
        c = self.archive("c.tar.gz", b"hip bytes")
        self.assertNotEqual(bc.sha256_file(a), bc.sha256_file(b))
        self.assertEqual(bc.content_conflict(bc.archive_files(a), bc.archive_files(b)), [])
        diffs = bc.content_conflict(bc.archive_files(a), bc.archive_files(c))
        self.assertEqual(diffs, [(self.OUT, bc.sha256_bytes(b"cuda bytes"), bc.sha256_bytes(b"hip bytes"))])
        self.assertEqual(bc.keep_or_refuse(self.base / "absent.tar.gz", a, self.key)[0], "write")
        self.assertEqual(bc.keep_or_refuse(a, b, self.key)[0], "same")
        verdict, line = bc.keep_or_refuse(a, c, self.key, "here")
        self.assertEqual(verdict, "conflict")
        for part in (self.key, bc.sha256_bytes(b"cuda bytes"), bc.sha256_bytes(b"hip bytes"), self.OUT):
            self.assertIn(part, line)
        junk = self.base / "junk.tar.gz"
        junk.write_bytes(b"not a tarball")
        self.assertEqual(bc.keep_or_refuse(junk, a, self.key)[0], "write")

    def test_promote_first_writes_same_is_a_no_op_and_other_bytes_are_refused(self):
        store = DirStore(self.store)
        d1, dest, inbox1 = self.upload("sm89", "000", self.archive("a.tar.gz", b"cuda bytes", "sm_89"))
        rc, log = self.promote(d1, store)
        self.assertEqual(rc, 0, log)
        self.assertIn("BINCACHE PROMOTED 1 refused_or_failed=0 already_same=0 conflicts=0", log)
        kept = (self.store / dest).read_bytes()
        self.assertFalse(inbox1.exists())

        # a second leg with the SAME files: nothing is written, the inbox object goes
        d2, _, inbox2 = self.upload("sm80", "000", self.archive("b.tar.gz", b"cuda bytes", "sm_80"))
        rc, log = self.promote(d2, store)
        self.assertEqual(rc, 0, log)
        self.assertIn("already_same=1 conflicts=0", log)
        self.assertEqual((self.store / dest).read_bytes(), kept)
        self.assertFalse(inbox2.exists())

        # a third leg with OTHER bytes under the same key: refused loudly, existing kept
        d3, _, inbox3 = self.upload("gfx942", "000", self.archive("c.tar.gz", b"hip bytes", "gfx942"))
        rc, log = self.promote(d3, store)
        self.assertEqual(rc, 4, log)
        self.assertIn("conflicts=1", log)
        self.assertTrue(log.strip().splitlines()[-1].startswith("BINCACHE PROMOTED 0"), log)
        self.assertIn("BINCACHE CONFLICT key=" + self.key, log)
        self.assertIn("existing sha256=" + bc.sha256_bytes(b"cuda bytes"), log)
        self.assertIn("upload sha256=" + bc.sha256_bytes(b"hip bytes"), log)
        self.assertEqual((self.store / dest).read_bytes(), kept)
        self.assertTrue(inbox3.exists(), "the refused upload stays in the inbox for inspection")
        self.assertEqual(store.copies, [dest])

    def test_promote_repairs_an_unreadable_object(self):
        store = DirStore(self.store)
        d, dest, _ = self.upload("sm89", "000", self.archive("a.tar.gz", b"cuda bytes"))
        (self.store / dest).parent.mkdir(parents=True, exist_ok=True)
        (self.store / dest).write_bytes(b"truncated")
        rc, log = self.promote(d, store)
        self.assertEqual(rc, 0, log)
        self.assertEqual(bc.archive_files(self.store / dest), {self.OUT: bc.sha256_bytes(b"cuda bytes")})

    def test_vendor_neutral_names_the_host_builds_only(self):
        self.assertTrue(bc.vendor_neutral(self.fields))
        self.assertTrue(bc.vendor_neutral(dict(script="bindings/build_core_host.sh", args=[])))
        self.assertTrue(bc.vendor_neutral(dict(script="bindings/merged_build.sh", args=["host", "--group", "g"])))
        self.assertFalse(bc.vendor_neutral(dict(script="bindings/merged_build.sh", args=["gpu", "--group", "g"],
                                                merged=[dict(dir="identical", stem="g")])))
        self.assertFalse(bc.vendor_neutral(dict(script="bindings/merged_build.sh", args=["host"],
                                                merged=[dict(dir="host"), dict(dir="identical")])))
        self.assertFalse(bc.vendor_neutral(dict(script="bindings/build_rf.sh", args=[])))


# The hot directory and the local cache never take other bytes under a key.
# Attached to the existing classes (their fixtures) so no parent test runs twice.
def _test_hot_directory_keeps_the_first_bytes_of_a_key(self):
    self.assertNotEqual(VARYING_SCRIPT, SCRIPT)
    hot = self.base / "hot"
    a = self.tree("a")
    (a / "bindings/build_fake.sh").write_text(VARYING_SCRIPT)
    out = self.base / "out"
    self.assertEqual(self.rbuild(a, self.write_map("a"), out, hot=hot), 0)
    key = self.provenance(out)[0][3]
    first = (hot / (key + ".tar.gz")).read_bytes()
    uploads = (out / "uploads.tsv").read_text()
    # the destination exists, so the hot archive is bypassed and the build runs,
    # writing other bytes under the same key
    self.assertEqual(self.rbuild(a, self.write_map("a2", leg="leg2"), out, hot=hot,
                                 extra={"FAKE_BYTES": "-hip"}), 0)
    row = self.provenance(out)[1]
    self.assertEqual(row[3], key)
    self.assertEqual(row[2], "bypass-destination-exists+built-not-uploaded:conflict")
    self.assertEqual((hot / (key + ".tar.gz")).read_bytes(), first)
    self.assertEqual((out / "uploads.tsv").read_text(), uploads, "a conflicting archive is never uploaded")


ReleaseDeclaredTests.test_hot_directory_keeps_the_first_bytes_of_a_key = \
    _test_hot_directory_keeps_the_first_bytes_of_a_key


def _test_local_cache_keeps_the_first_bytes_and_refreshes_equal_ones(self):
    a = self.tree(VARYING_SCRIPT)
    self.assertEqual(self.build(a), 0)
    key = self.rows()[-1][3]
    arc = self.cache / (key + ".tar.gz")
    first = bc.archive_files(arc)
    c = self.tree(VARYING_SCRIPT)
    (c / self.OUT).parent.mkdir(parents=True, exist_ok=True)
    (c / self.OUT).write_bytes(b"old")
    self.assertEqual(self.build(c, extra={"FAKE_BYTES": "-other"}), 0)
    self.assertEqual(self.rows()[-1][3], key)
    self.assertEqual(self.rows()[-1][2], "bypass-destination-exists+built-not-cached:conflict")
    self.assertEqual(bc.archive_files(arc), first)
    d = self.tree(VARYING_SCRIPT)
    (d / self.OUT).parent.mkdir(parents=True, exist_ok=True)
    (d / self.OUT).write_bytes(b"old")
    self.assertEqual(self.build(d), 0)
    self.assertEqual(self.rows()[-1][2], "bypass-destination-exists+built-cached")
    self.assertEqual(bc.archive_files(arc), first)


LocalCacheTests.test_local_cache_keeps_the_first_bytes_and_refreshes_equal_ones = \
    _test_local_cache_keeps_the_first_bytes_and_refreshes_equal_ones


# ---------------------------------------------------------------------------
# the one Mac cache (2026-11-07, lane/post-one-mac-cache)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


class OneMacCacheTests(unittest.TestCase):
    OUT = "python/mojolearn/identical/_mojolearn_fake.so"

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.repo = make_repo(Path(self.td.name) / "repo").resolve()
        (self.repo / self.OUT).parent.mkdir(parents=True)
        (self.repo / self.OUT).write_bytes(b"one-mac-cache-bytes")
        self.fields = fields(self.repo)
        self.key = bc.key_of(self.fields)

    def archive(self, directory, data=None):
        directory.mkdir(parents=True, exist_ok=True)
        if data is not None:
            (self.repo / self.OUT).write_bytes(data)
        arc = directory / (self.key + ".tar.gz")
        bc.pack(arc, self.key, self.fields, self.repo, [self.OUT], {})
        return arc

    def test_the_mac_cache_defaults_to_the_evidence_directory_and_follows_the_override(self):
        self.assertEqual(bc.mac_cache_dir({}), Path.home() / "mojolearn-evidence" / "bincache-mac")
        self.assertEqual(bc.mac_cache_dir({"MOJOLEARN_BINCACHE_DIR": "/x/y"}), Path("/x/y"))

    def test_the_release_compile_settings_are_the_mac_slot_and_byte_lm_ones(self):
        self.assertEqual(bc.MACOS_RELEASE_COMPILE_ENV,
                         {"MOJOLEARN_PACKAGE_BYTE_LM": "1", "MOJOLEARN_COMPILE_JOBS": "1"})
        slot = (REPO_ROOT / "tools" / "mac_slot.py").read_text()
        self.assertIn('MOJOLEARN_COMPILE_JOBS="1"', slot, "tools/mac_slot.py run pins one compiler worker")

    def test_group_environment_matches_build_release_wheel_sh(self):
        base = {"MOJOLEARN_GPU_ARCHS": "metal:3", "MOJOLEARN_HOST_OUTDIR": "/elsewhere", "KEEP": "1"}
        kind, output, env = bc.macos_merged_group(base, "host", "_mojolearn_native_host")
        self.assertEqual((kind, output), ("host", "python/mojolearn/host/_mojolearn_native_host.so"))
        self.assertNotIn("MOJOLEARN_GPU_ARCHS", env)
        self.assertNotIn("MOJOLEARN_HOST_OUTDIR", env)
        self.assertEqual((env["MOJOLEARN_NUMERIC_MODE"], env["MOJOLEARN_TARGET_COLUMN"], env["KEEP"]),
                         ("identical", "cpu", "1"))
        self.assertEqual((env["MOJOLEARN_BINCACHE_OUTPUTS"], env["MOJOLEARN_BINCACHE_SHELL"]), (output, "sh"))
        kind, output, env = bc.macos_merged_group(base, "identical", "_mojolearn_native")
        self.assertEqual((kind, output), ("gpu", "python/mojolearn/identical/_mojolearn_native.so"))
        self.assertEqual((env["MOJOLEARN_NUMERIC_MODE"], env["MOJOLEARN_GPU_ARCHS"]), ("identical", "metal:3"))
        self.assertNotIn("MOJOLEARN_TARGET_COLUMN", env)
        self.assertEqual(bc.macos_merged_group({}, "fast", "_mojolearn_native")[1],
                         "python/mojolearn/_mojolearn_native.so")
        self.assertEqual(base["MOJOLEARN_HOST_OUTDIR"], "/elsewhere", "the caller's environment is not changed")
        # The shell builder spells the same rules; a change there must change the helper.
        wheel = (REPO_ROOT / "packaging" / "macos" / "build_release_wheel.sh").read_text()
        for line in ('host) output="python/mojolearn/host/$stem.so"; kind=host ;;',
                     'fast) output="python/mojolearn/$stem.so"; kind=gpu ;;',
                     '*) output="python/mojolearn/$mode/$stem.so"; kind=gpu ;;',
                     "set -- env -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_HOST_OUTDIR MOJOLEARN_NUMERIC_MODE=identical "
                     "MOJOLEARN_TARGET_COLUMN=cpu",
                     'set -- env MOJOLEARN_NUMERIC_MODE="$mode"',
                     'export MOJOLEARN_BINCACHE_OUTPUTS="$output" MOJOLEARN_BINCACHE_SHELL=sh',
                     '"$@" $runner ./bindings/merged_build.sh "$kind" --group "$stem"'):
            self.assertIn(line, wheel)

    def test_copy_archives_copies_verified_archives_once_and_never_over_a_conflict(self):
        src, dst = Path(self.td.name) / "release", Path(self.td.name) / "mac"
        self.archive(src)
        (src / ("0" * 64 + ".tar.gz")).write_bytes(b"not an archive")
        got = bc.copy_archives(src, dst)
        self.assertEqual((got["copied"], got["present"], got["conflicts"], got["rejected"]),
                         (1, 0, [], ["0" * 64]))
        bc.verify_archive(str(dst / (self.key + ".tar.gz")), self.key, self.fields)
        self.assertFalse((dst / ("0" * 64 + ".tar.gz")).exists())
        again = bc.copy_archives(src, dst)
        self.assertEqual((again["copied"], again["present"]), (0, 1))
        # Same key, other bytes in the destination: reported, never replaced.
        other = Path(self.td.name) / "other"
        self.archive(other, data=b"different-bytes")
        shutil.copyfile(other / (self.key + ".tar.gz"), dst / (self.key + ".tar.gz"))
        before = (dst / (self.key + ".tar.gz")).read_bytes()
        conflict = bc.copy_archives(src, dst)
        self.assertEqual(conflict["conflicts"], [self.key])
        self.assertEqual((dst / (self.key + ".tar.gz")).read_bytes(), before)
        self.assertEqual(bc.copy_archives(src, src)["copied"], 0, "a directory is never copied onto itself")
        self.assertEqual(bc.copy_archives(Path(self.td.name) / "absent", dst)["copied"], 0)

    def test_a_seeded_mac_cache_archive_is_placed_instead_of_compiled(self):
        mac, release_dir = Path(self.td.name) / "mac", Path(self.td.name) / "release"
        self.archive(mac)
        release_dir.mkdir()
        self.assertEqual(bc.seed_archive(release_dir, self.key, self.fields,
                                         {"MOJOLEARN_BINCACHE_SEED_DIRS": str(mac)}), str(mac))
        bc.verify_archive(str(release_dir / (self.key + ".tar.gz")), self.key, self.fields)
