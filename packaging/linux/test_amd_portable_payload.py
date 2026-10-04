"""Unit checks for the AMD portable payload prototype (pure Python; no GPU).
Run: python3 packaging/linux/test_amd_portable_payload.py"""
import importlib.util, struct, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("amd_portable_payload", ROOT / "packaging/linux/amd_portable_payload.py")
R = _load("amd_portable", ROOT / "python/mojolearn/amd_portable.py")


def _fake_elf(kernel, comment=b"", pad=0):
    """Minimal AMDGPU ELF: a .strtab naming `<kernel>.kd`, a .comment, a
    .shstrtab and the section header table, in that file order."""
    names = b"\0.comment\0.shstrtab\0.strtab\0"
    strtab = kernel.encode() + b".kd\0"
    body = bytearray(64)
    body[0:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", body, 18, 224)
    off_str = len(body); body += strtab + b"\0" * pad
    off_com = len(body); body += comment
    while len(body) % 8:
        body += b"\0"
    off_names = len(body); body += names
    while len(body) % 8:
        body += b"\0"
    shoff = len(body)
    rows = [(0, 0, 0, 0, 1), (1, 1, off_com, len(comment), 1),
            (10, 3, off_names, len(names), 1), (20, 3, off_str, len(strtab), 1)]
    for name, typ, off, sz, al in rows:
        body += struct.pack("<IIQQQQIIQQ", name, typ, 0, 0, off, sz, 0, 0, al, 0)
    struct.pack_into("<Q", body, 0x20, 0)
    struct.pack_into("<Q", body, 0x28, shoff)
    struct.pack_into("<HH", body, 0x36, 56, 0)
    struct.pack_into("<HHH", body, 0x3A, 64, len(rows), 2)
    return bytes(body)


class Inject(unittest.TestCase):
    def test_adds_flag_balanced(self):
        src = "ctx.enqueue_function[kern[HD, TQ], kern[HD, TQ]](a, b)\nctx.enqueue_function[k](x)\n"
        out, n = B.inject_text(src)
        self.assertEqual(n, 2)
        self.assertIn("enqueue_function[kern[HD, TQ], kern[HD, TQ], dump_llvm=True](a, b)", out)
        self.assertIn("enqueue_function[k, dump_llvm=True](x)", out)

    def test_idempotent(self):
        out, _ = B.inject_text("c.enqueue_function[k](x)")
        self.assertEqual(B.inject_text(out), (out, 0))

    def test_unbalanced_refuses(self):
        with self.assertRaises(ValueError):
            B.inject_text("c.enqueue_function[k[1](x)")


class Downgrade(unittest.TestCase):
    def test_rules(self):
        text = ("define void @k(ptr nofree captures(none) %0) {\n"
                "  %2 = getelementptr inbounds nuw [4 x i8], ptr %0, i64 1\n"
                "  %3 = icmp samesign ult i32 1, 2\n  %4 = fmul contract float %1, f0x3F800000\n}\n"
                "attributes #2 = { nocallback nocreateundeforpoison nofree }\n")
        out, counts = B.downgrade(text)
        for gone in ("captures(", "samesign", "nuw", "nocreateundeforpoison", "f0x"):
            self.assertNotIn(gone, out)
        self.assertIn("0x3FF0000000000000", out)  # 1.0f widened to LLVM's double spelling
        self.assertEqual(counts["f0x-float-literal"], 1)
        self.assertEqual(B.fmf_inventory(out), {"fmul:contract": 1})


class EmbeddedIR(unittest.TestCase):
    def test_extract_and_refuse_two_kernels(self):
        one = (b"; ModuleID = 'm'\nsource_filename = \"a\"\ntarget triple = \"amdgcn-amd-amdhsa\"\n"
               b"define dso_local amdgpu_kernel void @kern_a(ptr %0) {\n ret void\n}\n\0")
        self.assertEqual(list(B.embedded_ir(b"xx" + one)), ["kern_a"])
        two = one.replace(b"}\n\0", b"}\ndefine amdgpu_kernel void @kern_b() {\n ret void\n}\n\0")
        with self.assertRaises(ValueError):
            B.embedded_ir(two)


class Runtime(unittest.TestCase):
    def manifest(self, files):
        return dict(schema=R.SCHEMA, code_format=R.CODE_FORMAT, identical_qualified=False,
                    targets=list(R.FAMILY_TARGETS), files=files,
                    bindings={"a.so": dict(kernels={"k": "a.ir/x.bc"})})

    def test_manifest(self):
        files = {"a.so": "1", "a.ir/x.bc": "2"}
        R.validate_manifest(self.manifest(files), dict(files))
        with self.assertRaises(R.AmdPortableError):
            R.validate_manifest(self.manifest(files), {"a.so": "1", "a.ir/x.bc": "3"})
        bad = self.manifest(files)
        bad["identical_qualified"] = True
        with self.assertRaises(R.AmdPortableError):
            R.validate_manifest(bad, dict(files))

    def test_family(self):
        doc = self.manifest({})
        self.assertEqual(R.family_target(doc, "gfx942:sramecc+:xnack-"), "amdgcn-amd-amdhsa--gfx942")
        for gfx in ("gfx1100", "gfx1201", "gfx908"):
            with self.assertRaises(R.AmdPortableError):
                R.family_target(doc, gfx)

    def test_trim_comment_keeps_tables(self):
        elf = _fake_elf("kern_b", comment=b"c" * 100)
        out = R.trim_comment(elf)
        self.assertLess(len(out), len(elf))
        rows = {r["name"]: r for r in R._sections(out)}
        self.assertLessEqual(rows[".comment"]["size"], 8)
        off = rows[".strtab"]["off"]
        self.assertEqual(out[off:off + 10], b"kern_b.kd\0")

    def test_patch_in_place(self):
        old = _fake_elf("kern_a", comment=b"x" * 64)
        host = b"HOST" * 10 + old + b"TAIL"
        self.assertEqual([n for _o, _s, n in R.embedded_objects(host)], [["kern_a"]])
        new = _fake_elf("kern_a", comment=b"y" * 200)  # bigger, fits once its comment is trimmed
        patched = R.patch_in_place(host, {"kern_a": new})
        self.assertEqual(len(patched), len(host))
        self.assertTrue(patched.startswith(b"HOST" * 10) and patched.endswith(b"TAIL"))
        self.assertEqual([n for _o, _s, n in R.embedded_objects(patched)], [["kern_a"]])
        with self.assertRaises(R.AmdPortableError):
            R.patch_in_place(host, {})  # an embedded kernel without code refuses
        with self.assertRaises(R.AmdPortableError):  # too big: moving is not implemented
            R.patch_in_place(host, {"kern_a": _fake_elf("kern_a", pad=4096)})


if __name__ == "__main__":
    unittest.main()
