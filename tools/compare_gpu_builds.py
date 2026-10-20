#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Compare two builds of the same commit and architecture, down to the GPU kernel.

WHY THIS EXISTS. On 2026-09-22 the gfx942 set of 0.8.14 built on two boxes matched in
62 of 66 binaries, and nothing in the repository compared two builds of one commit, so
the difference surfaced only from a hand-run sha256 list. A digest says THAT a binding
moved; this says WHICH embedded kernel moved and WHICH instructions, and whether any
floating-point instruction is among them.

What it does, for every .so present in both trees (or for two files):
  1. whole-file sha256; identical files stop here;
  2. every embedded GPU code object (ELF64 with e_machine AMDGPU 224 or CUDA 190) is
     cut out of the host binary and paired by position; code objects that differ are
     split into their FUNC symbols, and each function is compared byte for byte;
  3. for AMDGPU, when llvm-objdump is available (MOJOLEARN_LLVM_OBJDUMP, PATH, or
     Homebrew's /opt/homebrew/opt/llvm/bin), each differing function is disassembled
     for gfx942 and the differing instruction lines are counted and classified:
     FLOAT (a v_* instruction on f16/f32/f64 data or an MFMA) versus everything else.

It never decides that a difference is harmless. A difference with zero FLOAT lines is
still a difference: it is reported, the exit status is 1, and output identity still
has to be proven on the hardware (tools/identity_break.py --diff against the CPU column).

USAGE
    python3 tools/compare_gpu_builds.py A_DIR B_DIR        # two build-set trees
    python3 tools/compare_gpu_builds.py A.so B.so
    python3 tools/compare_gpu_builds.py --show 20 A B      # print up to 20 differing lines per kernel
    python3 tools/compare_gpu_builds.py --self-test        # the check must see a one-byte change
    python3 tools/compare_gpu_builds.py --canonical A B    # judge by the canonical kernel digest, not bytes
    python3 tools/compare_gpu_builds.py --manifest X.so    # one JSON line: file, code-object and kernel digests

CANONICAL DIGESTS (lane/amd-compile-speed, 2026-10-20). A parallel AMD compile can
give different file bytes from the same kernels: code objects in another order,
another symbol or note layout, another host-side object order. Each binary is
classified against the other at four levels, strongest first:
  IDENTICAL      whole-file sha256 equal;
  REORDERED      the same multiset of embedded code objects (sha256 of each, sorted),
                 placed differently; host bytes outside them may differ;
  SAME-KERNELS   every FUNC symbol of every code object has the same bytes under the
                 same name (kernel_digest: sorted name and code sha256); metadata,
                 notes, symbol tables or grouping differ;
  CODE-DIFFERS   at least one kernel's machine code differs, or the kernel name sets
                 differ (HOST-DIFFERS: a binary with no GPU code object differs). That is a compiler difference: record it for Modular
                 (docs/MODULAR_ASKS.md), never canonicalize it away.
Host code outside the code objects is reported (host sections, blanked host digest)
but not judged: a GPU binary's host side is CPU codegen, reproducible at any -j.
--canonical exits 0 when every pair is at least SAME-KERNELS. Nothing here says that a
CODE-DIFFERS build computes different results; output identity is decided on the
hardware (tools/identity_break.py), as before.

Exit 0: every compared binary is byte-identical (with --canonical: SAME-KERNELS or
better). 1: something differs. 2: usage error.
"""
import argparse
import difflib
import json
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

EM_CUDA, EM_AMDGPU = 190, 224
FLOAT_OP = re.compile(r"\bv_[a-z0-9_]*_(f16|f32|f64|bf16)\b|\bv_mfma|\bv_dot")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def code_objects(data):
    """Embedded ELF64 GPU code objects in a host binary, in file order."""
    out, i = [], data.find(b"\x7fELF", 1)
    while i != -1:
        if i + 0x40 <= len(data) and data[i + 4] == 2:
            machine = struct.unpack_from("<H", data, i + 18)[0]
            if machine in (EM_CUDA, EM_AMDGPU):
                shoff = struct.unpack_from("<Q", data, i + 0x28)[0]
                shentsize, shnum = struct.unpack_from("<HH", data, i + 0x3A)
                end = i + shoff + shentsize * shnum
                if shoff and end <= len(data):
                    out.append((machine, data[i:end]))
                    i = data.find(b"\x7fELF", end)
                    continue
        i = data.find(b"\x7fELF", i + 4)
    return out


def functions(blob):
    """{name: bytes} for every sized FUNC symbol of an ELF64 little-endian object."""
    shoff = struct.unpack_from("<Q", blob, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", blob, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", blob, shoff + k * shentsize) for k in range(shnum)]
    funcs = {}
    for sec in secs:
        if sec[1] != 2:  # SHT_SYMTAB
            continue
        strtab = secs[sec[6]]
        for off in range(sec[4], sec[4] + sec[5], 24):
            st_name, st_info, _, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", blob, off)
            if (st_info & 0xF) != 2 or not st_size or st_shndx >= len(secs):
                continue
            end = blob.index(b"\0", strtab[4] + st_name)
            name = blob[strtab[4] + st_name:end].decode(errors="replace")
            text = secs[st_shndx]  # file offset = section offset + (value - section address)
            start = text[4] + st_value - text[3]
            funcs[name] = blob[start:start + st_size]
    return funcs


def code_object_spans(data):
    """(start, end) of each embedded GPU code object, in file order (as code_objects)."""
    spans, i = [], data.find(b"\x7fELF", 1)
    while i != -1:
        if i + 0x40 <= len(data) and data[i + 4] == 2:
            machine = struct.unpack_from("<H", data, i + 18)[0]
            if machine in (EM_CUDA, EM_AMDGPU):
                shoff = struct.unpack_from("<Q", data, i + 0x28)[0]
                shentsize, shnum = struct.unpack_from("<HH", data, i + 0x3A)
                end = i + shoff + shentsize * shnum
                if shoff and end <= len(data):
                    spans.append((i, end))
                    i = data.find(b"\x7fELF", end)
                    continue
        i = data.find(b"\x7fELF", i + 4)
    return spans


def host_sections(data):
    """{section name: sha256} of the outer (host) ELF64; {} when it is not one."""
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return {}
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    if not shoff or shoff + shentsize * shnum > len(data):
        return {}
    secs = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + k * shentsize) for k in range(shnum)]
    names_off = secs[shstrndx][4] if shstrndx < len(secs) else 0
    out = {}
    for k, sec in enumerate(secs):
        end = data.find(b"\0", names_off + sec[0]) if names_off else -1
        name = data[names_off + sec[0]:end].decode(errors="replace") if end != -1 else f"#{k}"
        body = b"" if sec[1] == 8 else data[sec[4]:sec[4] + sec[5]]  # SHT_NOBITS has no bytes
        out[name if name not in out else f"{name}#{k}"] = sha256(body)
    return out


def manifest(data):
    """Digests of one binary at every level the canonical classification uses."""
    cos = code_objects(data)
    co_shas = [sha256(blob) for _, blob in cos]
    kernels = []
    for _, blob in cos:
        try:
            kernels.extend((name, sha256(code)) for name, code in functions(blob).items())
        except (struct.error, ValueError, IndexError):
            kernels.append(("<unparsed code object>", sha256(blob)))
    kernels.sort()
    host = bytearray(data)
    for s, e in code_object_spans(data):
        host[s:e] = bytes(e - s)  # blank the code objects in place, keep offsets
    return dict(sha256=sha256(data), size=len(data), code_objects=len(cos),
                code_object_sha256=co_shas,
                code_object_set=sha256("\n".join(sorted(co_shas)).encode()),
                kernel_names=sorted({n for n, _ in kernels}),
                kernel_digest=sha256("\n".join(f"{n} {h}" for n, h in kernels).encode()),
                host_blanked_sha256=sha256(bytes(host)),
                host_sections=host_sections(data))


def classify(ma, mb):
    """The strongest level at which two manifests agree (see the module docstring)."""
    if ma["sha256"] == mb["sha256"]:
        return "IDENTICAL"
    if not (ma["code_objects"] and mb["code_objects"]):
        return "HOST-DIFFERS"  # no GPU code to canonicalize: a host-code difference stays one
    if ma["code_object_set"] == mb["code_object_set"]:
        return "REORDERED"
    if ma["kernel_digest"] == mb["kernel_digest"]:
        return "SAME-KERNELS"
    return "CODE-DIFFERS"


CANONICAL_OK = ("IDENTICAL", "REORDERED", "SAME-KERNELS")


def objdump():
    for cand in (os.environ.get("MOJOLEARN_LLVM_OBJDUMP"), shutil.which("llvm-objdump"),
                 "/opt/rocm/llvm/bin/llvm-objdump", "/opt/homebrew/opt/llvm/bin/llvm-objdump"):
        if cand and os.path.exists(cand):
            return cand
    return None


def disassemble(tool, blob, name):
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(blob)
        f.flush()
        out = subprocess.run([tool, "-d", "--mcpu=gfx942", "--no-show-raw-insn", "--no-leading-addr",
                              f"--disassemble-symbols={name}", f.name],
                             capture_output=True, text=True).stdout
    lines = []
    for line in out.splitlines():
        line = line.split("//")[0].strip()
        if line and not line.endswith(":") and "file format" not in line and not line.startswith("Disassembly"):
            lines.append(line)
    return lines


def compare_files(a, b, show, tool):
    da, db = open(a, "rb").read(), open(b, "rb").read()
    if da == db:
        return True, []
    mfa, mfb = manifest(da), manifest(db)
    level = classify(mfa, mfb)
    report = [f"DIFFER {os.path.basename(a)}: sha256 {sha256(da)[:16]} vs {sha256(db)[:16]} class={level}"]
    moved_secs = sorted(n for n in set(mfa["host_sections"]) | set(mfb["host_sections"])
                        if mfa["host_sections"].get(n) != mfb["host_sections"].get(n))
    report.append(f"  host sections differing: {', '.join(moved_secs) or 'none'}; host bytes outside "
                  f"code objects {'equal' if mfa['host_blanked_sha256'] == mfb['host_blanked_sha256'] else 'differ'}")
    na, nb = set(mfa["kernel_names"]), set(mfb["kernel_names"])
    if na != nb:
        report.append(f"  kernel names: {len(na - nb)} only in A, {len(nb - na)} only in B")
    if level in ("REORDERED", "SAME-KERNELS"):
        return False, report
    ca, cb = code_objects(da), code_objects(db)
    if len(ca) != len(cb):
        report.append(f"  embedded GPU code objects: {len(ca)} vs {len(cb)} (cannot pair by position)")
        return False, report
    # PAIR BY KERNEL NAMES, NOT POSITION, when the order moved: the code object of B
    # whose FUNC name set equals A's (a parallel compile may emit them in another order).
    def names(blob):
        try:
            return frozenset(functions(blob))
        except (struct.error, ValueError, IndexError):
            return frozenset()
    by_names = {}
    for mach, xb in cb:
        by_names.setdefault(names(xb), []).append((mach, xb))
    paired = []
    for mach, xa in ca:
        cand = by_names.get(names(xa)) or []
        paired.append(((mach, xa), cand.pop(0) if cand and names(xa) else None))
    if any(p is None for _, p in paired):
        paired = list(zip(ca, cb))
    moved = 0
    for k, ((ma, xa), (_, xb)) in enumerate(paired):
        if xa == xb:
            continue
        moved += 1
        vendor = "amdgpu" if ma == EM_AMDGPU else "cuda"
        fa, fb = functions(xa), functions(xb)
        report.append(f"  code object {k} ({vendor}): {len(xa)} vs {len(xb)} bytes")
        diff_funcs = [n for n in fa if fa[n] != fb.get(n)]
        for n in sorted(set(fb) - set(fa)):
            report.append(f"    only in B: {n}")
        if not diff_funcs:
            report.append("    no function bytes differ (metadata, symbols or notes only)")
        for n in diff_funcs:
            if n not in fb:
                report.append(f"    only in A: {n}")
                continue
            nbytes = sum(p != q for p, q in zip(fa[n], fb[n])) + abs(len(fa[n]) - len(fb[n]))
            line = f"    kernel {n}: size {len(fa[n])} vs {len(fb[n])}, {nbytes} bytes differ"
            if ma == EM_AMDGPU and tool:
                la, lb = disassemble(tool, xa, n), disassemble(tool, xb, n)
                changed = [d for d in difflib.unified_diff(la, lb, lineterm="", n=0)
                           if d[:1] in "+-" and not d.startswith(("+++", "---"))]
                floats = [d for d in changed if FLOAT_OP.search(d)]
                line += f"; {len(changed)} instruction lines differ, {len(floats)} FLOAT"
                report.append(line)
                for d in (floats + [c for c in changed if c not in floats])[:show]:
                    report.append(f"      {d}")
            else:
                report.append(line + ("" if ma != EM_AMDGPU else "; no llvm-objdump, instructions not compared"))
    if not moved:
        report.append("  no embedded GPU code object differs (host code or data only)")
    return False, report


def pairs(a, b):
    if os.path.isfile(a) and os.path.isfile(b):
        return [(a, b, os.path.basename(a))]
    out = []
    for root, _, files in os.walk(a):
        for f in sorted(files):
            if f.endswith(".so"):
                rel = os.path.relpath(os.path.join(root, f), a)
                other = os.path.join(b, rel)
                out.append((os.path.join(root, f), other if os.path.exists(other) else None, rel))
    return out


def self_test():
    """The comparison must report a one-byte change inside an embedded kernel."""
    # A minimal AMDGPU ELF64: header, one .text section, one symtab, one strtab.
    names = b"\0.text\0.symtab\0.strtab\0"
    strtab = b"\0kernel_a\0"
    text = bytes(range(64))
    sym = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0) + struct.pack("<IBBHQQ", 1, 0x12, 0, 1, 0, len(text))
    body = text + sym + strtab + names
    off_text, off_sym = 0x40, 0x40 + len(text)
    off_str, off_names = off_sym + len(sym), off_sym + len(sym) + len(strtab)
    shoff = 0x40 + len(body)
    hdr = bytearray(0x40)
    hdr[:4], hdr[4], hdr[5] = b"\x7fELF", 2, 1
    struct.pack_into("<HHIQQQIHHHHHH", hdr, 16, 1, EM_AMDGPU, 1, 0, 0, shoff, 0, 0x40, 0, 0, 64, 5, 4)
    sh = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
          struct.pack("<IIQQQQIIQQ", 1, 1, 6, 0, off_text, len(text), 0, 0, 1, 0),
          struct.pack("<IIQQQQIIQQ", 7, 2, 0, 0, off_sym, len(sym), 3, 1, 8, 24),
          struct.pack("<IIQQQQIIQQ", 15, 3, 0, 0, off_str, len(strtab), 0, 0, 1, 0),
          struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, off_names, len(names), 0, 0, 1, 0)]
    obj = bytes(hdr) + body + b"".join(sh)
    host = b"HOSTCODE" * 8 + obj + b"TRAILER"
    bad = bytearray(host)
    bad[64 + 0x40 + 10] ^= 1  # one bit inside kernel_a's code
    with tempfile.TemporaryDirectory() as d:
        pa, pb, pc = (os.path.join(d, n) for n in ("a.so", "b.so", "c.so"))
        open(pa, "wb").write(host)
        open(pb, "wb").write(bytes(bad))
        open(pc, "wb").write(host)
        same, _ = compare_files(pa, pc, 0, None)
        diff, rep = compare_files(pa, pb, 0, None)
    # Two code objects swapped in place: the bytes differ, the canonical class is REORDERED.
    obj_b = bytes(bad[64:64 + len(obj)])
    swapped = classify(manifest(b"H" * 64 + obj + obj_b), manifest(b"H" * 64 + obj_b + obj))
    rep.append(f"swapped code objects: class={swapped}")
    ok = same and not diff and any("kernel kernel_a" in r for r in rep) and swapped == "REORDERED"
    print("self-test", "PASS" if ok else "FAIL", *rep, sep="\n  ")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--show", type=int, default=8, help="differing instruction lines to print per kernel")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--canonical", action="store_true",
                    help="exit 0 when every pair is SAME-KERNELS or better (module docstring)")
    ap.add_argument("--manifest", metavar="SO", help="print one JSON line of digests for one binary")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.manifest:
        m = manifest(open(args.manifest, "rb").read())
        m["file"] = args.manifest
        print(json.dumps(m, sort_keys=True))
        return 0
    if not (args.a and args.b):
        ap.error("two builds (directories or .so files) are required")
    tool = objdump()
    total = same = canon = 0
    for a, b, rel in pairs(args.a, args.b):
        total += 1
        if b is None:
            print(f"MISSING in B: {rel}")
            continue
        ok, rep = compare_files(a, b, args.show, tool)
        same += ok
        canon += ok or any(f"class={c}" in rep[0] for c in CANONICAL_OK)
        if not ok:
            rep[0] = rep[0].replace(os.path.basename(a), rel, 1)
            print("\n".join(rep))
    print(f"# {same} of {total} binaries byte-identical, {canon} of {total} canonically equal (SAME-KERNELS or better)"
          + ("" if tool else " (no llvm-objdump: AMD instructions not compared)"))
    if args.canonical:
        return 0 if canon == total else 1
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main())
