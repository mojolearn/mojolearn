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

Exit 0: every compared binary is byte-identical. 1: something differs. 2: usage error.
"""
import argparse
import difflib
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


def objdump():
    for cand in (os.environ.get("MOJOLEARN_LLVM_OBJDUMP"), shutil.which("llvm-objdump"),
                 "/opt/homebrew/opt/llvm/bin/llvm-objdump"):
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
    report = [f"DIFFER {os.path.basename(a)}: sha256 {sha256(da)[:16]} vs {sha256(db)[:16]}"]
    ca, cb = code_objects(da), code_objects(db)
    if len(ca) != len(cb):
        report.append(f"  embedded GPU code objects: {len(ca)} vs {len(cb)} (cannot pair)")
        return False, report
    moved = 0
    for k, ((ma, xa), (_, xb)) in enumerate(zip(ca, cb)):
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
    ok = same and not diff and any("kernel kernel_a" in r for r in rep)
    print("self-test", "PASS" if ok else "FAIL", *rep, sep="\n  ")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--show", type=int, default=8, help="differing instruction lines to print per kernel")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.a and args.b):
        ap.error("two builds (directories or .so files) are required")
    tool = objdump()
    total = same = 0
    for a, b, rel in pairs(args.a, args.b):
        total += 1
        if b is None:
            print(f"MISSING in B: {rel}")
            continue
        ok, rep = compare_files(a, b, args.show, tool)
        same += ok
        if not ok:
            rep[0] = rep[0].replace(os.path.basename(a), rel, 1)
            print("\n".join(rep))
    print(f"# {same} of {total} binaries byte-identical"
          + ("" if tool else " (no llvm-objdump: AMD instructions not compared)"))
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main())
