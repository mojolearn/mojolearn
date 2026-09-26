#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL-tier CUDA machine code: every embedded PTX module replaced by a
compressed fatbin holding the cubin `ptxas --fmad=false` makes of it, and
audited.

WHY. The Linux wheel's CUDA sets embed PTX text, and on the user's box the
Mojo runtime hands those bytes to `cuModuleLoadDataEx` (libAsyncRTMojoBindings;
its bundled nvPTXCompiler serves only compile-only mode, and the wheel does not
ship libNVPTX.so), so the DRIVER's JIT compiles our IDENTICAL kernels with
whatever ptxas that driver carries, fmad on. packaging/linux/ptx_contract.py
pins the rounding in the PTX (`.rn`, never contracted); this pass removes the
JIT itself: the machine code is made once, at build time, by a pinned ptxas
with contraction off, and the driver only loads it (and inflates it).
Measured on an RTX 4090 (2026-09-26, docs/lanes/NVIDIA_CUBIN_RESUME.md): the
0.8.19 release column JIT-compiled 794 modules from PTX; the same column from
fatbins reads the same bits in every cell.

HOW, IN PLACE. `cuModuleLoadDataEx` takes PTX, a cubin or a fatbin and tells
them apart by content (the binary forms carry their own size), and Mojo passes
the embedded bytes through unchanged. So a fatbin written over the PTX string,
NUL-padded to the PTX length, loads, and nothing is relinked: the same in-place
discipline as the `.rn` pass. A bare cubin rarely fits (sm_89: 337 of 1,108
distinct modules; the ELF skeleton and the long Mojo symbol names), a
zstd-compressed fatbin (`fatbinary --compress-all --compress-mode=size`)
almost always does (1,087 of 1,108).

HOW, FOR THE REST. A module whose fatbin is longer than its PTX (tiny kernels,
765 to 2,949 bytes of PTX, where the ELF skeleton dominates) is MOVED: its
fatbin goes into the NUL padding a converted module of the same binary left
behind, and every host-code reference to the PTX string is repointed. Mojo
reaches the string only through RIP-relative `lea` (measured on every IDENTICAL
binding of 0.8.19: 100% of references, no relocations), so a reference is the
7-byte `REX.W 8D modrm(rip) disp32` whose target is the string's address, and
repointing rewrites its disp32. Every candidate is decoded again by objdump
before it is touched. The old PTX is then zeroed, so a reference this missed
fails loudly (an invalid image) instead of quietly JIT-compiling.

A module that can be neither converted in place nor moved stays PTX only if it
is JIT-invariant (jit_invariant: small, no approximate instruction, every float
op rounding-pinned), so whatever ptxas the driver carries makes the same
arithmetic of it; it is reported by name. One such module per set in 0.8.19.

THE AUDIT refuses an IDENTICAL-tier CUDA binary that embeds a PTX module
outside that rule, no fatbin at all, a fatbin for another architecture, or a
fatbin carrying a non-ELF (PTX) image.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptx_contract import APPROX, is_identical_cuda, modules, plain_ops  # noqa: E402

_TARGET = re.compile(rb"^\s*\.target\s+(sm_\w+)", re.M)
_FATBIN_MAGIC = struct.pack("<I", 0xBA55ED50)
#: fatbin entry flag: the image was declared for the architecture-specific
#: target (`sm=90a`), measured with fatbinary 13.0 (sm=90 and sm=90a images of
#: the same cubin differ only in this bit).
_FATBIN_ARCH_SPECIFIC = 0x100000
_FATBIN_KIND_ELF = 2
#: The ptxas contract. -O3 is ptxas's default and its maximum.
PTXAS_FLAGS = ("--fmad=false", "-O3")
#: `--compress-all` with the pinned 12.5 fatbinary is LZ4, which pre-580
#: drivers load; CUDA 13's zstd (`--compress-mode=size`) is smaller but, like
#: a CUDA 13 cubin, raises the driver floor to 580.
FATBIN_FLAGS = tuple(os.environ.get("MOJOLEARN_FATBIN_FLAGS", "--compress-all").split())
#: A moved fatbin starts on this boundary (the fatbin header is u64 fields).
_ALIGN = 16
#: x86-64 `lea r64, [rip + disp32]`: REX.W (+R) 8D, ModRM mod=00 rm=101.
_LEA = re.compile(rb"[\x48\x4c]\x8d(?=[\x05\x0d\x15\x1d\x25\x2d\x35\x3d])", re.S)
_SHF_EXECINSTR = 0x4


def ptx_arch(text):
    m = _TARGET.search(text, 0, 4096)
    if not m:
        raise ValueError("PTX module without a .target sm_* line")
    return m.group(1).decode()


def fatbin_images(data, start):
    """[(kind, arch)] of the fatbin at `start`, or None if there is none there."""
    if data[start:start + 4] != _FATBIN_MAGIC or len(data) < start + 16:
        return None
    _ver, hsize, fsize = struct.unpack_from("<HHQ", data, start + 4)
    off, end, images = start + hsize, start + hsize + fsize, []
    if end > len(data) or hsize != 16:
        return None
    while off + 64 <= end:
        kind, _v, ehsize, payload = struct.unpack_from("<HHIQ", data, off)
        (sm,) = struct.unpack_from("<I", data, off + 28)
        (flags,) = struct.unpack_from("<Q", data, off + 40)
        if ehsize < 64 or payload == 0:
            return None
        images.append((kind, f"sm_{sm}" + ("a" if flags & _FATBIN_ARCH_SPECIFIC else "")))
        off += ehsize + payload
    return images or None


def fatbins(data):
    """(start, [(kind, arch)]) of every fatbin embedded in `data`."""
    out, i = [], data.find(_FATBIN_MAGIC)
    while i >= 0:
        imgs = fatbin_images(data, i)
        if imgs:
            out.append((i, imgs))
        i = data.find(_FATBIN_MAGIC, i + 1)
    return out


# ------------------------------------------------------------------ ELF
def _segments(data):
    """[(file offset, vaddr, filesz)] of the PT_LOAD segments of an ELF64 LE file."""
    (e_phoff,) = struct.unpack_from("<Q", data, 32)
    e_phentsize, e_phnum = struct.unpack_from("<HH", data, 54)
    out = []
    for i in range(e_phnum):
        p_type, _f, p_offset, p_vaddr, _pa, p_filesz = struct.unpack_from("<IIQQQQ", data, e_phoff + i * e_phentsize)
        if p_type == 1:
            out.append((p_offset, p_vaddr, p_filesz))
    return out


def _exec_sections(data):
    """[(file offset, vaddr, size)] of the executable sections."""
    (e_shoff,) = struct.unpack_from("<Q", data, 40)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 58)
    out = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        _name, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from("<IIQQQQ", data, o)
        if sh_type == 1 and sh_flags & _SHF_EXECINSTR:
            out.append((sh_offset, sh_addr, sh_size))
    return out


def _vaddr(segs, off):
    for po, pv, pf in segs:
        if po <= off < po + pf:
            return pv + off - po
    return None


def _segment_end(segs, off):
    for po, _pv, pf in segs:
        if po <= off < po + pf:
            return po + pf
    return None


def lea_refs(data, targets):
    """{target vaddr: [file offset of each `lea r64, [rip+disp32]` that computes it]}."""
    hits = {}
    for so, sa, sz in _exec_sections(data):
        for m in _LEA.finditer(data, so, so + sz):
            i = m.start()
            if i + 7 > so + sz:
                continue
            (disp,) = struct.unpack_from("<i", data, i + 3)
            t = sa + (i - so) + 7 + disp
            if t in targets:
                hits.setdefault(t, []).append(i)
    return hits


def _objdump_says_lea(path, data, off, objdump):
    """Decode 7 bytes at `off` independently: the candidate must be one whole lea."""
    segs = _segments(data)
    va = _vaddr(segs, off)
    r = subprocess.run([objdump, "-d", "--no-show-raw-insn", f"--start-address={va:#x}",
                        f"--stop-address={va + 7:#x}", str(path)], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if re.match(rf"\s*{va:x}:", ln)]
    return len(lines) == 1 and "lea" in lines[0]


# ------------------------------------------------------------------ build
def tool_version(tool):
    r = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()[-1]


def compile_ptx(text, arch, ptxas, fatbinary):
    """PTX bytes -> compressed fatbin bytes holding the fmad=false cubin.

    ptxas and fatbinary run in a scratch directory on FIXED relative names, so
    no path reaches the image (the cubin's .note.nv.tkinfo records the command
    line) and the output depends on the PTX and the tools alone."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "m.ptx"), "wb") as f:
            f.write(text)
        r = subprocess.run([ptxas, f"-arch={arch}", *PTXAS_FLAGS, "m.ptx", "-o", "m.cubin"],
                           capture_output=True, text=True, cwd=d)
        if r.returncode:
            raise RuntimeError(f"ptxas -arch={arch} failed: {r.stderr[-800:]}")
        r = subprocess.run([fatbinary, "--create=m.fatbin", f"--image3=kind=elf,sm={arch[3:]},file=m.cubin",
                            *FATBIN_FLAGS], capture_output=True, text=True, cwd=d)
        if r.returncode:
            raise RuntimeError(f"fatbinary failed: {r.stderr[-800:]}")
        with open(os.path.join(d, "m.fatbin"), "rb") as f:
            return f.read()


#: The largest PTX module the audit lets stay PTX, and only when it is
#: JIT-invariant. The one measured case (0.8.19, both arches): the
#: 2,949-byte integer-only `embedding_checks_embedding_ide*` kernel in
#: _mojolearn_embedding.so, whose fatbin (5.7 KB; ptxas unrolls it to 15.8 KB of
#: SASS) fits no free region of that binary, whose largest PTX span is 3.9 KB.
#: No release lane loads it (the 0.8.19 column passes with the JIT disabled).
MAX_LEFTOVER = 4096


def jit_invariant(text):
    """A PTX module any ptxas compiles to the same arithmetic: small, no
    approximate instruction, every float mul/add/sub/fma rounding-pinned."""
    return len(text) <= MAX_LEFTOVER and not APPROX.search(text) and not plain_ops(text)


def _patch_one(path, data, built, objdump):
    """In place where the fatbin fits, moved where it does not; returns (bytes, row)."""
    buf = bytearray(data)
    segs = _segments(data)
    row = {"file": str(path), "modules": 0, "in_place": 0, "moved": 0, "ptx_bytes": 0, "image_bytes": 0,
           "unplaced": []}
    free, pending = [], []
    for s, e in modules(data):
        text = data[s:e]
        img = built[hashlib.sha256(text).hexdigest()]
        row["modules"] += 1
        row["ptx_bytes"] += e - s
        row["image_bytes"] += len(img)
        if len(img) <= e - s:
            buf[s:e] = img + b"\0" * (e - s - len(img))
            row["in_place"] += 1
            lo = -(-(s + len(img) + 1) // _ALIGN) * _ALIGN
            if e - lo > 0:
                free.append([lo, e])
        else:
            pending.append((s, e, img, text))
    if pending:
        refs = lea_refs(data, {_vaddr(segs, s) for s, _e, _i, _t in pending})
        free.sort(key=lambda r: r[1] - r[0])
        moved_old, moved_new, moved_n = [], [], []
        for s, e, img, text in pending:
            old_va = _vaddr(segs, s)
            sites = refs.get(old_va, [])
            why = None
            if not sites:
                why = "no RIP-relative lea reaches it"
            elif objdump and not all(_objdump_says_lea(path, data, i, objdump) for i in sites):
                why = "a candidate reference is not a whole lea by objdump"
            slot = None
            if why is None:
                for r in free:  # best fit: the smallest region that holds it
                    if r[1] - r[0] >= len(img) and (_segment_end(segs, r[0]) or 0) >= r[0] + (e - s):
                        slot = r
                        break
                if slot is None:
                    why = f"no free region of {len(img)} bytes"
            if why:
                row["unplaced"].append({"sha256": hashlib.sha256(text).hexdigest()[:16], "ptx": e - s,
                                        "fatbin": len(img), "why": why, "jit_invariant": jit_invariant(text)})
                continue
            dest = slot[0]
            buf[dest:dest + len(img)] = img
            slot[0] = -(-(dest + len(img)) // _ALIGN) * _ALIGN
            free.sort(key=lambda r: r[1] - r[0])
            new_va = _vaddr(segs, dest)
            for i in sites:
                insn_va = _vaddr(segs, i)
                struct.pack_into("<i", buf, i + 3, new_va - (insn_va + 7))
            buf[s:e] = b"\0" * (e - s)
            row["moved"] += 1
            moved_old.append(old_va); moved_new.append(new_va); moved_n.append(len(sites))
        # the repointing, read back: nothing reaches a moved module's old
        # start any more, and each new home is reached exactly as often
        back = lea_refs(bytes(buf), set(moved_old) | set(moved_new))
        for old_va, new_va, n in zip(moved_old, moved_new, moved_n):
            if back.get(old_va) or len(back.get(new_va, [])) != n:
                raise AssertionError(f"{path}: the references to the module at {old_va:#x} did not all move")
    return bytes(buf), row


def patch_files(paths, ptxas, fatbinary, jobs=None, arch=None, objdump=None):
    """Convert every PTX module of every file; one JSON row per file.

    Distinct modules are compiled once across all files. `arch`, when given,
    must equal every module's .target (a set is one architecture)."""
    datas = {p: Path(p).read_bytes() for p in paths}
    todo = {}
    for p, data in datas.items():
        for s, e in modules(data):
            text = data[s:e]
            a = ptx_arch(text)
            if arch and a != arch:
                raise ValueError(f"{p}: a module targets {a}, the set is {arch}")
            todo.setdefault(hashlib.sha256(text).hexdigest(), (text, a))
    with cf.ThreadPoolExecutor(jobs or os.cpu_count() or 1) as ex:
        futs = {k: ex.submit(compile_ptx, t, a, ptxas, fatbinary) for k, (t, a) in todo.items()}
        built = {k: f.result() for k, f in futs.items()}
    rows = []
    for p, data in datas.items():
        new, row = _patch_one(p, data, built, objdump)
        if new != data:
            Path(p).write_bytes(new)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ audit
def audit_bytes(data, arch=None):
    """{ptx_modules, jit_invariant_ptx, fatbins, arches, errors}."""
    errors = []
    ptx = modules(data)
    loose = [(s, e) for s, e in ptx if not jit_invariant(data[s:e])]
    fbs = fatbins(data)
    arches = sorted({a for _, imgs in fbs for _, a in imgs})
    kinds = sorted({k for _, imgs in fbs for k, _ in imgs})
    if loose:
        errors.append(f"{len(loose)} PTX module(s) the driver would JIT that are not JIT-invariant"
                      f" (first {loose[0][1] - loose[0][0]} bytes, .target {ptx_arch(data[loose[0][0]:loose[0][1]])})")
    if not fbs:
        errors.append("no fatbin embedded")
    if any(k != _FATBIN_KIND_ELF for k in kinds):
        errors.append(f"a fatbin carries a non-ELF image (kinds {kinds}); PTX inside a fatbin is JIT too")
    if arch and any(a != arch for a in arches):
        errors.append(f"fatbin architecture(s) {arches}, the set is {arch}")
    return {"ptx_modules": len(ptx), "jit_invariant_ptx": len(ptx) - len(loose), "fatbins": len(fbs),
            "arches": arches, "errors": errors}


def audit_tree(root):
    """(errors, rows) for every IDENTICAL-tier CUDA binary under an unpacked wheel root."""
    root = Path(root)
    errors, rows = [], []
    for p in sorted(root.rglob("*.so")):
        rel = p.relative_to(root).as_posix()
        if not is_identical_cuda(rel):
            continue
        r = audit_bytes(p.read_bytes(), arch=Path(rel).parts[2])
        rows.append(dict(r, file=rel))
        errors.extend(f"{rel}: {e}" for e in r["errors"])
    return errors, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("patch", "audit"))
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--ptxas", default=os.environ.get("MOJOLEARN_PTXAS", "ptxas"))
    ap.add_argument("--fatbinary", default=os.environ.get("MOJOLEARN_FATBINARY", "fatbinary"))
    ap.add_argument("--objdump", default=shutil.which("objdump"),
                    help="cross-checks every repointed lea (default: objdump on PATH; required)")
    ap.add_argument("--arch", help="the set's architecture; every module must target it")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "patch":
        if not a.objdump:
            print("REFUSING: no objdump to cross-check the repointed references", file=sys.stderr)
            return 2
        print(json.dumps({"ptxas": tool_version(a.ptxas), "fatbinary": tool_version(a.fatbinary),
                          "ptxas_flags": PTXAS_FLAGS, "fatbinary_flags": FATBIN_FLAGS}))
        rows = patch_files(a.paths, a.ptxas, a.fatbinary, a.jobs, a.arch, a.objdump)
        for r in rows:
            print(json.dumps(r))
        return int(any(not u["jit_invariant"] for r in rows for u in r["unplaced"]))
    rc = 0
    for p in a.paths:
        r = audit_bytes(p.read_bytes(), a.arch)
        print(json.dumps(dict(r, file=str(p))))
        rc |= bool(r["errors"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
