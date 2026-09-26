#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL-tier CUDA machine code: every embedded PTX module replaced, in
place, by the cubin `ptxas --fmad=false` makes of it, and audited.

WHY. The Linux wheel's CUDA sets embed PTX text, and on the user's box the
Mojo runtime hands those bytes to `cuModuleLoadDataEx` (libAsyncRTMojoBindings;
the bundled nvPTXCompiler is used only in compile-only mode, and the wheel
does not ship libNVPTX.so), so the DRIVER's JIT compiles our IDENTICAL kernels
with whatever ptxas that driver carries, fmad on. packaging/linux/ptx_contract.py
pins the rounding in the PTX (`.rn`, never contracted); this pass removes the
JIT itself: the machine code is made once, at build time, by a pinned ptxas
with contraction off, and the driver only loads it.

HOW. `cuModuleLoadDataEx` takes PTX, a cubin or a fatbin and tells them apart
by content (a cubin is an ELF image and carries its own size), and Mojo passes
the embedded bytes through unchanged. So a cubin written over the PTX string,
NUL-padded to the PTX length, is a valid image and nothing is relinked, the
same in-place discipline as the `.rn` pass. It requires the cubin to be no
longer than the PTX text; a module that does not fit is left as PTX and
reported by name (the audit then refuses the set).

THE AUDIT refuses an IDENTICAL-tier CUDA binary that still embeds any PTX
module, or a cubin whose architecture is not the set's.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptx_contract import is_identical_cuda, modules  # noqa: E402

_TARGET = re.compile(rb"^\s*\.target\s+(sm_\w+)", re.M)
#: A CUDA cubin: 64-bit little-endian ELF with OSABI 0x33 (ELFOSABI_CUDA).
_CUBIN_MAGIC = b"\x7fELF\x02\x01\x01\x33"
_EM_CUDA = 190
#: The ptxas contract. -O3 is ptxas's default and its maximum.
PTXAS_FLAGS = ("--fmad=false", "-O3")


def ptx_arch(text):
    m = _TARGET.search(text, 0, 4096)
    if not m:
        raise ValueError("PTX module without a .target sm_* line")
    return m.group(1).decode()


def cubin_arch(image):
    """'sm_89' / 'sm_90a' from a cubin's ELF header, or None if it is not a cubin.

    CUDA ELF e_flags: the low byte is the SM number for the pre-CUDA-13 ABI
    (EF_CUDA_SM, 0xff), with the architecture-specific 'a' flag at bit 11
    (EF_CUDA_ACCELERATORS_V1 0x800) in e_flags as ptxas 12.x/13.0 writes it for
    sm_90a; ABI version 8 (CUDA 13) moves the SM to bits 8..15 with 'a' at
    bit 3. Both layouts are read; the result is checked against the set.
    """
    if len(image) < 64 or not image.startswith(_CUBIN_MAGIC):
        return None
    (e_machine,) = struct.unpack_from("<H", image, 18)
    if e_machine != _EM_CUDA:
        return None
    abiver = image[8]
    (flags,) = struct.unpack_from("<I", image, 48)
    if abiver >= 8:
        sm, accel = (flags >> 8) & 0xff, bool(flags & 0x8)
    else:
        sm, accel = flags & 0xff, bool(flags & 0x800)
    return f"sm_{sm}" + ("a" if accel else "")


def cubin_size(image):
    """The ELF image's extent: the end of its section header table, or its
    furthest section / program segment, whichever is further."""
    e_phoff, e_shoff = struct.unpack_from("<QQ", image, 32)
    e_phentsize, e_phnum, e_shentsize, e_shnum = struct.unpack_from("<HHHH", image, 54)
    end = max(64, e_shoff + e_shentsize * e_shnum, e_phoff + e_phentsize * e_phnum)
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_type, = struct.unpack_from("<I", image, off + 4)
        sh_offset, sh_size = struct.unpack_from("<QQ", image, off + 24)
        if sh_type != 8:  # SHT_NOBITS occupies no file bytes
            end = max(end, sh_offset + sh_size)
    return end


def cubins(data):
    """(start, arch) of every CUDA cubin embedded in `data`."""
    out, i = [], data.find(_CUBIN_MAGIC)
    while i >= 0:
        arch = cubin_arch(data[i:i + 64])
        if arch:
            out.append((i, arch))
        i = data.find(_CUBIN_MAGIC, i + 1)
    return out


def ptxas_version(ptxas):
    r = subprocess.run([ptxas, "--version"], capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()[-1]


def compile_ptx(text, arch, ptxas):
    """PTX bytes -> cubin bytes, `ptxas -arch=<arch> --fmad=false -O3`."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "m.ptx"), os.path.join(d, "m.cubin")
        with open(src, "wb") as f:
            f.write(text)
        r = subprocess.run([ptxas, f"-arch={arch}", *PTXAS_FLAGS, src, "-o", dst],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ptxas -arch={arch} failed: {r.stderr[-800:]}")
        with open(dst, "rb") as f:
            return f.read()


def patch_files(paths, ptxas, jobs=None, arch=None):
    """Replace every PTX module of every file in place; one JSON row per file.

    Distinct modules are compiled once across all files (a set embeds its
    1,770 distinct modules 3,481 times). `arch`, when given, must equal every
    module's .target (a set is one architecture)."""
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
        futs = {k: ex.submit(compile_ptx, t, a, ptxas) for k, (t, a) in todo.items()}
        built = {k: f.result() for k, f in futs.items()}
    rows = []
    for p, data in datas.items():
        buf = bytearray(data)
        row = {"file": str(p), "modules": 0, "converted": 0, "ptx_bytes": 0, "cubin_bytes": 0, "too_big": []}
        for s, e in modules(data):
            text = data[s:e]
            key = hashlib.sha256(text).hexdigest()
            cub = built[key]
            row["modules"] += 1
            row["ptx_bytes"] += e - s
            if len(cub) > e - s:
                row["too_big"].append({"sha256": key[:16], "ptx": e - s, "cubin": len(cub)})
                continue
            buf[s:e] = cub + b"\0" * (e - s - len(cub))
            row["converted"] += 1
            row["cubin_bytes"] += len(cub)
        if bytes(buf) != data:
            Path(p).write_bytes(bytes(buf))
        rows.append(row)
    return rows


def audit_bytes(data, arch=None):
    """{ptx_modules, cubins, arches, errors}."""
    ptx = modules(data)
    cbs = cubins(data)
    arches = sorted({a for _, a in cbs})
    errors = []
    if ptx:
        errors.append(f"{len(ptx)} PTX module(s) still embedded (the driver would JIT them)")
    if not cbs:
        errors.append("no cubin embedded")
    if arch and any(a != arch for a in arches):
        errors.append(f"cubin architecture(s) {arches}, the set is {arch}")
    return {"ptx_modules": len(ptx), "cubins": len(cbs), "arches": arches, "errors": errors}


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
    ap.add_argument("--arch", help="the set's architecture; every module must target it")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "patch":
        print(json.dumps({"ptxas": ptxas_version(a.ptxas), "flags": PTXAS_FLAGS}))
        rows = patch_files(a.paths, a.ptxas, a.jobs, a.arch)
        for r in rows:
            print(json.dumps(r))
        return int(any(r["too_big"] for r in rows))
    rc = 0
    for p in a.paths:
        r = audit_bytes(p.read_bytes(), a.arch)
        print(json.dumps(dict(r, file=str(p))))
        rc |= bool(r["errors"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
