#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL-tier CUDA machine code: every embedded PTX module replaced, in
place, by a compressed fatbin holding the cubin `ptxas --fmad=false` makes of
it, and audited.

WHY. The Linux wheel's CUDA sets embed PTX text, and on the user's box the
Mojo runtime hands those bytes to `cuModuleLoadDataEx` (libAsyncRTMojoBindings;
its bundled nvPTXCompiler serves only compile-only mode, and the wheel does not
ship libNVPTX.so), so the DRIVER's JIT compiles our IDENTICAL kernels with
whatever ptxas that driver carries, fmad on. packaging/linux/ptx_contract.py
pins the rounding in the PTX (`.rn`, never contracted); this pass removes the
JIT itself: the machine code is made once, at build time, by a pinned ptxas
with contraction off, and the driver only loads it (and inflates it).

HOW. `cuModuleLoadDataEx` takes PTX, a cubin or a fatbin and tells them apart
by content (both binary forms carry their own size), and Mojo passes the
embedded bytes through unchanged. So a binary image written over the PTX
string, NUL-padded to the PTX length, loads, and nothing is relinked: the same
in-place discipline as the `.rn` pass. It must be no longer than the PTX text.
A bare cubin is not: measured on the 0.8.19 sm_89 IDENTICAL set with ptxas
13.0, only 337 of 1,108 distinct modules fit (cubins total 19.1 MB against
25.0 MB of PTX, worst 6.6x). A fatbin with the cubin zstd-compressed
(`fatbinary --compress-all --compress-mode=size`) totals 4.2 MB and fits
1,087 of 1,108. The 21 that do not are tiny (765 to 2,949 bytes of PTX,
where the ELF skeleton dominates); they stay PTX, and the audit admits a
leftover PTX module only when it is JIT-invariant by construction: under
MAX_LEFTOVER bytes, no approximate instruction (APPROX) and no float op
without a rounding modifier, so any ptxas makes the same arithmetic of it.

THE AUDIT refuses an IDENTICAL-tier CUDA binary that embeds a PTX module
outside that rule, no fatbin at all, or a fatbin for another architecture.
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
FATBIN_FLAGS = ("--compress-all", "--compress-mode=size")
#: The largest PTX module the audit lets stay PTX (the biggest measured
#: leftover is 2,949 bytes).
MAX_LEFTOVER = 4096


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


def tool_version(tool):
    r = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()[-1]


def compile_ptx(text, arch, ptxas, fatbinary):
    """PTX bytes -> compressed fatbin bytes holding the fmad=false cubin."""
    with tempfile.TemporaryDirectory() as d:
        src, cub, fat = (os.path.join(d, n) for n in ("m.ptx", "m.cubin", "m.fatbin"))
        with open(src, "wb") as f:
            f.write(text)
        r = subprocess.run([ptxas, f"-arch={arch}", *PTXAS_FLAGS, src, "-o", cub], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ptxas -arch={arch} failed: {r.stderr[-800:]}")
        r = subprocess.run([fatbinary, f"--create={fat}", f"--image3=kind=elf,sm={arch[3:]},file={cub}",
                            *FATBIN_FLAGS], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"fatbinary failed: {r.stderr[-800:]}")
        with open(fat, "rb") as f:
            return f.read()


def jit_invariant(text):
    """A PTX module any ptxas compiles to the same arithmetic: small, no
    approximate instruction, every float mul/add/sub/fma rounding-pinned."""
    return len(text) <= MAX_LEFTOVER and not APPROX.search(text) and not plain_ops(text)


def patch_files(paths, ptxas, fatbinary, jobs=None, arch=None):
    """Replace every PTX module of every file in place; one JSON row per file.

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
        buf = bytearray(data)
        row = {"file": str(p), "modules": 0, "converted": 0, "ptx_bytes": 0, "image_bytes": 0,
               "left_ptx": [], "too_big": []}
        for s, e in modules(data):
            text = data[s:e]
            key = hashlib.sha256(text).hexdigest()
            img = built[key]
            row["modules"] += 1
            row["ptx_bytes"] += e - s
            if len(img) > e - s:
                entry = {"sha256": key[:16], "ptx": e - s, "fatbin": len(img)}
                (row["left_ptx"] if jit_invariant(text) else row["too_big"]).append(entry)
                continue
            buf[s:e] = img + b"\0" * (e - s - len(img))
            row["converted"] += 1
            row["image_bytes"] += len(img)
        if bytes(buf) != data:
            Path(p).write_bytes(bytes(buf))
        rows.append(row)
    return rows


def audit_bytes(data, arch=None):
    """{ptx_modules, jit_invariant_ptx, fatbins, arches, errors}."""
    errors = []
    ptx = modules(data)
    loose = [data[s:e] for s, e in ptx if not jit_invariant(data[s:e])]
    fbs = fatbins(data)
    arches = sorted({a for _, imgs in fbs for _, a in imgs})
    kinds = sorted({k for _, imgs in fbs for k, _ in imgs})
    if loose:
        errors.append(f"{len(loose)} PTX module(s) the driver would JIT that are not JIT-invariant"
                      f" (first {len(loose[0])} bytes, .target {ptx_arch(loose[0])})")
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
    ap.add_argument("--arch", help="the set's architecture; every module must target it")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "patch":
        print(json.dumps({"ptxas": tool_version(a.ptxas), "fatbinary": tool_version(a.fatbinary),
                          "ptxas_flags": PTXAS_FLAGS, "fatbinary_flags": FATBIN_FLAGS}))
        rows = patch_files(a.paths, a.ptxas, a.fatbinary, a.jobs, a.arch)
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
