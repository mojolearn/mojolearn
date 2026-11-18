#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL-tier Apple GPU code (AIR): fast math off, no approximate math intrinsic.

WHY (lane fb13-identity-audit, 2026-11-17). The Apple half of the compiled-
kernel audits (NVIDIA: packaging/linux/ptx_contract.py, AMD: packaging/linux/
hip_contract.py). The Metal compiler runs in the user's driver, on the AIR
(LLVM bitcode) the wheel embeds, so what decides Apple's arithmetic is in
that bitcode and can be read from it:

  * `!air.compile_options` must carry `air.compile.fast_math_disable`. With
    fast math on, the driver may pick `fast_` variants of every math call and
    reassociate (Metal Shading Language spec, "Math Functions": precise vs fast
    math). Every module of the 0.8.67-dev Apple build carries it, beside
    `air.compile.denorms_disable` (Apple flushes subnormals; the identical
    seams model that flush on every column).
  * No instruction may carry a fast-math flag other than `contract` (`fast`,
    `afn`, `arcp`, `reassoc`, `nnan`, `ninf`, `nsz` would license the driver
    to change results). `contract` is the build's contraction mode, the same
    on every backend (IDENTITY_PATHS row 9; tools/contraction_census.py).
  * No call to an `air.fast_*` intrinsic, and none to a precise transcendental
    (`air.exp*`, `air.log*`, `air.sin`, `air.cos`, `air.tan*`, `air.pow*`,
    `air.rsqrt`, ...): the MSL spec states those to a few ulp, not correctly
    rounded, so their bits are Apple's own. IDENTICAL routes through
    checks/numerics.mojo. `air.sqrt` is allowed: precise sqrt measured
    correctly rounded on normals on Apple silicon (checks/ieee_arith_check.mojo
    arm 2, M4, 0 mismatches; IDENTITY_PATHS row 10), with subnormals flushed
    exactly as `portable_sqrtf` returns them. `air.fmin`/`air.fmax` are exact.

The kernels named in packaging/linux/ptx_contract.APPROX_ALLOW (verify
sabotage kernels) may carry anything. Every AIR module is disassembled with
an LLVM `llvm-dis`, one worker per module across every core.

    python3 packaging/macos/air_contract.py audit <.so> [...]   # exit 1 on a refusal
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux"))
from ptx_contract import approx_allowed  # noqa: E402

#: The LLVM bitcode wrapper header (0x0B17C0DE): magic, version, offset, size, cputype.
WRAPPER = b"\xde\xc0\x17\x0b"
REQUIRED_OPTION = "air.compile.fast_math_disable"
_FLAGS = re.compile(r"=\s*(?:fadd|fsub|fmul|fdiv|frem|fneg|call|select|phi)((?:\s+(?:fast|afn|arcp|contract|reassoc|"
                    r"nnan|ninf|nsz))+)\b")
_CALL = re.compile(r"\bcall\b[^@\n]*@(air\.[\w.]+)\(")
_APPROX_CALL = re.compile(r"^air\.(?:fast_\w+|exp\w*|log\w*|sin\w*|cos\w*|tan\w*|pow\w*|rsqrt|cbrt|erf\w*|"
                          r"atan\w*|asin\w*|acos\w*|hypot|divide|recip\w*)\.")
_DEFINE = re.compile(r"^define\b[^@\n]*@([\w$.]+)\(", re.M)
_OPTIONS = re.compile(r'!"(air\.compile\.\w+)"')


def modules(data):
    """Every wrapped bitcode module embedded in `data` (the metallibs' AIR)."""
    out, pos = [], 0
    while True:
        i = data.find(WRAPPER, pos)
        if i < 0 or i + 20 > len(data):
            return out
        _, _, off, size, _ = struct.unpack_from("<IIIII", data, i)
        out.append(data[i:i + off + size])
        pos = i + max(off + size, 4)


def audit_text(ll):
    """(refused: [(kernel, what)], options) for one disassembled AIR module."""
    options = sorted(set(_OPTIONS.findall(ll)))
    refused = []
    kernels = _DEFINE.findall(ll) or ["<no definition>"]
    kernel = kernels[0]
    if REQUIRED_OPTION not in options:
        refused.append((kernel, "module lacks " + REQUIRED_OPTION))
    starts = [(m.start(), m.group(1)) for m in _DEFINE.finditer(ll)]

    def owner(pos):
        name = kernel
        for start, n in starts:
            if start > pos:
                break
            name = n
        return name

    for m in _FLAGS.finditer(ll):
        bad = sorted(set(m.group(1).split()) - {"contract"})
        if bad:
            refused.append((owner(m.start()), "fast-math flag " + " ".join(bad)))
    for m in _CALL.finditer(ll):
        if _APPROX_CALL.match(m.group(1)):
            refused.append((owner(m.start()), m.group(1)))
    return refused, options


def find_llvm_dis():
    for name in [os.environ.get("MOJOLEARN_LLVM_DIS", ""), "/opt/homebrew/opt/llvm/bin/llvm-dis",
                 "/usr/local/opt/llvm/bin/llvm-dis", "llvm-dis"]:
        path = shutil.which(name) if name else None
        if path:
            return path
    return None


def _disassemble(job):
    tool, blob = job
    r = subprocess.run([tool, "-", "-o", "-"], input=blob, capture_output=True)
    if r.returncode:
        return None, r.stderr.decode("utf8", "replace").strip()[:200]
    return audit_text(r.stdout.decode("utf8", "replace")), ""


def audit_file(path, tool, pool=None):
    blobs = modules(Path(path).read_bytes())
    report = {"file": str(path), "air_modules": len(blobs), "refused": [], "allowed": {}, "errors": []}
    if not blobs:
        report["errors"].append("no AIR module embedded")
        return report
    jobs = [(tool, b) for b in blobs]
    for result, err in (pool.map(_disassemble, jobs, chunksize=8) if pool else map(_disassemble, jobs)):
        if result is None:
            report["errors"].append("llvm-dis failed: " + err)
            continue
        for kernel, what in result[0]:
            why = approx_allowed(kernel)
            if why:
                report["allowed"][kernel] = why
            elif f"{kernel}: {what}" not in report["refused"]:
                report["refused"].append(f"{kernel}: {what}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("audit",))
    ap.add_argument("paths", nargs="+", type=Path)
    a = ap.parse_args()
    tool = find_llvm_dis()
    if not tool:
        print(json.dumps({"error": "no llvm-dis (set MOJOLEARN_LLVM_DIS; `brew install llvm`)"}))
        return 1
    rc = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for p in a.paths:
            r = audit_file(p, tool, pool)
            print(json.dumps(r))
            rc |= bool(r["refused"]) or bool(r["errors"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
