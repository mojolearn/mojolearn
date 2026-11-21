#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL-tier AMD GPU machine code: no instruction whose result is AMD's own.

WHY (lane fb13-identity-audit, 2026-11-17). The contract is the same bits on
Apple, NVIDIA and AMD and the host column. packaging/linux/ptx_contract.py
refuses NVIDIA's approximate PTX; this is the AMD half. gfx942 has hardware
instructions that are NOT correctly rounded (AMD "CDNA3 Instruction Set
Architecture", VOP1: V_SQRT_F32 / V_RSQ_F32 / V_RCP_F32 are 1 ulp,
V_EXP_F32 / V_LOG_F32 are base-2 approximations, V_SIN_F32 / V_COS_F32 take
revolutions with reduced precision; the F64 RCP/RSQ/SQRT are approximations
too). Mojo's stdlib lowers Float32 sqrt to a bare V_SQRT_F32 and log2/exp/
cos/tanh to V_LOG/V_EXP/V_COS on gfx942. In the 0.8.67-dev AMD wheel
(a74680919a57) exactly the kernels that carried approximate PTX on NVIDIA
carried these: IterativeImputer's |corr| (V_SQRT_F32), the 50 gbdt pointwise
kernels' fold stripe (V_LOG_F32), and the verify sabotage kernels.

WHAT IS ALLOWED. V_RCP_F32 inside LLVM's correctly rounded f32 division
(V_DIV_SCALE / V_DIV_FMAS / V_DIV_FIXUP around it) and inside its integer
division expansions (the 2^32 scale constants 0x4f7ffffe, 0x4f800000,
0x5f7ffffc, or a V_CVT between an integer and f32 next to it: the 24-bit
`expandDivRem24` form, rcp of the divisor, a one-ulp bump, trunc, correct;
seen as `(k + 1) mod 5` in sequence team_kernel): neither leaks the
approximation into a result. V_RCP_IFLAG_F32 is the integer-division seed.
The kernels named in ptx_contract.APPROX_ALLOW (verify sabotage kernels;
kernel names are the same on both vendors) may carry anything.

HOW. Every AMDGPU code object (ELF, e_machine 224) embedded in an IDENTICAL
.so is cut out and disassembled with an LLVM `llvm-objdump` that knows
amdgcn, one worker per object across every core (independent units).

    python3 packaging/linux/hip_contract.py audit <.so> [...]   # exit 1 on a refused kernel
    python3 packaging/linux/hip_contract.py find-disassembler   # exit 1 if none

The build (packaging/linux/build_sets.sh, VENDOR=hip) checks for the
disassembler before it compiles and audits the IDENTICAL set after.
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
import tempfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptx_contract import approx_allowed, vendor_math_build  # noqa: E402

EM_AMDGPU = 224
#: Not correctly rounded on gfx942: refused wherever they appear (outside APPROX_ALLOW).
APPROX_OPS = ("v_sqrt_f32", "v_sqrt_f64", "v_sqrt_f16", "v_rsq_f32", "v_rsq_f64", "v_rsq_f16", "v_exp_f32",
              "v_exp_f16", "v_exp_legacy_f32", "v_log_f32", "v_log_f16", "v_log_legacy_f32", "v_sin_f32",
              "v_sin_f16", "v_cos_f32", "v_cos_f16", "v_rcp_f64", "v_rcp_f16", "v_rcp_legacy_f32")
#: Refused unless it sits inside a correctly rounded division or an integer-division expansion.
RCP = "v_rcp_f32"
_OP = re.compile(r"^\s+(v_[a-z0-9_]+?)(?:_e32|_e64|_sdwa|_dpp)?\s")
_FDIV = ("v_div_scale_f32", "v_div_fmas_f32", "v_div_fixup_f32")
_INTDIV = re.compile(r"0x4f7ffffe|0x4f800000|0x5f7ffffc|0x2f800000|v_cvt_[ui]32_f32|v_cvt_f32_[ui]32")
#: v_rcp_f64 inside LLVM's correctly rounded f64 division (v_div_scale_f64 ... v_div_fixup_f64) is fine.
_FDIV64 = ("v_div_scale_f64", "v_div_fmas_f64", "v_div_fixup_f64")
WINDOW = 10


def code_objects(data):
    """Every AMDGPU ELF embedded in `data`, as bytes (section extents read from its own header)."""
    out, pos = [], 1
    while True:
        i = data.find(b"\x7fELF", pos)
        if i < 0:
            return out
        pos = i + 4
        if i + 64 > len(data) or data[i + 4] != 2 or struct.unpack_from("<H", data, i + 18)[0] != EM_AMDGPU:
            continue
        shoff, = struct.unpack_from("<Q", data, i + 40)
        shent, shnum = struct.unpack_from("<HH", data, i + 58)
        end = i + shoff + shent * shnum
        for k in range(shnum):
            o = i + shoff + k * shent
            if o + 40 > len(data):
                break
            typ, = struct.unpack_from("<I", data, o + 4)
            off, size = struct.unpack_from("<QQ", data, o + 24)
            if typ != 8:  # SHT_NOBITS occupies no file bytes
                end = max(end, i + off + size)
        out.append(data[i:end])
        pos = end


def refused_sites(asm):
    """[(kernel, op)] for every instruction of an `llvm-objdump -d` listing that the contract refuses."""
    lines = asm.split("\n")
    out, fn = [], "<no symbol>"
    for i, line in enumerate(lines):
        if line.endswith(">:") and "<" in line:
            fn = line.rsplit("<", 1)[1][:-2]
            continue
        m = _OP.match(line)
        if not m:
            continue
        op = m.group(1)
        if op in APPROX_OPS:
            near = "\n".join(lines[max(0, i - WINDOW):i + WINDOW])
            if op == "v_rcp_f64" and any(x in near for x in _FDIV64):
                continue
            out.append((fn, op))
        elif op == RCP:
            near = "\n".join(lines[max(0, i - WINDOW):i + WINDOW])
            if any(x in near for x in _FDIV) or _INTDIV.search(near):
                continue
            out.append((fn, op + " (outside a division expansion)"))
    return out


def find_disassembler():
    """An llvm-objdump that lists the amdgcn target, or None."""
    names = [os.environ.get("MOJOLEARN_LLVM_OBJDUMP", "")]
    names += [f"/usr/lib/llvm-{v}/bin/llvm-objdump" for v in range(22, 16, -1)]
    names += [f"llvm-objdump-{v}" for v in range(22, 16, -1)] + ["llvm-objdump", "/opt/homebrew/opt/llvm/bin/llvm-objdump"]
    for name in names:
        path = shutil.which(name) if name else None
        if not path:
            continue
        try:
            version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if re.search(r"^\s+amdgcn\s", version, re.M):
            return path
    return None


def _disassemble(job):
    objdump, mcpu, blob = job
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(blob)
        f.flush()
        r = subprocess.run([objdump, "-d", f"--mcpu={mcpu}", f.name], capture_output=True, text=True)
    if r.returncode:
        return None, r.stderr.strip()[:200]
    return refused_sites(r.stdout), ""


def audit_file(path, objdump, mcpu="gfx942", pool=None):
    """{file, code_objects, refused: ["kernel: op"], allowed: {kernel: reason}, errors: [...]}."""
    blobs = code_objects(Path(path).read_bytes())
    report = {"file": str(path), "code_objects": len(blobs), "refused": [], "allowed": {}, "errors": []}
    if not blobs:
        report["errors"].append("no AMDGPU code object embedded")
        return report
    vendor_math = vendor_math_build()  # the vendor-math measurement row (ptx_contract._VENDOR_MATH_ROW)
    jobs = [(objdump, mcpu, b) for b in blobs]
    results = pool.map(_disassemble, jobs, chunksize=8) if pool else map(_disassemble, jobs)
    for sites, err in results:
        if sites is None:
            report["errors"].append("llvm-objdump failed: " + err)
            continue
        for kernel, op in sites:
            why = approx_allowed(kernel, vendor_math)
            if why:
                report["allowed"][kernel] = why
            elif f"{kernel}: {op}" not in report["refused"]:
                report["refused"].append(f"{kernel}: {op}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("audit", "find-disassembler"))
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--mcpu", default="gfx942")
    a = ap.parse_args()
    objdump = find_disassembler()
    if a.mode == "find-disassembler":
        print(objdump or "no llvm-objdump with the amdgcn target (set MOJOLEARN_LLVM_OBJDUMP)")
        return 0 if objdump else 1
    if not objdump:
        print(json.dumps({"error": "no llvm-objdump with the amdgcn target (set MOJOLEARN_LLVM_OBJDUMP)"}))
        return 1
    rc = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for p in a.paths:
            r = audit_file(p, objdump, a.mcpu, pool)
            print(json.dumps(r))
            rc |= bool(r["refused"]) or bool(r["errors"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
