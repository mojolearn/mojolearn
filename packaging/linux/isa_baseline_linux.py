#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prove a Linux x86-64 set contains no instruction a target CPU may lack.

    python3 packaging/linux/isa_baseline_linux.py <dir-of-.so> [--json out]

THIS IS THE LINUX HALF OF `packaging/isa_baseline.py`, WRITTEN NINE MONTHS
LATE AND ONLY AFTER IT COST A RELEASE. That file's docstring already said the
whole thing, about arm64:

    `mojo build` defaults `--target-cpu` to the chip that ran the compiler.
    LLVM does not need to be ASKED to use an enabled feature. On a Mac that
    lacks the feature the result is SIGILL at the instruction, inside the
    extension, with no diagnostic the user can act on, and IT CANNOT
    REPRODUCE ON THE MACHINE THAT BUILT THE WHEEL.

Every word transfers to x86-64 and nobody transferred it. macOS pinned
`--target-cpu apple-m1` and gated the result; Linux pinned nothing and gated
nothing, so `mojo build` targeted the build box.

WHAT IT COST. mojolearn 0.3.0's Linux wheel, published 2026-08-30, carried
AVX-512 in its HOST code. On an L40 whose host was an AMD EPYC 7773X, a Zen 3
part with no AVX-512, every numeric tier died with SIGILL and a core dump
inside `cluster::estimator::kmeans_fit`, on

    vandps 0x6d7ee(%rip){1to4},%xmm2,%xmm2

`{1to4}` is an EVEX embedded broadcast. There is no `cpuid` dispatch in these
binaries, so it is unconditional: every AMD Zen 1, 2 and 3 host, most Intel
consumer parts and every Xeon before Skylake-SP would have crashed the same
way. It reproduced on none of the four boxes that built the sets, because all
four happened to have AVX-512.

WHAT THIS LOOKS FOR. The baseline is x86-64-v3, which is AVX2, FMA, BMI2 and
SSE4.2, Haswell 2013 and Zen 1 2017 onward. Anything above it is refused:

    zmm registers          the 512-bit register file
    {1toN} broadcasts      EVEX embedded broadcast, any width, including on
                           xmm and ymm operands. THIS IS THE ONE THAT BIT US
                           and it does not mention zmm anywhere.
    k0-k7 as operands      AVX-512 opmask registers
    named AVX-512 opcodes  vpternlog, vpermi2/t2, vpcompress, vpexpand,
                           vpdpbusd/wssd, vdpbf16ps, vp2intersect, kmov/kadd
                           and friends, vgatherpf/vscatterpf

Reads `objdump -d`, which is present on every build box. A file it cannot
disassemble is a FAILURE, not a pass, because an ungated binary is exactly
what this exists to prevent.
"""

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

#: Each entry is (label, compiled regex). A hit is a refusal.
FORBIDDEN = [
    ("zmm register", re.compile(r"%zmm\d")),
    ("EVEX embedded broadcast {1toN}", re.compile(r"\{1to\d+\}")),
    ("opmask register k0-k7", re.compile(r"%k[0-7]\b")),
    ("AVX-512 opcode", re.compile(
        r"\b(v?pternlog\w*|vperm[it]2\w*|vpcompress\w*|vpexpand\w*"
        r"|vpdpbusds?|vpdpwssds?|vdpbf16ps|vcvtne2ps2bf16|vcvtneps2bf16"
        r"|vp2intersect\w*|k(mov|add|and|or|xor|not|shift|test|unpck)\w*"
        r"|vgatherpf\w*|vscatterpf\w*|vrangep[sd]|vreducep[sd]|vfpclassp[sd]"
        r"|vscalefp[sd]|vrcp14\w*|vrsqrt14\w*)\b")),
]


def scan(path):
    """(findings, ninsn, ncpuid) for one ELF. Raises on anything it cannot read.

    `ncpuid` is the count of `cpuid` instructions, which is how a library
    says it dispatches on CPU features at RUNTIME. That distinction decides
    whether AVX-512 is a defect or not, and it is the reason this gate is not
    a blanket ban. Measured 2026-08-30 on the shipped 0.3.0 sm_80 set:

        libAsyncRTMojoBindings.so   cpuid 5   zmm 555   Modular's, dispatched
        libKGENCompilerRTShared.so  cpuid 5   zmm 555   Modular's, dispatched
        _mojolearn.so               cpuid 0   zmm  77   OURS, unguarded

    The MAX runtime carries plenty of AVX-512 and guards every bit of it.
    Ours had none of that guard and executed it unconditionally, which is why
    only ours crashed. A gate that also failed the vendored libraries would
    be a false positive on correct code, and a gate people learn to ignore is
    worse than no gate."""
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"objdump could not disassemble {path}: "
                           f"rc={out.returncode} {out.stderr.strip()[:200]}")
    ncpuid = sum(1 for ln in out.stdout.splitlines()
                 if ":\t" in ln and re.search(r"\bcpuid\b", ln))
    findings, n, fn = {}, 0, "<unknown>"
    for line in out.stdout.splitlines():
        if line.endswith(">:"):
            fn = line.split("<", 1)[-1].rstrip(">:")
            continue
        if ":\t" not in line:
            continue
        n += 1
        for label, rx in FORBIDDEN:
            if rx.search(line):
                d = findings.setdefault(label, {"count": 0, "first": None})
                d["count"] += 1
                if d["first"] is None:
                    d["first"] = {"function": fn[:120], "insn": line.split("\t", 1)[-1].strip()[:160]}
    return findings, n, ncpuid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if not shutil.which("objdump"):
        print("isa_baseline_linux: objdump not found. A set that cannot be "
              "disassembled cannot be cleared; install binutils.", file=sys.stderr)
        return 2
    sos = sorted(pathlib.Path(a.directory).rglob("*.so"))
    if not sos:
        print(f"isa_baseline_linux: no .so under {a.directory}", file=sys.stderr)
        return 2
    report, bad = {}, 0
    for so in sos:
        try:
            findings, n, ncpuid = scan(so)
        except RuntimeError as exc:
            print(f"  FAIL  {so.name}: {exc}")
            report[str(so)] = {"error": str(exc)}
            bad += 1
            continue
        rel = str(so.relative_to(a.directory))
        # A VENDORED LIBRARY THAT DISPATCHES IS NOT A DEFECT. Only our own
        # extensions are held to the baseline unconditionally; anything under
        # .libs/ is Modular's and is judged on whether it GUARDS what it uses.
        vendored = rel.startswith(".libs/") or "/.libs/" in rel
        report[rel] = {"instructions": n, "cpuid": ncpuid,
                       "vendored": vendored, "findings": findings}
        if findings and vendored and ncpuid > 0:
            print(f"  ok    {rel}  (vendored, {sum(d['count'] for d in findings.values())} "
                  f"above-baseline instructions GUARDED by {ncpuid} cpuid dispatch sites)")
            continue
        if findings:
            bad += 1
            print(f"  FAIL  {rel}  ({n} instructions)")
            for label, d in sorted(findings.items()):
                f = d["first"]
                print(f"          {label}: {d['count']} occurrence(s)")
                print(f"            first at {f['insn']}")
                print(f"            in {f['function'][:100]}")
        else:
            print(f"  ok    {rel}  ({n} instructions, baseline clean)")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"baseline": "x86-64-v3", "files": len(sos), "failing": bad,
             "report": report}, indent=2, sort_keys=True))
    print(f"\nisa_baseline_linux: {len(sos)} binaries, {bad} above the "
          f"x86-64-v3 baseline")
    if bad:
        print("REFUSING. These would SIGILL on any host without the feature, "
              "with no diagnostic, and would not reproduce on this build box.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
