#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL-tier CUDA PTX, made contraction-proof in place, and audited.

WHY. The Linux wheel's CUDA sets ship PTX only. The driver JIT compiles it
with fmad on, and ptxas (the JIT included) may fuse a plain `mul.f32` feeding
a plain `add.f32`/`sub.f32` into one FFMA, which rounds once where the source
rounded twice. PTX gives the escape by name: an instruction that carries an
explicit rounding modifier (`mul.rn.f32`) is never contracted. 0.8.19's
IDENTICAL PTX had 920 such contractable pairs (all `mul.f32 x, 0f33800000`
in gbdt `hist2_dither`, exact by construction, so bit-neutral), and rewriting
every plain mul/add/sub (f32, f64) to `.rn` made the fmad=true SASS equal the
fmad=false SASS in all 100 affected modules (ptxas 13.0).

WHY NOT AT THE SOURCE. `mojo build --fp-mode contract=off` does emit `.rn`
for every plain op, but it ALSO stops LLVM from fusing `a*b + c` across
statements, which the default (`contract=fast`) does on NVIDIA and AMD before
any PTX exists (measured 2026-09-25 on a probe kernel: sm_89/sm_90a emit
`fma.rn.f32`, gfx942 `v_fmac_f32`, under the default; `mul.rn`+`add.rn` and
`v_mul`+`v_add` under contract=off). Turning it off changes IDENTICAL bits on
every column, so it is a release-wide decision, not this pass.

THE REWRITE is length-preserving, so it edits the PTX string inside the
shared object without relinking. LLVM's NVPTX printer spells these lines

    \\tmul.f32 \\t%r4, %r2, %r3;

and the rewrite spends exactly the three whitespace bytes it adds:

    mul.rn.f32 %r4,%r2, %r3;

(leading tab dropped, " \\t" to " ", the first ", " to ","). A line in any
other shape is left alone and the audit then refuses it by name.

THE AUDIT refuses any IDENTICAL-tier CUDA PTX that still contains a mul, add,
sub, fma or mad on a floating type with no rounding modifier.
"""
import argparse
import json
from pathlib import Path
import re
import sys

#: A PTX module in a binary: `.version X.Y` up to the NUL that ends the string.
_VERSION = re.compile(rb"\.version \d+\.\d+")
_TARGET = re.compile(rb"^\s*\.target\s+sm_", re.M)
#: A float mul/add/sub/fma/mad with NO rounding modifier (.rn/.rz/.rm/.rp).
#: `mul.rn.ftz.f32` does not match: the modifier group admits only .ftz/.sat.
PLAIN = re.compile(
    rb"^[ \t]*(?:@!?%\w+[ \t]+)?(add|sub|mul|fma|mad)((?:\.(?:ftz|sat))*)"
    rb"\.(f16x2|f16|bf16x2|bf16|f32|f64)(?=[ \t])", re.M)
#: The exact line shape the rewrite knows how to lengthen for free.
_REWRITE = re.compile(rb"(?<=\n)\t(add|sub|mul)\.(f32|f64) \t([^,\n]+), ")
#: Approximate transcendental instructions (recorded, not refused).
APPROX = re.compile(rb"\b((?:sqrt|rsqrt|rcp|ex2|lg2|sin|cos|tanh|div)\.(?:approx|full)[.a-z0-9]*)")


def is_identical_cuda(relative):
    """`mojolearn/cuda/<arch>/identical/<name>.so`: the tier this pass owns."""
    parts = Path(relative).parts
    return (len(parts) == 5 and parts[0] == "mojolearn" and parts[1] == "cuda"
            and parts[3] == "identical" and parts[4].endswith(".so"))


def modules(data):
    """(start, end) of every PTX module string in `data`."""
    spans, last = [], -1
    for m in _VERSION.finditer(data):
        start = data.rfind(b"\0", 0, m.start()) + 1
        if start <= last:
            continue
        end = data.find(b"\0", m.start())
        end = len(data) if end < 0 else end
        if _TARGET.search(data, start, min(end, start + 4096)):
            spans.append((start, end))
            last = end
    return spans


def rewrite_ptx(text):
    """PTX text -> (same-length text with plain f32/f64 mul/add/sub made .rn, count)."""
    out, n = _REWRITE.subn(rb"\1.rn.\2 \3,", text)
    if len(out) != len(text):
        raise AssertionError("the .rn rewrite changed the PTX length")
    return out, n


def plain_ops(text):
    """Every float mul/add/sub/fma/mad line with no rounding modifier."""
    lines = []
    for m in PLAIN.finditer(text):
        end = text.find(b"\n", m.start())
        lines.append(text[m.start():end if end >= 0 else len(text)].strip().decode("ascii", "replace"))
    return lines


def patch_bytes(data):
    """A binary's bytes -> (bytes with every PTX module rewritten in place, modules, rewrites)."""
    buf = bytearray(data)
    spans = modules(data)
    total = 0
    for start, end in spans:
        new, n = rewrite_ptx(bytes(buf[start:end]))
        buf[start:end] = new
        total += n
    return bytes(buf), len(spans), total


def audit_bytes(data):
    """{modules, plain: [first offending lines], plain_count, approx: {op: count}}."""
    report = {"modules": 0, "plain_count": 0, "plain": [], "approx": {}}
    for start, end in modules(data):
        text = data[start:end]
        report["modules"] += 1
        bad = plain_ops(text)
        report["plain_count"] += len(bad)
        report["plain"].extend(bad[:max(0, 5 - len(report["plain"]))])
        for op in APPROX.findall(text):
            key = op.decode()
            report["approx"][key] = report["approx"].get(key, 0) + 1
    return report


def patch_file(path):
    data = Path(path).read_bytes()
    new, n_mod, n_rw = patch_bytes(data)
    if new != data:
        Path(path).write_bytes(new)
    return {"file": str(path), "modules": n_mod, "rewritten": n_rw}


def patch_tree(root):
    """Rewrite every IDENTICAL-tier CUDA binary under an unpacked wheel root."""
    root = Path(root)
    return [dict(patch_file(p), file=p.relative_to(root).as_posix())
            for p in sorted(root.rglob("*.so")) if is_identical_cuda(p.relative_to(root).as_posix())]


def audit_tree(root):
    """(errors, per-binary reports) for every IDENTICAL-tier CUDA binary under `root`."""
    root = Path(root)
    errors, rows = [], []
    for p in sorted(root.rglob("*.so")):
        rel = p.relative_to(root).as_posix()
        if not is_identical_cuda(rel):
            continue
        r = audit_bytes(p.read_bytes())
        rows.append({"file": rel, "ptx_modules": r["modules"], "plain_float_ops": r["plain_count"],
                     "approx": r["approx"]})
        if r["plain_count"]:
            errors.append(f"{rel}: {r['plain_count']} float mul/add/sub/fma without a rounding modifier"
                          f" in IDENTICAL PTX (the driver JIT may contract them), first: {r['plain']}")
    return errors, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("patch", "audit"))
    ap.add_argument("paths", nargs="+", type=Path, help=".so files (patched/audited whatever their path)")
    a = ap.parse_args()
    rc = 0
    for p in a.paths:
        if a.mode == "patch":
            print(json.dumps(patch_file(p)))
        else:
            r = audit_bytes(p.read_bytes())
            print(json.dumps({"file": str(p), "ptx_modules": r["modules"], "plain_float_ops": r["plain_count"],
                              "first": r["plain"], "approx": r["approx"]}))
            rc |= bool(r["plain_count"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
