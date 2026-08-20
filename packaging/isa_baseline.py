#!/usr/bin/env python3
"""Prove the shipped binary contains no instruction older Apple silicon lacks.

WHY DISASSEMBLY AND NOT A HEADER READ. arm64 Mach-O `cpusubtype` stays
`ARM64_ALL` whatever `--target-cpu` was, so a binary compiled for apple-m4
LOOKS identical to one compiled for apple-m1 to anything that reads the
header. `otool -l` cannot see this. The only way to know is to look at the
instructions.

WHY IT MATTERS. `mojo build` defaults `--target-cpu` to the chip that ran the
compiler. LLVM does not need to be ASKED to use an enabled feature: with
`+bf16` set it emits `bfdot`/`bfmmla` from ordinary float loops, with `+i8mm`
it emits `smmla`, with `+sme2` it emits SME. On a Mac that lacks the feature
the result is SIGILL at the instruction, inside the extension, with no
diagnostic the user can act on -- and it cannot reproduce on the machine that
built the wheel.

    feature   first Apple silicon    instructions this looks for
    bf16      M2                     bfdot bfmmla bfcvt bfcvtn bfcvtn2
    i8mm      M2                     smmla ummla usmmla
    sme/sme2  M4                     smstart smstop zero {za ...}, mova, addha

Exit 0 when clean, 1 when anything is found, 2 when the disassembler produced
nothing (which is a broken check, not a passing one -- the distinction that
makes this worth writing down).

Usage:  python packaging/isa_baseline.py <binary> [<binary> ...]
"""

import re
import subprocess
import sys

# THE PATTERN MUST MATCH otool's ACTUAL FORMAT, WHICH IS
#
#     <address>\t<mnemonic>\t<operands>
#
# and has NO hex-bytes column. The first version of this file required one,
# so it matched nothing and passed every binary it was given -- including a
# deliberately apple-m4-targeted control. It was "validated" against
# fabricated sample lines written in the same wrong shape, which is the
# uniform-test-data failure this repository keeps paying for: a check whose
# fixture shares the implementation's assumption tests the assumption, not
# the code. `self_test` below now takes its lines from a REAL disassembly.
#
# Mnemonics carry suffixes (`bfdot.4s`), so the boundary is on what FOLLOWS
# the mnemonic, not `\b`.
FORBIDDEN = {
    "bf16 (M2+)": r"^[0-9a-f]+\t(bfdot|bfmmla|bfcvt|bfcvtn|bfcvtn2)([.\t ]|$)",
    "i8mm (M2+)": r"^[0-9a-f]+\t(smmla|ummla|usmmla)([.\t ]|$)",
    "sme/sme2 (M4+)": r"^[0-9a-f]+\t(smstart|smstop|addha|addva)([.\t ]|$)",
}


def disassemble(path):
    out = subprocess.run(
        ["otool", "-tvV", path], capture_output=True, text=True
    )
    if out.returncode != 0:
        return None
    return out.stdout


def scan(path):
    text = disassemble(path)
    if text is None or text.count("\n") < 50:
        print(f"BROKEN {path}: disassembler produced nothing usable")
        return 2
    lines = text.splitlines()
    bad = 0
    for label, pat in FORBIDDEN.items():
        rx = re.compile(pat)
        hits = [l for l in lines if rx.match(l)]
        if hits:
            bad += 1
            print(f"FAIL {path}: {len(hits)} {label} instruction(s)")
            for h in hits[:3]:
                print(f"     {h.strip()}")
        else:
            print(f"  ok {path}: no {label}")
    if bad:
        print(
            f"FAIL {path}: built for a newer chip than the wheel tag admits. "
            "Check bindings/build.sh TARGET_FLAGS."
        )
        return 1
    print(f"PASS {path}: {len(lines)} disassembled lines, M1-safe")
    return 0


def self_test(path):
    """Prove the patterns can match, using REAL lines from a real binary.

    Takes the first disassembled instruction line, substitutes a forbidden
    mnemonic into it, and requires a match. A pattern that cannot match a line
    built from actual otool output is a pattern that will pass everything, and
    that is how this file shipped broken the first time.
    """
    text = disassemble(path)
    if text is None:
        print("SELFTEST BROKEN: cannot disassemble", path)
        return 2
    real = None
    for line in text.splitlines():
        if re.match(r"^[0-9a-f]+\t\w+", line):
            real = line
            break
    if real is None:
        print("SELFTEST BROKEN: no instruction line found in", path)
        return 2
    addr, _, rest = real.partition("\t")
    ok = True
    for label, pat in FORBIDDEN.items():
        mnem = re.search(r"\(([a-z0-9|]+)\)", pat).group(1).split("|")[0]
        _, _, ops = rest.partition("\t")
        planted = f"{addr}\t{mnem}\t{ops}" if ops else f"{addr}\t{mnem}"
        if not re.compile(pat).match(planted):
            print(f"SELFTEST FAIL {label}: cannot match {planted!r}")
            ok = False
    if not ok:
        return 2
    print(f"  selftest: all {len(FORBIDDEN)} patterns match real-format lines")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    rc = self_test(sys.argv[1])
    if rc:
        raise SystemExit(rc)
    raise SystemExit(max(scan(p) for p in sys.argv[1:]))
