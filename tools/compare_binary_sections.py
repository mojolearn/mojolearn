#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two built binaries by SECTION, not by whole-file digest and not by segment.

WHY THIS EXISTS. `tools/rf_nondeterminism/rf_claimfix_ab_body.template.sh` guarded A/B arm
independence with a whole-file sha256. `bindings/build_rf.sh` bakes a fresh `mktemp -d`
path into the output's install name, so two builds of BYTE-IDENTICAL source always differ
in roughly 54 bytes of install name, LC_UUID and code-signature slot. That guard could
never report its VOID branch. It was a verification that cannot fail, and a leg run under
it would have compared one program with itself and called the result two arms.

WHY SECTIONS AND NOT SEGMENTS. The `__TEXT` SEGMENT starts at file offset 0 and contains
the Mach-O header and the load commands, which is where the install name and the UUID
live. A segment-level digest therefore reports a false DIFFER, which is the same trap one
level up. Sections carry code and constants and nothing build-path dependent.

Metal AOT matters here: the embedded AIR for a Metal build lands in `__TEXT,__const`, so
an ordering instruction that reaches the GPU shows up as a change in a CONST section and
not in `__TEXT,__text`. Both are compared.

USAGE
    python3 tools/compare_binary_sections.py A.so B.so
    python3 tools/compare_binary_sections.py --expect same A.so B.so
    python3 tools/compare_binary_sections.py --expect differ A.so B.so

Exit status is 0 when the observation matches `--expect`, 1 when it does not. With no
`--expect` it is 0 when the sections are identical and 1 when they differ, and it always
PRINTS which sections moved rather than a count.
"""
import argparse
import hashlib
import re
import subprocess
import sys

# Load-command-bearing and signature-bearing regions are excluded by construction: we only
# ever read named SECTIONS. __LINKEDIT carries the code signature and has no sections.
SKIP_SEGMENTS = {"__LINKEDIT"}


def is_elf(path):
    with open(path, "rb") as fh:
        return fh.read(4) == b"\x7fELF"


def elf_sections(path):
    """Return {(",ELF"), section): (size, sha256)} for an ELF file, via readelf.

    The Linux half of this tool. `readelf -S -W` lists the sections and
    `readelf -x <name>` hex dumps one. Sections carrying build paths and build
    ids are skipped by name for the same reason __LINKEDIT is on Mach-O.
    """
    skip = {".comment", ".note.gnu.build-id", ".gnu_debuglink"}
    listing = subprocess.run(
        ["readelf", "-S", "-W", path], capture_output=True, text=True, check=True
    ).stdout
    names = []
    for line in listing.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+([0-9a-fA-F]+)", line)
        if m and m.group(2) != "NULL" and m.group(1) not in skip:
            names.append((m.group(1), int(m.group(3), 16)))
    result = {}
    for name, size in names:
        dump = subprocess.run(
            ["readelf", "-x", name, path], capture_output=True, text=True
        ).stdout
        body = []
        for line in dump.splitlines():
            m = re.match(r"^\s+0x[0-9a-fA-F]+\s+((?:[0-9a-fA-F]{2,8}\s+){1,4})", line)
            if m:
                body.append(m.group(1).strip())
        blob = " ".join(body).encode()
        result[("ELF", name)] = (size, hashlib.sha256(blob).hexdigest())
    return result


def sections(path):
    """Return {(segment, section): (size, sha256)}, Mach-O or ELF."""
    if is_elf(path):
        return elf_sections(path)
    out = subprocess.run(
        ["otool", "-l", path], capture_output=True, text=True, check=True
    ).stdout
    # Parse `otool -l` section records. Each has `sectname`, then `segname`, then `size`.
    found = []
    cur = {}
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"^sectname\s+(\S+)$", line)
        if m:
            cur = {"sect": m.group(1)}
            continue
        m = re.match(r"^segname\s+(\S+)$", line)
        if m and "sect" in cur:
            cur["seg"] = m.group(1)
            continue
        m = re.match(r"^size\s+(0x[0-9a-fA-F]+|\d+)$", line)
        if m and "seg" in cur and "sect" in cur:
            cur["size"] = int(m.group(1), 0)
            found.append((cur["seg"], cur["sect"], cur["size"]))
            cur = {}
    result = {}
    for seg, sect, size in found:
        if seg in SKIP_SEGMENTS:
            continue
        dump = subprocess.run(
            ["otool", "-s", seg, sect, path], capture_output=True, text=True
        ).stdout
        # Strip the leading "path:\nContents of (SEG,SECT) section" banner and the
        # per-line addresses, keeping only the hex payload.
        body = []
        for line in dump.splitlines():
            m = re.match(r"^[0-9a-fA-F]{8,16}\s+(.*)$", line.strip())
            if m:
                body.append(m.group(1))
        blob = " ".join(body).encode()
        result[(seg, sect)] = (size, hashlib.sha256(blob).hexdigest())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--expect", choices=("same", "differ"), default=None)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    la = args.label_a or args.a
    lb = args.label_b or args.b

    sa, sb = sections(args.a), sections(args.b)
    if not sa or not sb:
        print("REFUSED: no sections parsed; otool gave nothing usable")
        return 2

    keys = sorted(set(sa) | set(sb))
    moved = []
    for k in keys:
        va, vb = sa.get(k), sb.get(k)
        seg, sect = k
        if va is None or vb is None:
            moved.append(k)
            print(f"  ONLY-IN-ONE {seg},{sect}  {la}={va}  {lb}={vb}")
        elif va != vb:
            moved.append(k)
            print(f"  DIFFER      {seg},{sect}  size {va[0]} -> {vb[0]}  "
                  f"sha {va[1][:12]} -> {vb[1][:12]}")

    print(f"{len(keys)} sections compared, {len(moved)} moved, "
          f"{la} against {lb}")
    if not moved:
        print("  ALL SECTIONS IDENTICAL: these two builds are the SAME PROGRAM.")

    observed = "differ" if moved else "same"
    if args.expect:
        ok = observed == args.expect
        print(f"EXPECT {args.expect}  OBSERVED {observed}  "
              f"{'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    return 1 if moved else 0


if __name__ == "__main__":
    raise SystemExit(main())
