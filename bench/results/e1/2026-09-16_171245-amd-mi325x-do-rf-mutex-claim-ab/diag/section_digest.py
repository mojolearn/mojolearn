# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""sha256 of a shared library's CODE AND CONSTANT sections, not of the file.

WHY THIS EXISTS (2026-09-16, lane/rf-mutex-claim-acquire). The A/B leg
decided its two arms were independent programs by comparing the whole-file
sha256 of `_mojolearn_rf.so`. That comparison cannot fail on macOS, because
`bindings/build_rf.sh` builds into a fresh `mktemp` directory and the
compiler bakes that path into the install name, so two builds of the SAME
source always have different file digests. Its `DIGEST COLLISION` branch was
dead code, and the leg would have timed one program twice and reported
whatever it saw.

It also would not have caught the defect that was actually there. The repair
as first written was `_ = Atomic.load[ordering = Ordering.ACQUIRE](mutex)`,
an atomic load whose result is discarded, which the compiler deletes. Built
with and without the control define, every section of the artifact was
byte-identical. A file digest that differs for a reason unrelated to the code
says nothing about whether the code moved; the sections say it directly.

WHAT IS HASHED. The executable text and the read-only constants, and nothing
else. On Linux ELF that is `.text` and `.rodata`. On Mach-O it is
`__TEXT,__text` and `__TEXT,__const`, and `__TEXT,__const` is where this
project's embedded Metal AIR blobs live, so a kernel change shows up there.
Skipped on purpose: the Mach-O header and load commands (the install name),
`LC_UUID`, the code-signature slot, ELF build ids and `.comment`. Those move
between two builds of identical source and carry no code.

REFUSES RATHER THAN PASSES. If the file is not an object this reader
understands, or a named section is missing, it exits 2 with a message. A
guard that cannot fail is worth nothing, so a broken instrument must never
be indistinguishable from two independent arms. It shells out to nothing, so
a pod without binutils is not a reason for it to go quiet.

    python3 tools/rf_nondeterminism/section_digest.py <file> [<file> ...]

prints `<sha256>  <path>` per file, and with two or more files also prints
`SECTIONS IDENTICAL` or `SECTIONS DIFFER` and exits 1 when they are
identical, so a shell can branch on the exit status. Exit 2 is a REFUSAL,
which is never exit 1, so a caller cannot read a broken instrument as a
collision or the other way round.
"""
import hashlib
import struct
import sys

ELF_WANT = (".text", ".rodata")
MACHO_WANT = (("__TEXT", "__text"), ("__TEXT", "__const"))


def _refuse(msg):
    """Exit 2, never 1. Exit 1 means SECTIONS IDENTICAL, and a caller that
    cannot tell a broken instrument from a collision has a guard that says
    the same thing for two different reasons."""
    sys.stderr.write(msg + "\n")
    raise SystemExit(2)


def _elf_sections(data):
    """{name: bytes} for an ELF32 or ELF64 object, either endianness."""
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    if is64:
        e_shoff, = struct.unpack_from(end + "Q", data, 0x28)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
            end + "HHH", data, 0x3A)
        namefmt, offfmt, sizefmt = 0x00, 0x18, 0x20
        unpack = lambda o: struct.unpack_from(end + "Q", data, o)[0]
    else:
        e_shoff, = struct.unpack_from(end + "I", data, 0x20)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
            end + "HHH", data, 0x2E)
        namefmt, offfmt, sizefmt = 0x00, 0x10, 0x14
        unpack = lambda o: struct.unpack_from(end + "I", data, o)[0]

    def hdr(i):
        return e_shoff + i * e_shentsize

    strtab_off = unpack(hdr(e_shstrndx) + offfmt)
    out = {}
    for i in range(e_shnum):
        h = hdr(i)
        nameoff, = struct.unpack_from(end + "I", data, h + namefmt)
        sh_type, = struct.unpack_from(end + "I", data, h + 4)
        off = unpack(h + offfmt)
        size = unpack(h + sizefmt)
        z = data.index(b"\0", strtab_off + nameoff)
        name = data[strtab_off + nameoff:z].decode("ascii", "replace")
        if sh_type == 8:  # SHT_NOBITS occupies no file space
            out[name] = b""
        else:
            out[name] = data[off:off + size]
    return out


def _macho_sections(data):
    """{(segname, sectname): bytes} for a 64-bit Mach-O."""
    ncmds, = struct.unpack_from("<I", data, 16)
    off = 32
    out = {}
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmd == 0x19:  # LC_SEGMENT_64
            segname = data[off + 8:off + 24].rstrip(b"\0").decode()
            nsects, = struct.unpack_from("<I", data, off + 64)
            s = off + 72
            for _ in range(nsects):
                sectname = data[s:s + 16].rstrip(b"\0").decode()
                sec_off, = struct.unpack_from("<I", data, s + 48)
                size, = struct.unpack_from("<Q", data, s + 40)
                out[(segname, sectname)] = data[sec_off:sec_off + size]
                s += 80
        off += cmdsize
    return out


def digest(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] == b"\x7fELF":
        secs = _elf_sections(data)
        want = ELF_WANT
    elif data[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
        secs = _macho_sections(data)
        want = MACHO_WANT
    else:
        _refuse("section_digest: %s is not an ELF or Mach-O object "
                "(magic %s); REFUSING rather than reporting a digest"
                % (path, data[:4].hex()))
    h = hashlib.sha256()
    for key in want:
        if key not in secs:
            _refuse("section_digest: %s has no section %r; REFUSING, "
                    "because a missing section would hash the same for "
                    "every arm" % (path, key))
        h.update(secs[key])
    return h.hexdigest()


def main(argv):
    paths = argv[1:]
    if not paths:
        _refuse("usage: section_digest.py <file> [<file> ...]")
    digests = []
    for p in paths:
        d = digest(p)
        digests.append(d)
        print("%s  %s" % (d, p))
    if len(paths) > 1:
        if len(set(digests)) == 1:
            print("SECTIONS IDENTICAL")
            return 1
        print("SECTIONS DIFFER")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
