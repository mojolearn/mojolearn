"""`pcg_rng.mojo` against RAFT's and cuML's OWN arithmetic, cell for cell.

    pixi run mojo run -I . extratrees/mojo_only/pcg_rng_check.mojo

Reads `extratrees/tools/rng_oracle/pcg_reference.txt`, which is written by
`extratrees/tools/rng_oracle/main.cpp` -- a C++ program that COPIES RAFT's
`PCGenerator` and cuML's `fnv1a32` with the CUDA decorations defined away.
So this is not a tally we wrote twice: it is our transcription against theirs.

WHAT IT IS GATING BEYOND "the numbers match":

* **The seeding order.** `_init_pcg` discards a draw, THEN adds the seed, THEN
  discards another. Every from-memory `pcg32_srandom_r` gets some part of this
  wrong, and every stream in the file would move if we did.
* **`skipahead`.** Nine streams, offsets 0, 1, 5, 64, 1023, 10**6 and
  0xFFFFFFFF, so the advance is exercised with one bit set, with an exact power
  of two, with a run of ones, and with a 32-bit ladder -- not just with zero,
  where the whole function is a no-op.
* **The rejection loop, at a range where it fires.** `diff = 1000003` and
  `diff = 97` are prime; `diff = 0xFFFFFFFF` rejects almost never but has a
  huge `(-s) % s`; `diff = 2**63 + 1` puts the 64-bit path through the wide
  multiply's carry.
* **Floats by their BITS.** `String(Float32)` in this repository does not round
  trip (0.46% of float32 come back one ULP wrong), so the reference writes
  `<decimal>/<hexbits>` and only the hex half is compared. The decimal half is
  for a human reading the diff.
* **The fusion question.** `uniform_float` is a multiply then an add. If Mojo
  contracts it into an FMA and the reference does not, the `uf` cells go red;
  that is DEVIATION 142 being measured rather than asserted.
"""
from extratrees.mojo_only.pcg_rng import (
    PCGenerator,
    SplitKey,
    fnv1a32,
    key_for,
    uniform_float,
    uniform_int_u32,
    uniform_int_u64,
    uniform_threshold,
    wmul_64bit,
)
from std.memory import bitcast
from std.sys import exit

comptime ORACLE = "extratrees/tools/rng_oracle/pcg_reference.txt"


def parse_hex(text: String) raises -> UInt64:
    var v: UInt64 = 0
    var n = 0
    for b in text.as_bytes():
        var c = Int(b)
        var d: Int
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        else:
            raise Error("bad hex digit in '" + String(text) + "'")
        v = (v << 4) | UInt64(d)
        n += 1
    if n == 0 or n > 16:
        raise Error("bad hex width in '" + String(text) + "'")
    return v


def parse_dec_u64(text: String) raises -> UInt64:
    var v: UInt64 = 0
    var n = 0
    for b in text.as_bytes():
        var c = Int(b)
        if c < 48 or c > 57:
            raise Error("bad decimal digit in '" + String(text) + "'")
        v = v * 10 + UInt64(c - 48)
        n += 1
    if n == 0:
        raise Error("empty decimal")
    return v


def hexbits_of_field(text: String) raises -> UInt32:
    """`<decimal>/<hexbits>` -> the BITS. The decimal half is for humans only."""
    var parts = text.split("/")
    if len(parts) != 2:
        raise Error("expected <decimal>/<hexbits>, got '" + text + "'")
    return parse_hex(String(parts[1])).cast[DType.uint32]()


def float_of_field(text: String) raises -> Float32:
    """`<decimal>/<hexbits>` -> the Float32 those BITS name."""
    return bitcast[DType.float32](hexbits_of_field(text))


def take(lines: List[String], mut pos: Int, tag: String) raises -> List[String]:
    """Consume one line, require it to start with `tag`, return its fields."""
    if pos >= len(lines):
        raise Error("reference ended early, wanted '" + tag + "'")
    var parts = lines[pos].split(" ")
    if String(parts[0]) != tag:
        raise Error(
            "line " + String(pos) + ": expected '" + tag + "', got '" + lines[pos] + "'"
        )
    var out = List[String]()
    for p in parts:
        out.append(String(p))
    pos += 1
    return out^


def hex32(v: UInt32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(8):
        var nib = Int((v >> UInt32((7 - i) * 4)) & 0xF)
        out += String(DIGITS[byte=nib])
    return out


def hex64(v: UInt64) -> String:
    return hex32((v >> 32).cast[DType.uint32]()) + hex32(v.cast[DType.uint32]())


struct Tally(Copyable, Movable):
    var checked: Int
    var failed: Int
    var shown: Int

    def __init__(out self):
        self.checked = 0
        self.failed = 0
        self.shown = 0

    def ok(mut self, section: String, where: String, got: String, want: String):
        self.checked += 1
        if got != want:
            self.failed += 1
            if self.shown < 12:
                self.shown += 1
                print("  FAIL", section, where, "ours", got, "theirs", want)


def main() raises:
    print("PCG / fnv1a32 vs RAFT+cuML own arithmetic (" + ORACLE + "):")
    var f = open(ORACLE, "r")
    var text = f.read()
    f.close()

    var lines = List[String]()
    for raw in text.splitlines():
        var s = String(String(raw).strip())
        if s.byte_length() == 0:
            continue
        if s[byte=0] == "#":
            continue
        lines.append(s^)

    var pos = 0

    # --- pins -----------------------------------------------------------
    var raft_pin = take(lines, pos, "raft_pin")[1]
    var cuml_pin = take(lines, pos, "cuml_pin")[1]
    print("  RAFT", raft_pin)
    print("  cuML", cuml_pin)
    if raft_pin != "661a3b840c3300f95f053812a560c952c9d049a4":
        raise Error("reference was built against a different RAFT pin")
    if cuml_pin != "00094f7e4e4b5da3a968d193a4da6085fa38f11b":
        raise Error("reference was built against a different cuML pin")

    var t = Tally()

    # --- fnv1a32 --------------------------------------------------------
    var n_fnv = Int(take(lines, pos, "fnv_cases")[1])
    for _ in range(n_fnv):
        var r = take(lines, pos, "fnv")
        var h = parse_hex(r[1]).cast[DType.uint32]()
        var txt = parse_hex(r[2]).cast[DType.uint32]()
        var want = parse_hex(r[3]).cast[DType.uint32]()
        t.ok("fnv1a32", "(" + r[1] + "," + r[2] + ")", hex32(fnv1a32(h, txt)), hex32(want))

    # --- the (feature, tree, node) chain --------------------------------
    var n_chain = Int(take(lines, pos, "chain_cases")[1])
    for _ in range(n_chain):
        var r = take(lines, pos, "chain")
        var feature = parse_hex(r[1]).cast[DType.uint32]()
        var tree = parse_hex(r[2]).cast[DType.uint32]()
        var node = parse_hex(r[3]).cast[DType.uint32]()
        var want = parse_hex(r[4])
        var k = key_for(UInt64(0), tree, node, feature)
        t.ok(
            "key_for",
            "(f=" + r[1] + ",t=" + r[2] + ",n=" + r[3] + ")",
            hex64(k.subsequence),
            hex64(want),
        )

    # --- raw next_u32 ---------------------------------------------------
    var n_streams = Int(take(lines, pos, "streams")[1])
    for _ in range(n_streams):
        var r = take(lines, pos, "stream")
        var gen = PCGenerator(parse_hex(r[1]), parse_hex(r[2]), parse_hex(r[3]))
        var n = Int(r[4])
        for i in range(n):
            var v = take(lines, pos, "u32")
            t.ok(
                "next_u32",
                "seed=" + r[1] + " sub=" + r[2] + " off=" + r[3] + " i=" + String(i),
                hex32(gen.next_u32()),
                hex32(parse_hex(v[1]).cast[DType.uint32]()),
            )

    # --- raw next_u64 ---------------------------------------------------
    var n_u64_streams = Int(take(lines, pos, "u64_streams")[1])
    for _ in range(n_u64_streams):
        var r = take(lines, pos, "u64stream")
        var gen = PCGenerator(parse_hex(r[1]), parse_hex(r[2]), parse_hex(r[3]))
        var n = Int(r[4])
        for i in range(n):
            var v = take(lines, pos, "u64")
            t.ok(
                "next_u64",
                "seed=" + r[1] + " sub=" + r[2] + " off=" + r[3] + " i=" + String(i),
                hex64(gen.next_u64()),
                hex64(parse_hex(v[1])),
            )

    # --- uniform ints, 32-bit diff --------------------------------------
    var n_u32_cases = Int(take(lines, pos, "uint32_cases")[1])
    for _ in range(n_u32_cases):
        var r = take(lines, pos, "uint32")
        var gen = PCGenerator(parse_hex(r[1]), parse_hex(r[2]), parse_hex(r[3]))
        var start = parse_dec_u64(r[4]).cast[DType.uint32]()
        var diff = parse_dec_u64(r[5]).cast[DType.uint32]()
        var n = Int(r[6])
        for i in range(n):
            var v = take(lines, pos, "i32")
            t.ok(
                "uniform_int_u32",
                "sub=" + r[2] + " off=" + r[3] + " start=" + r[4] + " diff=" + r[5]
                + " i=" + String(i),
                String(uniform_int_u32(gen, start, diff)),
                String(parse_dec_u64(v[1]).cast[DType.uint32]()),
            )

    # --- uniform ints, 64-bit diff (the overload cuML instantiates) ------
    var n_u64_cases = Int(take(lines, pos, "uint64_cases")[1])
    for _ in range(n_u64_cases):
        var r = take(lines, pos, "uint64")
        var gen = PCGenerator(parse_hex(r[1]), parse_hex(r[2]), parse_hex(r[3]))
        var start = parse_dec_u64(r[4])
        var diff = parse_dec_u64(r[5])
        var n = Int(r[6])
        for i in range(n):
            var v = take(lines, pos, "i64")
            t.ok(
                "uniform_int_u64",
                "sub=" + r[2] + " off=" + r[3] + " start=" + r[4] + " diff=" + r[5]
                + " i=" + String(i),
                String(uniform_int_u64(gen, start, diff)),
                String(parse_dec_u64(v[1])),
            )

    # --- raw next_float, compared by BITS --------------------------------
    var n_f_streams = Int(take(lines, pos, "float_streams")[1])
    for _ in range(n_f_streams):
        var r = take(lines, pos, "floatstream")
        var gen = PCGenerator(parse_hex(r[1]), parse_hex(r[2]), parse_hex(r[3]))
        var n = Int(r[4])
        for i in range(n):
            var v = take(lines, pos, "f")
            t.ok(
                "next_float",
                "seed=" + r[1] + " sub=" + r[2] + " off=" + r[3] + " i=" + String(i),
                hex32(bitcast[DType.uint32](gen.next_float())),
                hex32(hexbits_of_field(v[1])),
            )

    # --- uniform floats, compared by BITS --------------------------------
    var n_uf_cases = Int(take(lines, pos, "ufloat_cases")[1])
    for _ in range(n_uf_cases):
        var r = take(lines, pos, "ufloat")
        var seed = parse_hex(r[1])
        var sub = parse_hex(r[2])
        var off = parse_hex(r[3])
        var lo = float_of_field(r[4])
        var hi = float_of_field(r[5])
        var n = Int(r[6])
        var gen = PCGenerator(seed, sub, off)
        for i in range(n):
            var v = take(lines, pos, "uf")
            t.ok(
                "uniform_float",
                "sub=" + r[2] + " off=" + r[3] + " lo=" + r[4] + " hi=" + r[5]
                + " i=" + String(i),
                hex32(bitcast[DType.uint32](uniform_float(gen, lo, hi))),
                hex32(hexbits_of_field(v[1])),
            )
        # OURS: uniform_threshold on a fresh generator must reproduce the FIRST
        # draw of that stream exactly, because it is the same call with the
        # offset pinned to 0. Only checkable where off == 0.
        if off == 0:
            var k = SplitKey(seed, sub)
            var again = uniform_threshold(k, lo, hi)
            var expect = PCGenerator(seed, sub, 0)
            t.ok(
                "uniform_threshold",
                "sub=" + r[2] + " lo=" + r[4] + " hi=" + r[5],
                hex32(bitcast[DType.uint32](again)),
                hex32(bitcast[DType.uint32](uniform_float(expect, lo, hi))),
            )

    _ = take(lines, pos, "end")

    # --- wmul_64bit, self-evident cells ---------------------------------
    # The oracle exercises wmul through uniform_int_u64, which only ever reads
    # the HIGH word plus a comparison on the low one. These pin the low word
    # too, against products small enough to be written out by hand.
    var w = wmul_64bit(UInt64(0xFFFFFFFFFFFFFFFF), UInt64(0xFFFFFFFFFFFFFFFF))
    t.ok("wmul_64bit", "(2^64-1)^2 hi", hex64(w[0]), hex64(UInt64(0xFFFFFFFFFFFFFFFE)))
    t.ok("wmul_64bit", "(2^64-1)^2 lo", hex64(w[1]), hex64(UInt64(1)))
    w = wmul_64bit(UInt64(1) << 32, UInt64(1) << 32)
    t.ok("wmul_64bit", "2^32*2^32 hi", hex64(w[0]), hex64(UInt64(1)))
    t.ok("wmul_64bit", "2^32*2^32 lo", hex64(w[1]), hex64(UInt64(0)))
    w = wmul_64bit(UInt64(0xDEADBEEF), UInt64(0xCAFEBABE))
    t.ok("wmul_64bit", "32x32 hi", hex64(w[0]), hex64(UInt64(0)))
    t.ok("wmul_64bit", "32x32 lo", hex64(w[1]), hex64(UInt64(0xB092AB7B88CF5B62)))
    w = wmul_64bit(UInt64(0xDEADBEEFCAFEBABE), UInt64(0x1234567890ABCDEF))
    t.ok("wmul_64bit", "64x64 hi", hex64(w[0]), hex64(UInt64(0x0FD5BDEEE268600E)))
    t.ok("wmul_64bit", "64x64 lo", hex64(w[1]), hex64(UInt64(0x773285AE1C447D62)))

    print("  checked", t.checked, "cells,", t.failed, "failed")
    if t.failed != 0:
        exit(1)
    print("  PASS")
