#!/usr/bin/env python3
"""DEVIATION 2701: the exact host reference for gemm/checks/gemm_seam_probe.mojo.

Reads one or more probe logs, regenerates every triple from the SEAM_WORDS
line in the probe's own order and flushes, computes a*b+acc EXACTLY as a
rational, rounds it to binary32 under three candidate semantics, hashes each
result stream with the probe's FNV-1a 64, and names which semantics each
device lane's hash equals:

  rtf   round-to-nearest-even to binary32, then flush subnormals to signed zero
        (the contract, `mojolearn.identical.gemm.fp32.v1` sections 4 and 5c)
  fbr   if 0 < |exact| < 2^-126 the result is the signed zero of the exact
        value, else round-to-nearest-even (what NVIDIA's fma.rn.ftz was
        measured to do at a=0x3f7fffff b=0x00800000 acc=+0)
  none  round-to-nearest-even with subnormals kept

    python3 tools/gemm_seam_probe_reference.py <probe.log> [<probe.log> ...]

Exit 0 when every lane of every log matches exactly one semantics, 1 otherwise.
"""
import re
import struct
import sys
from fractions import Fraction

MIN_NORMAL = Fraction(1, 2 ** 126)
MAX_FINITE = Fraction((2 ** 24 - 1) * 2 ** 104)
FNV_OFF = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def f32_from_word(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def word_from_f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ftz_word(w):
    if (w & 0x7F800000) == 0 and (w & 0x007FFFFF) != 0:
        return w & 0x80000000
    return w


def round_to_binary32(x, neg_zero):
    """RN-even of the exact rational x to a binary32 word, subnormals kept,
    overflow to infinity. `neg_zero` decides the sign of an exact zero."""
    if x == 0:
        return 0x80000000 if neg_zero else 0
    sign = 0x80000000 if x < 0 else 0
    ax = -x if x < 0 else x
    # Find e with 2^e <= ax < 2^(e+1).
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    if Fraction(2) ** e > ax:
        e -= 1
    elif Fraction(2) ** (e + 1) <= ax:
        e += 1
    assert Fraction(2) ** e <= ax < Fraction(2) ** (e + 1)
    if e < -126:
        # Subnormal range: quantum 2^-149, no hidden bit.
        q = -149
    else:
        q = e - 23
    scaled = ax / Fraction(2) ** q
    m = scaled.numerator // scaled.denominator
    rem = scaled - m
    half = Fraction(1, 2)
    if rem > half or (rem == half and (m & 1) == 1):
        m += 1
    # Renormalize on mantissa overflow (m == 2^24) or subnormal promotion.
    if q == -149:
        if m >= 2 ** 23:
            # Became the smallest normal (or more): exponent field 1.
            exp_field = 1
            mant = m - 2 ** 23
            if mant >= 2 ** 23:
                exp_field += 1
                mant -= 2 ** 23
            return sign | (exp_field << 23) | mant
        return sign | m
    if m == 2 ** 24:
        m = 2 ** 23
        q += 1
    e_field = q + 23 + 127
    if e_field >= 255:
        return sign | 0x7F800000
    return sign | (e_field << 23) | (m - 2 ** 23)


def exact_fma(wa, wb, wc):
    fa, fb, fc = (Fraction(f32_from_word(w)) for w in (wa, wb, wc))
    p = fa * fb
    x = p + fc
    # Sign of an exact zero result: both addends zero -> negative only if both
    # negative; a nonzero cancellation -> +0 (RN).
    neg_zero = False
    if x == 0:
        p_neg = ((wa ^ wb) & 0x80000000) != 0
        p_zero = p == 0
        c_neg = (wc & 0x80000000) != 0
        c_zero = fc == 0
        if p_zero and c_zero:
            neg_zero = p_neg and c_neg
        else:
            neg_zero = False
    return x, neg_zero


def semantics_words(wa, wb, wc):
    x, neg_zero = exact_fma(wa, wb, wc)
    none_w = round_to_binary32(x, neg_zero)
    rtf_w = ftz_word(none_w)
    if x != 0 and abs(x) < MIN_NORMAL:
        fbr_w = 0x80000000 if x < 0 else 0
    else:
        fbr_w = none_w
    return rtf_w, fbr_w, none_w


def fnv1a(words):
    h = FNV_OFF
    for w in words:
        for k in range(4):
            h ^= (w >> (8 * k)) & 0xFF
            h = (h * FNV_PRIME) & MASK64
    return h


def reference_for(words):
    flushed = [ftz_word(w) for w in words]
    count = len(words)
    n = count ** 3
    streams = {"rtf": [], "fbr": [], "none": []}
    boundary_set = []
    for i in range(n):
        wa = flushed[i % count]
        wb = flushed[(i // count) % count]
        wc = flushed[i // (count * count)]
        rtf_w, fbr_w, none_w = semantics_words(wa, wb, wc)
        streams["rtf"].append(rtf_w)
        streams["fbr"].append(fbr_w)
        streams["none"].append(none_w)
        if rtf_w != fbr_w:
            boundary_set.append((i, wa, wb, wc, rtf_w, fbr_w))
    hashes = {k: fnv1a(v) for k, v in streams.items()}
    return hashes, streams, boundary_set


def parse(path):
    words = None
    lanes = {}
    column = None
    for line in open(path):
        line = line.strip()
        if line.startswith("SEAM_PROBE "):
            m = re.search(r"column=(\S+)", line)
            column = m.group(1) if m else "?"
        elif line.startswith("SEAM_WORDS"):
            words = [int(t, 16) for t in line.split()[1:]]
        elif line.startswith("SEAM_HASH "):
            m = re.search(r"lane=(\S+) fnv1a64=([0-9a-f]+)", line)
            lanes[m.group(1)] = int(m.group(2), 16)
    return column, words, lanes


def main(paths):
    ok = True
    cache = {}
    for path in paths:
        column, words, lanes = parse(path)
        if words is None or not lanes:
            print(f"{path}: no SEAM_WORDS or SEAM_HASH lines")
            ok = False
            continue
        key = tuple(words)
        if key not in cache:
            cache[key] = reference_for(words)
        hashes, streams, boundary_set = cache[key]
        print(f"== {path}: column={column} triples={len(streams['rtf'])} "
              f"boundary_triples(rtf!=fbr)={len(boundary_set)}")
        for k, h in hashes.items():
            print(f"   reference {k:5s} fnv1a64={h:016x}")
        for lane, h in lanes.items():
            names = [k for k, v in hashes.items() if v == h]
            verdict = ",".join(names) if names else "NONE OF THE THREE"
            print(f"   lane {lane:8s} fnv1a64={h:016x} -> {verdict}")
            if len(names) != 1:
                ok = False
        for (i, wa, wb, wc, r, f) in boundary_set[:8]:
            print(f"   boundary i={i} a={wa:08x} b={wb:08x} acc={wc:08x} rtf={r:08x} fbr={f:08x}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
