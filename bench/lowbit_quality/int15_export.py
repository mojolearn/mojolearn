# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Export vectors of the quality lane's `int15` arithmetic for the host
oracle's cross-check.

Lane lane/lowbit-int15, 2026-09-29. Contract
`gemm/IDENTICAL_LOWBIT_CONTRACT.md` section 6.6. Reader:
`gemm/checks/gemm_int15_sim_check.mojo`.

WHAT THIS IS. `arith.py` beside this file is the quality lane's reference
arithmetic, byte for byte (`ARITH_PIN` holds its git blob hash and the
commit of `lane/lowbit-quality` it was read at; this script recomputes the
hash of the file it imports and REFUSES when the two differ). It is the
arithmetic whose held-out perplexity was measured. This script runs it on
float32 operands and writes what it computed: the operands' bits, the
codes, the exponents and the float32 product. It computes nothing of its
own.

    python3 bench/lowbit_quality/int15_export.py --out <file.bin>

Runs on a box that has PyTorch (the shared pod), never on the laptop.

THE FILE. Little-endian 32-bit words, nothing else.
    0            magic 0x35314951
    1            format version, 1
    2            the number of cases
    3..7         the git blob hash of arith.py, five words, high word first
    per case     m, n, k
                 A bits          m * k words
                 B bits          n * k words
                 A codes         m * k words (int32)
                 A exponents     m words (int32)
                 B codes         n * k words (int32)
                 B exponents     n words (int32)
                 C bits          m * n words, C = A @ B^T under int15 x int15

THE OPERANDS. Seeded, so the file is a function of this script and of
`arith.py`. Values over many binades with 24-bit significands, and the rows
the rule has an opinion about: a row of zeros, a NaN, an infinity,
subnormals, a row below 2^-114, ties of the rounding, the clamp at the top
of the range, and rows large and small enough that `ea + eb` leaves
`[-126, 127]` at both ends.
"""
import argparse
import hashlib
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arith  # noqa: E402

MAGIC = 0x35314951
VERSION = 1


def blob_hash(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def pinned_hash():
    return open(os.path.join(HERE, "ARITH_PIN")).read().split()[0]


def operands(gen, rows, k, planted):
    """`rows x k` float32: a normal times a per-value power of two over
    eight binades times a per-row power of two over sixty."""
    x = torch.randn(rows, k, generator=gen, dtype=torch.float32)
    x = x * torch.pow(2.0, torch.randint(-4, 4, (rows, k), generator=gen).to(torch.float32))
    x = x * torch.pow(2.0, torch.randint(-30, 30, (rows, 1), generator=gen).to(torch.float32))
    if not planted:
        return x
    r = 0

    def row():
        nonlocal r
        r += 1
        return r - 1

    if rows >= 12 and k >= 8:
        x[row()] = 0.0                                   # a row of zeros
        i = row(); x[i, 0] = float("nan")                # a NaN codes to 0
        i = row(); x[i, 1] = float("-inf")               # an infinity wins the absmax
        i = row(); x[i] = 0.0; x[i, 0::3] = 2.0 ** -120; x[i, 1::3] = -(2.0 ** -124)   # below 2^-114
        i = row(); x[i] = torch.arange(k, dtype=torch.float32) * 0.5 - 60.0; x[i, 0] = 8192.0  # ties
        i = row(); x[i, 0::2] = 16383.75; x[i, 1::2] = -16382.5                        # the clamp
        i = row(); x[i, 0::2] = 1.0e-40; x[i, 1::2] = 3.0                              # subnormals flush
        i = row(); x[i] = x[i] * 2.0 ** 60; x[i, 0] = 2.0 ** 100                       # a large row
        i = row(); x[i] = x[i] * 2.0 ** -60; x[i, 0] = 2.0 ** -100                     # a small row
        i = row(); x[i] = 0.0; x[i, 0] = 3.0e38                                        # near the top
        i = row(); x[i] = 0.0; x[i, 0] = 2.0 ** -126; x[i, 1] = -(2.0 ** -126)         # the smallest normal
        i = row(); x[i] = -0.0; x[i, 2] = -1.0                                         # negative zeros
    return x


def words_u32(t):
    return t.contiguous().view(torch.int32).reshape(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    here = blob_hash(os.path.join(HERE, "arith.py"))
    pin = pinned_hash()
    if here != pin:
        print(f"REFUSED: arith.py has blob hash {here}, ARITH_PIN names {pin}")
        return 3
    gen = torch.Generator().manual_seed(15)
    shapes = [(4, 6, 33, False), (16, 24, 257, True), (12, 12, 64, True), (3, 5, 4097, False),
              (1, 40, 1000, False), (2, 2, 8192, False), (33, 17, 31, True)]
    out = [MAGIC, VERSION, len(shapes)] + [int(here[i:i + 8], 16) for i in range(0, 40, 8)]
    nan_cells = inf_cells = zero_cells = cells = 0
    for m, n, k, planted in shapes:
        a = operands(gen, m, k, planted)
        b = operands(gen, n, k, planted)
        pa = arith.prepare(a, "int15")
        pb = arith.prepare(b, "int15")
        c = arith.product_prepared(pa, pb)
        assert c.dtype == torch.float32 and c.shape == (m, n), (c.dtype, c.shape)
        qa = pa.codes.to(torch.int32)
        qb = pb.codes.to(torch.int32)
        assert torch.equal(qa.to(torch.float64), pa.codes) and torch.equal(qb.to(torch.float64), pb.codes)
        assert int(qa.abs().max()) <= 16383 and int(qb.abs().max()) <= 16383
        out += [m, n, k]
        out += words_u32(a) + words_u32(b)
        out += qa.reshape(-1).tolist() + pa.e.to(torch.int32).reshape(-1).tolist()
        out += qb.reshape(-1).tolist() + pb.e.to(torch.int32).reshape(-1).tolist()
        out += words_u32(c)
        cells += m * n
        nan_cells += int(torch.isnan(c).sum())
        inf_cells += int(torch.isinf(c).sum())
        zero_cells += int((c == 0).sum())
        print(f"case {m}x{n}x{k} planted={planted} e_a=[{int(pa.e.min())},{int(pa.e.max())}] "
              f"e_b=[{int(pb.e.min())},{int(pb.e.max())}] codes at the clamp: "
              f"{int((qa.abs() == 16383).sum()) + int((qb.abs() == 16383).sum())}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack(f"<{len(out)}I", *[w & 0xFFFFFFFF for w in out]))
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"wrote {args.out}: {len(out)} words, {len(shapes)} cases, {cells} cells "
          f"({nan_cells} NaN, {inf_cells} infinite, {zero_cells} zero), torch {torch.__version__}, "
          f"arith.py blob {here}, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
