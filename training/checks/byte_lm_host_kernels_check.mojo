# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Bitwise check of the byte LM host kernels against the oracles (DEVIATION 2624).

`tools/byte_lm_host_gate.py` qualifies the kernels end to end at the admitted
profile on retained GPU captures. This check reaches what that data cannot:

  - GEMM at shapes with scalar tails, one SIMD vector, eight-chain groups, and
    more than one leaf (the balanced tree and its carry); rows split across
    calls; the reversed fold against the oracle on reversed columns;
  - the deferred flush's fallback, FORCED on every group, and TRIGGERED by
    planted accumulators that go subnormal and then stay tiny, with a self
    test that those plants really do change the answer when the flush is
    skipped (a plant that could not expose a missing flush proves nothing);
  - signed zeros and subnormal operands;
  - RMS norm, RoPE and a whole block at head_dim 8 (SIMD) and 6 (scalar), at
    full and short lengths;
  - the loss kernel at vocab 256 and 300 (a three-leaf denominator fold), with
    underflowing exponentials and ignored targets.

Every comparison is by bits. Run from the repository root:

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/byte_lm_host_kernels_check.mojo

Raises (nonzero exit) on any mismatch.
"""

from std.memory import bitcast

from checks.numerics import identical_mul_add
from gemm.checks.gemm_oracle import OP_NT, gemm_oracle
from training.byte_lm_host_kernels import (
    block_fast,
    ce_causal_mean_loss_fast,
    flushed,
    gemm_nt_rows,
    pack_nt,
    rms_norm_fast,
    rope_fast,
)
from training.checks.loss_oracle import CeConfig, IGNORE_INDEX_DEFAULT, ce_forward_oracle
from transformer.checks.transformer_fixture import ScorePlant, TransformerDims, TransformerWeights
from transformer.checks.transformer_oracle import (
    TransformerKVCache,
    apply_rope_into,
    build_rope_table,
    rms_norm_into,
    transformer_block_oracle,
)


struct Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed ^ UInt64(0x9E3779B97F4A7C15)

    def next(mut self) -> UInt64:
        var x = self.state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.state = x
        return x

    def uniform(mut self) -> Float32:
        """Exactly representable values in [-1, 1)."""
        return Float32(Int(self.next() >> 40)) / Float32(8388608.0) - Float32(1.0)


def random_list(mut rng: Rng, n: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(rng.uniform() * scale)
    return out^


def differing(x: List[Float32], y: List[Float32]) -> Int:
    if len(x) != len(y):
        return len(x) + len(y) + 1
    var count = 0
    for i in range(len(x)):
        if bitcast[DType.uint32](x[i]) != bitcast[DType.uint32](y[i]):
            count += 1
    return count


def report(name: String, bad: Int) -> Int:
    if bad == 0:
        print("  ok  ", name)
    else:
        print("  FAIL", name, bad, "values differ")
    return bad


def plant_tiny_chains(mut a: List[Float32], mut b: List[Float32], m: Int, n: Int, k: Int):
    """Row 0 of A and every row of B start with products 1e-40 then -1.1e-40
    and continue with exact zeros. With the flush at every step the chain is
    +0.0, then -0.0, then a zero; without it the accumulator stays -1e-41."""
    if k < 3:
        return
    a[0] = Float32(1.0e-20)
    a[1] = Float32(1.1e-20)
    for p in range(2, k):
        a[p] = Float32(0.0)
    for j in range(n):
        b[j * k] = Float32(1.0e-20)
        b[j * k + 1] = Float32(-1.0e-20)
    if m > 1 and k > 3:
        a[k + 3] = Float32(-0.0)
    if m > 2 and k > 4:
        a[2 * k + 4] = Float32(1.0e-45)
    if n > 2 and k > 5:
        b[2 * k + 5] = Float32(-0.0)


def unflushed_cell(a: List[Float32], b: List[Float32], i: Int, j: Int, k: Int) -> Float32:
    var acc = Float32(0.0)
    for p in range(k):
        acc = identical_mul_add(a[i * k + p], b[j * k + p], acc)
    return acc


def check_gemm(mut rng: Rng, m: Int, n: Int, k: Int, plant: Bool) raises -> Int:
    var a = random_list(rng, m * k, Float32(1.0))
    var b = random_list(rng, n * k, Float32(1.0))
    if plant:
        plant_tiny_chains(a, b, m, n, k)
    var want = gemm_oracle(a, b, OP_NT, m, n, k)
    var bt = pack_nt(b, n, k)
    var tag = String("gemm m=") + String(m) + " n=" + String(n) + " k=" + String(k)
    if plant:
        tag += " planted"
    var bad = 0
    if plant and k <= 128:
        # The plant must be able to expose a missing flush.
        var exposed = 0
        for j in range(n):
            if bitcast[DType.uint32](unflushed_cell(a, b, 0, j, k)) != bitcast[DType.uint32](want[j]):
                exposed += 1
        if exposed == 0:
            print("  FAIL", tag, "plant does not change the unflushed chain")
            bad += 1
    for mode in range(2):
        var got = List[Float32](length=m * n, fill=Float32(0.0))
        gemm_nt_rows(a, bt, n, k, 0, m, got, False, mode == 1)
        var name = tag + " force_redo=" + String(mode == 1)
        bad += report(name, differing(got, want))
    var half = m // 2
    var top = List[Float32](length=half * n, fill=Float32(0.0))
    var bottom = List[Float32](length=(m - half) * n, fill=Float32(0.0))
    gemm_nt_rows(a, bt, n, k, 0, half, top)
    gemm_nt_rows(a, bt, n, k, half, m, bottom)
    for q in range(len(bottom)):
        top.append(bottom[q])
    bad += report(tag + " rows split across calls", differing(top, want))
    if k <= 128:
        var ar = List[Float32](length=m * k, fill=Float32(0.0))
        var br = List[Float32](length=n * k, fill=Float32(0.0))
        for i in range(m):
            for p in range(k):
                ar[i * k + p] = a[i * k + (k - 1 - p)]
        for j in range(n):
            for p in range(k):
                br[j * k + p] = b[j * k + (k - 1 - p)]
        var want_r = gemm_oracle(ar, br, OP_NT, m, n, k)
        var got_r = List[Float32](length=m * n, fill=Float32(0.0))
        gemm_nt_rows(a, bt, n, k, 0, m, got_r, True)
        bad += report(tag + " reversed fold", differing(got_r, want_r))
    return bad


def check_rms(mut rng: Rng, dm: Int, l: Int) raises -> Int:
    """No dims: the norm takes the width alone, so a width that is not a
    SIMD multiple can be reached."""
    var x = random_list(rng, l * dm, Float32(2.0))
    var w = random_list(rng, dm, Float32(1.5))
    x[1] = Float32(-0.0)
    x[2] = Float32(1.0e-40)
    w[0] = Float32(3.0e-42)
    var sumsq = List[Float32]()
    var want = List[Float32]()
    rms_norm_into(x, w, l, dm, sumsq, want)
    var tag = String("rms_norm dm=") + String(dm) + " l=" + String(l)
    return report(tag, differing(rms_norm_fast(x, flushed(w), l, dm), want))


def check_rope(mut rng: Rng, nh: Int, hd: Int, l: Int) raises -> Int:
    var dims = TransformerDims(nh * hd, nh, nh, hd, 2 * nh * hd, l)
    var rope = build_rope_table(dims)
    var src = random_list(rng, l * nh * hd, Float32(3.0))
    src[0] = Float32(-0.0)
    src[3] = Float32(2.0e-41)
    var want = List[Float32]()
    apply_rope_into(src, nh, hd, 1, l, 0, rope, want)
    var tag = String("rope nh=") + String(nh) + " hd=" + String(hd) + " l=" + String(l)
    return report(tag, differing(rope_fast(src, nh, hd, l, rope), want))


def check_block(mut rng: Rng, dm: Int, nh: Int, nkv: Int, hd: Int, inter: Int, positions: Int, l: Int) raises -> Int:
    var dims = TransformerDims(dm, nh, nkv, hd, inter, positions)
    var qw = nh * hd
    var kw = nkv * hd
    var w = TransformerWeights(dims)
    w.norm1_w = random_list(rng, dm, Float32(1.2))
    w.w_q = random_list(rng, qw * dm, Float32(0.4))
    w.w_k = random_list(rng, kw * dm, Float32(0.4))
    w.w_v = random_list(rng, kw * dm, Float32(0.4))
    w.w_o = random_list(rng, dm * qw, Float32(0.4))
    w.norm2_w = random_list(rng, dm, Float32(1.2))
    w.w_gate = random_list(rng, inter * dm, Float32(0.4))
    w.w_up = random_list(rng, inter * dm, Float32(0.4))
    w.w_down = random_list(rng, dm * inter, Float32(0.4))
    w.w_q[5] = Float32(-0.0)
    w.w_down[7] = Float32(4.0e-43)
    var x = random_list(rng, l * dm, Float32(1.0))
    x[4] = Float32(-0.0)
    x[9] = Float32(1.0e-39)
    var rope = build_rope_table(dims)
    var cache = TransformerKVCache(1, dims, l, 0)
    var st = transformer_block_oracle(w, x, 1, l, cache, rope, ScorePlant.none())
    var tensors = List[List[Float32]]()
    tensors.append(List[Float32]())
    tensors.append(flushed(w.norm1_w))
    tensors.append(pack_nt(w.w_q, qw, dm))
    tensors.append(pack_nt(w.w_k, kw, dm))
    tensors.append(pack_nt(w.w_v, kw, dm))
    tensors.append(pack_nt(w.w_o, dm, qw))
    tensors.append(flushed(w.norm2_w))
    tensors.append(pack_nt(w.w_gate, inter, dm))
    tensors.append(pack_nt(w.w_up, inter, dm))
    tensors.append(pack_nt(w.w_down, dm, inter))
    var got = block_fast(tensors, 1, x, l, dims, rope)
    var tag = String("block dm=") + String(dm) + " hd=" + String(hd) + " kv=" + String(nkv) + " ff=" + String(inter) + " l=" + String(l)
    return report(tag, differing(got, st.residual2_out))


def check_loss(mut rng: Rng, n: Int, vocab: Int) raises -> Int:
    var logits = random_list(rng, n * vocab, Float32(8.0))
    var targets = List[Int32](capacity=n)
    for i in range(n):
        targets.append(Int32(Int(rng.next() >> 33) % vocab))
    # A row whose exponentials underflow almost everywhere, a flat row, and an
    # ignored target.
    for v in range(vocab):
        logits[v] = Float32(-100.0)
        logits[vocab + v] = Float32(0.25)
    logits[3] = Float32(90.0)
    if n > 2:
        targets[2] = Int32(IGNORE_INDEX_DEFAULT)
    var want = ce_forward_oracle(logits, targets, CeConfig.causal_lm(vocab)).loss[0]
    var got = ce_causal_mean_loss_fast(logits, targets, vocab)
    var want_l: List[Float32] = [want]
    var got_l: List[Float32] = [got]
    return report(String("loss n=") + String(n) + " vocab=" + String(vocab), differing(got_l, want_l))


def main() raises:
    var rng = Rng(UInt64(2624))
    var bad = 0
    print("byte LM host kernels against the oracles, by bits")
    bad += check_gemm(rng, 3, 256, 32, False)
    bad += check_gemm(rng, 3, 256, 32, True)
    bad += check_gemm(rng, 4, 64, 32, True)
    bad += check_gemm(rng, 5, 32, 64, True)
    bad += check_gemm(rng, 4, 16, 32, True)
    bad += check_gemm(rng, 3, 40, 8, True)
    bad += check_gemm(rng, 3, 7, 5, True)
    bad += check_gemm(rng, 2, 70, 128, True)
    bad += check_gemm(rng, 2, 33, 130, True)
    bad += check_gemm(rng, 3, 40, 300, True)
    bad += check_gemm(rng, 4, 1, 256, False)
    bad += check_rms(rng, 32, 32)
    bad += check_rms(rng, 13, 7)
    bad += check_rope(rng, 4, 8, 32)
    bad += check_rope(rng, 1, 6, 7)
    bad += check_rope(rng, 2, 10, 3)
    bad += check_block(rng, 32, 4, 2, 8, 64, 32, 32)
    bad += check_block(rng, 32, 4, 2, 8, 64, 32, 5)
    bad += check_block(rng, 24, 4, 4, 6, 40, 16, 16)
    bad += check_block(rng, 24, 4, 2, 6, 40, 16, 1)
    bad += check_loss(rng, 64, 256)
    bad += check_loss(rng, 5, 300)
    if bad != 0:
        raise Error("byte LM host kernels check: " + String(bad) + " failures")
    print("PASS")
