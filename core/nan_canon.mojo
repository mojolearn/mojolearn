# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE NaN WORD FOR EVERY COLUMN, written where a float32 result leaves the
device (or, on the CPU column, where the host kernel has finished the
caller's array): lane fx-nan-canon, 2026-11-19.

WHY. IEEE 754 says WHEN an operation returns a NaN (inf - inf, 0 * inf,
0 / 0, sqrt of a negative) and leaves the word to the implementation.
Measured (checks/ieee_arith_check.mojo, IDENTITY_PATHS row 39): NVIDIA
0x7fffffff, Apple and an ARM host 0x7fc00000, AMD and an x86 host
0xffc00000. So an `identical` result that holds a computed NaN was not the
same bits on the three vendors, and neither was the hash of the returned
bytes. A GEMM reaches one from finite inputs: two products overflow to
+inf and -inf and the fold adds them
(gemm/checks/gemm_rtf_boundary_check.mojo, the +-FLT_MAX draw at k = 300).

WHAT (the experiment MOJOLEARN_IDENTITY_FAST_NAN_CANON; DEFAULT ON since
2026-11-20, row `identity-nan-canon` of
bench/results/paper_compare/owed-pre0867.json; see NAN_CANON_ON below):
every NaN word of a downloaded float32 result becomes NAN_CANON_BITS, the
quiet NaN with a clear sign and an empty payload (the word
`metrics/checks/pinned_sum.mojo` `canonicalize_nan` already writes for the
metric scalars). No other word moves: an infinity, a signed zero, a
subnormal and every finite value keep their bits, because the test is an
integer compare on the word and the only store is the constant.

WHERE, AND WHY NOT IN THE KERNEL. The pass runs on the host over the
caller's pages right after the copy that filled them: one read of words
the copy has just written, and a store only where a NaN is. A select per
cell inside the GEMM leaf loop would sit on the hot path of every cell of
every launch on every vendor for a word that almost no result contains.
Nothing is added on the device.

NOT EVERY float32 BUFFER HOLDS FLOATS. `x_prep/device.mojo` moves int32
words (the LabelBinarizer output, the count cells) through float32 device
buffers and `core/staged_download.mojo` `download_f32_into`; the int32 word
-1 is 0xffffffff, which reads as a NaN. So the transport itself does NOT
canonicalize: a door whose buffer holds float32 results asks for it
(`download_f32_into_canon`, or `nan_canon_f32` after its own copy), and a
door that moves words keeps the plain copy.

Under `-D MOJOLEARN_IDENTITY_FAST_NAN_CANON_OFF` every entry here compiles to
nothing at its call sites (`comptime if NAN_CANON_ON`).
"""
from std.memory import bitcast
from std.sys import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count_work

#: The one NaN word: quiet bit set, sign clear, payload empty. Equal to
#: `metrics/checks/pinned_sum.mojo` CANONICAL_NAN_BITS and to the constant
#: IDENTITY_PATHS Clause B names.
comptime NAN_CANON_BITS: UInt32 = 0x7FC00000

#: The switch. `identical` only: `fast` promises no bits. DEFAULT ON since 2026-11-20.
# This is a cross-vendor identity fix, not a speed arm: a computed NaN was three different words on the three
# vendors. Its cost, A/B round 3, 2026-11-20 (lane fx-verdicts5; NVIDIA L40S host 2ef132a6de04, wheel 28a780604c and
# its B set, base and arm back to back in one pod job, tag nvc-ab-20261120T063324Z-c630ad; one scored run per arm;
# digests and quality equal on both cells, which hold no NaN). Base -> arm ms: neural/gemm 73.55 -> 67.42 (0.917;
# the row's other base in the job is 61.74, so the row's own spread is 19 %: no cost readable), transformer-decode
# 1,086.0 -> 1,095.6 (1.009; the row's three bases in the job are 1,086.0 to 1,086.8 and an arm that does not touch
# it reads 1,089.4, so +0.9 % is above the row's base spread: a cost of at most 9.6 ms a call, one reading). DEFAULT
# as the identity fix it is: no cost on the GEMM row, under 1 % on the decode row. OWED: name the door the decode row
# passes (lane fx-nan-canon read the decode rows as returning tokens through no canon door) and drop the pass there
# if its buffer cannot hold a computed NaN. BITS CHANGE on every column together, only in results that hold a NaN:
# NVIDIA 0x7fffffff, AMD and x86 host 0xffc00000 -> 0x7fc00000.
# -D MOJOLEARN_IDENTITY_FAST_NAN_CANON_OFF returns the vendor's word (kept until the release's one verification
# has recorded the columns; then an owed deletion).
# OWED, doors that still return the vendor's NaN word (lane fx-nan-canon's list, each needs its buffer confirmed to
# hold floats): the dequantize and widening doors gemm/host/linalg_native.mojo (a passed-through input NaN); qr /
# svdvals / eigen (decomposition/linalg_door_impl.mojo); estimator outputs that leave through a raw
# `enqueue_copy(dst_ptr=...)` or a private `download_f32` (kernel_methods, mixture, gaussian_process, cholesky,
# x_cnn, x_ann distances, resample statistics, embedding); the cross-entropy gradient training/estimator.mojo;
# state exports through plain `download_f32_into` (byte-LM state and gradients).
comptime NAN_CANON_ON = (
    not is_defined["MOJOLEARN_IDENTITY_FAST_NAN_CANON_OFF"]() and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
)


@always_inline
def nan_canon_span_f32(p: MutPointer[Float32, MutUntrackedOrigin], n: Int) -> Int:
    """Every NaN word of `p[0:n]` replaced by NAN_CANON_BITS; returns how
    many words were NaN. Eight lanes at a time, an integer test on the bits
    (`SIMD.ne(x, x)` reads False on a NaN lane: core/host_simd_identical.mojo
    `isnan_v`), and the scalar walk only over a vector that holds one. A
    span with no NaN is read and never written. Runs on the calling thread:
    `nan_canon_f32` is the split."""
    comptime W = 8
    var absmask = SIMD[DType.uint32, W](0x7FFFFFFF)
    var inf = SIMD[DType.uint32, W](0x7F800000)
    var canon = bitcast[DType.float32](NAN_CANON_BITS)
    var hits = 0
    var i = 0
    var body = n - n % W
    while i < body:
        var bits = bitcast[DType.uint32, W](p.unsafe_load[width=W](i)) & absmask
        if bits.gt(inf).reduce_or():
            for j in range(W):  # driver-loop: over the 8 lanes of the one vector that holds a NaN
                if (bitcast[DType.uint32](p.unsafe_load(i + j)) & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
                    p.unsafe_store(i + j, canon)
                    hits += 1
        i += W
    while i < n:
        if (bitcast[DType.uint32](p.unsafe_load(i)) & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
            p.unsafe_store(i, canon)
            hits += 1
        i += 1
    return hits


def nan_canon_f32(p: MutPointer[Float32, MutUntrackedOrigin], n: Int) -> Int:
    """`nan_canon_span_f32` over `p[0:n]`, split into contiguous spans
    across the host's cores above the grain floor (the split of
    core/staged_download.mojo `_copy_out`). Each word is a function of
    itself alone, so the result does not depend on the split. Returns the
    number of NaN words found (summed over the spans). The caller owns
    `p[0:n]` and keeps it alive for the call."""
    # parallel: independent float32 words over contiguous spans across the host's cores
    if n <= 0:
        return 0
    var tasks = host_predict_task_count_work(n, n)
    if tasks <= 1:
        return nan_canon_span_f32(p, n)
    # 1024 floats = one 4 KiB page: whole pages per span
    var span = ((host_predict_chunk(n, tasks) + 1023) // 1024) * 1024
    var base = Int(p)
    var found = List[Int](length=tasks, fill=0)
    var found_ptr = rebind[MutPointer[Int, MutUntrackedOrigin]](found.unsafe_ptr())

    def _span(t: Int) {imm base, imm n, imm span, imm found_ptr}:
        var lo = t * span
        if lo >= n:
            return
        var hi = min(lo + span, n)
        var at = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=base + lo * 4)
        found_ptr.unsafe_store(t, nan_canon_span_f32(at, hi - lo))

    host_parallelize(_span, tasks)
    var hits = 0
    for t in range(tasks):  # driver-loop: over the tasks' counts (at most one per core)
        hits += found[t]
    _ = found^
    return hits
