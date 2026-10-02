# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Sorted unique classes and inverse codes (`np.unique(y, return_inverse=True)`)
for one numeric label buffer: the HOST TWIN of `core/label_encode_device.mojo`
(cpu-gpu-cleanup w2-pyglue, 2026-10-02).

The GPU binding (`bindings/_mojolearn.mojo::unique_inverse`) runs the device
pipeline; the core host binding (`bindings/_mojolearn_core_host.mojo`) runs
`host_unique_inverse` below under the same name on a CPU-only install. Both
take the same steps, in the same order:

  1. KEY. Each label's 64 bits (`kind` 0: float64, 1: int64) become an
     unsigned key that orders as the value does (`label_sort_key`). `-0.0`
     is folded onto `+0.0` first, so the two are one class. A NaN raises
     the status flag (the caller refuses it).
  2. SORT. A stable LSD radix sort of (key, row) pairs by 8-bit digits,
     low digits first. A stable sort by the full key is one permutation
     (ties keep row order), so the device's two 32-bit-half sorts and the
     host's eight digit passes give the same order.
  3. FLAG. flag[j] = 1 where sorted position j starts a new key.
  4. SCAN. The exclusive scan of the flags; class id = scan + flag - 1.
  5. EMIT. codes[row(j)] = class id; at a flagged position the class value
     is the row's own 64 bits, the FIRST row of that class in row order
     (the `_labels.py` ORDER RULE's first-seen representative).

Integers and bit moves only: the two routes give the same bytes."""


comptime LABEL_KIND_F64 = 0
comptime LABEL_KIND_I64 = 1
comptime _SIGN = UInt64(0x8000000000000000)


@always_inline
def label_is_nan(bits: UInt64, kind: Int) -> Bool:
    """True for a float64 NaN pattern (exponent all ones, mantissa non-zero)."""
    if kind != LABEL_KIND_F64:
        return False
    return (bits & UInt64(0x7FFFFFFFFFFFFFFF)) > UInt64(0x7FF0000000000000)


@always_inline
def label_sort_key(bits: UInt64, kind: Int) -> UInt64:
    """An unsigned key in the value's order. float64: CUB's TwiddleIn after
    folding -0.0 onto +0.0; int64: the sign bit flipped."""
    if kind == LABEL_KIND_I64:
        return bits ^ _SIGN
    var b = bits
    if b == _SIGN:
        b = UInt64(0)
    if (b & _SIGN) != UInt64(0):
        return ~b
    return b ^ _SIGN


def host_unique_inverse(
    src: MutPointer[UInt64, MutUntrackedOrigin], n: Int, kind: Int,
    classes: MutPointer[UInt64, MutUntrackedOrigin],
    codes: MutPointer[Int32, MutUntrackedOrigin],
) -> Int:
    """Steps 1-5 above over `n` labels at `src`. Returns the class count,
    or -2 when a label is NaN (nothing is promised about the outputs then).
    `classes` holds `n` slots, `codes` `n`."""
    # 1. key
    var key = List[UInt64](unsafe_uninit_length=n)
    var row = List[UInt32](unsafe_uninit_length=n)
    var tk = List[UInt64](unsafe_uninit_length=n)
    var tr = List[UInt32](unsafe_uninit_length=n)
    var nan = False
    for i in range(n):
        var b = src.unsafe_load(i)
        if label_is_nan(b, kind):
            nan = True
        key[i] = label_sort_key(b, kind)
        row[i] = UInt32(i)
    if nan:
        return -2
    # 2. stable LSD radix sort by 8-bit digits
    var counts = List[Int](length=256, fill=0)
    for p in range(8):
        var shift = UInt64(8 * p)
        for d in range(256):
            counts[d] = 0
        for i in range(n):
            counts[Int((key[i] >> shift) & UInt64(255))] += 1
        var run = 0
        for d in range(256):
            var c = counts[d]
            counts[d] = run
            run += c
        for i in range(n):
            var d = Int((key[i] >> shift) & UInt64(255))
            var dst = counts[d]
            counts[d] = dst + 1
            tk[dst] = key[i]
            tr[dst] = row[i]
        for i in range(n):
            key[i] = tk[i]
            row[i] = tr[i]
    # 3. flag, 4. exclusive scan, 5. emit
    var scan = 0
    for j in range(n):
        var flag = 1 if j == 0 or key[j] != key[j - 1] else 0
        var cls = scan + flag - 1
        var r = Int(row[j])
        codes.unsafe_store(r, Int32(cls))
        if flag == 1:
            classes.unsafe_store(cls, src.unsafe_load(r))
        scan += flag
    return scan
