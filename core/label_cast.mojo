# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The HOST TWIN of `core/label_cast_device.mojo` (lane ag-thin-glue,
2026-10-19), for the CPU-only install: the caller's label array as it is
(core/cast_in.mojo) widened to 64-bit label words, `host_unique_inverse`,
and the k class words narrowed back to the caller's items, by the same
element functions as the device kernels, so both columns give the same
bytes."""
from std.memory import memcpy
from sequence.ops import FP
from core.cast_in import CastIn, cast_raw_bits, cast_w64_bits, narrow_w64
from core.label_encode import host_unique_inverse

comptime _BP = MutPointer[UInt8, MutUntrackedOrigin]


def host_unique_inverse_cast(src: CastIn, label_kind: Int, classes_addr: Int, codes_addr: Int) -> Int:
    """`device_unique_inverse_cast`'s contract on the host: returns k, or -2
    for a NaN label."""
    var n = src.rows
    var size = src.itemsize
    # the bytes in whole words (the tail zero-padded), so a word read never
    # passes the caller's buffer
    var raw = List[Float32](length=src.words(), fill=Float32(0))
    memcpy(dest=_BP(unsafe_from_address=Int(raw.unsafe_ptr())), src=_BP(unsafe_from_address=src.start()), count=src.nbytes())
    var rp = FP(unsafe_from_address=Int(raw.unsafe_ptr()))
    var w64 = List[UInt64](length=n, fill=UInt64(0))
    for i in range(n):  # host twin: one label per row, as _widen_kernel
        w64[i] = cast_w64_bits(cast_raw_bits(rp, src.origin() + i * src.s0, size), src.kind, size)
    var cls = List[UInt64](length=n, fill=UInt64(0))
    var k = host_unique_inverse(
        MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(w64.unsafe_ptr())), n, label_kind,
        MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(cls.unsafe_ptr())),
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes_addr),
    )
    if k >= 0:
        var out = _BP(unsafe_from_address=classes_addr)
        for c in range(k):  # host twin: one class per item, as _narrow_kernel
            var v = narrow_w64(cls[c], src.kind, size)
            for b in range(size):  # the item's bytes, little-endian
                out.unsafe_store(c * size + b, UInt8((v >> UInt64(8 * b)) & UInt64(0xFF)))
    _ = raw^
    _ = w64^
    _ = cls^
    return k
