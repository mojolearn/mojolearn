# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 636 over a borrowed host buffer, for the CPU host binding
(`bindings/_mojolearn_svm_host.mojo`) only. Moved out of
`svm/impl/svm_parameter.mojo` (cpu-gpu-cleanup c-svm, 2026-10-02): the GPU
binding checks X on the device after its one upload
(`svm/checks/device_select.mojo::check_finite_device`), so no GPU module
imports the host pool for it any more.

DEVIATION 2665 (2026-09-11): one threaded pass. The predicate is
`isfinite`'s by bits (the exponent all ones is a NaN or an infinity), the
reported index is the first failing flat index, and the message is
`check_finite_list`'s. Spans of SVM_FINITE_SPAN cells run on the host pool;
the lowest span with a failing cell holds the first one.
"""

from std.memory import bitcast
from core.host_parallel import host_parallelize

comptime SVM_FINITE_SPAN = 1 << 18


def _first_nonfinite_span(
    values: MutPointer[Float32, MutUntrackedOrigin], lo: Int, hi: Int
) -> Int:
    """The first `i` in `[lo, hi)` whose exponent bits are all ones, or -1.
    Sixteen cells at a time OR `magnitude + 0x00800000`, whose bit 31 is set
    exactly when the magnitude bits are at or above 0x7F800000; a flagged
    block is then walked cell by cell."""
    var i = lo
    while i + 16 <= hi:
        var acc = UInt32(0)
        for j in range(16):
            acc |= (
                bitcast[DType.uint32](values.unsafe_load(i + j)) & UInt32(0x7FFFFFFF)
            ) + UInt32(0x00800000)
        if (acc & UInt32(0x80000000)) != UInt32(0):
            break
        i += 16
    while i < hi:
        if (bitcast[DType.uint32](values.unsafe_load(i)) & UInt32(0x7FFFFFFF)) >= UInt32(
            0x7F800000
        ):
            return i
        i += 1
    return -1


def check_finite_ptr(
    values: MutPointer[Float32, MutUntrackedOrigin], n: Int, what: String
) raises:
    """`check_finite_list` over `n` borrowed cells (DEVIATION 2665)."""
    if n <= 0:
        return
    var spans = (n + SVM_FINITE_SPAN - 1) // SVM_FINITE_SPAN
    var first = -1
    if spans == 1:
        first = _first_nonfinite_span(values, 0, n)
    else:
        var firsts = List[Int](length=spans, fill=-1)
        var fp = firsts.unsafe_ptr()

        def _span_task(s: Int) {imm values, imm fp, imm n}:
            var lo = s * SVM_FINITE_SPAN
            var hi = lo + SVM_FINITE_SPAN
            if hi > n:
                hi = n
            fp.unsafe_store(s, _first_nonfinite_span(values, lo, hi))

        host_parallelize(_span_task, spans)
        # `firsts` is read past the join, so it outlives every task.
        for s in range(spans):
            if firsts[s] >= 0:
                first = firsts[s]
                break
    if first >= 0:
        raise Error(
            "svm: " + what + " contains a non-finite value at flat index "
            + String(first) + " (DEVIATION 636: a NaN or inf input cannot be"
            " fitted; a computed NaN has a vendor-specific payload)"
        )
