# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host row helpers of the classification, ranking and clustering metrics'
native owner (bindings/metrics_exp_native.mojo): the label input step's
remap of dense class codes onto a caller's label order (what
`_metrics_impl._label_map` gathered through the base binding's
`gather_i32`), and the contingency matrix's dtype cast (what
`Array.astype` did). Pure functions of their inputs, no Python."""
from core.host_parallel import host_parallelize


comptime _MEXP_GATHER_CHUNK = 65536  # rows per host-pool task, as core/labels_score_encode.mojo's _DENSE_CHUNK
comptime _MEXP_I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _MEXP_IP = MutPointer[Int, MutUntrackedOrigin]


def mexp_gather_i32(
    table: MutPointer[Int32, MutUntrackedOrigin], nt: Int,
    codes: MutPointer[Int32, MutUntrackedOrigin], n: Int,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) -> Int:
    """`dst[i] = table[codes[i]]` for the n rows (codes in [0, nt)). Returns
    the first row whose mapped value is -1 (a label outside the caller's
    order), or -1 when there is none.

    Lane vdf-xm-labels (2026-11-24): the rows are split into contiguous runs
    of `_MEXP_GATHER_CHUNK` across the host pool (was one serial loop over
    every row). Each run writes its own rows and the first miss of its run;
    the answer is the smallest run's first miss, an integer minimum, so the
    written codes and the returned row are the serial loop's whatever the
    split (no bit moves). The device twin for device-resident codes is
    core/metric_labels_device.mojo `metric_gather_resident` (same table
    rule, same first-miss answer)."""
    if n <= 0:
        return -1
    var units = (n + _MEXP_GATHER_CHUNK - 1) // _MEXP_GATHER_CHUNK
    var firsts = List[Int](length=units, fill=-1)
    var firsts_at = Int(firsts.unsafe_ptr())
    var table_at = Int(table)
    var codes_at = Int(codes)
    var dst_at = Int(dst)
    def run(unit: Int) {imm table_at, imm nt, imm codes_at, imm n, imm dst_at, imm firsts_at}:
        var tp = _MEXP_I32P(unsafe_from_address=table_at)
        var cp = _MEXP_I32P(unsafe_from_address=codes_at)
        var dp = _MEXP_I32P(unsafe_from_address=dst_at)
        var first = -1
        var stop = min(n, (unit + 1) * _MEXP_GATHER_CHUNK)
        for i in range(unit * _MEXP_GATHER_CHUNK, stop):  # native data loop (not a driver loop): rows of one gather run
            var c = Int(cp[i])
            var v = Int32(-1)
            if c >= 0 and c < nt:
                v = tp[c]
            dp[i] = v
            if v < 0 and first < 0:
                first = i
        _MEXP_IP(unsafe_from_address=firsts_at)[unit] = first
    host_parallelize(run, units)
    var first = -1
    for u in range(units):  # driver-loop: over the gather runs, the first run with a miss
        if firsts[u] >= 0:
            first = firsts[u]
            break
    _ = firsts^  # written through its address above
    return first


def mexp_cast_i64(
    src: MutPointer[Int64, MutUntrackedOrigin], n: Int, dst_addr: Int, code: Int,
):
    """The n Int64 counts as int32 (code 0), float64 (code 1) or float32
    (code 2), the conversions `Array.astype` made."""
    if code == 0:
        var d = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
        for i in range(n):
            d[i] = src[i].cast[DType.int32]()
    elif code == 1:
        var d = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=dst_addr)
        for i in range(n):
            d[i] = src[i].cast[DType.float64]()
    else:
        var d = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
        for i in range(n):
            d[i] = src[i].cast[DType.float32]()
