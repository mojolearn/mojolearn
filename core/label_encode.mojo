# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Sorted unique classes and inverse codes (`np.unique(y, return_inverse=True)`)
for one numeric label buffer: the HOST TWIN of `core/label_encode_device.mojo`
(cpu-gpu-cleanup w2-pyglue, 2026-10-02).

The GPU binding (`bindings/_mojolearn.mojo::unique_inverse`) runs the device
pipeline; the core host binding (`bindings/_mojolearn_core_host.mojo`) runs
`host_unique_inverse` below under the same name on a CPU-only install. Both
take the same steps, in the same order:

  1. KEY. Each label's 64 bits (`kind` 0: float64, 1: int64, 2: uint64) become an
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
from core.host_parallel import host_parallelize


comptime LABEL_KIND_F64 = 0
comptime LABEL_KIND_I64 = 1
comptime LABEL_KIND_U64 = 2
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
    folding -0.0 onto +0.0; int64: the sign bit flipped; uint64: unchanged."""
    if kind == LABEL_KIND_U64:
        return bits
    if kind == LABEL_KIND_I64:
        return bits ^ _SIGN
    var b = bits
    if b == _SIGN:
        b = UInt64(0)
    if (b & _SIGN) != UInt64(0):
        return ~b
    return b ^ _SIGN


comptime _UI_CHUNK = 65536  # labels per host-pool task: each task keys, counts or scatters tens of thousands of labels, so the pool's dispatch cost is amortized whatever the label count
comptime _U64P = MutPointer[UInt64, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime _INTP = MutPointer[Int, MutUntrackedOrigin]


def host_unique_inverse(
    src: MutPointer[UInt64, MutUntrackedOrigin], n: Int, kind: Int,
    classes: MutPointer[UInt64, MutUntrackedOrigin],
    codes: MutPointer[Int32, MutUntrackedOrigin],
) -> Int:
    """Steps 1-5 above over `n` labels at `src`. Returns the class count,
    or -2 when a label is NaN (nothing is promised about the outputs then).
    `classes` holds `n` slots, `codes` `n`.

    Every step runs across the host pool in `_UI_CHUNK` label runs
    (round2-fixes-2; perf audit 3: it was one thread, eight full passes):
    the keys per run; each radix pass a per-run digit histogram, ONE
    digit-major prefix over (digit, run) and a per-run stable scatter (a
    label's destination is the same as the serial pass's, so the order is
    the same permutation); a pass whose digit is the same for every label is
    the identity and is skipped; the flags counted per run, one prefix over
    the runs, then each run emits its codes and class values. Same bytes."""
    if n <= 0:
        return 0
    var units = (n + _UI_CHUNK - 1) // _UI_CHUNK
    # 1. key
    var key = List[UInt64](unsafe_uninit_length=n)
    var row = List[UInt32](unsafe_uninit_length=n)
    var tk = List[UInt64](unsafe_uninit_length=n)
    var tr = List[UInt32](unsafe_uninit_length=n)
    var nan_runs = List[Int](length=units, fill=0)
    var src_at = Int(src)
    var key_at = Int(key.unsafe_ptr())
    var row_at = Int(row.unsafe_ptr())
    var nan_at = Int(nan_runs.unsafe_ptr())
    def keyed(unit: Int) {imm n, imm kind, imm src_at, imm key_at, imm row_at, imm nan_at}:
        var sp = _U64P(unsafe_from_address=src_at)
        var kp = _U64P(unsafe_from_address=key_at)
        var rp = _U32P(unsafe_from_address=row_at)
        var stop = min(n, (unit + 1) * _UI_CHUNK)
        for i in range(unit * _UI_CHUNK, stop):  # native data loop (not a driver loop): labels of one key run
            var b = sp[i]
            if label_is_nan(b, kind):
                _INTP(unsafe_from_address=nan_at)[unit] = 1
            kp[i] = label_sort_key(b, kind)
            rp[i] = UInt32(i)
    host_parallelize(keyed, units)
    for u in range(units):  # driver-loop: over the key runs (one NaN flag each)
        if nan_runs[u] != 0:
            return -2
    # 2. stable LSD radix sort by 8-bit digits
    var hist = List[Int](length=units * 256, fill=0)
    var totals = List[Int](length=256, fill=0)
    var hist_at = Int(hist.unsafe_ptr())
    var cur_k = key_at
    var cur_r = row_at
    var alt_k = Int(tk.unsafe_ptr())
    var alt_r = Int(tr.unsafe_ptr())
    for p in range(8):  # driver-loop: over the eight digit passes
        var shift = UInt64(8 * p)
        var in_k = cur_k
        def count(unit: Int) {imm n, imm shift, imm in_k, imm hist_at}:
            var kp = _U64P(unsafe_from_address=in_k)
            var hp = _INTP(unsafe_from_address=hist_at) + unit * 256
            for d in range(256):  # driver-loop: over the 256 digit counters of one run
                hp[d] = 0
            var stop = min(n, (unit + 1) * _UI_CHUNK)
            for i in range(unit * _UI_CHUNK, stop):  # native data loop (not a driver loop): labels of one count run
                hp[Int((kp[i] >> shift) & UInt64(255))] += 1
        host_parallelize(count, units)
        var single = False
        for d in range(256):  # driver-loop: over the 256 digits (their totals)
            var t = 0
            for u in range(units):  # driver-loop: over the runs of one digit
                t += hist[u * 256 + d]
            totals[d] = t
            if t == n:
                single = True
        if single:
            continue  # every label has this digit: the stable pass is the identity
        var run = 0
        for d in range(256):  # driver-loop: over the digits, digit-major prefix of the run offsets
            for u in range(units):  # driver-loop: over the runs of one digit
                var c = hist[u * 256 + d]
                hist[u * 256 + d] = run
                run += c
        var in_r = cur_r
        var out_k = alt_k
        var out_r = alt_r
        def scatter(unit: Int) {imm n, imm shift, imm in_k, imm in_r, imm out_k, imm out_r, imm hist_at}:
            var kp = _U64P(unsafe_from_address=in_k)
            var rp = _U32P(unsafe_from_address=in_r)
            var ok = _U64P(unsafe_from_address=out_k)
            var orr = _U32P(unsafe_from_address=out_r)
            var hp = _INTP(unsafe_from_address=hist_at) + unit * 256
            var stop = min(n, (unit + 1) * _UI_CHUNK)
            for i in range(unit * _UI_CHUNK, stop):  # native data loop (not a driver loop): labels of one scatter run, in row order (stable)
                var d = Int((kp[i] >> shift) & UInt64(255))
                var dst = hp[d]
                hp[d] = dst + 1
                ok[dst] = kp[i]
                orr[dst] = rp[i]
        host_parallelize(scatter, units)
        alt_k = cur_k
        alt_r = cur_r
        cur_k = out_k
        cur_r = out_r
    # 3. flag, 4. exclusive scan, 5. emit
    var base = List[Int](length=units, fill=0)
    var base_at = Int(base.unsafe_ptr())
    var sk = cur_k
    var sr = cur_r
    def flags(unit: Int) {imm n, imm sk, imm base_at}:
        var kp = _U64P(unsafe_from_address=sk)
        var c = 0
        var stop = min(n, (unit + 1) * _UI_CHUNK)
        for j in range(unit * _UI_CHUNK, stop):  # native data loop (not a driver loop): sorted positions of one flag run
            if j == 0 or kp[j] != kp[j - 1]:
                c += 1
        _INTP(unsafe_from_address=base_at)[unit] = c
    host_parallelize(flags, units)
    var scan = 0
    for u in range(units):  # driver-loop: over the flag runs, exclusive prefix
        var c = base[u]
        base[u] = scan
        scan += c
    var codes_at = Int(codes)
    var classes_at = Int(classes)
    def emit(unit: Int) {imm n, imm sk, imm sr, imm base_at, imm src_at, imm codes_at, imm classes_at}:
        var kp = _U64P(unsafe_from_address=sk)
        var rp = _U32P(unsafe_from_address=sr)
        var sp = _U64P(unsafe_from_address=src_at)
        var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes_at)
        var clp = _U64P(unsafe_from_address=classes_at)
        var running = _INTP(unsafe_from_address=base_at)[unit]
        var stop = min(n, (unit + 1) * _UI_CHUNK)
        for j in range(unit * _UI_CHUNK, stop):  # native data loop (not a driver loop): sorted positions of one emit run
            var flag = 1 if j == 0 or kp[j] != kp[j - 1] else 0
            var cls = running + flag - 1
            var r = Int(rp[j])
            cp[r] = Int32(cls)
            if flag == 1:
                clp[cls] = sp[r]
            running += flag
    host_parallelize(emit, units)
    _ = key^  # the buffers above are read and written through their addresses
    _ = row^
    _ = tk^
    _ = tr^
    _ = nan_runs^
    _ = hist^
    _ = base^
    return scan
