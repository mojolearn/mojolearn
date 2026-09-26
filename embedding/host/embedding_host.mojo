# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The Embedding layer's forward gather and backward fold on the host, for a
box with no GPU (lane/cpu-training-embedding-ivf, 2026-09-15; the embedding
and embedding-sort lanes of tools/identity_break.py).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
and no GPU binding imports this file. The profile is
`mojolearn.identical.embedding.fp32.v1`; the specification is
`embedding/IDENTICAL_EMBEDDING_CONTRACT.md` and the normative host answer is
`embedding/checks/embedding_oracle.mojo`. This file is NOT that oracle. It
restates, statement for statement, what the DEVICE entry the GPU binding
calls launches (`embedding/checks/embedding_identical.mojo`,
`identical_embedding_forward_into` and `_emb_backward_launch`, and
`embedding/checks/embedding_sort.mojo` for PLAN_SORT), because the cells a
CPU column is diffed against are the device's bits and the device and the
oracle are two spellings the contract says agree, not one spelling.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  `host_embedding_forward`   `emb_gather_kernel`: one cell per thread,
                             `Y[t, j] = ftz(ftz(W[ids[t] * d + j]))`. No
                             launch when `d < 1` or `T < 1` (the launch's
                             early return), so nothing is written.
  `host_embedding_backward`  `_emb_backward_launch`, in its launch order:
                             no launch at all when `V < 1` or `d < 1`; the
                             seed kernel's `+0.0` STORED in every cell
                             unless the call carries (7.4: a carried cell
                             keeps its bits, no flush, exactly as the
                             device leaves an untouched buffer); with no
                             positions, the pad row alone; the run structure
                             (below); the fold kernel, one cell per thread,
                             `acc` read from the buffer and walked ascending
                             over its run `acc = ftz(ftz(acc) + ftz(dy))`,
                             `ftz(acc)` stored, and a cell whose run is
                             empty left untouched; then the pad row kernel
                             storing `+0.0` over `padding_idx`'s row.
  PLAN_SCAN                  `emb_counts_kernel` (the padding row counts
                             zero), `emb_run_begin_kernel` (the exclusive
                             prefix sum, V + 1 entries), `emb_perm_kernel`
                             (each run's positions in ascending `t`).
  PLAN_SORT                  `embedding_sort_runs`: `size` the smallest power
                             of two at or above `T` (1 for `T` = 1), the
                             64-bit key `(id << 32) | position` with the
                             padding positions and the slack at the
                             all-ones sentinel (`_pack`), the bitonic
                             compare-exchange network over `span` and
                             `stride` (`_exchange`; every pass's pairs are
                             disjoint, so a serial walk of one pass is that
                             pass's parallel result), the run boundaries by
                             lower bound on `v << 32` (`_runs`), and the
                             positions decoded from the sorted keys
                             (`_perm`). The contract's clause (d) holds its
                             counts, begins and permutation equal to
                             PLAN_SCAN's; this spelling does not assume it.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` makes the pad row kernel
store `-0.0` instead of `+0.0` (the device file's own
`MOJOLEARN_EMB_SABOTAGE_PAD_ROW_NEG_ZERO` arm, which the 2026-09-14 exposure
record reads DIVERGENT on nine of nine `embedding` train cells). A carried
microbatch and a sorted plan store the same `-0.0`, so the lanes' carry and
plan checks still pass and the cell is a wrong answer, not a refusal. Read
back by `embedding_host_sabotage`.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the embedding and embedding-sort lanes is the
measurement.
"""
from std.sys.compile import is_defined

from checks.numerics import ftz
from embedding.checks.embedding_oracle import EmbConfig


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime EMBEDDING_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `embedding/checks/embedding_sort.mojo`'s plan codes and sentinel, restated
#: by value because that file imports `max.gpu`; the host binding's
#: `embedding_backward` refuses any other plan code in the GPU binding's words.
comptime HOST_PLAN_SCAN = 0
comptime HOST_PLAN_SORT = 1
comptime HOST_SORT_SENTINEL = UInt64(0xFFFFFFFFFFFFFFFF)


def host_embedding_forward(
    weight: List[Float32],
    ids: List[Int32],
    n_positions: Int,
    cfg: EmbConfig,
    yp: MutPointer[Float32, MutUntrackedOrigin],
):
    """`emb_gather_kernel` over every cell (module docstring). The caller has
    refused the shape, the ids and a nonfinite `W`, as the GPU binding does
    before its upload."""
    var width = cfg.width
    if width < 1 or n_positions < 1:
        return
    for t in range(n_positions):
        var v = Int(ids[t])
        comptime if EMBEDDING_HOST_SABOTAGE:
            # THE FORWARD SABOTAGE ARM (lane/inference-embedding-ivf-cholesky,
            # 2026-09-15): gather the NEXT row, wrong on purpose, so a saved
            # table's CPU lookup (the shipped embedding_infer binding) is
            # seen to fail its recorded bytes.
            v = (v + 1) % cfg.vocab
        for j in range(width):
            yp.unsafe_store(t * width + j, ftz(ftz(weight[v * width + j])))


def _host_runs_scan(
    ids: List[Int32],
    n_positions: Int,
    cfg: EmbConfig,
    mut counts: List[Int32],
    mut run_begin: List[Int32],
    mut perm: List[Int32],
):
    """PLAN_SCAN's three kernels (module docstring)."""
    var vocab = cfg.vocab
    var pad = cfg.padding_idx
    for v in range(vocab):
        if v == pad:
            counts[v] = Int32(0)
            continue
        var c = Int32(0)
        for t in range(n_positions):
            if Int(ids[t]) == v:
                c = c + Int32(1)
        counts[v] = c
    var acc = Int32(0)
    for v in range(vocab):
        run_begin[v] = acc
        acc = acc + counts[v]
    run_begin[vocab] = acc
    for v in range(vocab):
        if v == pad:
            continue
        var w = Int(run_begin[v])
        var hi = Int(run_begin[v + 1])
        if w >= hi:
            continue
        for t in range(n_positions):
            if Int(ids[t]) == v:
                perm[w] = Int32(t)
                w += 1


def _host_lower(keys: List[UInt64], size: Int, needle: UInt64) -> Int:
    """`_lower`, the lower bound over the sorted keys."""
    var lo = 0
    var hi = size
    while lo < hi:
        var mid = (lo + hi) // 2
        if keys[mid] < needle:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _host_runs_sort(
    ids: List[Int32],
    n_positions: Int,
    cfg: EmbConfig,
    mut counts: List[Int32],
    mut run_begin: List[Int32],
    mut perm: List[Int32],
):
    """`embedding_sort_runs` (module docstring)."""
    var n = n_positions
    var vocab = cfg.vocab
    var size = 1
    while size < n:
        size *= 2
    var keys = List[UInt64](length=size, fill=HOST_SORT_SENTINEL)
    for i in range(size):
        var key = HOST_SORT_SENTINEL
        if i < n:
            var v = Int(ids[i])
            if v != cfg.padding_idx:
                key = ((UInt64(v) & UInt64(0xFFFFFFFF)) << 32) | (UInt64(i) & UInt64(0xFFFFFFFF))
        keys[i] = key
    var span = 2
    while span <= size:
        var stride = span // 2
        while stride > 0:
            for i in range(size):
                var j = i ^ stride
                if j <= i or i >= size:
                    continue
                var a = keys[i]
                var b = keys[j]
                if (a > b) == ((i & span) == 0):
                    keys[i] = b
                    keys[j] = a
            stride //= 2
        span *= 2
    for v in range(vocab + 1):
        var lo = _host_lower(keys, size, UInt64(v) << 32)
        run_begin[v] = Int32(lo)
        if v < vocab:
            var hi = _host_lower(keys, size, UInt64(v + 1) << 32)
            counts[v] = Int32(hi - lo)
    for i in range(size):
        var key = keys[i]
        if key != HOST_SORT_SENTINEL:
            perm[i] = Int32(Int(key & UInt64(0xFFFFFFFF)))


def _host_pad_row(mut dw: List[Float32], cfg: EmbConfig):
    """`emb_pad_row_kernel`: `+0.0` STORED over `padding_idx`'s row."""
    var base = cfg.padding_idx * cfg.width
    for j in range(cfg.width):
        comptime if EMBEDDING_HOST_SABOTAGE:
            # THE SABOTAGE ARM: `-0.0`, wrong on purpose; see
            # EMBEDDING_HOST_SABOTAGE.
            dw[base + j] = Float32(-0.0)
        else:
            dw[base + j] = Float32(0.0)


def host_embedding_backward(
    mut dw: List[Float32],
    dy: List[Float32],
    ids: List[Int32],
    n_positions: Int,
    cfg: EmbConfig,
    plan: Int,
) raises:
    """`_emb_backward_launch` (module docstring). `dw` is the `V * d` buffer
    the GPU binding uploads: the carried gradient when `cfg.accumulate`, any
    contents otherwise (the seed stores `+0.0`). The caller has refused the
    plan code, the shape, the ids and every nonfinite input."""
    if plan != HOST_PLAN_SCAN and plan != HOST_PLAN_SORT:
        raise Error("embedding: unknown execution plan")
    var vocab = cfg.vocab
    var width = cfg.width
    if vocab < 1 or width < 1:
        return
    var cells = vocab * width
    if not cfg.accumulate:
        for i in range(cells):
            dw[i] = Float32(0.0)
    if n_positions < 1:
        if cfg.has_padding():
            _host_pad_row(dw, cfg)
        return
    var counts = List[Int32](length=vocab, fill=Int32(0))
    var run_begin = List[Int32](length=vocab + 1, fill=Int32(0))
    var perm = List[Int32](length=n_positions, fill=Int32(0))
    if plan == HOST_PLAN_SORT:
        _host_runs_sort(ids, n_positions, cfg, counts, run_begin, perm)
    else:
        _host_runs_scan(ids, n_positions, cfg, counts, run_begin, perm)
    for cell in range(cells):
        var v = cell // width
        var j = cell - v * width
        var lo = Int(run_begin[v])
        var hi = Int(run_begin[v + 1])
        if lo >= hi:
            continue
        var acc = dw[cell]
        for r in range(lo, hi):
            var t = Int(perm[r])
            acc = ftz(ftz(acc) + ftz(dy[t * width + j]))
        dw[cell] = ftz(acc)
    if cfg.has_padding():
        _host_pad_row(dw, cfg)
