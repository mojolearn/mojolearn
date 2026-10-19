# SPDX-License-Identifier: Apache-2.0
"""Epoch order and batch assembly for tape trainers, one seed for every vendor.

The epoch order is the trainers' existing device permutation (OP_MLP_PERM,
sequence/mlp.mojo): six Feistel rounds keyed by mlp_epoch_key(seed, epoch),
cycle-walked into [0, n). Every thread computes its position's row alone,
with integers only, so the order is a pure function of (seed, epoch, n) and
the launch is parallel over positions. This module launches that operation
with its own arguments rather than a copy of it, so an order built here is
the MLP's and the recurrent trainer's order bit for bit.

Batch assembly reuses OP_GATHER_ROWS (sequence/ops.mojo), one thread per
(row, column) of the batch. Row indices travel as float32 words, as they
always have in these trainers, so n is below 2^24.

Dropout lives on the tape (Tape.dropout); its stream is documented in
autograd/data_ops.mojo.
"""
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_GATHER_ROWS, OP_MLP_PERM, OP_SEQ_IOTA
from sequence.mlp import mlp_epoch_key, mlp_perm_args

# Row indices are float32 words: every integer below 2^24 is exact.
comptime DATA_MAX_ROWS = 1 << 24


def _check_rows(n: Int) raises:
    if n < 1 or n >= DATA_MAX_ROWS:
        raise Error("row count must be in [1, 2^24)")


def permutation[E: Exec](mut ex: E, seed: UInt64, epoch: Int, n: Int, dst_idx: FP) raises:
    """Write epoch `epoch`'s order of rows 0..n-1 into dst_idx (n float32 words).

    The order depends only on (seed, epoch, n): the same on every executor
    and every run, and a different order for each epoch.
    """
    _check_rows(n)
    if epoch < 0:
        raise Error("epoch must be >= 0")
    ex.launch[OP_MLP_PERM](mlp_perm_args(n, mlp_epoch_key(seed, epoch), dst_idx), n)


def epoch_order[E: Exec](
    mut ex: E, shuffle: Bool, seed: UInt64, epoch: Int, n: Int, dst_idx: FP,
) raises:
    """permutation() when shuffle is set, else rows in order (0, 1, ..., n-1)."""
    if shuffle:
        permutation(ex, seed, epoch, n, dst_idx)
        return
    _check_rows(n)
    var a = Args()
    a.p0 = dst_idx
    ex.launch[OP_SEQ_IOTA](a, n)


def gather_rows[E: Exec](
    mut ex: E, X: FP, idx: FP, start: Int, count: Int, D: Int, dst: FP,
) raises:
    """dst[r, c] = X[idx[start + r], c] for r < count, c < D (row-major, D columns).

    Assembles a batch from an epoch order: `idx` holds row numbers as
    float32 words (permutation, epoch_order), `X` the full row-major data.
    """
    if count < 1 or D < 1 or start < 0:
        raise Error("gather_rows needs count >= 1, D >= 1 and start >= 0")
    var a = Args()
    a.p0 = X
    a.p1 = idx
    a.p2 = dst
    a.i0 = count
    a.i1 = D
    a.i2 = start
    ex.launch[OP_GATHER_ROWS](a, count * D)
