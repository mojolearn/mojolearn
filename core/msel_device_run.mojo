# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/msel_device.mojo` (wave 8 run split): the parallel
forest's device offset rebase (called by the GPU base image's parallel table),
the fold doors' typed address checks, and the KMeans and PCA fold fits the
doors call with the GIL released. Typed Mojo only: no Python import."""
from core.msel_device import device_rebase_offsets_i32
from core.neural_context import process_ctx
from core.msel_device_x import msel_fold_entry_context, MSEL_CONTEXT_SLOT
from decomposition.pca_fold import pca_fit_fold, PCAFoldFit
from cluster.kmeans_fold import kmeans_fit_fold, KMeansFoldFit

comptime _MSEL_SLOT = MSEL_CONTEXT_SLOT


def msel_rebase_offsets_i32_device_run(ra: Int, nr: Int, la: Int, p: Int, no: Int, d: Int) raises:
    """The parallel forest's merged int32 tree offsets, on the device. The
    GPU base image's parallel table entry (bindings/parallel_base_entry_device.mojo)
    calls it with the GIL released by its caller; the Python door
    `msel_rebase_offsets_i32` had no caller left once the parallel forest
    merge read the table (lane calltable-parallel-archive, 2026-11-11)."""
    if p < 1 or nr < 1 or no < 1 or ra == 0 or la == 0 or d == 0:
        raise Error("msel_rebase_offsets_i32: bad sizes")
    # the part lengths are k scalars (one per device shard): checked here
    var lens = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=la)
    var total = 0
    var outs = 1
    for s in range(p):  # small-loop(p: one length per device shard): shard bookkeeping, not data
        var l = Int(lens.unsafe_load(s))
        if l < 1:
            raise Error("msel_rebase_offsets_i32: every part holds its leading offset")
        total += l
        outs += l - 1
    if total != nr or outs != no:
        raise Error("msel_rebase_offsets_i32: part lengths disagree with the sizes")
    var ctx = process_ctx[_MSEL_SLOT]()
    if not device_rebase_offsets_i32(ctx, ra, nr, la, p, no, d):
        raise Error("msel_rebase_offsets_i32: a merged offset leaves int32")


def _mdx_f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr <= 0 or addr % 4 != 0:
        raise Error("pca_transform_fold: pointers must be positive and float32-aligned")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _mdx_u32_ptr(addr: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if addr <= 0 or addr % 4 != 0:
        raise Error("kmeans fold: pointers must be positive and 4-byte aligned")
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=addr)


def kmeans_fit_fold_run(
    base_id: Int, base_rows: Int, index_id: Int, ns: Int, nf: Int, nc: Int,
    cp: MutPointer[Float32, MutUntrackedOrigin], lp: MutPointer[UInt32, MutUntrackedOrigin],
    mi: Int, tl: Float64, sd: UInt64, ninit: Int, ii: Int, mm: Int, ovs: Float64,
) raises -> KMeansFoldFit:
    """`kmeans_fit_fold_binding`'s fit on this build's fold context."""
    var ctx = msel_fold_entry_context()
    return kmeans_fit_fold(ctx, base_id, base_rows, index_id, ns, nf, nc, cp, lp,
                           mi, tl, sd, ninit, ii, mm, ovs)


def pca_fit_fold_run(
    base_id: Int, base_rows: Int, index_id: Int,
    cp: MutPointer[Float32, MutUntrackedOrigin], mp: MutPointer[Float32, MutUntrackedOrigin],
    ep: MutPointer[Float32, MutUntrackedOrigin], rp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int, nf: Int, nc: Int, dense: Bool, keep_mode: Int, frac: Float64,
) raises -> PCAFoldFit:
    """`pca_fit_fold_binding`'s fit on this build's fold context."""
    var ctx = msel_fold_entry_context()
    return pca_fit_fold(ctx, base_id, base_rows, index_id, cp, mp, ep, rp, sp, nr, nf, nc, dense, keep_mode, frac)
