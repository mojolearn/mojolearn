# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Apple FAST Samba op candidates (lane afn-samba, 2026-10-03) are gone.

Every switch of this file was an opt-in arm that no A/B measured; the
names below stay as False constants for their importers
(bindings/_mojolearn_training.mojo reads AFN_SAMBA_FUSE), and the two fused
entries the Python provider still names refuse in every build.
"""

# TOMBSTONE: MOJOLEARN_AFN_SAMBA_FUSE (record read: EXP:2363 DELETED, superseded), MOJOLEARN_AFN_SAMBA_ARENA, MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT, MOJOLEARN_AFN_SAMBA_EMB_ATOMIC, MOJOLEARN_AFN_SAMBA_HEAD_GEMM (MERGED-UNMEASURED opt-in arms, EXP:653-657) deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at f6652fc1ac.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_SAMBA_FUSE.patch (one patch: the arms, the arena, the device admission, the atomic embedding scatter, the head GEMMs and the standalone op bodies).
# TOMBSTONE: MOJOLEARN_AFN_SAMBA_ALL and MOJOLEARN_AFN_EPI_ALL (umbrella reads, owner another lane) deleted here 2026-11-15 by lane/expdel2-32-neural; code recoverable at f6652fc1ac.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_SAMBA_ALL.expdel2-32-neural.patch
from max.gpu.host import DeviceContext


comptime AFN_SAMBA_ALL = False
comptime AFN_SAMBA_FUSE = False
comptime AFN_SAMBA_ARENA = False
comptime AFN_SAMBA_DEVICE_ADMIT = False
comptime AFN_SAMBA_EMB_ATOMIC = False
comptime AFN_SAMBA_HEAD_GEMM = False
comptime AFN_SAMBA_OPS = False
comptime AFN_SAMBA_FLAGS = False


def afn_tail_train_host(
    ctx: DeviceContext,
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    row_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dh_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dnw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dhw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    nw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    hw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    targets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
    eps: Float32,
    ignore_index: Int,
    reduction: Int,
    num_items: Int,
    label_smoothing: Float32,
) raises -> Int:
    """Refuses: the fused Apple FAST tail is not in any build."""
    raise Error("mojolearn samba afn: fused entries are not in this build")


def afn_embedding_backward_tied_host(
    ctx: DeviceContext,
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dy_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    pair_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_positions: Int,
    vocab: Int,
    width: Int,
) raises -> Int:
    """Refuses: the fused Apple FAST tied embedding gradient is not in any build."""
    raise Error("mojolearn samba afn: fused entries are not in this build")
