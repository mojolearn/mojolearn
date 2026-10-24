# SPDX-License-Identifier: Apache-2.0
# Native adapter tensor descriptors and storage copies, shared by the whole-model
# training entry (adapters/native/training.mojo). No math. The per-operation
# dispatcher run_operation was removed on 2026-10-24 (docs/TOMBSTONES.md):
# framework adapters run whole models only and must not get a per-op path back.
from autograd.rankn import ShapeN, RN_MAX_RANK
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY


comptime WORDS = 3 + RN_MAX_RANK  # address, kind (0=f32, 1=i32 bits), rank, dimensions
comptime IP = MutPointer[Int, MutUntrackedOrigin]


def descriptor_shape(words: IP, offset: Int) raises -> ShapeN:
    var rank = words[offset + 2]
    if rank < 0 or rank > RN_MAX_RANK:
        raise Error("adapter: rank must be between zero and eight")
    var dims = List[Int]()
    for j in range(rank):  # small-loop(rank<=8): descriptor metadata
        dims.append(words[offset + 3 + j])
    var shape = ShapeN.of(dims)
    if words[offset] == 0 or words[offset] % 4 != 0:
        raise Error("adapter: non-null four-byte-aligned storage required")
    if words[offset + 1] != 0 and words[offset + 1] != 1:
        raise Error("adapter: storage must be float32 or int32")
    return shape


def copy_words[E: Exec](mut ex: E, src: FP, dst: FP, count: Int) raises:
    var args = Args()
    args.p0 = src
    args.p1 = dst
    ex.launch[OP_COPY](args, count)
