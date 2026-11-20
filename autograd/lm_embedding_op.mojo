# SPDX-License-Identifier: Apache-2.0
"""The byte LM trainer's embedding backward as an executor op (lane
r1-restore-measured-winners, 2026-11-09).

    x = lm_embedding(table (vocab, width), ids (n int32 words))   OPS_LM_EMBEDDING

The forward, the id refusal and the recorded words are `OPS_EMBEDDING`'s
(autograd/embedding_ops.mojo `embedding_refuse_ids`, `embedding_forward_into`:
Y[t, j] = ftz(ftz(W[ids[t], j]))). The recorded op is one tape custom node
(`LM_EMBEDDING_TAG`) over [table] whose VJP is, on a device session, the
trainer's own embedding backward, restored from the hand step lane x7 deleted
(training/byte_lm.mojo `_byte_backward_tape` at `5b5490960d`, its
`byte_backward_autograd` embedding seam): `identical_embedding_backward_prerefused_into`
(autograd/kernels/embedding.mojo, seams E0-E4) with the trainer's plan pick,
verbatim:

    PLAN_SORT if vocab * positions >= EMB_AUTO_SORT_MIN_CELLS else PLAN_SCAN

PLAN_SORT there is the stable LSD radix sort of (id, position) pairs
(autograd/kernels/embedding_sort.mojo EMB_RADIX_SORT, I11 measured winner:
radix / bitonic NVIDIA 0.675 / 0.489, AMD 0.920 / 0.958) where the generic
tape op runs the bitonic network (log2(S) (log2(S) + 1) / 2 exchange
launches); PLAN_SCAN is the trainer's counts / block prefix / perm kernels.
The pick is a work-size rule (V * T against the embedding lane's 2^26), not a
shape.

Bits: none move. Contract 6 clause (d) (embedding/IDENTICAL_EMBEDDING_CONTRACT.md)
holds counts, run_begin, perm and dW bit-identical across both plans and both
implementations on every vendor and the host column: each (v, j) cell folds
its positions in ascending t from +0.0. The host column runs the executor's
embedding backward (autograd/embedding_ops.mojo `embedding_backward_into`,
row tasks), which is those words.

`-D MOJOLEARN_BYTE_LM_GENERIC_OPS` makes the byte LM program capture the
generic `OPS_EMBEDDING` instead (the A/B arm: tonight's capture).
"""
from max.gpu.host import DeviceBuffer
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from embedding.checks.embedding_oracle import EmbConfig
from embedding.checks.embedding_identical import EMB_AUTO_SORT_MIN_CELLS
from autograd.kernels.embedding_sort import PLAN_SCAN, PLAN_SORT
from autograd.kernels.embedding import emb_run_scratch_ints, identical_embedding_backward_prerefused_into

#: The session op code (autograd/ops_api.mojo) and the tape custom tag
#: (lane r1 reserved op codes 93-95 and custom tags 120-122).
comptime OPS_LM_EMBEDDING = 93
comptime LM_EMBEDDING_TAG = 120

comptime _F = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]


def lm_embedding_scratch_words(vocab: Int, n_positions: Int) -> Int:
    """Executor words the VJP's run scratch needs (counts vocab, run_begin
    vocab + 1, perm n), reserved with the forward."""
    return emb_run_scratch_ints(vocab, n_positions)


def lm_embedding_plan(vocab: Int, n_positions: Int) -> Int:
    """The trainer's pick, verbatim (training/byte_lm.mojo at 5b5490960d)."""
    return PLAN_SORT if vocab * n_positions >= EMB_AUTO_SORT_MIN_CELLS else PLAN_SCAN


def lm_embedding_backward_device(
    ex: DeviceExec, ids: FP, dy: FP, dw: FP, scratch: FP, n_positions: Int, vocab: Int, width: Int,
    padding_idx: Int,
) raises:
    """dw (vocab, width) = the run-sorted scatter-add of dy (n, width) over
    ids, written whole (seeded +0.0, the padding row +0.0), enqueued on the
    executor's stream. The ids passed the forward's refusal (prerefused).
    PLAN_SORT synchronizes once before it releases its key scratch, as the
    trainer's step did."""
    var ctx = ex.ctx.copy()
    var nv = vocab if vocab > 0 else 1
    var nn = n_positions if n_positions > 0 else 1
    var sa = Int(scratch)
    var ids_b = DeviceBuffer[DType.int32](ctx, _I(unsafe_from_address=Int(ids)), nn, owning=False)
    var dy_b = DeviceBuffer[DType.float32](ctx, _F(unsafe_from_address=Int(dy)), n_positions * width if n_positions * width > 0 else 1, owning=False)
    var dw_b = DeviceBuffer[DType.float32](ctx, _F(unsafe_from_address=Int(dw)), vocab * width if vocab * width > 0 else 1, owning=False)
    var counts = DeviceBuffer[DType.int32](ctx, _I(unsafe_from_address=sa), nv, owning=False)
    var run_begin = DeviceBuffer[DType.int32](ctx, _I(unsafe_from_address=sa + 4 * vocab), vocab + 1, owning=False)
    var perm = DeviceBuffer[DType.int32](ctx, _I(unsafe_from_address=sa + 4 * (2 * vocab + 1)), nn, owning=False)
    var cfg = EmbConfig(vocab, width, padding_idx, False)
    identical_embedding_backward_prerefused_into(
        ctx, dw_b, dy_b, ids_b, counts, run_begin, perm, n_positions, cfg, lm_embedding_plan(vocab, n_positions),
    )
    _ = ids_b^
    _ = dy_b^
    _ = dw_b^
    _ = counts^
    _ = run_begin^
    _ = perm^
