# SPDX-License-Identifier: Apache-2.0
"""`-D MOJOLEARN_IDN_TRAIN_ONE_WAIT` (lane decode-resident, 2026-10-28), the
IDENTICAL byte-LM resident train step's deferred checks. Default OFF; a build
without the define never reaches this module's functions.

The synchronous step reads, on the host, in order: the two token-range
scans (two waits), each decoder layer's call refusal (a wait per layer),
the per-layer completion waits (`_byte_layer_sync`), the loss, the gradient
scan, and the four post-update state scans. Under the define the token
range is ONE wait (it stays in front of the embedding gather: an
out-of-range id must never be gathered), every per-layer refusal goes into
a `LlamaDeferredChecks` window, the per-layer waits are dropped (every
launch is on the one in-order context; the eager-scratch release enqueues
its frees, DEVIATION 2520), and the loss, gradient, parameter, moment and
negative-moment checks are ONE status launch (`afn_step_status_kernel`, the
same predicates by bits) read back with the window and the loss at the
step's last wait. When every check is clean the step's bytes are the
synchronous step's: the same kernels ran in the same order on the same
inputs. When any is not, the caller rolls the update back (the shadow the
update itself takes) and replays the synchronous step, which raises
today's message and recovers as today. No arithmetic moves: no bit moves.

What still waits inside the step: the fused attention forward's regime scan
and corner flag (kept on the host so the backward reuses the forward scan,
IDN_ATTN_BWD_SCAN_REUSE), the autograd tape's own wait and the fused
backward's regime reads, and the step-glue update's wait before its handle
swaps (NVIDIA and AMD shipped arms)."""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from core.device_scan import (
    NONFINITE_NONE,
    SCAN_BLOCKS,
    SCAN_TPB,
    _scan_blocks,
    min_partials_kernel,
    token_oob_partial_kernel,
)
from core.step_phase import (
    step_count_d2h,
    step_count_device_alloc,
    step_count_host_alloc,
    step_count_launch,
    step_count_sync,
)
from transformer.impl.llama.fused_attention import IDN_TRAIN_ONE_WAIT, LlamaDeferredChecks
from training.byte_lm_afn import (
    AFN_SCAN_MAX_BLOCKS,
    AFN_ST_CELLS,
    AFN_TPB,
    afn_reset_kernel,
    afn_step_status_kernel,
)

comptime _OW_TOKEN_WORDS = 2
comptime _OW_HOST_WORDS = _OW_TOKEN_WORDS + AFN_ST_CELLS + 1
"""Pinned words: the two token-range answers, the status cells, the loss bits."""


struct ByteOneWait(Movable):
    """A trainer's one-wait scratch, allocated on its first deferred step."""

    var window: LlamaDeferredChecks
    var tok_part: DeviceBuffer[DType.int32]
    var tok_res: DeviceBuffer[DType.int32]
    var status: DeviceBuffer[DType.int32]
    var host: HostBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        self.window = LlamaDeferredChecks(ctx, True, False)
        step_count_device_alloc()
        self.tok_part = ctx.enqueue_create_buffer[DType.int32](_OW_TOKEN_WORDS * SCAN_BLOCKS)
        step_count_device_alloc()
        self.tok_res = ctx.enqueue_create_buffer[DType.int32](_OW_TOKEN_WORDS)
        step_count_device_alloc()
        self.status = ctx.enqueue_create_buffer[DType.int32](AFN_ST_CELLS)
        step_count_host_alloc()
        self.host = ctx.enqueue_create_host_buffer[DType.int32](_OW_HOST_WORDS)


def one_wait_require_tokens(
    ctx: DeviceContext,
    mut ow: ByteOneWait,
    mut ids_dev: DeviceBuffer[DType.int32],
    mut targets_dev: DeviceBuffer[DType.int32],
    m: Int,
    vocab: Int,
) raises:
    """`byte_require_tokens_device` with ONE wait: the two
    `token_oob_partial_kernel` scans at their shipped grids, each folded on
    the device by `min_partials_kernel`, both answers home in one copy. Same
    message. The wait also completes the uploads queued before it."""
    if m < 0 or m > Int(NONFINITE_NONE):
        raise Error("byte LM: token count outside Int32 index range")
    var blocks = 0
    if m > 0:
        blocks = _scan_blocks(m)
        step_count_launch()
        ctx.enqueue_function[token_oob_partial_kernel](
            ow.tok_part.unsafe_ptr(), ids_dev.unsafe_ptr(), Int32(m), Int32(vocab),
            grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
        )
        step_count_launch()
        ctx.enqueue_function[token_oob_partial_kernel](
            ow.tok_part.unsafe_ptr() + SCAN_BLOCKS, targets_dev.unsafe_ptr(), Int32(m), Int32(vocab),
            grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
        )
    step_count_launch()
    ctx.enqueue_function[min_partials_kernel](
        ow.tok_res.unsafe_ptr(), ow.tok_part.unsafe_ptr(), Int32(blocks),
        grid_dim=(1, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    step_count_launch()
    ctx.enqueue_function[min_partials_kernel](
        ow.tok_res.unsafe_ptr() + 1, ow.tok_part.unsafe_ptr() + SCAN_BLOCKS, Int32(blocks),
        grid_dim=(1, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=ow.host.unsafe_ptr(), src_buf=ow.tok_res)
    step_count_sync()
    ctx.synchronize()
    var hp = ow.host.unsafe_ptr()
    if hp.unsafe_load(0) != NONFINITE_NONE or hp.unsafe_load(1) != NONFINITE_NONE:
        raise Error("byte LM: token ID outside configured vocabulary")


def one_wait_step_finish(
    ctx: DeviceContext,
    mut ow: ByteOneWait,
    mut loss: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut param: DeviceBuffer[DType.float32],
    mut m: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Tuple[Bool, Float32]:
    """The step's ONE wait: the status launch (loss, gradients, parameters,
    first and second moments finite; no negative second moment), its cells,
    the loss and the deferred refusal window home in one wait. Returns
    (clean, loss); `clean` is False when any check the synchronous step
    makes would have refused."""
    if n < 0 or n > Int(NONFINITE_NONE):
        raise Error("byte LM: state length outside Int32 index range")
    step_count_launch()
    ctx.enqueue_function[afn_reset_kernel](
        ow.status.unsafe_ptr(), loss.unsafe_ptr(), Int32(1), Int32(0),
        grid_dim=(1, 1, 1), block_dim=(AFN_ST_CELLS, 1, 1),
    )
    var blocks = (n + AFN_TPB - 1) // AFN_TPB
    if blocks > AFN_SCAN_MAX_BLOCKS:
        blocks = AFN_SCAN_MAX_BLOCKS
    if blocks < 1:
        blocks = 1
    step_count_launch()
    ctx.enqueue_function[afn_step_status_kernel](
        ow.status.unsafe_ptr(), loss.unsafe_ptr(), grad.unsafe_ptr(),
        param.unsafe_ptr(), m.unsafe_ptr(), v.unsafe_ptr(), Int32(n),
        grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
    )
    var hp = ow.host.unsafe_ptr()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=hp + _OW_TOKEN_WORDS, src_buf=ow.status)
    var loss_cell = loss.create_sub_buffer[DType.float32](0, 1)
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=hp.bitcast[Float32]() + _OW_TOKEN_WORDS + AFN_ST_CELLS, src_buf=loss_cell)
    ow.window.enqueue_readback(ctx)
    step_count_sync()
    ctx.synchronize()
    _ = loss_cell^
    var value = hp.bitcast[Float32]().unsafe_load(_OW_TOKEN_WORDS + AFN_ST_CELLS)
    var clean = ow.window.clean_after_wait()
    for c in range(AFN_ST_CELLS):  # small-loop(AFN_ST_CELLS: status cells, a constant eight): one first-index word per predicate
        if hp.unsafe_load(_OW_TOKEN_WORDS + c) != NONFINITE_NONE:
            clean = False
    return (clean, value)
