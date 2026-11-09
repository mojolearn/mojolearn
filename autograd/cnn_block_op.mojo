# SPDX-License-Identifier: Apache-2.0
"""CNNClassifier's hand-tuned conv block and head as executor ops with
their backward rules (lane r1-restore-measured-winners, 2026-11-09).

    y      = cnn_conv_block(x (N, C, H, W), w (OC, C, k, k), b (1, OC))
             OPS_CNN_CONV_BLOCK  ints [pad, pool (0 / 1), pool size]
             = maxpool(relu(conv(x) + b)) (stride 1, dilation 1), or
               relu(conv(x) + b) without the pool
    logits = cnn_linear(h (n, d_in), w (d_out, d_in), b (1, d_out))
             OPS_CNN_LINEAR  = h w^T + b (torch's head layout, no transpose)

Lane x8 (2026-11-09) moved CNNClassifier onto one captured program and
replaced these kernels with the tape's generic rank-4 conv, ReLU, pool and
linear, without a measurement. Andrew (2026-11-09, AGENTS.md "A measured
winner is never deleted on reasoning"): the tuned kernels come back INSIDE
the one runtime path, as ops of the program. The bodies are the ones the
old hand epoch ran (x_cnn/device.mojo `conv_block_forward_into[True]` and
`linear_forward_into[True]`, autograd/kernels/cnn.mojo
`conv_block_backward_into[True]` and `linear_backward_into[True]`: the
CNN's pinned GEMM plan, im2col / col2im, the fused ReLU + max pool and its
winners), called with the arguments the deleted rules autograd/rules_cnn.mojo
(`CnnRules._block` / `_head`, recoverable at f59d913b0b) passed. The
plumbing stays tonight's: the weights are the session's resident parameters,
the epoch is Program.fit, predict is the resident session's replay.

The x_cnn entries launch on `cnn_ctx()` (x_cnn/device.mojo), the context
every x_cnn entry uses, and each ends in its own wait (as in the hand
epoch). A device session that runs these ops must be a DeviceExec over that
same context (sequence/model_device_dispatch.mojo opens the CNN sessions
with `DeviceExec(cnn_ctx())` under CNN_FUSED_OPS), so the session's
launches and the entries share one in-order queue: no cross-queue wait.

Recorded: one custom node over [x, w, b] each (CNN_CONV_BLOCK_TAG,
CNN_LINEAR_TAG, rules in autograd/rules_step_ops.mojo StepOpRules). The
entries overwrite every word of dW, db and dx (the hand path reused those
buffers with no fill), so no destination is zero-filled; the tape keeps its
plain accumulation contract (writes_total False). The block's input
gradient is computed only when x requires one (block 0's input is the data,
as `need_dx` was False there in the hand epoch). The forward saves the
im2col matrix and the conv output in tape storage within the x_cnn tape
budget (`neural_tape_budget_bytes`, per block), else the backward
recomputes them (the hand epoch's `saved` rule).

The host column runs the same blocks through the x_cnn host twins
(autograd/cnn_block_host.mojo, reached through `Exec.cnn_op_host`, so no
GPU module imports host threads); those split their element and GEMM work
across cores.

A/B: `-D MOJOLEARN_CNN_GENERIC_OPS` captures tonight's generic chain
(OPS_CONV2D, OPS_RELU, OPS_MAXPOOL2D, OPS_TRANSPOSE + OPS_LINEAR) instead;
RUN OWED on NVIDIA and AMD (board rows algos/cnn-clf fit and predict).
"""
from std.sys.compile import is_defined
from sequence.ops import FP
from x_cnn.ops import FP as XFP, IP as XIP, NN14_BOUNDED_IM2COL, neural_tape_budget_bytes
from x_cnn.shape_words import (
    conv_params, pool_params, CP_N, CP_C, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, PP_OH, PP_OW,
)
from x_cnn.device import conv_block_forward_into, linear_forward_into
from autograd.kernels.cnn import conv_block_backward_into, linear_backward_into

#: The session op codes (autograd/ops_api.mojo) and the tape custom tags
#: (lane r1 reservation: codes 90-92, tags 117-119).
comptime OPS_CNN_CONV_BLOCK = 90
comptime OPS_CNN_LINEAR = 91
comptime CNN_CONV_BLOCK_TAG = 117
comptime CNN_LINEAR_TAG = 118

#: Default: the CNN program captures the tuned ops. The define is the B arm
#: (tonight's generic tape chain) of the A/B.
comptime CNN_FUSED_OPS = not is_defined["MOJOLEARN_CNN_GENERIC_OPS"]()

#: Host-arm operation words (`Exec.cnn_op_host`).
comptime CNN_HOST_BLOCK_FORWARD = 0
comptime CNN_HOST_BLOCK_BACKWARD = 1
comptime CNN_HOST_LINEAR_FORWARD = 2
comptime CNN_HOST_LINEAR_BACKWARD = 3


@always_inline
def cnn_xp(p: FP) -> XFP:
    return XFP(unsafe_from_address=Int(p))


@always_inline
def cnn_xi(p: FP) -> XIP:
    return XIP(unsafe_from_address=Int(p))


def cnn_block_words(
    n: Int, c: Int, h: Int, w: Int, oc: Int, ks: Int, pad: Int, pool: Bool, ps: Int,
) raises -> Tuple[List[Int32], List[Int32]]:
    """The block's conv and pool launch words, as the hand classifier's
    `_conv_params` / `_pool_params` built them for CNNClassifier's layers
    (stride 1, dilation 1, one group, a bias; the pool window = stride =
    `ps`, no padding, floor mode). The pool words are empty without a pool."""
    var raw: List[Int] = [n, c, h, w, oc, ks, ks, 1, 1, pad, pad, 1, 1, 0, 0, 1, 0]
    var cprm = conv_params(raw)
    var pprm = List[Int32]()
    if pool:
        var praw: List[Int] = [
            n, oc, Int(cprm[CP_OH]), Int(cprm[CP_OW]), ps, ps, ps, ps, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
        ]
        pprm = pool_params(praw)
    return (cprm^, pprm^)


def cnn_block_out_dims(cprm: List[Int32], pprm: List[Int32], pool: Bool) -> Tuple[Int, Int]:
    """The block output's (height, width)."""
    if pool:
        return (Int(pprm[PP_OH]), Int(pprm[PP_OW]))
    return (Int(cprm[CP_OH]), Int(cprm[CP_OW]))


def cnn_block_saved_sizes(cprm: List[Int32]) -> Tuple[Int, Int]:
    """Words of the saved im2col matrix (one word under the bounded im2col,
    whose tiles the entries keep in their own workspace) and conv output."""
    var ckk = Int(cprm[CP_C]) * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var rows = Int(cprm[CP_N]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var cols = 1 if NN14_BOUNDED_IM2COL else rows * ckk
    return (cols, rows * Int(cprm[CP_OC]))


def cnn_block_keep(cprm: List[Int32]) -> Bool:
    """The hand classifier's tape rule: save the im2col matrix and the conv
    output when they fit the x_cnn tape budget (-1: unbounded)."""
    var sz = cnn_block_saved_sizes(cprm)
    var budget = neural_tape_budget_bytes()
    return budget < 0 or 4 * (sz[0] + sz[1]) <= budget


# ---------------------------------------------------------------- device
def cnn_block_forward_device(
    x: FP, w: FP, b: FP, cprm: List[Int32], pprm: List[Int32], pool: Bool, dst: FP, idx: FP,
    save_cols: FP, save_y: FP,
) raises:
    """The hand epoch's block forward (`fit_epoch_r_device`'s
    conv_block_forward_into[True] call) over session device buffers; a null
    save buffer is not saved."""
    conv_block_forward_into[True](
        cnn_xp(x), cnn_xp(w), cnn_xp(b), cprm, pprm, pool, cnn_xp(dst), cnn_xi(idx), Int(save_cols), Int(save_y),
    )


def cnn_block_backward_device(
    x: FP, w: FP, b: FP, g: FP, idx: FP, cprm: List[Int32], pprm: List[Int32], pool: Bool, need_dx: Bool,
    gx: FP, gw: FP, gb: FP, save_cols: FP, save_y: FP,
) raises:
    """`CnnRules._block`'s conv_block_backward_into[True] call, verbatim
    (without dx the entry is handed the bias gradient there, never written)."""
    conv_block_backward_into[True](
        cnn_xp(x), cnn_xp(w), cnn_xp(b), cnn_xp(g), cnn_xi(idx), cprm, pprm, pool, need_dx,
        cnn_xp(gx) if need_dx else cnn_xp(gb), cnn_xp(gw), cnn_xp(gb), Int(save_cols), Int(save_y),
    )


def cnn_linear_forward_device(x: FP, w: FP, b: FP, n: Int, d_in: Int, d_out: Int, dst: FP) raises:
    """The hand epoch's head (`linear_forward_into[True]`: the CNN's pinned
    NT GEMM and its bias kernel)."""
    linear_forward_into[True](cnn_xp(x), cnn_xp(w), cnn_xp(b), n, d_in, d_out, cnn_xp(dst))


def cnn_linear_backward_device(
    x: FP, w: FP, g: FP, n: Int, d_in: Int, d_out: Int, gx: FP, gw: FP, gb: FP,
) raises:
    """`CnnRules._head`'s linear_backward_into[True] call, verbatim."""
    linear_backward_into[True](cnn_xp(x), cnn_xp(w), cnn_xp(g), n, d_in, d_out, cnn_xp(gx), cnn_xp(gw), cnn_xp(gb))
