# SPDX-License-Identifier: Apache-2.0
"""The host arm of OPS_CNN_CONV_BLOCK and OPS_CNN_LINEAR
(autograd/cnn_block_op.mojo): the device ops' words on the CPU column. A
host module: only the host executor (sequence/exec.mojo
`HostExec.cnn_op_host`) imports it.

It calls the x_cnn host twins the hand classifier's host column ran
(x_cnn/host/ops_host.mojo `conv_block_forward_into`,
`conv_block_backward_into`, `linear_forward_into`, `linear_backward_into`,
reached through the deleted `fit_epoch_r_into`, recoverable at f59d913b0b):
the same im2col, pinned GEMM and element bodies as the device, their
element and GEMM work split across cores (`run` / `gemm_host_into`).

Buffer orders (`cnn_op_host_run`):
    CNN_HOST_BLOCK_FORWARD    [x, w, b, dst, idx, cols, y]          words [pool, keep]
    CNN_HOST_BLOCK_BACKWARD   [x, w, b, g, idx, gx, gw, gb, cols, y] words [pool, need_dx, keep]
    CNN_HOST_LINEAR_FORWARD   [x, w, b, dst]                        words [n, d_in, d_out]
    CNN_HOST_LINEAR_BACKWARD  [x, w, g, gx, gw, gb]                 words [n, d_in, d_out]
"""
from sequence.ops import FP
from x_cnn.host.ops_host import (
    conv_block_forward_into, conv_block_backward_into, linear_forward_into, linear_backward_into,
)
from autograd.cnn_block_op import (
    cnn_xp, cnn_xi,
    CNN_HOST_BLOCK_FORWARD, CNN_HOST_BLOCK_BACKWARD, CNN_HOST_LINEAR_FORWARD, CNN_HOST_LINEAR_BACKWARD,
)


def cnn_op_host_run(op: Int, bufs: List[FP], cprm: List[Int32], pprm: List[Int32], words: List[Int]) raises:
    if op == CNN_HOST_BLOCK_FORWARD:
        if len(bufs) != 7 or len(words) != 2:
            raise Error("ops: cnn_conv_block host forward takes 7 buffers and 2 words")
        conv_block_forward_into(
            cnn_xp(bufs[0]), cnn_xp(bufs[1]), cnn_xp(bufs[2]), cnn_xp(bufs[3]), cnn_xi(bufs[4]), cprm, pprm,
            words[0] != 0, words[1] != 0, cnn_xp(bufs[5]), cnn_xp(bufs[6]),
        )
        return
    if op == CNN_HOST_BLOCK_BACKWARD:
        if len(bufs) != 10 or len(words) != 3:
            raise Error("ops: cnn_conv_block host backward takes 10 buffers and 3 words")
        conv_block_backward_into(
            cnn_xp(bufs[0]), cnn_xp(bufs[1]), cnn_xp(bufs[2]), cnn_xp(bufs[3]), cnn_xi(bufs[4]),
            cnn_xp(bufs[5]), cnn_xp(bufs[6]), cnn_xp(bufs[7]), cprm, pprm, words[0] != 0, words[1] != 0,
            words[2] != 0, cnn_xp(bufs[8]), cnn_xp(bufs[9]),
        )
        return
    if op == CNN_HOST_LINEAR_FORWARD:
        if len(bufs) != 4 or len(words) != 3:
            raise Error("ops: cnn_linear host forward takes 4 buffers and 3 words")
        linear_forward_into(
            cnn_xp(bufs[0]), cnn_xp(bufs[1]), cnn_xp(bufs[2]), cnn_xp(bufs[3]), words[0], words[1], words[2],
        )
        return
    if op == CNN_HOST_LINEAR_BACKWARD:
        if len(bufs) != 6 or len(words) != 3:
            raise Error("ops: cnn_linear host backward takes 6 buffers and 3 words")
        linear_backward_into(
            cnn_xp(bufs[0]), cnn_xp(bufs[1]), cnn_xp(bufs[2]), cnn_xp(bufs[3]), cnn_xp(bufs[4]), cnn_xp(bufs[5]),
            words[0], words[1], words[2],
        )
        return
    raise Error("ops: unknown cnn host operation")
