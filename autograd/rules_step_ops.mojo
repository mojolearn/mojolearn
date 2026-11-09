# SPDX-License-Identifier: Apache-2.0
"""The backward rules of a session program that trains through a block step
op (lane x7-byte-lm-on-executor, 2026-11-09; EXECUTOR_X1_API.md section 2).

`StepOpRules` is `ImportRules` (the rules every captured program's
backward uses: the rank-N and import families) plus the attention step's
tag, ATTN_STEP_TAG (autograd/block_step_ops.mojo): a recorded OPS_ATTN_STEP
node over [x, weights] whose VJP is the transformer stage rule over the
stages the recorded run kept in its (weights, state) entry. Every other tag
is ImportRules' own, so a program without a step op differentiates exactly
as before.

The rule reads the session's resident step entries, so the struct carries
the address of the session's `BlockStepScratch` (`OpsSession.steps`; a
session is heap-allocated by `ops_session_open` and never moves). It is
built per backward by the session's training loop (Program.fit,
autograd/program_runtime.mojo `train_program_control`) and never stored on
the tape. On a host session (lane x7b-step-ops-host-arm) the same tag runs
the executor's host arm (`Exec.step_backward`: HostExec's
`host_attn_step_backward`, autograd/block_step_host.mojo): the backward
oracle over the stages the host recorded run kept, the rule's host body.

Lane x56d-mamba-samba-train-and-host (2026-11-09): the three Mamba step ops'
tags (M1_STEP_TAG, M2_STEP_TAG, M3_STEP_TAG) on the same contract: a recorded
run of OPS_M{1,2,3}_STEP is one custom node over [x, weights], its VJP the
Mamba stage rules over the stages the recorded run kept
(`mamba_step_backward`), on the host the host arm's Mamba backward
(`Exec.step_backward`, which finds the entry by its operands).
"""
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args
from autograd.tape import BackwardRules
from autograd.import_api import ImportRules
from autograd.block_step_ops import (
    BlockStepScratch, ATTN_STEP_TAG, attn_step_backward, step_tag_code, mamba_step_backward,
)
from autograd.lm_head_v2_op import LM_HEAD_V2_TAG, lm_head_v2_backward_device
from autograd.small_mlp_op import SMALL_MLP_TAG, small_mlp_backward_device
from autograd.cnn_block_op import (
    CNN_CONV_BLOCK_TAG, CNN_LINEAR_TAG, CNN_HOST_BLOCK_BACKWARD, CNN_HOST_LINEAR_BACKWARD, cnn_block_words,
    cnn_block_backward_device, cnn_linear_backward_device,
)
# Lane r1-restore-measured-winners: the byte LM trainer's embedding backward
# (autograd/lm_embedding_op.mojo); the host arm is the executor's embedding
# backward (the same words, contract 6 clause (d)).
from autograd.lm_embedding_op import LM_EMBEDDING_TAG, lm_embedding_backward_device
from autograd.embedding_ops import AG_EMB_PLAN_AUTO, embedding_plan, embedding_perm_words, embedding_backward_into

comptime _ScratchRef = MutPointer[BlockStepScratch, MutUntrackedOrigin]


@fieldwise_init
struct StepOpRules(BackwardRules, ImplicitlyCopyable, Movable):
    """ImportRules plus the block step ops' recorded-run tags."""

    #: address of the session's BlockStepScratch (0: no step entries)
    var scratch: Int

    def supports(self, tag: Int) -> Bool:
        if step_tag_code(tag) != 0:
            return self.scratch != 0
        if tag == LM_HEAD_V2_TAG or tag == SMALL_MLP_TAG or tag == CNN_CONV_BLOCK_TAG or tag == CNN_LINEAR_TAG or tag == LM_EMBEDDING_TAG:
            return True
        return ImportRules().supports(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag == LM_HEAD_V2_TAG or tag == LM_EMBEDDING_TAG:
            # the rule writes all of d_hidden and d_weight (lm_embedding:
            # every cell of d_table, seeded +0.0, the padding row +0.0)
            return False
        if tag == SMALL_MLP_TAG:
            # every launch overwrites its whole destination (dx only when
            # x wants a gradient; otherwise nobody reads that edge)
            return False
        if tag == CNN_CONV_BLOCK_TAG or tag == CNN_LINEAR_TAG:
            # the x_cnn entries overwrite every word of dx, dW and db (the
            # hand epoch reused those buffers with no fill); a block's dx
            # left unwritten belongs to an input that needs no gradient
            return False
        if step_tag_code(tag) != 0:
            # edge 0 (x): the rule writes all of d_x; edge 1 (the packed
            # weights): only the weight pieces' slots, the gaps stay zero.
            return edge != 0
        return ImportRules().zeroed_destination(tag, saved, edge)

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag == LM_HEAD_V2_TAG or tag == SMALL_MLP_TAG or tag == CNN_CONV_BLOCK_TAG or tag == CNN_LINEAR_TAG or tag == LM_EMBEDDING_TAG:
            return False
        if step_tag_code(tag) != 0:
            return False
        return ImportRules().writes_total(tag, saved, edge)

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == ATTN_STEP_TAG:
            if self.scratch == 0 or len(gradients) != 2:
                raise Error("ops: attention_step backward needs the session's step entries")
            # saved: p0 the weights, p1 the kv state, p2 the state words
            # (the entry's key, `OpsSession._block_step`)
            comptime if E.IS_DEVICE:
                var sc = _ScratchRef(unsafe_from_address=self.scratch)
                attn_step_backward(rebind[DeviceExec](ex), sc[], saved.p0, saved.p1, saved.p2,
                                   upstream, gradients[0], gradients[1])
            else:
                # the host runs every launch before it returns: the upstream
                # gradient is final here
                ex.step_backward(saved.p0, saved.p1, saved.p2, upstream, gradients[0], gradients[1])
            return
        var code = step_tag_code(tag)
        if code != 0:
            if self.scratch == 0 or len(gradients) != 2:
                raise Error("ops: a Mamba step backward needs the session's step entries")
            # saved: p0 the weights, p1 the state (p2 the words, unused: the
            # entry is keyed by its weights and state)
            comptime if E.IS_DEVICE:
                var sc = _ScratchRef(unsafe_from_address=self.scratch)
                mamba_step_backward(rebind[DeviceExec](ex), sc[], code, saved.p0, saved.p1, upstream,
                                    gradients[0], gradients[1])
            else:
                ex.step_backward(saved.p0, saved.p1, saved.p2, upstream, gradients[0], gradients[1])
            return
        if tag == LM_HEAD_V2_TAG:
            # saved: p0 the int32 codes, p1 the row maxima, p2 the row
            # denominators; i0 rows, i1 vocab, i2 width (OpsSession._lm_head_v2)
            if len(gradients) != 2 or len(inputs) != 2:
                raise Error("ops: lm_head_v2 backward takes [hidden, weight]")
            comptime if E.IS_DEVICE:
                lm_head_v2_backward_device(rebind[DeviceExec](ex), gradients[0], gradients[1], inputs[0], inputs[1],
                                           saved.p0, saved.p1, saved.p2, upstream, saved.i0, saved.i1, saved.i2)
            else:
                # the host arm (autograd/lm_head_v2_host.mojo) through the executor
                var hb: List[FP] = [gradients[0], gradients[1], inputs[0], inputs[1], saved.p0, saved.p1, saved.p2,
                                    upstream]
                ex.lm_head_v2_host(True, hb, saved.i0, saved.i1, saved.i2)
            return
        if tag == SMALL_MLP_TAG:
            # saved: p0 act, p1 incoming, p2 dhidden, p3 the GEMM workspace;
            # i0 rows, i1 in, i2 hid, i3 out, i4 workspace floats, i5 x wants
            # a gradient (OpsSession._small_mlp); inputs [x, w1, b1, w2, b2]
            if len(gradients) != 5 or len(inputs) != 5:
                raise Error("ops: small_mlp backward takes [x, w1, b1, w2, b2]")
            var bb: List[FP] = [inputs[0], inputs[1], inputs[3], saved.p0, upstream, saved.p1, saved.p2, saved.p3,
                                gradients[0], gradients[1], gradients[2], gradients[3], gradients[4]]
            comptime if E.IS_DEVICE:
                var bd: List[Int] = [saved.i0, saved.i1, saved.i2, saved.i3, saved.i4, saved.i5]
                small_mlp_backward_device(rebind[DeviceExec](ex), bb, bd)
            else:
                # the host arm (autograd/small_mlp_host.mojo) through the executor;
                # bufs[7] (the device workspace) is unread there
                var hd: List[Int] = [saved.i0, saved.i1, saved.i2, saved.i3, saved.i4, saved.i5]
                ex.small_mlp_host(True, bb, hd)
            return
        if tag == CNN_CONV_BLOCK_TAG:
            # saved: p0 the pool winners, p1 / p2 the kept im2col matrix and
            # conv output (null: recomputed); i0..i3 N, C, H, W, i4 OC, i5 k,
            # i6 pad, i7 pool, i8 pool size, i9 dx wanted, i10 kept
            # (OpsSession._cnn_conv_block). CnnRules._block's call, verbatim.
            if len(gradients) != 3 or len(inputs) != 3:
                raise Error("ops: cnn_conv_block backward takes [x, w, b]")
            var pool = saved.i7 != 0
            var need_dx = saved.i9 != 0
            var wd = cnn_block_words(saved.i0, saved.i1, saved.i2, saved.i3, saved.i4, saved.i5, saved.i6, pool, saved.i8)
            var cprm = wd[0].copy()
            var pprm = wd[1].copy()
            comptime if E.IS_DEVICE:
                cnn_block_backward_device(inputs[0], inputs[1], inputs[2], upstream, saved.p0, cprm, pprm, pool, need_dx,
                                          gradients[0], gradients[1], gradients[2], saved.p1, saved.p2)
            else:
                var kept = saved.i10 != 0
                var hb: List[FP] = [inputs[0], inputs[1], inputs[2], upstream, saved.p0, gradients[0], gradients[1],
                                    gradients[2], saved.p1 if kept else upstream, saved.p2 if kept else upstream]
                var hw: List[Int] = [1 if pool else 0, 1 if need_dx else 0, 1 if kept else 0]
                ex.cnn_op_host(CNN_HOST_BLOCK_BACKWARD, hb, cprm, pprm, hw)
            return
        if tag == CNN_LINEAR_TAG:
            # saved: i0 rows, i1 inputs, i2 outputs (OpsSession._cnn_linear).
            # CnnRules._head's call, verbatim.
            if len(gradients) != 3 or len(inputs) != 3:
                raise Error("ops: cnn_linear backward takes [h, w, b]")
            comptime if E.IS_DEVICE:
                cnn_linear_backward_device(inputs[0], inputs[1], upstream, saved.i0, saved.i1, saved.i2,
                                           gradients[0], gradients[1], gradients[2])
            else:
                var hb: List[FP] = [inputs[0], inputs[1], upstream, gradients[0], gradients[1], gradients[2]]
                var hw: List[Int] = [saved.i0, saved.i1, saved.i2]
                ex.cnn_op_host(CNN_HOST_LINEAR_BACKWARD, hb, List[Int32](), List[Int32](), hw)
            return
        if tag == LM_EMBEDDING_TAG:
            # saved: p0 the int32 ids, p1 the run scratch; i0 positions,
            # i1 vocab, i2 width, i3 padding_idx (OpsSession._lm_embedding)
            if len(gradients) != 1:
                raise Error("ops: lm_embedding backward takes [table]")
            if Int(gradients[0]) == 0:
                return
            comptime if E.IS_DEVICE:
                lm_embedding_backward_device(rebind[DeviceExec](ex), saved.p0, upstream, gradients[0], saved.p1,
                                             saved.i0, saved.i1, saved.i2, saved.i3)
            else:
                var plan = embedding_plan(AG_EMB_PLAN_AUTO, saved.i1, saved.i0)
                var run_begin = saved.p1
                var perm = run_begin + (saved.i1 + 1)
                var keys = perm + embedding_perm_words(plan, saved.i0)
                embedding_backward_into(ex, saved.p0, upstream, gradients[0], saved.i0, saved.i1, saved.i2, saved.i3,
                                        False, plan, run_begin, perm, keys)
            return
        ImportRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
