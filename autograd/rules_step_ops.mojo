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
"""
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args
from autograd.tape import BackwardRules
from autograd.import_api import ImportRules
from autograd.block_step_ops import BlockStepScratch, ATTN_STEP_TAG, attn_step_backward

comptime _ScratchRef = MutPointer[BlockStepScratch, MutUntrackedOrigin]


@fieldwise_init
struct StepOpRules(BackwardRules, ImplicitlyCopyable, Movable):
    """ImportRules plus the attention step op's tag."""

    #: address of the session's BlockStepScratch (0: no step entries)
    var scratch: Int

    def supports(self, tag: Int) -> Bool:
        if tag == ATTN_STEP_TAG:
            return self.scratch != 0
        return ImportRules().supports(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag == ATTN_STEP_TAG:
            # edge 0 (x): the rule writes all of d_x; edge 1 (the packed
            # weights): only the nine piece slots, the gaps stay zero.
            return edge != 0
        return ImportRules().zeroed_destination(tag, saved, edge)

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag == ATTN_STEP_TAG:
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
        ImportRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
