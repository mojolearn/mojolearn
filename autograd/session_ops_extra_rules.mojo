# SPDX-License-Identifier: Apache-2.0
# THE BACKWARD RULES OF GATHER ALONG AN AXIS AND LOGSUMEXP (sub-lane session-ops-extra, 2026-11-25).
#
# Two custom tags, answered by LayerRules (autograd/session_layer_rules.mojo, the `# session-ops-extra` group): a
# session reaches them only after one of its ops recorded a node (OpsSession.layer_nodes, tested once per backward
# call), so no other session pays a compare for them. The tags sit in this sub-lane's opcode range (6980..6999),
# clear of every other family's tags.
#
#   SOX_GATHER_TAG    d x = the run-sorted scatter-add of the upstream gradient over the gather's flat keys: the
#                     embedding backward's fold (OP_EMB_FOLD over the runs the forward built with
#                     embedding_runs_into), one thread per cell of x, the sources of a cell added in ascending
#                     output position from +0.0, so an index gathered twice gets both terms in one fixed order on
#                     every vendor (no atomics). A cell no index names gets +0.0. The index takes none.
#   SOX_LSE_TAG       d x = g * exp(x - logsumexp(x)) along the axis (OP_SOX_LSE_VJP), 0 on a row whose result is not
#                     finite.
# Each rule writes every word of its destination once (zeroed_destination and writes_total False); the tape adds
# the destination into the input's gradient. A second-order backward refuses both tags by name
# (autograd/higher_ops.mojo).
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from autograd.tape import BackwardRules
from autograd.embedding_ops import OP_EMB_FOLD
from autograd.session_ops_extra_ops import OP_SOX_LSE_VJP
from autograd.session_ops_extra_sabotage import SOX_SABOTAGE_GATHER_GRAD, SOX_SABOTAGE_GATHER_GRAD_HOST

comptime SOX_GATHER_TAG = 6990
comptime SOX_LSE_TAG = 6991
comptime SOX_TAG_FIRST = 6990
comptime SOX_TAG_LAST = 6991


@fieldwise_init
struct SessionOpsExtraRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The VJPs of gather along an axis and logsumexp (tags 6990, 6991)."""

    @staticmethod
    def owns(tag: Int) -> Bool:
        return tag >= SOX_TAG_FIRST and tag <= SOX_TAG_LAST

    def supports(self, tag: Int) -> Bool:
        return tag >= SOX_TAG_FIRST and tag <= SOX_TAG_LAST

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == SOX_GATHER_TAG:
            # saved: p1 perm, p2 run_begin (the runs of the flat keys over x's C cells); i1 C
            var f = Args()
            f.p0 = upstream
            f.p1 = saved.p1
            f.p2 = saved.p2
            f.p3 = gradients[0]
            f.i0 = 1
            f.i1 = -1
            comptime if (SOX_SABOTAGE_GATHER_GRAD and E.IS_DEVICE) or (SOX_SABOTAGE_GATHER_GRAD_HOST and not E.IS_DEVICE):
                f.i1 = 0  # SABOTAGE: flat cell 0 of x's gradient folded as padding (+0.0)
            f.i2 = 0
            ex.launch[OP_EMB_FOLD](f, saved.i1)
            return
        if tag == SOX_LSE_TAG:
            # saved: p0 x's dense (rows, cols) values (the axis last); i0 rows, i1 cols
            var a = Args()
            a.p0 = saved.p0
            a.p1 = output
            a.p2 = upstream
            a.p3 = gradients[0]
            a.i0 = saved.i1
            ex.launch[OP_SOX_LSE_VJP](a, saved.i0 * saved.i1)
            return
        raise Error("session-ops-extra rules: unknown tag")
