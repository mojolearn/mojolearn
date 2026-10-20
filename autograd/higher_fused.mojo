# SPDX-License-Identifier: Apache-2.0
# THE FUSED TAPE OPERATIONS UNDER create_graph (STUB, lane ag-ho-core).
#
# autograd/higher.mojo lower_vjp sends the fused node kinds here: LINEAR,
# LOSS_ELEM, LOSS_REDUCED, LOSS_CE, LOSS_SCE, DROPOUT and EMBEDDING. This
# stub refuses every one with the section 2 error of
# docs/plans/AUTOGRAD_HIGHER_ORDER.md, so the refusal state exists and is
# checked before lane H2 (ag-ho-fused) replaces this file whole with the
# recorded lowerings of section 1.2 (EMBEDDING keeps the refusal).
from sequence.exec_trait import Exec
from autograd.tape import Tape, Tensor
from autograd.higher_ops import higher_refusal


def lower_fused_vjp[E: Exec](mut tape: Tape, mut ex: E, i: Int, upstream: Tensor, edge: Int) raises -> Tensor:
    """The recorded VJP of fused node i into its input `edge` for the
    upstream gradient tensor `upstream`. Refused in this version."""
    raise Error(higher_refusal(tape._nodes[i].op))
