# SPDX-License-Identifier: Apache-2.0
"""Public training residual-dropout layer; GPU forward and paired backward."""
from std.python import Python,PythonObject
from std.python._cpython import GILReleased
from core.neural_context import neural_ctx
from core.hostptr import f32_ptr
from bindings.residual_dropout_common_unpack import residual_dropout_args
from training.residual_dropout_common_run import residual_dropout_addresses
from training.residual_dropout import residual_dropout_device
from training.residual_dropout_boundary_run import residual_dropout_dev_run,residual_dropout_backward_dev_run


# Lane ne3-c-blocks-embedding (row N117): the same layer on training device
# arrays (train_dev handles, training/dev_tensors.mojo). Nothing crosses the
# bus but the refusal scans' words: the inputs are read and the outputs
# written in place on the device, so a training step that keeps its
# activations resident pays no upload of x / residual and no download of the
# output. The launches are `residual_dropout_into`, the host entry's own:
# no bit moves. The handles must be distinct (the backward reads its upstream
# gradient after writing dx).
def _rd_dev_handles(handles: PythonObject) raises -> List[Int]:
    if len(handles)!=3:
        raise Error("residual_dropout_dev requires three device array handles")
    var out = List[Int]()
    for i in range(3):  # small-loop(three handles of one layer call)
        out.append(Int(py=handles[i]))
    if out[0]==out[1] or out[0]==out[2] or out[1]==out[2]:
        raise Error("residual_dropout outputs must be disjoint from inputs")
    return out^
