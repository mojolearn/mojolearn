# SPDX-License-Identifier: Apache-2.0
"""CPU-only residual-dropout forward/backward with the exact shared graph."""
from std.math import isfinite
from checks.numerics import ftz
from training.residual_dropout_contract import (
    NN59_DROPOUT_RESIDUAL,residual_dropout_admit,nn_dropout_cell,nn_dropout_residual_cell,
)
comptime _HP = MutPointer[Float32,MutUntrackedOrigin]


def residual_dropout_host[BACKWARD: Bool](first: _HP,second: _HP,x: _HP,residual: _HP,
    n: Int,offset: Int,p: Float32,seed_lo: Int,seed_hi: Int,stream: Int) raises -> Int:
    var scale = residual_dropout_admit(n,offset,p,seed_lo,seed_hi,stream)
    for i in range(n):
        if not isfinite(x[i]):
            raise Error("residual_dropout refuses nonfinite input")
        comptime if not BACKWARD:
            if not isfinite(residual[i]):
                raise Error("residual_dropout refuses nonfinite residual")
    # Match device failure behavior: produce into owned temporary output,
    # then publish only after the entire layer's finite-output admission.
    var output = List[Float32](length=n,fill=Float32(0))
    var dside = List[Float32](length=n if BACKWARD else 0,fill=Float32(0))
    comptime if NN59_DROPOUT_RESIDUAL:
        for i in range(n):
            comptime if BACKWARD:
                output[i] = nn_dropout_cell(x[i],p,scale,UInt32(seed_lo),UInt32(seed_hi),UInt32(stream),offset+i)
                dside[i] = x[i]
            else:
                output[i] = nn_dropout_residual_cell(x[i],residual[i],p,scale,UInt32(seed_lo),UInt32(seed_hi),UInt32(stream),offset+i)
    else:
        var dropped = List[Float32](length=n,fill=Float32(0))
        for i in range(n):
            dropped[i] = nn_dropout_cell(x[i],p,scale,UInt32(seed_lo),UInt32(seed_hi),UInt32(stream),offset+i)
        for i in range(n):
            comptime if BACKWARD:
                output[i] = dropped[i]
                dside[i] = x[i]
            else:
                output[i] = ftz(ftz(residual[i])+ftz(dropped[i]))
    for i in range(n):
        if not isfinite(output[i]):
            raise Error("residual_dropout produced a nonfinite output")
    for i in range(n):
        first[i] = output[i]
        comptime if BACKWARD:
            second[i] = dside[i]
    return n
