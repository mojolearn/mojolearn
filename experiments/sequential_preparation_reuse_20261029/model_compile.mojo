# Compilation fixture only; never executed for this change.
from autograd.model_descriptor import NativeModelRequest, model_request_execute
from autograd.model import parse_model, model_forward
from autograd.model_execution import model_float_words
from autograd.tape import Tape
from sequence.exec import HostExec
from sequence.ops import FP


@export
def sequential_model_compile(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) abi("C") -> Int:
    try:
        var request = NativeModelRequest(op, aa, na, pp, np, ff, nf)
        var ex = HostExec()
        return model_request_execute(ex, op, request)
    except:
        return -1


@export
def sequential_forward_compile(aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                               parameters: Int, gradients: Int, data: Int, rows: Int, train: Bool) abi("C") -> Int:
    try:
        var request = NativeModelRequest(2, aa, na, pp, np, ff, nf)
        var spec = parse_model(request.program, model_float_words(request.floats))
        var ex = HostExec()
        var tape = Tape()
        var x = tape.variable(ex, FP(unsafe_from_address=data), rows, spec.n_inputs, False)
        var output = model_forward(tape, ex, spec, FP(unsafe_from_address=parameters),
                                   FP(unsafe_from_address=gradients), x, train, UInt64(0), 0)
        return Int(tape.value(output))
    except:
        return -1
