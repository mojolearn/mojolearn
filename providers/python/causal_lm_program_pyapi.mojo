# SPDX-License-Identifier: Apache-2.0
# Python door of the CausalLM decode program (lane x45c-causal-generate-program-
# decode): unpacks the shell's flat descriptor lists, keeps the model's program
# owner and calls the one native read (transformer/causal_lm_program.mojo).
# No arithmetic; the GPU bindings (_mojolearn_transformer, _mojolearn_mamba)
# register it as `causal_lm_program_read_v1`.
from std.python import Python, PythonObject
from transformer.causal_lm_program import (
    CausalLMProgram, causal_program_open, causal_program_matches, causal_program_read,
)


struct CausalProgramOwner(Movable, Writable):
    """The model's cached decode program, held by the CausalLM shell in the
    model's native fields (`_decode_program`)."""
    var program: CausalLMProgram

    def __init__(out self, var program: CausalLMProgram):
        self.program = program^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CausalProgramOwner(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _ints(v: PythonObject) raises -> List[Int]:
    var n = Int(py=v.__len__())
    var out = List[Int](capacity=n)
    for i in range(n):  # driver-loop: over descriptor words (argument unpacking, no data)
        out.append(Int(py=v[i]))
    return out^


def _floats(v: PythonObject) raises -> List[Float32]:
    var n = Int(py=v.__len__())
    var out = List[Float32](capacity=n)
    for i in range(n):  # driver-loop: over descriptor floats (argument unpacking, no data)
        out.append(Float32(Float64(py=v[i])))
    return out^


def causal_lm_program_read_py(owner: PythonObject, a: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """(owner, [wrote, forwards, bad token, alive, layer words...]): the
    model's program (reused when it serves these weights at the current
    weight epoch, else a new one) after one read."""
    var ai = _ints(a)
    var pi = _ints(p)
    var fi = _floats(f)
    var own = owner
    var reuse = False
    if own is not PythonObject(None):
        reuse = causal_program_matches(own.downcast_value_ptr[CausalProgramOwner]()[].program, ai, pi, fi)
    if not reuse:
        own = PythonObject(alloc=CausalProgramOwner(causal_program_open(ai, pi, fi)))
    var ret = causal_program_read(own.downcast_value_ptr[CausalProgramOwner]()[].program, ai, pi, fi)
    var out = Python.list()
    for i in range(len(ret)):  # driver-loop: over result words (no data)
        out.append(ret[i])
    return Python.tuple(own, out)
