# SPDX-License-Identifier: Apache-2.0
# Python door of the CausalLM decode program (lane x45c-causal-generate-program-
# decode): unpacks the shell's flat descriptor lists, keeps the model's program
# owner and calls the one native read (transformer/causal_lm_program.mojo).
# No arithmetic; the GPU bindings (_mojolearn_transformer, _mojolearn_mamba)
# register it as `causal_lm_program_read_v1`.
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from std.reflection import reflect
from bindings.py_type_registry import py_type_absent, py_type_attach
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


def causal_program_owner_type(mut module: PythonModuleBuilder) raises:
    """Register `_CausalProgramOwner` on `module`. Mojo's Python type registry
    is per interpreter, and the transformer and Mamba bindings both export the
    decode program reader: the first module to load builds the type, every
    later one attaches that same class (the bindings/native_identity_type.mojo
    pattern). Module setup only, with the GIL held."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[CausalProgramOwner]()
    except error:
        if not py_type_absent(error, String(reflect[CausalProgramOwner].name())):
            raise error
        _ = module.add_type[CausalProgramOwner]("_CausalProgramOwner")
        return
    py_type_attach(module.module, "_CausalProgramOwner", existing)
