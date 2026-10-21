# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The argument words of the public optimizers' entries, shared by the GPU
door (bindings/optimizer_public_pyapi.mojo, where the words are listed) and
the CPU door (bindings/optimizer_public_host_pyapi.mojo): unpacking only, no
device code (lane neural-optimizers, 2026-10-21)."""
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception

from sequence.lr_table import LrSpec
from training.optimizer_public_common import OptHyper, PublicStep


def optimizer_python_error(error: Error) -> PyObjectPtr:
    """Raise the Python class a kernel error names (`ValueError: ...`), once
    without its prefix; any other error raises `Exception`."""
    var message = String(error)
    if message.startswith("ValueError: "):
        return raise_python_exception(Error(message[byte=12:]), ExceptionType.ValueError)
    if message.startswith("TypeError: "):
        return raise_python_exception(Error(message[byte=11:]), ExceptionType.TypeError)
    if message.startswith("RuntimeError: "):
        return raise_python_exception(Error(message[byte=14:]), ExceptionType("PyExc_RuntimeError"))
    return raise_python_exception(error)


def int_words(words: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(words)):  # driver-loop: over argument words (a fixed count, or one per tensor)
        out.append(Int(py=words[i]))
    return out^


def float_words(words: PythonObject) raises -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(words)):  # driver-loop: over argument words (a fixed count)
        out.append(Float64(py=words[i]))
    return out^


def step_request(
    words: PythonObject, floats: PythonObject, p_addrs: PythonObject, p_bytes: PythonObject,
    g_addrs: PythonObject, g_bytes: PythonObject,
) raises -> PublicStep:
    """The step words (see the module docstring) as the kernel's request."""
    var w = int_words(words)
    var f = float_words(floats)
    if len(w) != 23 or len(f) != 12:
        raise Error("TypeError: optimizer_public_step takes 23 words and 12 floats")
    var req = PublicStep()
    req.j = w[0]
    req.t = w[1]
    req.hyper = OptHyper(w[2], w[3], w[4] != 0, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7])
    req.has_sched = w[5] >= 0
    req.sched = LrSpec(w[5], w[6], w[7], w[8], f[8], f[9], f[10], f[11])
    req.n_total = w[9]
    req.offsets = w[10]
    req.init = w[11]
    req.g_mode = w[12]
    req.a_count = w[13]
    req.tokens = w[14]
    req.handle = w[15]
    req.resident_wanted = w[16]
    req.pristine = w[17] != 0
    req.m_addr = w[18]
    req.v_addr = w[19]
    req.p_dev = w[20] != 0
    req.g_dev = w[21] != 0
    req.device_op = w[22] != 0
    req.p_addr = int_words(p_addrs)
    req.p_bytes = int_words(p_bytes)
    req.g_addr = int_words(g_addrs)
    req.g_bytes = int_words(g_bytes)
    if req.offsets == 0 or req.init == 0:
        raise Error("TypeError: optimizer_public_step: null registry buffer")
    if len(req.p_addr) != len(req.p_bytes) or len(req.g_addr) != len(req.g_bytes):
        raise Error("TypeError: optimizer_public_step: an address list and its size list differ in length")
    return req^
