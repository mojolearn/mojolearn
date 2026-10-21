# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the public optimizers on the CPU binding
(`_mojolearn_training_host`; lane neural-optimizers, 2026-10-21): the GPU
door's entries and words (bindings/optimizer_public_pyapi.mojo) over
training/host/optimizer_public_host.mojo. Each entry unpacks its argument
words, calls ONE typed kernel entry and writes the typed result into the
caller's result buffer (or returns one integer)."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr

from bindings.optimizer_public_words import int_words, optimizer_python_error, step_request
from training.host.optimizer_public_host import (
    host_close, host_download, host_open, host_upload, public_accumulate_host, public_clip_host,
    public_step_host,
)
from training.optimizer_public_common import F64P, zero_tensors


def optimizer_public_step_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_step(words, floats, p_addrs, p_bytes, g_addrs,
    g_bytes, out_addr) on the host. Returns 0."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 7:
            raise Error("TypeError: optimizer_public_step expects seven arguments")
        var req = step_request(args[0], args[1], args[2], args[3], args[4], args[5])
        var out = F64P(unsafe_from_address=Int(py=args[6]))
        with GILReleased(Python()):
            public_step_host(req).write(out)
        return PythonObject(0).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_public_io_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_io(words, addrs, nbytes, out_addr): a CPU install has
    no device; the query answers 0 and every transfer is refused."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 4:
            raise Error("TypeError: optimizer_public_io expects four arguments")
        var w = int_words(args[0])
        var out = F64P(unsafe_from_address=Int(py=args[3]))
        if len(w) != 10:
            raise Error("TypeError: optimizer_public_io takes 10 words")
        if w[0] != 0:
            raise Error("RuntimeError: mojolearn optimizer: this build has no device-resident parameter "
                        "entries (a CPU-only install)")
        out[0] = Float64(w[5])
        out[1] = 0.0
        return PythonObject(0).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_public_clip_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_clip(addrs, nbytes, max_norm, out_addr) on the host.
    Returns 0."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 4:
            raise Error("TypeError: optimizer_public_clip expects four arguments")
        var addrs = int_words(args[0])
        var nbytes = int_words(args[1])
        var max_norm = Float32(Float64(py=args[2]))
        var out = F64P(unsafe_from_address=Int(py=args[3]))
        with GILReleased(Python()):
            var r = public_clip_host(addrs, nbytes, max_norm)
            out[0] = Float64(r[0])
            out[1] = Float64(r[1])
        return PythonObject(0).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_public_accumulate_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_accumulate(in_addrs, nbytes, out_addrs, a, tokens) on
    the host. Returns N."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 5:
            raise Error("TypeError: optimizer_public_accumulate expects five arguments")
        var in_addrs = int_words(args[0])
        var nbytes = int_words(args[1])
        var out_addrs = int_words(args[2])
        var a = Int(py=args[3])
        var tokens = Int(py=args[4])
        var n = 0
        with GILReleased(Python()):
            n = public_accumulate_host(in_addrs, nbytes, out_addrs, a, tokens)
        return PythonObject(n).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_public_zero_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_zero(addrs, nbytes): `zero_grad`. Returns the tensor
    count."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 2:
            raise Error("TypeError: optimizer_public_zero expects two arguments")
        var addrs = int_words(args[0])
        var nbytes = int_words(args[1])
        with GILReleased(Python()):
            zero_tensors(addrs, nbytes)
        return PythonObject(len(addrs)).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_resident_open_host_binding(n_total: PythonObject) raises -> PythonObject:
    """A host slot of `n_total` floats per moment, zero filled. Returns the handle."""
    return PythonObject(host_open(Int(py=n_total)))


def optimizer_resident_close_host_binding(handle: PythonObject) raises -> PythonObject:
    """Free a host slot. Returns the handle."""
    return PythonObject(host_close(Int(py=handle)))


def optimizer_resident_download_host_binding(
    handle: PythonObject, m_addr: PythonObject, v_addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """The slot's moments to host memory. Returns `n_total`."""
    return PythonObject(host_download(Int(py=handle), Int(py=m_addr), Int(py=v_addr), Int(py=n_total)))


def optimizer_resident_upload_host_binding(
    handle: PythonObject, m_addr: PythonObject, v_addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """Host memory into the slot's moments. Returns `n_total`."""
    return PythonObject(host_upload(Int(py=handle), Int(py=m_addr), Int(py=v_addr), Int(py=n_total)))
