# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the public optimizers on the CPU binding
(`_mojolearn_training_host`; lane neural-optimizers, 2026-10-21; lane
optimizers-thin-glue): the GPU door's entries (bindings/optimizer_public_pyapi.mojo)
over training/host/optimizer_public_host.mojo, through the shared door
bindings/optimizer_public_door.mojo (`HostOptKernels`)."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr

from bindings.optimizer_public_door import (
    OptKernels, entry_accumulate, entry_clip, entry_io, entry_load_state, entry_state, entry_state_set,
    entry_step,
)
from training.host.optimizer_public_host import (
    host_close, host_download, host_open, host_upload, public_accumulate_host, public_clip_host,
    public_step_host,
)
from training.optimizer_public_common import PublicStep, PublicStepOut


struct HostOptKernels(OptKernels):
    """The CPU binding's kernel entries, each with the GIL released. A CPU
    install has no device: the io query answers 0 and every transfer is
    refused."""

    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        var out = PublicStepOut(0, -1, False, False, Float32(0.0), Float32(0.0), Float32(0.0), -1, False)
        with GILReleased(Python()):
            out = public_step_host(req)
        return out

    @staticmethod
    def io(req: PublicStep, op_code: Int, tensor: Int) raises -> Tuple[Int, Int]:
        if op_code != 0:
            raise Error("RuntimeError: mojolearn optimizer: this build has no device-resident parameter "
                        "entries (a CPU-only install)")
        return (req.handle, 0)

    @staticmethod
    def clip(addrs: List[Int], nbytes: List[Int], max_norm: Float32) raises -> Tuple[Float32, Float32]:
        var r = (Float32(0.0), Float32(0.0))
        with GILReleased(Python()):
            r = public_clip_host(addrs, nbytes, max_norm)
        return r

    @staticmethod
    def accumulate(
        in_addrs: List[Int], nbytes: List[Int], out_addrs: List[Int], a_count: Int, tokens: Int
    ) raises -> Int:
        var n = 0
        with GILReleased(Python()):
            n = public_accumulate_host(in_addrs, nbytes, out_addrs, a_count, tokens)
        return n

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        return host_download(handle, m_addr, v_addr, n)

    @staticmethod
    def upload(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        return host_upload(handle, m_addr, v_addr, n)


def optimizer_door_step_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_step(opt, grads, max_norm, op, tokens) on the host."""
    return entry_step[HostOptKernels](py_self, packed)


def optimizer_door_io_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_io(opt, op, tensors, name, tensor) on the host."""
    return entry_io[HostOptKernels](py_self, packed)


def optimizer_door_state_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_state(opt, which) on the host."""
    return entry_state[HostOptKernels](py_self, packed)


def optimizer_door_state_set_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_state_set(opt, which, value) on the host."""
    return entry_state_set[HostOptKernels](py_self, packed)


def optimizer_door_load_state_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_load_state(opt, t, ea, c1, es, c2, bi, c3) on the host."""
    return entry_load_state[HostOptKernels](py_self, packed)


def optimizer_door_clip_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_clip(grads, max_norm, norm_type, error_if_nonfinite, kwargs) on the host."""
    return entry_clip[HostOptKernels](py_self, packed)


def optimizer_door_accumulate_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_accumulate(microbatch_grads, tokens) on the host."""
    return entry_accumulate[HostOptKernels](py_self, packed)


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
