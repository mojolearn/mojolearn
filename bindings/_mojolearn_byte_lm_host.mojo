# SPDX-License-Identifier: Apache-2.0
"""CPU inference binding for the byte-level decoder language model (DEVIATION 2610).

HOST ONLY. No DeviceContext, no kernel, no GPU, and nothing imported from
`training/byte_lm.mojo`, which is device code. Every entry validates the
shape and the span lengths before it dereferences an address, copies the
inputs into owned Lists, computes, and only then writes the caller's output.

The loss comes back as its IEEE-754 bit pattern, an int, so the gate compares
bytes and nothing on the Python side rounds it.
"""
from std.math import isfinite
from std.memory import bitcast
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, read_f32, read_i32
from checks.numerics import GLOBAL_NUMERIC_MODE
from training.byte_lm_config import ByteConfig
from training.byte_lm_host import (
    byte_host_logits,
    byte_host_logits_threaded,
    byte_host_loss,
    byte_host_sabotage_compiled,
)
# DEVIATION 2680. The CPU training step. This import is what puts the host
# backward pass inside the certified CPU inference artifact, so the loss gate's
# 33 of 33 and the DEVIATION 2612 sabotage catch must both be unmoved
# afterwards; either one shifting is a stop, not something to reconcile.
from training.byte_lm_host_backward import (
    byte_host_adamw,
    byte_host_train_step,
)


comptime BYTE_HOST_MAX_LOGITS = 268435456


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("byte LM host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def _host_config(shape: PythonObject) raises -> ByteConfig:
    if len(shape) != 9:
        raise Error("byte LM host: expected 9 shape integers (B,L,DM,H,KV,HD,FF,layers,vocab)")
    var values = List[Int]()
    for i in range(9):
        values.append(_index(shape[i]))
    var cfg = ByteConfig(values[0], values[1], values[2], values[3],
                         values[4], values[5], values[6], values[7], values[8])
    cfg.validate()
    return cfg^


def byte_lm_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def byte_lm_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def byte_lm_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(byte_host_sabotage_compiled())


def byte_lm_host_profile_binding(shape: PythonObject) raises -> PythonObject:
    var cfg = _host_config(shape)
    return PythonObject(cfg.profile())


def _thread_count(threads: PythonObject) raises -> Int:
    var count = _index(threads)
    if count < 0 or count > 1024:
        raise Error("byte LM host: threads must be in [0, 1024] (0: one per physical core)")
    return count


def byte_lm_host_logits_binding(addresses: PythonObject, dims: PythonObject,
                                shape: PythonObject, threaded: PythonObject,
                                threads: PythonObject) raises -> PythonObject:
    """Logits for addresses [params f32 (n_total), ids i32 (batch * length),
    out logits f32 (batch * length * vocab)] and dims [batch, length].
    `threaded` is 0 for the reference path, 1 for the threaded path of
    DEVIATIONS 2616 and 2640, which runs on at most `threads` threads (0: one
    per physical core). Returns the number of logits written."""
    var cfg = _host_config(shape)
    var use_threads = _index(threaded)
    if use_threads != 0 and use_threads != 1:
        raise Error("byte LM host: threaded must be 0 or 1")
    var thread_count = _thread_count(threads)
    if len(addresses) != 3 or len(dims) != 2:
        raise Error("byte LM host: expected 3 addresses and 2 dims")
    var batch = _index(dims[0])
    var length = _index(dims[1])
    if batch <= 0 or batch > 1048576 or length <= 0 or length > cfg.length:
        raise Error("byte LM host: batch in [1, 2^20] and length in [1, configured length]")
    var m = batch * length
    if m > BYTE_HOST_MAX_LOGITS // cfg.vocab_size:
        raise Error("byte LM host: logits exceed the admitted span")
    var params_addr = _index(addresses[0])
    var ids_addr = _index(addresses[1])
    var out_addr = _index(addresses[2])
    var params = read_f32(params_addr, cfg.n_total())
    var ids = read_i32(ids_addr, m)
    var logits: List[Float32]
    if use_threads == 1:
        logits = byte_host_logits_threaded(params, ids, batch, length, cfg, thread_count)
    else:
        logits = byte_host_logits(params, ids, batch, length, cfg)
    var out = f32_ptr(out_addr)
    for i in range(len(logits)):
        out.unsafe_store(i, logits[i])
    return PythonObject(len(logits))


def _write_span(address: Int, values: List[Float32]) raises:
    """Copy a host list into a caller-owned buffer at `address`.

    `raises` because `f32_ptr` refuses a null address, and a function that
    calls a raising function must say so."""
    var out = f32_ptr(address)
    for i in range(len(values)):
        out.unsafe_store(i, values[i])


def byte_lm_host_train_step_binding(addresses: PythonObject, shape: PythonObject,
                                    scalars: PythonObject,
                                    completed: PythonObject) raises -> PythonObject:
    """One CPU training step (DEVIATION 2680). Reference path only.

    `addresses` is eight, the first four read and the last four written:

        params f32 (n_total)        m f32 (n_total)      v f32 (n_total)
        ids i32 (batch * (length + 1))
        grad f32 (n_total)          post_p f32 (n_total)
        post_m f32 (n_total)        post_v f32 (n_total)

    `scalars` is five floats, the AdamW configuration `(lr, beta1, beta2, eps,
    weight_decay)`, spelled the one way `byte_validate_optimizer` allows: no
    clipping, no momentum, no dampening, no nesterov. `completed` is the number
    of steps ALREADY taken, so the optimizer's `t` is `completed + 1`.

    Returns the loss's IEEE-754 bits, as `byte_lm_host_loss` does, because a
    loss compared as a float is a loss compared with a tolerance.

    NO THREADED PATH. The threaded forward (DEVIATION 2640) has no backward
    twin and must not grow one by accident: a weight gradient sums over every
    row of the batch, so unlike the forward it crosses every thread boundary
    and needs a fixed cross-thread fold rather than threads accumulating as
    they finish.
    """
    var cfg = _host_config(shape)
    if len(addresses) != 8:
        raise Error("byte LM host: train step expects 8 addresses")
    if len(scalars) != 5:
        raise Error("byte LM host: train step expects 5 optimizer scalars")
    var completed_steps = _index(completed)
    if completed_steps < 0:
        raise Error("byte LM host: completed_steps must not be negative")
    var n = cfg.n_total()
    var params = read_f32(_index(addresses[0]), n)
    var m_state = read_f32(_index(addresses[1]), n)
    var v_state = read_f32(_index(addresses[2]), n)
    var ids = read_i32(_index(addresses[3]), cfg.batch * (cfg.length + 1))
    var opt = byte_host_adamw(
        Float32(Float64(py=scalars[0])), Float32(Float64(py=scalars[1])),
        Float32(Float64(py=scalars[2])), Float32(Float64(py=scalars[3])),
        Float32(Float64(py=scalars[4])),
    )
    var step = byte_host_train_step(params, m_state, v_state, ids, cfg, opt,
                                   completed_steps)
    _write_span(_index(addresses[4]), step.grad)
    _write_span(_index(addresses[5]), step.param)
    _write_span(_index(addresses[6]), step.m_state)
    _write_span(_index(addresses[7]), step.v_state)
    return PythonObject(Int(bitcast[DType.uint32](step.loss)))


def byte_lm_host_loss_binding(addresses: PythonObject, shape: PythonObject,
                              threaded: PythonObject, threads: PythonObject) raises -> PythonObject:
    """Loss for addresses [params f32 (n_total), ids i32 (batch * (length + 1))].
    `threaded` and `threads` as for logits. Returns the mean loss's IEEE-754 bits."""
    var cfg = _host_config(shape)
    if len(addresses) != 2:
        raise Error("byte LM host: expected 2 addresses")
    var use_threads = _index(threaded)
    if use_threads != 0 and use_threads != 1:
        raise Error("byte LM host: threaded must be 0 or 1")
    var thread_count = _thread_count(threads)
    var params = read_f32(_index(addresses[0]), cfg.n_total())
    var ids = read_i32(_index(addresses[1]), cfg.batch * (cfg.length + 1))
    var loss = byte_host_loss(params, ids, cfg, use_threads == 1, thread_count)
    return PythonObject(Int(bitcast[DType.uint32](loss)))


# ===========================================================================
# DEVIATION 2614. The three host helpers `python/mojolearn/_buffer.py::_native`
# resolves, mirrored from `bindings/_mojolearn.mojo` (`cast_f64_to_f32_binding`
# :659, `all_finite_f32_binding` :904, `all_finite_f64_binding` :933), so the
# Python layer works on a box where the GPU base binding is not built. Native
# either way, not a Python copy. The finiteness scan is the same one element
# at a time with the same first-failure exit; the cast is one `Float32(x)` per
# element, so its SIMD width changes no bit.
# ===========================================================================


def all_finite_f32_binding(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """1 if every one of the `n` float32 values at `addr` is finite, else 0."""
    var p = f32_ptr(Int(py=addr))
    var count = Int(py=n)
    if count < 0:
        raise Error("all_finite_f32: n must be non-negative, got " + String(count))
    var ok: Int = 1
    with GILReleased(Python()):
        for i in range(count):
            if not isfinite(p.unsafe_load(i)):
                ok = 0
                break
    return PythonObject(ok)


def all_finite_f64_binding(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`all_finite_f32_binding` over float64 values."""
    var p = f64_ptr(Int(py=addr))
    var count = Int(py=n)
    if count < 0:
        raise Error("all_finite_f64: n must be non-negative, got " + String(count))
    var ok: Int = 1
    with GILReleased(Python()):
        for i in range(count):
            if not isfinite(p.unsafe_load(i)):
                ok = 0
                break
    return PythonObject(ok)


def cast_f64_to_f32_binding(src_addr: PythonObject, dst_addr: PythonObject,
                            n: PythonObject) raises -> PythonObject:
    """Write `Float32(src[i])` to `dst[i]` for `i` in `[0, n)`. Returns 0."""
    var count = Int(py=n)
    if count < 0:
        raise Error("cast_f64_to_f32: n must be non-negative, got " + String(count))
    if count == 0:
        return PythonObject(0)
    var sp = f64_ptr(Int(py=src_addr))
    var dp = f32_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        var i = 0
        var body = count - (count % 8)
        while i < body:
            var v = sp.unsafe_load[width=8](i)
            dp.unsafe_store[width=8](i, v.cast[DType.float32]())
            i += 8
        while i < count:
            dp.unsafe_store(i, sp.unsafe_load(i).cast[DType.float32]())
            i += 1
    return PythonObject(0)


@export
def PyInit__mojolearn_byte_lm_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_byte_lm_host")
        module.def_function[byte_lm_host_numeric_mode_binding]("byte_lm_host_numeric_mode")
        module.def_function[byte_lm_host_vendor_binding]("byte_lm_host_vendor")
        module.def_function[byte_lm_host_sabotage_binding]("byte_lm_host_sabotage")
        module.def_function[byte_lm_host_profile_binding]("byte_lm_host_profile")
        module.def_function[byte_lm_host_logits_binding]("byte_lm_host_logits")
        module.def_function[byte_lm_host_loss_binding]("byte_lm_host_loss")
        module.def_function[byte_lm_host_train_step_binding]("byte_lm_host_train_step")
        module.def_function[all_finite_f32_binding]("all_finite_f32")
        module.def_function[all_finite_f64_binding]("all_finite_f64")
        module.def_function[cast_f64_to_f32_binding]("cast_f64_to_f32")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_byte_lm_host: ", error))
