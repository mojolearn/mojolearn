# SPDX-License-Identifier: Apache-2.0
# Same host model computations/copy order as the compiled public doors. The
# provider and legacy native doors share these owners; no Python ABI lives here.
from std.memory import bitcast
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from core.host_lanes import HostF32Ptr, host_f32_copy, host_f32_uninit
from training.byte_lm_config import ByteConfig
from training.byte_lm_host_io import byte_host_store_f32, byte_host_store_i32
from training.byte_lm_host import byte_host_logits, byte_host_logits_threaded, byte_host_next_threaded, byte_host_loss
from training.byte_lm_host_backward import byte_host_adamw, byte_host_train_step

comptime BYTE_HOST_MAX_LOGITS = 268435456


def _read_f32_rows(address: Int, n: Int) raises -> List[Float32]:
    """`read_f32` through the chunked copy: an owned list of the `n` floats at
    `address`, every element written by the copy."""
    var src = f32_ptr(address)
    if n < 0:
        raise Error("byte LM host: negative float32 copy length")
    var out = host_f32_uninit(n)
    if n > 0:
        host_f32_copy(rebind[HostF32Ptr](out.unsafe_ptr()), rebind[HostF32Ptr](src), n)
    return out^


def _write_span(address: Int, values: List[Float32]) raises:
    """Copy a host list into a caller-owned buffer at `address`.

    `raises` because `f32_ptr` refuses a null address, and a function that
    calls a raising function must say so."""
    var out = f32_ptr(address)
    if len(values) > 0:
        host_f32_copy(rebind[HostF32Ptr](out), rebind[HostF32Ptr](values.unsafe_ptr()), len(values))


def byte_host_logits_native(addresses: List[Int], dims: List[Int], cfg: ByteConfig,
                            use_threads: Int, thread_count: Int) raises -> Int:
    var batch = dims[0]
    var length = dims[1]
    if batch <= 0 or batch > 1048576 or length <= 0 or length > cfg.length:
        raise Error("byte LM host: batch in [1, 2^20] and length in [1, configured length]")
    var m = batch * length
    if m > BYTE_HOST_MAX_LOGITS // cfg.vocab_size:
        raise Error("byte LM host: logits exceed the admitted span")
    var params_addr = addresses[0]
    var ids_addr = addresses[1]
    var out_addr = addresses[2]
    var params = read_f32(params_addr, cfg.n_total())
    var ids = read_i32(ids_addr, m)
    var logits: List[Float32]
    if use_threads == 1:
        logits = byte_host_logits_threaded(params, ids, batch, length, cfg, thread_count)
    else:
        logits = byte_host_logits(params, ids, batch, length, cfg)
    byte_host_store_f32(f32_ptr(out_addr), logits)
    return len(logits)


def byte_host_next_native(addresses: List[Int], dims: List[Int], cfg: ByteConfig,
                          thread_count: Int) raises -> Int:
    var batch = dims[0]
    var length = dims[1]
    if batch <= 0 or batch > 1048576 or length <= 0 or length > cfg.length:
        raise Error("byte LM host next: batch in [1, 2^20] and length in [1, configured length]")
    var params = read_f32(addresses[0], cfg.n_total())
    var ids = read_i32(addresses[1], batch * length)
    var next = byte_host_next_threaded(params, ids, batch, length, cfg, thread_count)
    byte_host_store_i32(i32_ptr(addresses[2]), next, batch)
    return batch


def byte_host_train_native(addresses: List[Int], cfg: ByteConfig, scalars: List[Float64],
                           completed_steps: Int) raises -> Int:
    var n = cfg.n_total()
    # The three registry-sized inputs come in through the chunked copy (lane
    # neural-pass8) and go into the step OWNED; the step updates them in place.
    var params = _read_f32_rows(addresses[0], n)
    var m_state = _read_f32_rows(addresses[1], n)
    var v_state = _read_f32_rows(addresses[2], n)
    var ids = read_i32(addresses[3], cfg.batch * (cfg.length + 1))
    var opt = byte_host_adamw(
        Float32(scalars[0]), Float32(scalars[1]),
        Float32(scalars[2]), Float32(scalars[3]),
        Float32(scalars[4]),
    )
    var step = byte_host_train_step(params^, m_state^, v_state^, ids, cfg, opt,
                                   completed_steps)
    _write_span(addresses[4], step.grad)
    _write_span(addresses[5], step.param)
    _write_span(addresses[6], step.m_state)
    _write_span(addresses[7], step.v_state)
    return Int(bitcast[DType.uint32](step.loss))


def byte_host_loss_native(addresses: List[Int], cfg: ByteConfig,
                          use_threads: Int, thread_count: Int) raises -> Int:
    var params = read_f32(addresses[0], cfg.n_total())
    var ids = read_i32(addresses[1], cfg.batch * (cfg.length + 1))
    var loss = byte_host_loss(params, ids, cfg, use_threads == 1, thread_count)
    return Int(bitcast[DType.uint32](loss))
