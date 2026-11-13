# SPDX-License-Identifier: Apache-2.0
# Same host model computations/copy order as the compiled public doors. The
# provider and legacy native doors share these owners; no Python ABI lives here.
# Lane x7b-step-ops-host-arm (2026-11-09): the training step and the training
# loss run the trainer's captured program on a host session
# (training/byte_lm_program.mojo ByteHostProgram), the algorithm of the GPU
# columns; logits and next bytes keep the host inference forward.
# Lane restore-byte-lm-host-step (2026-11-13): that program route is the B arm
# -D MOJOLEARN_BYTE_LM_PROGRAM_STEP, as on the GPU columns; the default is the
# composed host step and loss again (training/byte_lm_host_backward.mojo,
# training/byte_lm_host.mojo byte_host_loss), the words of the GPU columns'
# restored hand step (0.8.55 / 0.8.66).
from std.memory import bitcast
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from core.host_lanes import HostF32Ptr, host_f32_copy, host_f32_uninit
from training.byte_lm_config import ByteConfig, BYTE_LM_PROGRAM_STEP
from training.byte_lm_host import byte_host_loss
from training.byte_lm_host_backward import byte_host_train_step
from training.byte_lm_host_io import byte_host_store_f32, byte_host_store_i32
from training.byte_lm_host import byte_host_logits, byte_host_logits_threaded, byte_host_next_threaded
from training.checks.optimizer_contract import OPT_ADAMW, OptimizerConfig
from training.byte_lm_program import byte_program_fit, byte_program_eval
from training.byte_lm_host_program import (
    byte_host_program_open, byte_host_program_load, byte_host_program_load_params, byte_host_program_store,
    byte_host_program_workers,
)

comptime BYTE_HOST_MAX_LOGITS = 268435456


def _read_f32_rows(address: Int, n: Int) raises -> List[Float32]:
    """`read_f32` through the chunked copy: an owned list of the `n` floats at
    `address`, every element written by the copy (restored with the composed
    step from 5fff22515a)."""
    var src = f32_ptr(address)
    if n < 0:
        raise Error("byte LM host: negative float32 copy length")
    var out = host_f32_uninit(n)
    if n > 0:
        host_f32_copy(rebind[HostF32Ptr](out.unsafe_ptr()), rebind[HostF32Ptr](src), n)
    return out^


def _write_span(address: Int, values: List[Float32]) raises:
    """Copy a host list into a caller-owned buffer at `address`."""
    var dst = f32_ptr(address)
    if len(values) > 0:
        host_f32_copy(rebind[HostF32Ptr](dst), rebind[HostF32Ptr](values.unsafe_ptr()), len(values))


def byte_host_adamw(lr: Float32, beta1: Float32, beta2: Float32,
                    eps: Float32, weight_decay: Float32) -> OptimizerConfig:
    """The byte LM's optimizer, spelled the one way its own validator allows:
    positive-lr AdamW with no clipping, momentum, dampening or nesterov
    (`byte_validate_optimizer`), so the remaining fields are zero."""
    return OptimizerConfig(OPT_ADAMW, lr, beta1, beta2, eps, weight_decay,
                           Float32(0.0), Float32(0.0), False, Float32(0.0))


def byte_host_optimizer(scalars: List[Float64]) raises -> OptimizerConfig:
    """The admitted optimizer scalars (lr, beta1, beta2, eps, weight_decay
    first) as the program's AdamW."""
    if len(scalars) < 5:
        raise Error("byte LM host: the optimizer takes five scalars")
    return byte_host_adamw(Float32(scalars[0]), Float32(scalars[1]), Float32(scalars[2]), Float32(scalars[3]),
                           Float32(scalars[4]))


def byte_host_eval_optimizer() -> OptimizerConfig:
    """The optimizer a loss-only program carries (never stepped)."""
    return byte_host_adamw(Float32(1e-3), Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0.0))


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
    """One stateless training step on the CPU column: addresses = [params,
    m, v (read), ids (batch * (length + 1) int32), grad, post_p, post_m,
    post_v (written)]. Default: the composed host step. Under
    MOJOLEARN_BYTE_LM_PROGRAM_STEP: the trainer's program on a host session
    for this call (load, one Program.fit step, store). Returns the loss's
    bits."""
    if len(addresses) != 8:
        raise Error("byte LM host: train step expects eight buffers")
    for i in range(8):  # driver-loop: over the eight admitted training buffers
        if addresses[i] == 0:
            raise Error("byte LM host: null training buffer")
    if completed_steps < 0:
        raise Error("byte LM host: completed_steps must not be negative")
    comptime if not BYTE_LM_PROGRAM_STEP:
        # The composed host step (default): the parameters and moments come
        # in through the chunked copy OWNED and the step updates them in
        # place. The inputs are read before any output is written, so an
        # output buffer may be an input buffer (the session's k-step call).
        var n = cfg.n_total()
        var params = _read_f32_rows(addresses[0], n)
        var m_state = _read_f32_rows(addresses[1], n)
        var v_state = _read_f32_rows(addresses[2], n)
        var ids = read_i32(addresses[3], cfg.batch * (cfg.length + 1))
        var step = byte_host_train_step(params^, m_state^, v_state^, ids, cfg, byte_host_optimizer(scalars),
                                        completed_steps)
        _write_span(addresses[4], step.grad)
        _write_span(addresses[5], step.param)
        _write_span(addresses[6], step.m_state)
        _write_span(addresses[7], step.v_state)
        return Int(bitcast[DType.uint32](step.loss))
    var e = byte_host_program_open(cfg, byte_host_optimizer(scalars))
    byte_host_program_load(e, addresses[0], addresses[1], addresses[2], completed_steps)
    var loss = List[Float32](length=1, fill=Float32(0))
    _ = byte_program_fit(e, addresses[3], 1, Int(loss.unsafe_ptr()))
    byte_host_program_store(e, addresses[5], addresses[6], addresses[7], addresses[4])
    return Int(bitcast[DType.uint32](loss[0]))


def byte_host_loss_native(addresses: List[Int], cfg: ByteConfig, use_threads: Int, thread_count: Int) raises -> Int:
    """The training forward's mean loss on one batch (addresses = [params,
    ids (batch * (length + 1) int32)]): the composed host forward by default,
    the trainer's program on a host session (no backward) under
    MOJOLEARN_BYTE_LM_PROGRAM_STEP. `use_threads` 0 runs every launch on the calling
    thread; 1 runs on at most `thread_count` threads (0: one per physical
    core). Same bits either way. Returns the loss's bits."""
    if len(addresses) != 2 or addresses[0] == 0 or addresses[1] == 0:
        raise Error("byte LM host: the loss expects non-null parameters and ids")
    if use_threads != 0 and use_threads != 1:
        raise Error("byte LM host: threaded must be 0 or 1")
    comptime if not BYTE_LM_PROGRAM_STEP:
        # The composed host loss (default): the oracle forward on the
        # calling thread, or the threaded forward with its fast loss (same
        # bits, byte_lm_host_kernels_check `check_loss`).
        var params = read_f32(addresses[0], cfg.n_total())
        var ids = read_i32(addresses[1], cfg.batch * (cfg.length + 1))
        return Int(bitcast[DType.uint32](byte_host_loss(params, ids, cfg, use_threads == 1, thread_count)))
    var e = byte_host_program_open(cfg, byte_host_eval_optimizer())
    byte_host_program_workers(e, 1 if use_threads == 0 else thread_count)
    byte_host_program_load_params(e, addresses[0])
    var loss = byte_program_eval(e, addresses[1])
    return Int(bitcast[DType.uint32](loss))
