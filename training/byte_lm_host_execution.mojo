# SPDX-License-Identifier: Apache-2.0
# Same host model computations/copy order as the compiled public doors. The
# provider and legacy native doors share these owners; no Python ABI lives here.
# Lane x7b-step-ops-host-arm (2026-11-09): the training step and the training
# loss run the trainer's captured program on a host session
# (training/byte_lm_program.mojo ByteHostProgram), the algorithm of the GPU
# columns; logits and next bytes keep the host inference forward.
from std.memory import bitcast
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from training.byte_lm_config import ByteConfig
from training.byte_lm_host_io import byte_host_store_f32, byte_host_store_i32
from training.byte_lm_host import byte_host_logits, byte_host_logits_threaded, byte_host_next_threaded
from training.checks.optimizer_contract import OPT_ADAMW, OptimizerConfig
from training.byte_lm_program import (
    ByteHostProgram, byte_host_program_open, byte_host_program_load, byte_host_program_load_params,
    byte_host_program_store, byte_program_fit, byte_program_eval,
)

comptime BYTE_HOST_MAX_LOGITS = 268435456


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
    post_v (written)]. The trainer's program on a host session for this
    call: load, one Program.fit step, store. Returns the loss's bits."""
    if len(addresses) != 8:
        raise Error("byte LM host: train step expects eight buffers")
    for i in range(8):  # driver-loop: over the eight admitted training buffers
        if addresses[i] == 0:
            raise Error("byte LM host: null training buffer")
    if completed_steps < 0:
        raise Error("byte LM host: completed_steps must not be negative")
    var e = byte_host_program_open(cfg, byte_host_optimizer(scalars))
    byte_host_program_load(e, addresses[0], addresses[1], addresses[2], completed_steps)
    var loss = List[Float32](length=1, fill=Float32(0))
    _ = byte_program_fit(e, addresses[3], 1, Int(loss.unsafe_ptr()))
    byte_host_program_store(e, addresses[5], addresses[6], addresses[7], addresses[4])
    return Int(bitcast[DType.uint32](loss[0]))


def byte_host_loss_native(addresses: List[Int], cfg: ByteConfig) raises -> Int:
    """The training forward's mean loss on one batch (addresses = [params,
    ids (batch * (length + 1) int32)]): the trainer's program on a host
    session, no backward. Returns the loss's bits."""
    if len(addresses) != 2 or addresses[0] == 0 or addresses[1] == 0:
        raise Error("byte LM host: the loss expects non-null parameters and ids")
    var e = byte_host_program_open(cfg, byte_host_eval_optimizer())
    byte_host_program_load_params(e, addresses[0])
    var loss = byte_program_eval(e, addresses[1])
    return Int(bitcast[DType.uint32](loss))
