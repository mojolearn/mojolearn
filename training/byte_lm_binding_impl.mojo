# SPDX-License-Identifier: Apache-2.0
"""Typed session state and helpers behind the byte LM GPU binding
(`bindings/_mojolearn_byte_lm.mojo`): the owned session, the opt-in context
keeper, span validation, the input-unchanged guard, the on-device bit compare,
flag and loss publication, export staging and the launch-cost probe. No
Python here; the binding unpacks its arguments and calls these."""
from core.hostptr import f32_ptr, read_f32, copy_f32, list_i32
from std.gpu import block_dim, block_idx, thread_idx
from core.device_zero import enqueue_fill
from std.ffi import _Global
from std.memory import bitcast, memcmp
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from training.byte_lm_afn import BYTE_LM_FAST_APPLE
from training.checks.optimizer_contract import OptimizerConfig
from training.byte_lm_config import ByteConfig
from training.byte_lm import ByteTrainer
from training.byte_lm_logits import ByteLogitsScratch


struct ByteLMSession(Movable, Writable):
    """Python-owned device lifetime; no global handles or retained host pointers.

    Upstream LlamaModel owns its layers across calls:
    transformers/models/llama/modeling_llama.py:334-359,402-412.
    This binding retains our existing ByteTrainer without changing its kernels.
    """
    var ctx: Optional[DeviceContext]
    var trainer: Optional[ByteTrainer]
    #: DEVIATION 2942: the logits forward's call-shaped buffers, kept across
    #: `byte_lm_session_logits` calls of the same shape. Inference-only
    #: state; the trainer never reads it. Released before the trainer.
    var logits_scratch: Optional[ByteLogitsScratch]
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.trainer = Optional[ByteTrainer]()
        self.logits_scratch = Optional[ByteLogitsScratch]()
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteLMSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("ByteLMSession")

    def __deinit__(deinit self):
        # Buffers must die before their context, and the frees they enqueue
        # must drain before the context goes (DEVIATION 2520, below).
        _ = self.logits_scratch^
        _ = self.trainer^
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        _ = self.ctx^

    def close(mut self) raises:
        if self.busy:
            raise Error("byte LM: session is busy")
        self.usable = False
        if self.ctx:
            self.ctx.value().synchronize()
        self.logits_scratch = None
        self.trainer = None
        # DEVIATION 2520: releasing the trainer enqueues its buffer frees on
        # the context's stream; destroying the context with those frees in
        # flight left the MAX runtime's allocator lock held on an RTX 4090
        # pod, and the next context's first enqueueCreateBuffer blocked in
        # pthread_mutex_lock forever (native backtrace, run 6, 2026-09-11).
        # Draining here is what the passing variant did.
        if self.ctx:
            self.ctx.value().synchronize()
        self.ctx = None


struct _ContextKeeper(Defaultable, Movable):
    """DEVIATION 2513: an OPT-IN process-lifetime DeviceContext.

    On one RTX 4090 pod every DeviceContext created after another was
    destroyed in the same process never returned from its first use, while a
    second context created while the first was still alive always did.
    When MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1 is set at a call that creates a
    context, `ensure()` creates ONE separate context here first and never
    releases it, so a live context exists whenever any later per-call or
    resident context is created. The per-call path, the resident session
    and the teardown order are unchanged; this is a diagnostic mitigation,
    OFF by default, and not a claim about the cause.

    Storage: `std.ffi._Global`, the pattern the trees and RF bindings use
    for their export registries (ET_EXPORTS, RF_EXPORTS). There is no
    module-level `var` in this tree on purpose; the _Global slot is
    runtime-owned, created once by name, and lives until process teardown.
    """
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()

    def ensure(mut self) raises:
        if not self.ctx:
            self.ctx = DeviceContext()

    def active(self) -> Bool:
        return Bool(self.ctx)


comptime BYTE_LM_CONTEXT_KEEPER = _Global[StorageType=_ContextKeeper,
    name="MojoByteLMContextKeeperIdentical", init_fn=_ContextKeeper.__init__]



def _span_cells(index: Int, shape: ByteConfig) raises -> Int:
    if index == 3 or index == 9:
        return shape.n_tensors()
    if index == 4:
        return shape.batch * (shape.length + 1)
    if index == 10:
        return 1
    return shape.n_total()


def _validate_addresses(addresses: List[Int], action: Int, shape: ByteConfig) raises:
    # All spans use validated runtime-profile lengths. Validate addition before any
    # pointer construction/dereference; reject null, misalignment and wraparound.
    for i in range(11):
        if i == 8 and action == 0:
            if addresses[i] != 0:
                raise Error("byte LM eval requires null gradient output; no gradient is computed")
            continue
        var size_bytes = _span_cells(i, shape) * 4
        if addresses[i] <= 0 or addresses[i] % 4 != 0 or addresses[i] > Int(0x7FFFFFFFFFFFFFFF) - size_bytes:
            raise Error("byte LM: null/misaligned/overflowing span at address slot " + String(i))
    # Inputs may share storage because all are copied before numerical work.
    # Every output must be disjoint from all inputs and other outputs.
    for i in range(5, 11):
        if i == 8 and action == 0:
            continue
        for j in range(i):  # small-loop(i: earlier address slots, at most 10): overlap checks between pointer spans, not data
            if j == 8 and action == 0:
                continue
            if (addresses[i] < addresses[j] + _span_cells(j, shape) * 4
                and addresses[j] < addresses[i] + _span_cells(i, shape) * 4):
                raise Error("byte LM: output overlaps another live span")


def _read_f32(address: Int, n: Int) raises -> List[Float32]:
    return read_f32(address, n)


def _require_inputs_unchanged(addr: List[Int], initial_p: List[Float32], initial_m: List[Float32],
                              initial_v: List[Float32], flags: List[Bool], ids: List[Int32],
                              shape: ByteConfig) raises:
    """The input-unchanged guard (lane py-runtime round 2; it was Python's
    before/after `tobytes` copies of every parameter-sized input each step):
    the borrowed parameter, m, v, flag and token buffers must still hold the
    words this call read from them. Three n-float memcmps against the owned
    copies, no extra copy."""
    var n = shape.n_total()
    var same = memcmp(f32_ptr(addr[0]), initial_p.unsafe_ptr(), n) == 0
    same = same and memcmp(f32_ptr(addr[1]), initial_m.unsafe_ptr(), n) == 0
    same = same and memcmp(f32_ptr(addr[2]), initial_v.unsafe_ptr(), n) == 0
    var n_ids = shape.batch * (shape.length + 1)
    var ids_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[4])
    same = same and memcmp(ids_ptr, ids.unsafe_ptr(), n_ids) == 0
    var flags_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[3])
    for i in range(shape.n_tensors()):  # small-loop(n_tensors: momentum flags, one per parameter tensor): compares per tensor flags, not data
        if (flags_ptr.unsafe_load(i) == 1) != flags[i]:
            same = False
    if not same:
        raise Error("Byte-LM native call changed an input state/token buffer")


def _write_f32(address: Int, values: List[Float32]) raises:
    copy_f32(values.unsafe_ptr(), f32_ptr(address), len(values))


def _btick(on: Bool, mut t: Int, name: String):
    """DEVIATION 2499: host-side phase timer for the binding, the same line
    shape as the transformer binding's `_btick` and the block timers
    (`timing <name> <ms> ms`), behind `MOJOLEARN_TRANSFORMER_TIMING=1`.
    No device wait here: every phase this brackets is host work, or ends
    with a `download_f32` (which waits inside) or the trainer's own waits."""
    if not on:
        return
    var now = Int(perf_counter_ns())
    print(
        "timing " + name + " " + String(Float64(now - t) / 1000000.0) + " ms"
    )
    t = now


def _bbytes(on: Bool, name: String, n_bytes: Int):
    if not on:
        return
    print("timing " + name + " " + String(n_bytes) + " bytes")


def _bits_differ_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """flag[0] = 1 when any a[i], b[i] differ in bits. Every writer stores
    the same value, so the result does not depend on thread order."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        if bitcast[DType.uint32](a.unsafe_load(i)) != bitcast[DType.uint32](b.unsafe_load(i)):
            flag.unsafe_store(0, Int32(1))


comptime _BITS_TPB = 256


def _require_same_bits(
    ctx: DeviceContext, before: List[Float32], mut after: DeviceBuffer[DType.float32], n: Int,
) raises:
    """The caller's host state against the device state it must equal, bit
    for bit, compared ON THE DEVICE: one bulk upload of `before`, one
    compare launch, one int read back (cpu3-bindings; the host walk it
    replaces downloaded `after` and compared n floats on the CPU)."""
    if len(before) != n:
        raise Error("byte LM: authoritative state length mismatch")
    if n <= 0:
        return
    var d_before = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d_before, src_ptr=before.unsafe_ptr())
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, flag, Int32(0))
    ctx.enqueue_function[_bits_differ_kernel](
        d_before.unsafe_ptr(), after.unsafe_ptr(), Int32(n), flag.unsafe_ptr(),
        grid_dim=((n + _BITS_TPB - 1) // _BITS_TPB, 1, 1), block_dim=(_BITS_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=host, src_buf=flag)
    ctx.synchronize()
    var bad = host.unsafe_ptr().unsafe_load(0)
    _ = d_before^
    _ = flag^
    _ = host^
    if bad != Int32(0):
        raise Error("byte LM: authoritative state bits mismatch")



def _require_binding_profile() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and not BYTE_LM_FAST_APPLE:
        raise Error("byte LM binding requires IDENTICAL (or FAST on Apple)")
    if String(COMPILED_VENDOR) != "cuda" and String(COMPILED_VENDOR) != "hip" and String(COMPILED_VENDOR) != "metal":
        raise Error("byte LM binding requires CUDA, HIP or Metal")


def _validate_slot_table(addresses: List[Int], cells: List[Int], n_inputs: Int) raises:
    """The slot-table form of `_validate_addresses`: slot `i` names
    `cells[i]` four-byte cells; slots `[0, n_inputs)` are inputs, the rest
    outputs. Every slot is non-null, four-aligned and cannot wrap. Inputs
    may share storage (each is copied before any device work); every output
    must be disjoint from every other slot. Validated before any pointer is
    built, as the eleven-slot form does."""
    if len(addresses) != len(cells):
        raise Error("byte LM: expected " + String(len(cells)) + " addresses")
    for i in range(len(cells)):  # small-loop(cells: the call's address slots, at most a handful): checks slot sizes, no data
        var size_bytes = cells[i] * 4
        if addresses[i] <= 0 or addresses[i] % 4 != 0 or addresses[i] > Int(0x7FFFFFFFFFFFFFFF) - size_bytes:
            raise Error("byte LM: null/misaligned/overflowing span at address slot " + String(i))
    for i in range(n_inputs, len(cells)):  # small-loop(cells: the call's address slots, at most a handful): checks output slot overlap, no data
        for j in range(len(cells)):  # small-loop(cells: the call's address slots, at most a handful): pairs of slots for the overlap check, no data
            if j == i:
                continue
            if (addresses[i] < addresses[j] + cells[j] * 4
                and addresses[j] < addresses[i] + cells[i] * 4):
                raise Error("byte LM: output overlaps another live span")



def _read_flags(address: Int, n_tensors: Int) raises -> List[Bool]:
    var flags_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    var flags = List[Bool]()
    for i in range(n_tensors):  # small-loop(n_tensors: one flag per parameter tensor): validates per tensor flags, not data
        var flag = flags_ptr.unsafe_load(i)
        if flag != 0 and flag != 1:
            raise Error("byte LM: momentum flags must be exactly 0 or 1")
        flags.append(flag == 1)
    return flags^


def _read_ids(address: Int, count: Int) -> List[Int32]:
    # lane cpu4-python: one memcpy of the caller's ids (the per-token append
    # loop is gone); the List goes up as is (`afn_upload_ids`) and the range
    # refusal is the device's (`byte_require_tokens_device`)
    var ids_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    return list_i32(ids_ptr, count)


def _write_flags(address: Int, flags: List[Bool]):
    var flags_out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(flags)):
        flags_out.unsafe_store(i, Int32(flags[i]))


def _write_loss(address: Int, loss: Float32):
    var loss_out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    loss_out.unsafe_store(0, loss)


def _nonfinite(value: Float32) -> Bool:
    return (bitcast[DType.uint32](value) & UInt32(0x7F800000)) == UInt32(0x7F800000)


def _same_optimizer_bits(a: OptimizerConfig, b: OptimizerConfig) -> Bool:
    """Bitwise equality of every optimizer scalar, signed zero included:
    the spelling `_byte_lm_run`'s resident admission uses."""
    return (a.kind == b.kind and a.nesterov == b.nesterov
        and bitcast[DType.uint32](a.lr) == bitcast[DType.uint32](b.lr)
        and bitcast[DType.uint32](a.beta1) == bitcast[DType.uint32](b.beta1)
        and bitcast[DType.uint32](a.beta2) == bitcast[DType.uint32](b.beta2)
        and bitcast[DType.uint32](a.eps) == bitcast[DType.uint32](b.eps)
        and bitcast[DType.uint32](a.weight_decay) == bitcast[DType.uint32](b.weight_decay)
        and bitcast[DType.uint32](a.momentum) == bitcast[DType.uint32](b.momentum)
        and bitcast[DType.uint32](a.dampening) == bitcast[DType.uint32](b.dampening)
        and bitcast[DType.uint32](a.max_norm) == bitcast[DType.uint32](b.max_norm))


def _require_open(session: ByteLMSession) raises:
    if session.busy:
        raise Error("byte LM: session is busy")
    if not session.trainer or not session.ctx:
        raise Error("byte LM: session is not open")
    if not session.usable:
        raise Error("byte LM: session lost; restore a retained export")


def _mark_if_lost(mut session: ByteLMSession):
    """After a failed call: a trainer whose `healthy` stayed False is a lost
    context (its rollback re-scan did not answer); the session is then
    unusable until it is closed, and `byte_lm_session_info` says so."""
    if session.trainer:
        if not session.trainer.value().healthy:
            session.usable = False


def _admit_session_scalars(session: ByteLMSession, completed: Int, cfg: OptimizerConfig,
                           flags: List[Bool], shape: ByteConfig) raises:
    """The per-step scalar admission kept from `_byte_lm_run` (design 3,
    item 3): profile, completed step, optimizer bits and the flags against
    `buf_initialized`. Scalars and `n_tensors` bools; no array."""
    if session.trainer.value().config.profile() != shape.profile():
        raise Error("byte LM: resident model shape mismatch")
    if session.trainer.value().completed_steps != completed:
        raise Error("byte LM: resident completed-step mismatch")
    if not _same_optimizer_bits(session.trainer.value().optimizer, cfg):
        raise Error("byte LM: resident optimizer mismatch")
    if len(flags) != len(session.trainer.value().buffers.buf_initialized):
        raise Error("byte LM: resident flags mismatch")
    for i in range(len(flags)):  # small-loop(flags: one momentum flag per parameter tensor): compares per tensor flags, not data
        if flags[i] != session.trainer.value().buffers.buf_initialized[i]:
            raise Error("byte LM: resident flags mismatch")


def _stage_download(ctx: DeviceContext, mut host: HostBuffer[DType.float32],
                    mut buf: DeviceBuffer[DType.float32], n: Int) raises:
    """`n` floats of a device buffer into a pinned host buffer the caller
    owns (the export staging of design section 3); waits inside."""
    if n < 1:
        return
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    ctx.synchronize()



# TOMBSTONE: byte_lm_launch_probe (dead native entry: tool-only launch-cost probe) deleted 2026-10-26 by @@SHA@@; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/byte_lm_launch_probe.patch


def _write_flags_n(address: Int, flags: List[Bool], n: Int):
    """`_write_flags` over the first `n` flags."""
    var flags_out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(n):
        flags_out.unsafe_store(i, Int32(flags[i]))
