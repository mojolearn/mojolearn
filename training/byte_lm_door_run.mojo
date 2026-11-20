# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of the byte-LM door (bindings/_mojolearn_byte_lm.mojo): the
private schedule descriptor table (session value, operations, C entry) and
the typed bodies of the replica, layer-pool and offload step/export/open
doors. Typed Mojo values and addresses only, no Python import
(tools/python_free_dirs.sh). The door unpacks the Python arguments (session
value, address lists, scalars), calls these with the GIL held exactly as
before (the GIL is also the native objects' exclusion lock), and packs the
result (lane rs-embedding-bytelm, 2026-11-12). The bodies moved unchanged,
so the bits are unchanged."""
from std.memory import bitcast
from core.hostptr import f32_ptr
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from checks.numerics import GLOBAL_NUMERIC_MODE
from training.byte_lm_binding_impl import (
    _require_binding_profile, _validate_slot_table,
    _read_f32, _read_flags, _read_ids, _write_flags,
)
from training.checks.optimizer_contract import OptimizerConfig
from training.checks.train_loop import download_f32_into
from training.byte_lm_config import ByteConfig
from training.byte_lm import byte_validate_state, byte_validate_optimizer
from training.byte_lm_model_pool import ByteModelPool
from training.byte_lm_offload import ByteOffloadedReplay
from training.byte_lm_parallel import ByteParallelTrainer


# ---------------------------------------------------------------------------
# Typed bodies of the session doors (split from the MIXED doors).
# ---------------------------------------------------------------------------


def byte_step_shards(addr: List[Int], count: Int, shape: ByteConfig) raises -> List[List[Int32]]:
    """The ids of each logical shard of one step: the slot table checked,
    then one memcpy per shard (the device checks the ids)."""
    var n_ids = shape.batch * (shape.length + 1)
    var cells = List[Int]()
    for i in range(count):  # driver-loop: over count: logical shards of the step: per shard slot sizes, no data
        cells.append(n_ids)
    _validate_slot_table(addr, cells, count)
    var shards = List[List[Int32]]()
    for i in range(count):  # driver-loop: over count: logical shards of the step: one memcpy of each shard's ids, the device checks them
        shards.append(_read_ids(addr[i], n_ids))
    return shards^


struct ByteOpenState(Movable):
    """The validated host state an `open` uploads: parameters, the two
    moments and the per-tensor initialized flags."""
    var p: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var flags: List[Bool]

    def __init__(out self, var p: List[Float32], var m: List[Float32], var v: List[Float32], var flags: List[Bool]):
        self.p = p^
        self.m = m^
        self.v = v^
        self.flags = flags^


def byte_open_state(addr: List[Int], cfg_shape: ByteConfig, completed: Int) raises -> ByteOpenState:
    """The open doors' state admission: the slot table, the four reads and
    `byte_validate_state`, in that order."""
    var cells: List[Int] = [cfg_shape.n_total(), cfg_shape.n_total(), cfg_shape.n_total(), cfg_shape.n_tensors()]
    _validate_slot_table(addr, cells, 4)
    var p = _read_f32(addr[0], cells[0])
    var m = _read_f32(addr[1], cells[1])
    var v = _read_f32(addr[2], cells[2])
    var flags = _read_flags(addr[3], cells[3])
    byte_validate_state(p, m, v, flags, completed, cfg_shape)
    return ByteOpenState(p^, m^, v^, flags^)


def byte_parallel_export_run(mut owner: ByteParallelTrainer, addr: List[Int], rank: Int, gradients: Bool) raises -> Int:
    """The replica export after the door admitted the rank and the address
    list: returns the completed step count."""
    ref tr = owner.trainers[rank]
    ref ctx = owner.contexts[rank]
    var n = tr.config.n_total()
    var count = 1 if gradients else 4
    var cells = List[Int]()
    for i in range(count):  # driver-loop: over count: one or four exported arrays: per array cell counts, not data
        cells.append(tr.config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr, cells, 0)
    tr.validate_device_state(ctx, tr.completed_steps)
    if owner.pool_optimizer:
        for i in range(len(owner.trainers)):  # driver-loop: over replica trainers
            owner.trainers[i].validate_device_state(owner.contexts[i], tr.completed_steps)
    # DEVIATION 3120: every array goes from the device straight into the
    # caller's memory (`download_f32_into`, one transfer each): no pinned
    # host buffer, no element loop into a List, no second host copy. The
    # pooled m/v land at each owner's `optimizer_first`, which is where the
    # old List concatenation put them (owners are contiguous, in order,
    # and cover [0, n); checked below before any copy). On a raise the
    # caller's arrays hold an unspecified partial export.
    if gradients:
        if tr.grad_step != tr.completed_steps:
            raise Error("byte LM parallel: no committed gradient")
        download_f32_into(ctx, tr.buffers.grad, n, f32_ptr(addr[0]))
    else:
        if owner.pool_optimizer:
            var at = 0
            for i in range(len(owner.trainers)):  # small-loop(trainers: one trainer per replica device): checks range offsets per replica, not data
                if owner.trainers[i].buffers.optimizer_first != at:
                    raise Error("byte LM parallel: pooled optimizer ranges are not contiguous")
                at += owner.trainers[i].buffers.optimizer_count
            if at != n:
                raise Error("byte LM parallel: pooled optimizer ranges do not cover the model")
        download_f32_into(ctx, tr.buffers.param, n, f32_ptr(addr[0]))
        if owner.pool_optimizer:
            for i in range(len(owner.trainers)):  # driver-loop: over replica trainers, one download each
                var first = owner.trainers[i].buffers.optimizer_first
                var owned = owner.trainers[i].buffers.optimizer_count
                download_f32_into(owner.contexts[i], owner.trainers[i].buffers.m_state, owned,
                                  f32_ptr(addr[1] + 4 * first))
                download_f32_into(owner.contexts[i], owner.trainers[i].buffers.v_state, owned,
                                  f32_ptr(addr[2] + 4 * first))
        else:
            download_f32_into(ctx, tr.buffers.m_state, n, f32_ptr(addr[1]))
            download_f32_into(ctx, tr.buffers.v_state, n, f32_ptr(addr[2]))
        _write_flags(addr[3], tr.buffers.buf_initialized)
    return tr.completed_steps


def byte_model_pool_export_run(mut owner: ByteModelPool, addr: List[Int], gradients: Bool) raises -> Int:
    """The layer-pool canonical export after the door admitted the rank and
    the address list: returns the completed step count."""
    var n = owner.config.n_total()
    var count = 1 if gradients else 4
    var cells = List[Int]()
    for i in range(count):  # driver-loop: over count: one or four exported arrays: per array slot sizes, no data
        cells.append(owner.config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr,cells,0)
    # N156: each array goes straight into the caller's memory (export_into);
    # every chunk is validated before the first copy.
    if gradients:
        owner.export_into([3], [addr[0]])
    else:
        owner.export_into([0, 1, 2], [addr[0], addr[1], addr[2]])
        _write_flags(addr[3],owner.flags)
    return owner.completed


def byte_offload_export_run(mut owner: ByteOffloadedReplay, addr: List[Int], gradients: Bool) raises -> Int:
    """The offloaded replay's canonical export after the door admitted the
    rank and the address list: returns the completed step count."""
    var n = owner.config.n_total()
    var count = 1 if gradients else 4
    var cells = List[Int]()
    for i in range(count):  # driver-loop: over count: one or four exported arrays: per array slot sizes, no data
        cells.append(owner.config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr,cells,0)
    # N156: each array goes straight into the caller's memory (export_into);
    # every chunk is validated before the first copy.
    if gradients:
        owner.export_into([3], [addr[0]])
    else:
        owner.export_into([0, 1, 2], [addr[0], addr[1], addr[2]])
        _write_flags(addr[3],owner.flags)
    return owner.completed


# ---------------------------------------------------------------------------
# Private byte-LM schedule descriptor and factory (Program pooled/offload lane,
# 2026-10-28). One compiled table drives the replica, layer-pool and offload
# sessions above with integer descriptors only: the tokenizer-host owner
# (bindings/parallel_byte_native.mojo) retains this image's provider, the
# session value minted by byte_lm_schedule_create_v1 and its handle, and never
# selects per-instance Python callables. The numerical sessions are unchanged.
# ---------------------------------------------------------------------------
comptime BYTE_LM_SCHEDULE_ABI_NAME = "mojolearn.training.byte_lm_schedule.v1"
comptime BYTE_LM_SCHEDULE_OPERATION_COUNT = 15
comptime BYTE_SCHEDULE_REPLICA = 0
comptime BYTE_SCHEDULE_MODEL_POOL = 1
comptime BYTE_SCHEDULE_OFFLOAD = 2
comptime BYTE_SCHEDULE_MAGIC = 0x42594C4D53434844
comptime BYTE_SCHEDULE_CLOSE = 0
comptime BYTE_SCHEDULE_OPEN = 1
comptime BYTE_SCHEDULE_STEP = 2
comptime BYTE_SCHEDULE_EXPORT = 3
comptime BYTE_SCHEDULE_ROLLBACK = 4
comptime BYTE_SCHEDULE_OWNERSHIP = 5
comptime BYTE_SCHEDULE_SET_LR = 6
comptime BYTE_SCHEDULE_SHARD_GRADIENT = 7
comptime BYTE_SCHEDULE_APPLY_GRADIENT = 8
comptime BYTE_SCHEDULE_FOLD_RESET = 9
comptime BYTE_SCHEDULE_SHARD_GRADIENT_FOLD = 10
comptime BYTE_SCHEDULE_FOLD_ADD = 11
comptime BYTE_SCHEDULE_FOLD_EXPORT = 12
comptime BYTE_SCHEDULE_KIND = 13
comptime BYTE_SCHEDULE_CAPABILITY = 14


struct ByteScheduleSession(Movable, Writable):
    """The one session value of a byte-LM schedule. Construction is lazy:
    no device work until OPEN. Only `kind`'s member is ever opened."""
    var magic: Int
    var kind: Int
    var replica: ByteParallelTrainer
    var pool: ByteModelPool
    var offload: ByteOffloadedReplay

    def __init__(out self, kind: Int):
        self.magic = BYTE_SCHEDULE_MAGIC
        self.kind = kind
        self.replica = ByteParallelTrainer()
        self.pool = ByteModelPool()
        self.offload = ByteOffloadedReplay()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteScheduleSession(kind=", self.kind, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _byte_schedule_kind(value: Int) raises -> Int:
    if value != BYTE_SCHEDULE_REPLICA and value != BYTE_SCHEDULE_MODEL_POOL and value != BYTE_SCHEDULE_OFFLOAD:
        raise Error("byte LM schedule: unknown schedule kind")
    return value


def _byte_schedule_capability(kind: Int) -> Int:
    # Bit i set: operation i is implemented for this schedule. The replica
    # session has every operation; the layer pool has no split-step/fold
    # door; the offloaded replay also has no resident optimizer ownership.
    if kind == BYTE_SCHEDULE_REPLICA:
        return (1 << BYTE_LM_SCHEDULE_OPERATION_COUNT) - 1
    var common = (1 << BYTE_SCHEDULE_CLOSE) | (1 << BYTE_SCHEDULE_OPEN) | (1 << BYTE_SCHEDULE_STEP) | (
        1 << BYTE_SCHEDULE_EXPORT) | (1 << BYTE_SCHEDULE_ROLLBACK) | (1 << BYTE_SCHEDULE_KIND) | (1 << BYTE_SCHEDULE_CAPABILITY)
    if kind == BYTE_SCHEDULE_MODEL_POOL:
        return common | (1 << BYTE_SCHEDULE_OWNERSHIP)
    return common


def _byte_schedule_session(address: Int) raises -> MutPointer[ByteScheduleSession, MutUntrackedOrigin]:
    if address == 0:
        raise Error("byte LM schedule: missing admitted session")
    var session = MutPointer[ByteScheduleSession, MutUntrackedOrigin](unsafe_from_address=address)
    if session[].magic != BYTE_SCHEDULE_MAGIC:
        raise Error("byte LM schedule: handle is not a schedule session")
    _ = _byte_schedule_kind(session[].kind)
    return session


def _byte_schedule_shards(addresses: List[Int], first: Int, count: Int, shape: ByteConfig) raises -> List[List[Int32]]:
    if count < 1 or len(addresses) != first + count:
        raise Error("byte LM parallel: logical shard count mismatch")
    var n_ids = shape.batch * (shape.length + 1)
    var slots = List[Int]()
    var cells = List[Int]()
    for i in range(count):  # driver-loop: over count: logical shards of the step: per shard slot sizes, no data
        slots.append(addresses[first + i])
        cells.append(n_ids)
    _validate_slot_table(slots, cells, count)
    var shards = List[List[Int32]]()
    for i in range(count):  # driver-loop: over count: logical shards of the step: one memcpy of each shard's ids, the device checks them
        shards.append(_read_ids(slots[i], n_ids))
    return shards^


def _byte_schedule_losses(address: Int, losses: List[Float32], count: Int) raises -> Int:
    if len(losses) != count:
        raise Error("RuntimeError: parallel step returned wrong loss count")
    var out = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # small-loop(count: one loss per logical shard): copies shard losses, no data
        out.store(i, Float64(losses[i]))
    return count


def _byte_schedule_open(mut session: ByteScheduleSession, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    # a = [handle, parameters, m, v, flags]; p = [completed, kind, nesterov,
    # pooled, n_dims, dims..., shards, n_devices, devices...]; f = the eight
    # optimizer scalars (lr, beta1, beta2, eps, weight_decay, momentum,
    # dampening, max_norm).
    _require_binding_profile()
    if len(a) != 5 or len(f) != 8 or len(p) < 6:
        raise Error("byte LM schedule: invalid open descriptor")
    var completed = p[0]
    if completed < 0 or completed >= 1000000:
        raise Error("byte LM: completed step outside admitted bound")
    if p[2] != 0 and p[2] != 1 or p[3] != 0 and p[3] != 1:
        raise Error("byte LM: action/nesterov must be 0 or 1")
    var n_dims = p[4]
    if n_dims != 9 and n_dims != 10 or len(p) < 5 + n_dims + 2:
        raise Error("byte LM schedule: invalid shape descriptor")
    var chunked = False
    if n_dims == 10:
        if p[14] != 0 and p[14] != 1:
            raise Error("byte LM: chunked head selector must be 0 or 1")
        chunked = p[14] == 1
    var shape = ByteConfig(p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], chunked)
    shape.validate()
    var at = 5 + n_dims
    var shards = p[at]
    var n_devices = p[at + 1]
    if n_devices < 1 or len(p) != at + 2 + n_devices:
        raise Error("byte LM schedule: invalid device descriptor")
    var device_ids = List[Int]()
    for i in range(n_devices):  # driver-loop: over n_devices: GPU device ordinals of the session: ids, not data
        device_ids.append(p[at + 2 + i])
    var cfg = OptimizerConfig(p[1], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]),
        Float32(f[5]), Float32(f[6]), p[2] == 1, Float32(f[7]))
    byte_validate_optimizer(cfg)
    var addr: List[Int] = [a[1], a[2], a[3], a[4]]
    var cells: List[Int] = [shape.n_total(), shape.n_total(), shape.n_total(), shape.n_tensors()]
    _validate_slot_table(addr, cells, 4)
    var params = _read_f32(addr[0], cells[0])
    var m = _read_f32(addr[1], cells[1])
    var v = _read_f32(addr[2], cells[2])
    var flags = _read_flags(addr[3], cells[3])
    byte_validate_state(params, m, v, flags, completed, shape)
    # Retain the GIL: it is also the native object's exclusion lock.
    if session.kind == BYTE_SCHEDULE_REPLICA:
        session.replica.open(device_ids, shards, params, m, v, flags, completed, cfg, shape, p[3] == 1)
    elif session.kind == BYTE_SCHEDULE_MODEL_POOL:
        if p[3] != 0:
            raise Error("byte LM schedule: optimizer pooling belongs to the replica schedule")
        session.pool.open(device_ids, shards, params, m, v, flags, completed, cfg, shape)
    else:
        if p[3] != 0:
            raise Error("byte LM schedule: optimizer pooling belongs to the replica schedule")
        session.offload.open(device_ids, shards, params, m, v, flags, completed, cfg, shape)
    return completed


def _byte_schedule_export_canonical(mut session: ByteScheduleSession, a: List[Int], gradients: Bool) raises -> Int:
    # Layer-pool and offload export: the canonical state, no replica rank.
    var config = session.pool.config.copy() if session.kind == BYTE_SCHEDULE_MODEL_POOL else session.offload.config.copy()
    var n = config.n_total()
    var count = 1 if gradients else 4
    if len(a) != 1 + count:
        raise Error("byte LM schedule: invalid export descriptor")
    var addr = List[Int]()
    var cells = List[Int]()
    for i in range(count):  # driver-loop: over count: one or four exported arrays: per array slot sizes, no data
        addr.append(a[1 + i])
        cells.append(config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr, cells, 0)
    # N156: each array goes straight into the caller's memory (export_into).
    if session.kind == BYTE_SCHEDULE_MODEL_POOL:
        if gradients:
            session.pool.export_into([3], [addr[0]])
        else:
            session.pool.export_into([0, 1, 2], [addr[0], addr[1], addr[2]])
            _write_flags(addr[3], session.pool.flags)
        return session.pool.completed
    if gradients:
        session.offload.export_into([3], [addr[0]])
    else:
        session.offload.export_into([0, 1, 2], [addr[0], addr[1], addr[2]])
        _write_flags(addr[3], session.offload.flags)
    return session.offload.completed


def _byte_schedule_export_replica(mut owner: ByteParallelTrainer, a: List[Int], rank: Int, gradients: Bool) raises -> Int:
    # The replica export of byte_lm_parallel_export_binding, over descriptors.
    if rank < 0 or rank >= len(owner.trainers):
        raise Error("byte LM parallel: invalid replica index")
    ref tr = owner.trainers[rank]
    ref ctx = owner.contexts[rank]
    var n = tr.config.n_total()
    var count = 1 if gradients else 4
    if len(a) != 1 + count:
        raise Error("byte LM schedule: invalid export descriptor")
    var addr = List[Int]()
    var cells = List[Int]()
    for i in range(count):  # driver-loop: over count: one or four exported arrays: per array cell counts, not data
        addr.append(a[1 + i])
        cells.append(tr.config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr, cells, 0)
    tr.validate_device_state(ctx, tr.completed_steps)
    if owner.pool_optimizer:
        for i in range(len(owner.trainers)):  # driver-loop: over replica trainers
            owner.trainers[i].validate_device_state(owner.contexts[i], tr.completed_steps)
    if gradients:
        if tr.grad_step != tr.completed_steps:
            raise Error("byte LM parallel: no committed gradient")
        download_f32_into(ctx, tr.buffers.grad, n, f32_ptr(addr[0]))
        return tr.completed_steps
    if owner.pool_optimizer:
        var at = 0
        for i in range(len(owner.trainers)):  # small-loop(trainers: one trainer per replica device): checks range offsets per replica, not data
            if owner.trainers[i].buffers.optimizer_first != at:
                raise Error("byte LM parallel: pooled optimizer ranges are not contiguous")
            at += owner.trainers[i].buffers.optimizer_count
        if at != n:
            raise Error("byte LM parallel: pooled optimizer ranges do not cover the model")
    download_f32_into(ctx, tr.buffers.param, n, f32_ptr(addr[0]))
    if owner.pool_optimizer:
        for i in range(len(owner.trainers)):  # driver-loop: over replica trainers, one download each
            var first = owner.trainers[i].buffers.optimizer_first
            var owned = owner.trainers[i].buffers.optimizer_count
            download_f32_into(owner.contexts[i], owner.trainers[i].buffers.m_state, owned, f32_ptr(addr[1] + 4 * first))
            download_f32_into(owner.contexts[i], owner.trainers[i].buffers.v_state, owned, f32_ptr(addr[2] + 4 * first))
    else:
        download_f32_into(ctx, tr.buffers.m_state, n, f32_ptr(addr[1]))
        download_f32_into(ctx, tr.buffers.v_state, n, f32_ptr(addr[2]))
    _write_flags(addr[3], tr.buffers.buf_initialized)
    return tr.completed_steps


def _byte_schedule_ownership(mut session: ByteScheduleSession, a: List[Int], p: List[Int]) raises -> Int:
    # a = [handle, out Int words]; p = [capacity]. Replica rows are five words
    # (first, count, moment, rollback, reduction bytes); layer-pool rows are
    # seven (owner index, first, count, parameter, moment, rollback, gradient bytes).
    if len(a) != 2 or a[1] == 0 or len(p) != 1:
        raise Error("byte LM schedule: invalid ownership descriptor")
    var words = List[Int]()
    if session.kind == BYTE_SCHEDULE_REPLICA:
        ref owner = session.replica
        owner.require_open()
        for i in range(len(owner.trainers)):  # driver-loop: over trainers: one trainer per replica device: per replica ranges, no data
            ref b = owner.trainers[i].buffers
            words.append(b.optimizer_first)
            words.append(b.optimizer_count)
            words.append(4 * (len(b.m_state) + len(b.v_state)))
            words.append(4 * (len(b.shadow_p) + len(b.shadow_m) + len(b.shadow_v)))
            var reduction_bytes = 0
            if owner.pool_optimizer:
                reduction_bytes = 4 * (len(owner.pool_totals[i]) + len(owner.pool_incoming[i]))
            elif i == 0:
                reduction_bytes = 4 * len(owner.total.value())
            words.append(reduction_bytes)
    elif session.kind == BYTE_SCHEDULE_MODEL_POOL:
        session.pool.require_open()
        for i in range(len(session.pool.chunks)):  # driver-loop: over chunks: one model chunk per owner: per chunk ranges, no data
            ref chunk = session.pool.chunks[i]
            words.append(chunk.owner)
            words.append(chunk.first)
            words.append(len(chunk.p))
            words.append(4 * len(chunk.p))
            words.append(4 * (len(chunk.m) + len(chunk.v)))
            words.append(4 * (len(chunk.shadow_p) + len(chunk.shadow_m) + len(chunk.shadow_v)))
            words.append(4 * len(chunk.g))
    else:
        raise Error("NotImplementedError: offloaded state lives on the host; no resident GPU optimizer owners")
    if len(words) > p[0]:
        raise Error("byte LM schedule: ownership output is too short")
    var out = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[1])
    for i in range(len(words)):  # small-loop(words: captured ownership record words): copies record words, no data
        out[i] = words[i]
    return len(words)


def _byte_schedule_require(session: ByteScheduleSession, op: Int) raises:
    if (_byte_schedule_capability(session.kind) >> op) & 1 == 0:
        raise Error("NotImplementedError: byte-LM schedule has no operation " + String(op))


def byte_lm_schedule_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == BYTE_SCHEDULE_CAPABILITY:
        if len(a) != 0 or len(p) != 1 or len(f) != 0:
            raise Error("byte LM schedule: capability takes one schedule kind")
        return _byte_schedule_capability(_byte_schedule_kind(p[0]))
    if len(a) < 1:
        raise Error("byte LM schedule: missing admitted session")
    var session = _byte_schedule_session(a[0])
    _byte_schedule_require(session[], op)
    var kind = session[].kind
    if op == BYTE_SCHEDULE_KIND:
        return kind
    if op == BYTE_SCHEDULE_CLOSE:
        if kind == BYTE_SCHEDULE_REPLICA:
            session[].replica.close()
        elif kind == BYTE_SCHEDULE_MODEL_POOL:
            session[].pool.close()
        else:
            session[].offload.close()
        return 0
    if op == BYTE_SCHEDULE_OPEN:
        return _byte_schedule_open(session[], a, p, f)
    if op == BYTE_SCHEDULE_ROLLBACK:
        if kind == BYTE_SCHEDULE_REPLICA:
            session[].replica.require_open()
            session[].replica.rollback()
            return session[].replica.trainers[0].completed_steps
        if kind == BYTE_SCHEDULE_MODEL_POOL:
            session[].pool.require_open()
            session[].pool.rollback()
            return session[].pool.completed
        session[].offload.require_open()
        session[].offload.rollback()
        return session[].offload.completed
    if op == BYTE_SCHEDULE_OWNERSHIP:
        return _byte_schedule_ownership(session[], a, p)
    _require_binding_profile()
    if op == BYTE_SCHEDULE_STEP:
        # a = [handle, losses float64[count], shard ids...]; p = [completed, count].
        if len(p) != 2 or len(a) < 3 or a[1] == 0:
            raise Error("byte LM schedule: invalid step descriptor")
        var completed = p[0]
        var count = p[1]
        if kind == BYTE_SCHEDULE_REPLICA:
            ref owner = session[].replica
            owner.require_open()
            if completed != owner.trainers[0].completed_steps:
                raise Error("byte LM parallel: completed-step mismatch")
            if count != owner.logical_shards:
                raise Error("logical shard count mismatch")
            var losses = owner.step(_byte_schedule_shards(a, 2, count, owner.trainers[0].config))
            try:
                return _byte_schedule_losses(a[1], losses, count)
            except error:
                owner.rollback()
                raise error
        if kind == BYTE_SCHEDULE_MODEL_POOL:
            ref owner = session[].pool
            owner.require_open()
            if completed != owner.completed:
                raise Error("byte LM parallel: completed-step mismatch")
            if count != owner.logical_shards:
                raise Error("logical shard count mismatch")
            var losses = owner.step(_byte_schedule_shards(a, 2, count, owner.config))
            try:
                return _byte_schedule_losses(a[1], losses, count)
            except error:
                owner.rollback()
                raise error
        ref owner = session[].offload
        owner.require_open()
        if completed != owner.completed:
            raise Error("byte LM parallel: completed-step mismatch")
        if count != owner.logical_shards:
            raise Error("logical shard count mismatch")
        var losses = owner.step(_byte_schedule_shards(a, 2, count, owner.config))
        try:
            return _byte_schedule_losses(a[1], losses, count)
        except error:
            owner.rollback()
            raise error
    if op == BYTE_SCHEDULE_EXPORT:
        # p = [rank, gradients]
        if len(p) != 2 or p[1] != 0 and p[1] != 1:
            raise Error("byte LM schedule: invalid export descriptor")
        if kind == BYTE_SCHEDULE_REPLICA:
            session[].replica.require_open()
            return _byte_schedule_export_replica(session[].replica, a, p[0], p[1] == 1)
        if kind == BYTE_SCHEDULE_MODEL_POOL:
            session[].pool.require_open()
        else:
            session[].offload.require_open()
        if p[0] != 0:
            raise Error("byte model pool: canonical export has no replica rank; use rank=0")
        return _byte_schedule_export_canonical(session[], a, p[1] == 1)
    # The remaining operations are the replica's split-step and fold doors;
    # the capability check above already refused them on other schedules.
    ref owner = session[].replica
    owner.require_open()
    var n = owner.trainers[0].config.n_total()
    var n_ids = owner.trainers[0].config.batch * (owner.trainers[0].config.length + 1)
    if op == BYTE_SCHEDULE_SET_LR:
        if len(a) != 1 or len(f) != 1:
            raise Error("byte LM schedule: invalid learning-rate descriptor")
        owner.set_learning_rate(Float32(f[0]))
        return Int(bitcast[DType.uint32](owner.trainers[0].optimizer.lr))
    if op == BYTE_SCHEDULE_SHARD_GRADIENT or op == BYTE_SCHEDULE_SHARD_GRADIENT_FOLD:
        # a = [handle, ids, (gradient out,) loss float64 out]; p = [completed].
        var fold = op == BYTE_SCHEDULE_SHARD_GRADIENT_FOLD
        if len(p) != 1 or len(a) != (3 if fold else 4) or a[len(a) - 1] == 0:
            raise Error("byte LM schedule: invalid shard-gradient descriptor")
        if p[0] != owner.trainers[0].completed_steps:
            raise Error("byte LM parallel: completed-step mismatch")
        var slots: List[Int] = [a[1]]
        var cells: List[Int] = [n_ids]
        if not fold:
            slots.append(a[2])
            cells.append(n)
        _validate_slot_table(slots, cells, 1)
        var loss = Float32(0)
        if fold:
            loss = owner.shard_gradient_fold(_read_ids(a[1], n_ids))
        else:
            loss = owner.shard_gradient(_read_ids(a[1], n_ids))
            download_f32_into(owner.contexts[0], owner.trainers[0].buffers.grad, n, f32_ptr(a[2]))  # DEVIATION 3120
        MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[len(a) - 1])[0] = Float64(loss)
        return 0
    if op == BYTE_SCHEDULE_APPLY_GRADIENT:
        if len(a) != 2 or len(p) != 1:
            raise Error("byte LM schedule: invalid apply-gradient descriptor")
        if p[0] != owner.trainers[0].completed_steps:
            raise Error("byte LM parallel: completed-step mismatch")
        var slots: List[Int] = [a[1]]
        var cells: List[Int] = [n]
        _validate_slot_table(slots, cells, 1)
        owner.apply_gradient(f32_ptr(a[1]))
        return owner.trainers[0].completed_steps
    if op == BYTE_SCHEDULE_FOLD_RESET:
        if len(a) == 1:
            owner.fold_clear()
            return 0
        if len(a) != 2:
            raise Error("byte LM schedule: invalid fold descriptor")
        var slots: List[Int] = [a[1]]
        var cells: List[Int] = [n]
        _validate_slot_table(slots, cells, 1)
        owner.fold_reset(f32_ptr(a[1]))
        return 1
    if op == BYTE_SCHEDULE_FOLD_ADD or op == BYTE_SCHEDULE_FOLD_EXPORT:
        if len(a) != 2:
            raise Error("byte LM schedule: invalid fold descriptor")
        var slots: List[Int] = [a[1]]
        var cells: List[Int] = [n]
        _validate_slot_table(slots, cells, 1 if op == BYTE_SCHEDULE_FOLD_ADD else 0)
        if op == BYTE_SCHEDULE_FOLD_ADD:
            owner.fold_add(f32_ptr(a[1]))
            return 0
        owner.fold_export_into(f32_ptr(a[1]))  # DEVIATION 3120
        return n
    raise Error("byte LM schedule: unknown operation")


def byte_lm_schedule_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                        result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= BYTE_LM_SCHEDULE_OPERATION_COUNT:
            raise Error("byte LM schedule: unknown operation")
        if result == 0 or (nf > 0 and ff == 0):
            raise Error("byte LM schedule: null scalar or result span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # driver-loop: over nf: at most the eight optimizer scalars: reads scalars, no data
            f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = byte_lm_schedule_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ByteLmScheduleTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(BYTE_LM_SCHEDULE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(byte_lm_schedule_v1)


comptime BYTE_LM_SCHEDULE_TABLE = SharedSlot[ByteLmScheduleTable, ("MojolearnByteLmScheduleTableIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnByteLmScheduleTableFastV3")]
