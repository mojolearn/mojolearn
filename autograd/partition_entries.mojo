# SPDX-License-Identifier: Apache-2.0
# Native lifetime table and typed entries for model/tensor/pipeline
# partitioned captured graphs. The Python door (partition_pyapi.mojo)
# converts arrays and options and builds results.
from std.ffi import _Global
from std.memory import alloc, memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.ops_api import ops_session, _OpsStore
from autograd.partition_plan import GraphBytes, partition_plan, column_bytes
from autograd.partition_runtime import PartitionedProgram, PartitionEvent
from autograd.partition_stage import stage_input_bytes
from autograd.partition_checkpoint import partition_checkpoint, partition_from_checkpoint, restore_partition_state
from autograd.distributed_entries import distributed_optimizer_config
from autograd.program_io import HostArray
from autograd.rankn import ShapeN
from autograd.optim import optimizer_slots
from core.cast_in import CastIn
from training.checkpoint import read_checkpoint_unchecked, save_checkpoint
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _PARTITION_NAME = "MojoPartitionedGraphsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoPartitionedGraphsFast"
comptime _PARTITIONS = _Global[StorageType=_OpsStore, name=_PARTITION_NAME, init_fn=_OpsStore.__init__]


def register_partition[E: Exec & Movable & Deinitable, KIND: Int](var d: PartitionedProgram[E]) raises -> Int:
    var p = alloc[PartitionedProgram[E]](1)
    p.init_pointee_move(d^)
    var s = _PARTITIONS.get_or_create_ptr()
    var id = s[].next_id
    s[].next_id += 1
    s[].ids.append(id)
    s[].addr.append(Int(p))
    s[].kind.append(KIND)
    return id


def partition_slot(id: Int, kind: Int) raises -> Int:
    var s = _PARTITIONS.get_or_create_ptr()
    for k in range(len(s[].ids)):  # driver-loop: live native partition handles
        if s[].ids[k] == id and s[].kind[k] == kind:
            return k
    raise Error("partitioned program is closed or belongs to another binding")


def partition[E: Exec & Movable & Deinitable, KIND: Int](id: Int) raises -> UnsafePointer[PartitionedProgram[E], MutUntrackedOrigin]:
    var slot = partition_slot(id, KIND)
    return UnsafePointer[PartitionedProgram[E], MutUntrackedOrigin](unsafe_from_address=_PARTITIONS.get_or_create_ptr()[].addr[slot])


def partition_open_native[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, devices: List[Int], stages: Int,
    tensor_shards: Int, microbatches: Int, optimizer: Int,
    parameters: List[Int], seed_int: Int, var executors: List[E],
) raises -> Int:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("partitioned programs require identical mode")
    var s = ops_session[E, KIND](sid)
    var p = s[].program(pid)
    var oi = s[]._opt_index(optimizer)
    var o = s[].opts[oi]
    var ip: List[Int] = [o.cfg.kind, 1 if o.cfg.nesterov else 0, o.sched, o.sched_step]
    var fp: List[Float32] = [o.cfg.lr, o.cfg.momentum, o.cfg.beta1, o.cfg.beta2, o.cfg.eps, o.cfg.weight_decay, o.gamma]
    var values = List[GraphBytes]()
    var trainable = List[Bool](length=len(p.params), fill=False)
    for k in range(len(parameters)):  # driver-loop: selected trainable source handles
        var found = False
        var h = parameters[k]
        for j in range(len(p.params)):  # driver-loop: captured parameter handles
            if p.params[j] == h:
                if trainable[j]:
                    raise Error("partitioned program: duplicate trainable parameter")
                trainable[j] = True
                found = True
        if not found:
            raise Error("partitioned program: trainable handle is not in capture")
    for j in range(len(p.params)):  # driver-loop: canonical source parameter bytes
        var integer = s[].int_words[s[]._slot(p.params[j])]
        if len(parameters) == 0:
            trainable[j] = not integer
        if trainable[j] and integer:
            raise Error("partitioned program: integer parameters cannot be trainable")
        var words = List[Float32](length=p.param_shapes[j].count(), fill=Float32(0))
        s[].download(p.params[j], FP(unsafe_from_address=Int(words.unsafe_ptr())))
        values.append(GraphBytes(p.param_shapes[j], words^, integer))
    if seed_int < 0:
        raise Error("partitioned program: seed must be a nonnegative signed 64-bit integer")
    var plan = partition_plan(p, values, trainable, stages, tensor_shards)
    var d = PartitionedProgram[E](plan^, devices, executors^, o.cfg, ip, fp, microbatches, UInt64(seed_int))
    # Existing optimizer moments are column-packed exactly as their parameters.
    var ck = partition_checkpoint(d)
    var clocks = -1
    var slots = optimizer_slots(o.cfg)
    var scalar = 7 + Int(ck.ext_ints[10])
    for j in range(len(d.plan.parameters)):  # driver-loop: complete source optimizer migration
        if not d.plan.trainable[j]:
            continue
        var original = d.plan.origins[j]
        var si = -1
        for k in range(len(s[].opt_states)):  # driver-loop: optimizer state lookup
            if s[].opt_states[k].opt == oi and s[].opt_states[k].param == p.params[original]:
                si = k
        var t = s[].opt_states[si].t if si >= 0 else 0
        if clocks >= 0 and t != clocks:
            raise Error("partitioned program: selected parameters have different optimizer clocks")
        clocks = t
        if si < 0:
            continue
        for which in range(2):  # small-loop(2: first and second optimizer moment): byte migration
            if (which == 0 and not slots[0]) or (which == 1 and not slots[1]):
                continue
            var sh = p.param_shapes[original]
            var words = List[Float32](length=sh.count(), fill=Float32(0))
            var ptr = s[].opt_states[si].s1 if which == 0 else s[].opt_states[si].s2
            s[].ex.download(FP(unsafe_from_address=Int(words.unsafe_ptr())), ptr, sh.count())
            var state = GraphBytes(sh, words^)
            if d.plan.column_width[j] > 0:
                state = column_bytes(state, d.plan.column_first[j], d.plan.column_width[j])
            var dst = FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) if which == 0 else FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr()))
            memcpy(dest=dst + ck.offsets[j], src=state.words.unsafe_ptr(), count=state.shape.count())
        ck.ext_floats[scalar + 3 * j] = s[].opt_states[si].pw1
        ck.ext_floats[scalar + 3 * j + 1] = s[].opt_states[si].pw2
        ck.ext_floats[scalar + 3 * j + 2] = s[].opt_states[si].mu_prod
    ck.t = max(clocks, 0)
    ck.ext_ints[4] = Int64(o.epoch)
    ck.ext_ints[5] = Int64(ck.t)
    restore_partition_state(d, ck)
    return register_partition[E, KIND](d^)


def partition_load_native[E: Exec & Movable & Deinitable, KIND: Int](
    file: String, devices: List[Int], var executors: List[E],
) raises -> Int:
    var ck = read_checkpoint_unchecked(file)
    var plan = partition_from_checkpoint(ck)
    var ip = List[Int]()
    var fp = List[Float32]()
    for k in range(4):  # small-loop(4: optimizer option integers): metadata
        ip.append(Int(ck.ext_ints[12 + k]))
    for k in range(7):  # small-loop(7: optimizer option floats): metadata
        fp.append(ck.ext_floats[k])
    var cfg = distributed_optimizer_config(ip, fp)
    var d = PartitionedProgram[E](plan^, devices, executors^, cfg, ip, fp, Int(ck.ext_ints[2]), ck.seed)
    restore_partition_state(d, ck)
    return register_partition[E, KIND](d^)


def partition_run_native[E: Exec & Movable & Deinitable, KIND: Int](
    id: Int, srcs: List[CastIn], shapes: List[ShapeN],
) raises -> GraphBytes:
    var d = partition[E, KIND](id)
    d[].ready()
    if len(srcs) != d[].plan.program.n_in:
        raise Error("partitioned run: input count mismatch")
    var inputs = List[GraphBytes]()
    for k in range(len(srcs)):  # driver-loop: native dtype conversion and byte staging
        var integer = d[].plan.program.in_kinds[k] != 0
        inputs.append(stage_input_bytes(d[].workers[0].session.ex, srcs[k], shapes[k], integer))
    return d[].run(inputs)


def partition_save_native[E: Exec & Movable & Deinitable, KIND: Int](id: Int, file: String) raises -> Int:
    var d = partition[E, KIND](id)
    var ck = partition_checkpoint(d[])
    return save_checkpoint(file, ck).bytes


struct PartitionState(Movable):
    """The parameters, first and second moments (in that order in
    `arrays`), and the counters of a partitioned program."""
    var arrays: List[List[HostArray]]
    var epoch: Int
    var step: Int
    var seed: UInt64
    var stages: Int
    var tensor_shards: Int
    var microbatches: Int

    def __init__(out self):
        self.arrays = List[List[HostArray]]()
        self.epoch = 0
        self.step = 0
        self.seed = 0
        self.stages = 0
        self.tensor_shards = 0
        self.microbatches = 0


def partition_state_native[E: Exec & Movable & Deinitable, KIND: Int](id: Int) raises -> PartitionState:
    var d = partition[E, KIND](id)
    var ck = partition_checkpoint(d[])
    var out = PartitionState()
    for which in range(3):  # small-loop(3: parameter and two optimizer arrays): native ABI outputs
        var arrays = List[HostArray]()
        for j in range(len(d[].plan.parameters)):  # driver-loop: transformed parameter tensors
            var sh = d[].plan.parameters[j].shape
            var a = HostArray(sh)
            var ptr = FP(unsafe_from_address=Int(ck.param.unsafe_ptr()))
            if which == 1:
                ptr = FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr()))
            elif which == 2:
                ptr = FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr()))
            memcpy(dest=a.data.unsafe_ptr(), src=ptr + ck.offsets[j], count=sh.count())
            arrays.append(a^)
        out.arrays.append(arrays^)
    out.epoch = d[].epoch
    out.step = d[].step
    out.seed = d[].seed
    out.stages = d[].plan.stages
    out.tensor_shards = d[].plan.tensor_shards
    out.microbatches = d[].microbatches
    return out^


struct PartitionResidency(Movable):
    var devices: List[Int]
    var counts: List[Int]
    var owners: List[Int]
    var placements: List[Int]
    var stages: Int
    var tensor_shards: Int
    var microbatches: Int
    var partitioned_products: Int

    def __init__(out self):
        self.devices = List[Int]()
        self.counts = List[Int]()
        self.owners = List[Int]()
        self.placements = List[Int]()
        self.stages = 0
        self.tensor_shards = 0
        self.microbatches = 0
        self.partitioned_products = 0


def partition_residency_native[E: Exec & Movable & Deinitable, KIND: Int](id: Int) raises -> PartitionResidency:
    var d = partition[E, KIND](id)
    d[].ready()
    var out = PartitionResidency()
    for rank in range(len(d[].workers)):  # driver-loop: actual device-resident parameter words
        out.devices.append(d[].devices[rank])
        var count = 0
        for j in range(len(d[].plan.parameters)):  # driver-loop: authoritative owned parameters
            if d[].workers[rank].owns[j]:
                count += d[].plan.parameters[j].shape.count()
        out.counts.append(count)
    for j in range(len(d[].plan.parameters)):  # driver-loop: unique parameter owners
        out.owners.append(d[].devices[d[].plan.owners[j] % len(d[].workers)])
    for j in range(d[].plan.program.n_nodes()):  # driver-loop: execution plan witness
        out.placements.append(d[].devices[d[].plan.placement[j] % len(d[].workers)])
    out.stages = d[].plan.stages
    out.tensor_shards = d[].plan.tensor_shards
    out.microbatches = d[].microbatches
    out.partitioned_products = d[].plan.partitioned_products
    return out^


def partition_events[E: Exec & Movable & Deinitable, KIND: Int](id: Int) raises -> List[PartitionEvent]:
    var d = partition[E, KIND](id)
    d[].ready()
    return d[].events.copy()


def partition_ready[E: Exec & Movable & Deinitable, KIND: Int](id: Int) raises:
    var d = partition[E, KIND](id)
    d[].ready()


def partition_close_native[E: Exec & Movable & Deinitable, KIND: Int](id: Int) raises:
    var table = _PARTITIONS.get_or_create_ptr()
    var slot = partition_slot(id, KIND)
    var p = partition[E, KIND](id)
    _ = table[].ids.pop(slot)
    _ = table[].addr.pop(slot)
    _ = table[].kind.pop(slot)
    try:
        p[].close()
    except err:
        p.destroy_pointee()
        p.free()
        raise err
    p.destroy_pointee()
    p.free()
